#!/usr/bin/env python3
"""Run V18 backbone inference in four geometric-TTA streams (Sec. 2.3 of
the paper) and write a single canonical-frame prediction file.

Streams (all in canonical (image_a anchor, no flip) coordinates):
  - id        : feed (image_a, image_b)
  - swap      : feed (image_b, image_a) with inverted LG matches; map back
                via (theta, r) -> (-theta, -r)
  - hflip     : feed hflipped images, x-flipped LG match field; map back via
                (theta, r) -> (-theta, +r)
  - swap+hflip: combine both; (theta, r) -> (+theta, -r)

Equal-weight blend across the four streams (heading via unit-vector mean,
range via arithmetic mean). No held-out validation file is required.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast


# Architecture and dataset utilities are imported from the user's training
# module. The release ships a clean copy of train.py with that module name.
from train import (
    CrossAttentionPoseNetV15, DualImageDatasetV15,
    safe_normalize, log_normalized_to_range,
    extract_group_json_sort_key, AMP_DTYPE,
)


class SwappedDataset(DualImageDatasetV15):
    """Returns dataset items with image_a/image_b roles swapped, and the
    LightGlue match matrix inverted accordingly. The resulting prediction
    is in B-anchor frame; mapping back is handled by the caller."""

    def __getitem__(self, i):
        json_path = self.json_paths[i]
        with open(json_path, 'rb') as f:
            data = json.load(f)
        orig_a = self._resolve_image_path(data['image_a'])
        orig_b = self._resolve_image_path(data['image_b'])
        an = Path(orig_a).stem
        bn = Path(orig_b).stem
        candidates = [
            os.path.join(self.match_dir, an, f'{bn}.npz'),
            os.path.join(self.match_dir, an, f'{bn}_matches.npz'),
            os.path.join(self.match_dir, an, f'{an}_{bn}_matches.npz'),
        ]
        npz_path = next((c for c in candidates if os.path.exists(c)), None)
        if npz_path is None:
            raise FileNotFoundError(f'Match npz not found: {candidates}')
        with np.load(npz_path) as z:
            k0, k1, m = z['keypoints0'], z['keypoints1'], z['matches']

        # Invert match mapping so that k1 acts as source (B-anchor).
        n1 = len(k1)
        m_inv = np.full(n1, -1, dtype=m.dtype)
        good = (m >= 0) & (m < n1)
        m_inv[m[good]] = np.where(good)[0]
        good_inv = (m_inv >= 0) & (m_inv < len(k0))
        src = k1[good_inv]
        dst = k0[m_inv[good_inv]]
        x_src, y_src = src[:, 0] * self.sx, src[:, 1] * self.sy
        x_dst, y_dst = dst[:, 0] * self.sx, dst[:, 1] * self.sy
        xi = np.clip(x_src, 0, 223.9999).astype(np.int32)
        yi = np.clip(y_src, 0, 223.9999).astype(np.int32)
        match_field = np.zeros((2, 224, 224), dtype=np.float32)
        match_field[0, yi, xi] = (x_dst - x_src).astype(np.float32)
        match_field[1, yi, xi] = (y_dst - y_src).astype(np.float32)
        disp_tensor = torch.from_numpy(match_field)

        nm = int(good_inv.sum())
        if nm >= 2:
            dx = (dst[:, 0] - src[:, 0]) / 640.0
            dy = (dst[:, 1] - src[:, 1]) / 480.0
            mag = np.sqrt(dx ** 2 + dy ** 2)
            ms = np.array([nm / 1000.0, mag.mean(),
                           dx.mean(), dy.mean(),
                           dx.std(), dy.std()], dtype=np.float32)
        else:
            ms = np.zeros(6, dtype=np.float32)
        ms_t = torch.from_numpy(ms)

        depth_a, sa = self._load_depth(orig_a)
        depth_b, sb = self._load_depth(orig_b)
        depth_pair = np.stack([depth_b, depth_a], axis=0)
        ds = np.concatenate([sb, sa])

        with Image.open(orig_b) as im:
            ia_t = self.DINOV2_TRANSFORM(im.convert('RGB'))
        with Image.open(orig_a) as im:
            ib_t = self.DINOV2_TRANSFORM(im.convert('RGB'))

        return (ia_t, ib_t, disp_tensor, ms_t,
                torch.from_numpy(depth_pair), torch.from_numpy(ds), json_path)


class HflipWrapper(Dataset):
    """Apply horizontal flip to the items of any base dataset."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        out = list(self.base[i])
        out[0] = torch.flip(out[0], dims=[-1])
        out[1] = torch.flip(out[1], dims=[-1])
        d = out[2].clone()
        d = torch.flip(d, dims=[2])
        d[0] = -d[0]
        out[2] = d
        ms = out[3].clone()
        ms[2] = -ms[2]
        out[3] = ms
        return tuple(out)


def build_model(weights_path, device):
    model = CrossAttentionPoseNetV15(
        dinov2_model='dinov2_vitl14',
        n_cross_layers=2, cross_heads=8, cross_dropout=0.1,
        use_spatial=False, max_correction_scale=0.02,
    )
    model.to(device)
    model = torch.nn.DataParallel(model)
    state = torch.load(weights_path, weights_only=True, map_location=device)
    res = model.load_state_dict(state, strict=False)
    if res.missing_keys:
        print(f'  missing keys: {res.missing_keys[:3]}...')
    model.train(mode=False)
    return model


def run_inference(model, loader, device):
    """Run forward pass over the loader and return per-pair (heading_deg,
    range) in the order yielded by the loader."""
    results = []
    t0 = time.time()
    total = len(loader)
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            ia, ib, disp, ms, dp, ds, jpaths = batch
            ia = ia.to(device, non_blocking=True)
            ib = ib.to(device, non_blocking=True)
            disp = disp.to(device, non_blocking=True)
            ms = ms.to(device, non_blocking=True)
            dp = dp.to(device, non_blocking=True)
            ds = ds.to(device, non_blocking=True)
            with autocast(device_type='cuda', dtype=AMP_DTYPE):
                out = model(ia, ib, disp, ms, dp, ds)
            xy = safe_normalize(out[:, :2].float(), dim=-1)
            pd = torch.rad2deg(torch.atan2(xy[:, 1], xy[:, 0]))
            pr = log_normalized_to_range(out[:, 2].float()).clamp(-132, 132)
            for jp, h, r in zip(jpaths, pd.cpu().tolist(), pr.cpu().tolist()):
                results.append((jp, float(h), float(r)))
            if (bi + 1) % 100 == 0:
                rate = (bi + 1) / (time.time() - t0)
                eta = (total - bi - 1) / rate / 60
                print(f'    [{bi+1}/{total}]  rate={rate:.2f} batch/s  '
                      f'ETA={eta:.1f} min', flush=True)
    return results


def write_predictions(results, out_path):
    results.sort(key=lambda x: extract_group_json_sort_key(x[0]))
    with open(out_path, 'w') as f:
        for _, h, r in results:
            f.write(f'{h:.6f} {r:.6f}\n')
    return [(h, r) for _, h, r in results]


def map_to_canonical(pred, mode):
    """Map raw stream prediction to canonical (image_a anchor, no-flip) frame."""
    pred = np.asarray(pred, dtype=np.float64)
    out = pred.copy()
    if mode == 'id':
        pass
    elif mode == 'swap':
        out[:, 0] = -pred[:, 0]
        out[:, 1] = -pred[:, 1]
    elif mode == 'hflip':
        out[:, 0] = -pred[:, 0]
        # range unchanged
    elif mode == 'swap_hflip':
        # swap negates both, then hflip negates heading, net heading: identity,
        # net range: negate.
        out[:, 1] = -pred[:, 1]
    else:
        raise ValueError(f'unknown mode {mode}')
    return out


def equal_blend(streams_canonical):
    """Equal-weight blend; heading via unit-vector mean, range via mean."""
    sumc = 0.0
    sums = 0.0
    sumr = 0.0
    K = len(streams_canonical)
    for s in streams_canonical:
        sumc = sumc + np.cos(np.deg2rad(s[:, 0]))
        sums = sums + np.sin(np.deg2rad(s[:, 0]))
        sumr = sumr + s[:, 1]
    out = np.zeros((len(streams_canonical[0]), 2), dtype=np.float64)
    out[:, 0] = np.rad2deg(np.arctan2(sums, sumc))
    out[:, 1] = sumr / K
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True,
                    help='Trained backbone weights (.pth).')
    ap.add_argument('--test_image_dir', required=True)
    ap.add_argument('--test_json_dir', required=True)
    ap.add_argument('--match_dir', required=True,
                    help='Directory of pre-extracted LightGlue match files.')
    ap.add_argument('--depth_compact', required=True,
                    help='Prefix for depth-compact files (without _keys.json '
                         'etc).')
    ap.add_argument('--out_dir', default='outputs')
    ap.add_argument('--batch_size', type=int, default=96)
    ap.add_argument('--num_workers', type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda')
    print('[1/3] Building backbone...', flush=True)
    model = build_model(args.weights, device)

    base_kwargs = dict(
        image_dir=args.test_image_dir,
        json_dir=args.test_json_dir,
        match_dir=args.match_dir,
        depth_compact_path=args.depth_compact,
        has_gt=False, force_image_ext='.webp', augment=False,
        depth_size=64, use_spatial=False,
    )

    streams = []
    for mode in ['id', 'swap', 'hflip', 'swap_hflip']:
        print(f'[2/3] Stream "{mode}"', flush=True)
        if mode == 'id':
            ds = DualImageDatasetV15(**base_kwargs)
        elif mode == 'swap':
            ds = SwappedDataset(**base_kwargs)
        elif mode == 'hflip':
            ds = HflipWrapper(DualImageDatasetV15(**base_kwargs))
        else:  # swap_hflip
            ds = HflipWrapper(SwappedDataset(**base_kwargs))

        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=True, prefetch_factor=2)
        results = run_inference(model, loader, device)
        out_path = os.path.join(args.out_dir, f'pred_{mode}.txt')
        raw = write_predictions(results, out_path)
        canonical = map_to_canonical(raw, mode)
        streams.append(canonical)

    print('[3/3] Equal-weight 4-stream blend in canonical frame', flush=True)
    blended = equal_blend(streams)
    np.savetxt(os.path.join(args.out_dir, 'pred_blended.txt'),
               blended, fmt='%.6f')
    print(f'    Wrote {os.path.join(args.out_dir, "pred_blended.txt")}')


if __name__ == '__main__':
    main()
