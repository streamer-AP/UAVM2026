#!/usr/bin/env python3
"""Extract LightGlue + SuperPoint matches in SuperGlue-compatible .npz schema.

Optimization: SuperPoint features are extracted ONCE per image and cached,
then reused across all pairs. This is ~100x faster than naive per-pair extraction.

Output:
  Train: out_dir/{group_id}/{stem_a}_{stem_b}_matches.npz
  Test:  out_dir/{stem_a}/{stem_b}.npz

Resume: existing .npz are skipped.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_lightglue(device="cuda", max_kp=2048):
    from lightglue import LightGlue, SuperPoint
    extractor = SuperPoint(max_num_keypoints=max_kp).to(device).eval()
    matcher = LightGlue(features='superpoint').to(device).eval()
    return extractor, matcher


def img_to_tensor(path, device):
    img = Image.open(path).convert('RGB')
    if img.size != (640, 480):
        img = img.resize((640, 480), Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


@torch.no_grad()
def extract_sp_features(extractor, paths, device):
    """Extract SuperPoint features for a list of images. Returns list of feat dicts."""
    feats = []
    for p in paths:
        t = img_to_tensor(str(p), device)
        f = extractor({'image': t})
        # Detach from graph but keep on GPU for fast LightGlue access
        feats.append({k: v.detach() for k, v in f.items() if torch.is_tensor(v)})
    return feats


@torch.no_grad()
def match_pair(matcher, feats_a, feats_b):
    out = matcher({'image0': feats_a, 'image1': feats_b})
    kp0 = feats_a['keypoints'][0].cpu().numpy().astype(np.float32)
    kp1 = feats_b['keypoints'][0].cpu().numpy().astype(np.float32)
    pair = out['matches'][0].cpu().numpy()
    scores = out['scores'][0].cpu().numpy().astype(np.float32)

    N = kp0.shape[0]
    matches_arr = np.full((N,), -1, dtype=np.int64)
    confidence = np.zeros((N,), dtype=np.float32)
    if pair.shape[0] > 0:
        idx_a = pair[:, 0].astype(np.int64)
        idx_b = pair[:, 1].astype(np.int64)
        # Filter valid bounds
        valid = (idx_a >= 0) & (idx_a < N)
        matches_arr[idx_a[valid]] = idx_b[valid]
        confidence[idx_a[valid]] = scores[valid]
    return {
        'keypoints0': kp0,
        'keypoints1': kp1,
        'matches': matches_arr,
        'match_confidence': confidence,
    }


def is_valid_image(name):
    return name.lower().endswith(('.jpeg', '.jpg', '.png', '.webp')) and not name.startswith('.')


def list_train_groups(image_dir):
    base = Path(image_dir)
    return sorted([d.name for d in base.iterdir() if d.is_dir()])


def list_train_images(image_dir, gid):
    d = Path(image_dir) / gid
    return sorted([p for p in d.iterdir() if is_valid_image(p.name)])


def collect_test_pairs(test_json_dir):
    base = Path(test_json_dir)
    pairs = []
    for sub in sorted(base.iterdir()):
        if not sub.is_dir():
            continue
        for jf in sorted(sub.glob('*.json')):
            with open(jf) as f:
                d = json.load(f)
            pairs.append((d['image_a'], d['image_b']))
    return pairs


def process_train_group(extractor, matcher, gid, image_dir, out_dir, device, max_kp):
    img_dir = Path(image_dir) / gid
    out_dir_g = Path(out_dir) / gid
    out_dir_g.mkdir(parents=True, exist_ok=True)
    imgs = list_train_images(image_dir, gid)
    if len(imgs) != 54:
        return 0, 0, 0.0, 0.0, 0.0

    # Skip if all 2916 already exist
    expected_count = len(imgs) * len(imgs)
    existing = sum(1 for _ in out_dir_g.glob('*_matches.npz'))
    if existing >= expected_count:
        return 0, existing, 0.0, 0.0, 0.0

    t0 = time.time()
    # Extract once, cache
    feats = extract_sp_features(extractor, imgs, device)
    extract_t = time.time() - t0

    t1 = time.time()
    done = 0
    skip = 0
    for i, ia in enumerate(imgs):
        stem_a = ia.stem
        for j, ib in enumerate(imgs):
            stem_b = ib.stem
            out_path = out_dir_g / f'{stem_a}_{stem_b}_matches.npz'
            if out_path.exists():
                skip += 1
                continue
            try:
                res = match_pair(matcher, feats[i], feats[j])
                np.savez(out_path, **res)
                done += 1
            except Exception as e:
                print(f'  FAIL {gid}/{stem_a}_{stem_b}: {e}')
    match_t = time.time() - t1
    elapsed = time.time() - t0
    return done, skip, elapsed, extract_t, match_t


def process_test(extractor, matcher, pairs, image_dir, out_dir, device, max_kp,
                 group_size=54):
    """For test, group pairs by image_a stem to reuse features.
    Many test pairs share the same image_a, so we group and cache.
    """
    image_dir = Path(image_dir)
    out_dir = Path(out_dir)

    # Build per-image_a list of (image_b)
    by_a = {}
    for ra, rb in pairs:
        by_a.setdefault(ra, []).append(rb)

    print(f'  {len(by_a)} unique image_a sources, total {sum(len(v) for v in by_a.values())} pairs')

    t0 = time.time()
    done = 0
    skip = 0
    pair_idx = 0
    for src_idx, (ra, rb_list) in enumerate(sorted(by_a.items())):
        stem_a = Path(ra).stem
        out_dir_a = out_dir / stem_a
        out_dir_a.mkdir(parents=True, exist_ok=True)

        # Filter not-yet-done
        todo_b = []
        for rb in rb_list:
            stem_b = Path(rb).stem
            out_path = out_dir_a / f'{stem_b}.npz'
            if out_path.exists():
                skip += 1
            else:
                todo_b.append(rb)

        if not todo_b:
            pair_idx += len(rb_list)
            continue

        # Extract image_a features once
        full_a = image_dir / ra
        if not full_a.exists():
            full_a = image_dir / (stem_a + '.webp')
        try:
            t_a = img_to_tensor(str(full_a), device)
            with torch.no_grad():
                feats_a = extractor({'image': t_a})
                feats_a = {k: v.detach() for k, v in feats_a.items() if torch.is_tensor(v)}
        except Exception as e:
            print(f'  FAIL extract {stem_a}: {e}')
            continue

        # Process all (a, b) pairs
        for rb in todo_b:
            stem_b = Path(rb).stem
            full_b = image_dir / rb
            if not full_b.exists():
                full_b = image_dir / (stem_b + '.webp')
            try:
                t_b = img_to_tensor(str(full_b), device)
                with torch.no_grad():
                    feats_b = extractor({'image': t_b})
                    feats_b = {k: v.detach() for k, v in feats_b.items() if torch.is_tensor(v)}
                    res = match_pair(matcher, feats_a, feats_b)
                np.savez(out_dir_a / f'{stem_b}.npz', **res)
                done += 1
            except Exception as e:
                print(f'  FAIL {stem_a}_{stem_b}: {e}')
            pair_idx += 1

        if (src_idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = max(done, 1) / elapsed
            total_pairs = sum(len(v) for v in by_a.values())
            eta_h = (total_pairs - done - skip) / max(rate, 1e-6) / 3600
            print(f'  src [{src_idx+1}/{len(by_a)}] pair [{pair_idx}/{total_pairs}] done={done} skip={skip} rate={rate:.1f}/s ETA={eta_h:.1f}h', flush=True)

    return done, skip, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['train', 'test'], required=True)
    ap.add_argument('--train_image_dir', default='./pairUAV/train_tour')
    ap.add_argument('--test_image_dir', default='./pairUAV/test_tour')
    ap.add_argument('--test_json_dir', default='./pairUAV/test')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max_kp', type=int, default=2048)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--start_group', type=int, default=0)
    ap.add_argument('--end_group', type=int, default=None)
    ap.add_argument('--hash_split', default=None,
                    help='For test split, format "i/N": process only pairs where hash(stem_a) % N == i')
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'Loading LightGlue + SuperPoint...')
    t0 = time.time()
    extractor, matcher = load_lightglue(device, max_kp=args.max_kp)
    print(f'  Loaded in {time.time()-t0:.1f}s on {device}')

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.split == 'train':
        gids = list_train_groups(args.train_image_dir)
        end = args.end_group if args.end_group is not None else len(gids)
        gids = gids[args.start_group:end]
        print(f'Processing {len(gids)} train groups')
        total_done = 0
        t_start = time.time()
        for idx, gid in enumerate(gids):
            done, skip, elapsed, ext_t, match_t = process_train_group(
                extractor, matcher, gid, args.train_image_dir, str(out_dir), device, args.max_kp)
            total_done += done
            elapsed_total = time.time() - t_start
            rate = total_done / max(elapsed_total, 1e-6)
            remaining_groups = len(gids) - idx - 1
            eta_h = remaining_groups * 2916 / max(rate, 1e-6) / 3600
            print(f'[{idx+1}/{len(gids)}] group {gid}: +{done} done, skip {skip} | extract {ext_t:.1f}s, match {match_t:.1f}s | total {total_done} pairs, ETA {eta_h:.1f}h', flush=True)
    else:
        print('Collecting test pairs...')
        pairs = collect_test_pairs(args.test_json_dir)
        print(f'  {len(pairs)} test pairs')
        if args.hash_split:
            import hashlib
            i_str, n_str = args.hash_split.split('/')
            shard_i = int(i_str); shard_n = int(n_str)
            pairs_filtered = []
            for ra, rb in pairs:
                stem_a = Path(ra).stem
                # Stable MD5-based hash
                h = int(hashlib.md5(stem_a.encode()).hexdigest()[:8], 16)
                if h % shard_n == shard_i:
                    pairs_filtered.append((ra, rb))
            print(f'  hash_split {args.hash_split}: filtered to {len(pairs_filtered)} pairs', flush=True)
            pairs = pairs_filtered
        done, skip, elapsed = process_test(extractor, matcher, pairs, args.test_image_dir, str(out_dir), device, args.max_kp)
        print(f'\nDone: {done} written, {skip} skipped, total time {elapsed/3600:.1f}h')


if __name__ == '__main__':
    main()
