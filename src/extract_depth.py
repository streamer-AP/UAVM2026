#!/usr/bin/env python3
"""Extract compact depth features from DepthAnythingV2-Large.

For each image we run DepthAnythingV2-Large, then store an 8-dimensional
summary of the resulting depth map (mean, std, median, 10th/90th percentile,
horizontal/vertical edge density, log range) plus a 64x64 compact map.

Output (in directory `--out_dir`, with prefix `--prefix`, default
`depth_compact_test`):

  <prefix>_keys.json   list of image stems, one per row of stats/maps
  <prefix>_stats.npy   float32 array of shape (N, 8)
  <prefix>_maps.npy    float16 array of shape (N, 64, 64)

The dataset class reads these three files via mmap. The runtime is dominated
by the depth-model forward pass; on a single A100 it takes about 30 minutes
for the 51k test images.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from PIL import Image


# Default model. Override with --model_id if needed.
DEFAULT_MODEL = 'depth-anything/Depth-Anything-V2-Large-hf'
DEFAULT_DEPTH_SIZE = 64


def is_valid_image(name):
    return (not name.startswith('.')) and \
           name.lower().endswith(('.jpeg', '.jpg', '.png', '.webp'))


def gather_image_paths(image_dir):
    """Collect (stem, path) tuples for every valid image. Supports both flat
    directories (e.g. test_tour/) and nested ones (e.g. train_tour/<group>/)."""
    image_dir = os.path.abspath(image_dir)
    pairs = []
    entries = sorted(os.listdir(image_dir))
    for name in entries:
        full = os.path.join(image_dir, name)
        if os.path.isdir(full) and not name.startswith('.'):
            for f in sorted(os.listdir(full)):
                if is_valid_image(f):
                    stem = f'{name}_{os.path.splitext(f)[0]}'
                    pairs.append((stem, os.path.join(full, f)))
        elif is_valid_image(name):
            pairs.append((os.path.splitext(name)[0], full))
    return pairs


def compute_stats(depth):
    """8-dim stats summarising the (full-resolution) depth map."""
    d = depth.flatten()
    dmin, dmax = d.min(), d.max()
    rng = dmax - dmin
    if rng > 1e-6:
        d_norm = (d - dmin) / rng
    else:
        d_norm = np.zeros_like(d)
    return np.array([
        d_norm.mean(),
        d_norm.std(),
        np.median(d_norm),
        np.percentile(d_norm, 10),
        np.percentile(d_norm, 90),
        np.abs(np.diff(depth, axis=0)).mean() / max(rng, 1e-6),
        np.abs(np.diff(depth, axis=1)).mean() / max(rng, 1e-6),
        math.log1p(rng),
    ], dtype=np.float32)


def compact_map(depth, size=DEFAULT_DEPTH_SIZE):
    img = Image.fromarray(depth.astype(np.float32), mode='F')
    resized = np.array(img.resize((size, size), Image.BILINEAR),
                       dtype=np.float32)
    dmin, dmax = resized.min(), resized.max()
    if dmax - dmin > 1e-6:
        resized = (resized - dmin) / (dmax - dmin)
    else:
        resized = np.zeros_like(resized)
    return resized.astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image_dir', required=True,
                    help='Root directory of images. Supports flat (one '
                         'level of .webp files) or nested (group/*.webp).')
    ap.add_argument('--out_dir', required=True,
                    help='Where to write <prefix>_{keys.json,stats.npy,maps.npy}.')
    ap.add_argument('--prefix', default='depth_compact_test')
    ap.add_argument('--batch_size', type=int, default=6)
    ap.add_argument('--depth_size', type=int, default=DEFAULT_DEPTH_SIZE)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--model_id', default=DEFAULT_MODEL)
    args = ap.parse_args()

    keys_path = os.path.join(args.out_dir, f'{args.prefix}_keys.json')
    stats_path = os.path.join(args.out_dir, f'{args.prefix}_stats.npy')
    maps_path = os.path.join(args.out_dir, f'{args.prefix}_maps.npy')
    if all(os.path.exists(p) for p in (keys_path, stats_path, maps_path)):
        print(f'Already exists: {args.out_dir}/{args.prefix}_*; nothing to do.')
        return
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'Loading model {args.model_id} on {args.device} ...', flush=True)
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModelForDepthEstimation.from_pretrained(args.model_id).to(args.device)
    model.train(mode=False)
    print('  loaded.', flush=True)

    pairs = gather_image_paths(args.image_dir)
    print(f'  {len(pairs)} images discovered.', flush=True)

    keys = []
    stats_list = []
    maps_list = []

    t0 = time.time()
    for batch_start in range(0, len(pairs), args.batch_size):
        chunk = pairs[batch_start:batch_start + args.batch_size]
        images = [Image.open(p).convert('RGB') for _, p in chunk]
        inputs = processor(images=images, return_tensors='pt').to(args.device)
        with torch.no_grad():
            out = model(**inputs)
        pred = out.predicted_depth  # (B, H, W) at the model's native size

        for i, (stem, _) in enumerate(chunk):
            d = pred[i]
            h, w = images[i].height, images[i].width
            d_full = torch.nn.functional.interpolate(
                d.unsqueeze(0).unsqueeze(0), size=(h, w),
                mode='bicubic', align_corners=False,
            ).squeeze().cpu().numpy().astype(np.float32)
            keys.append(stem)
            stats_list.append(compute_stats(d_full))
            maps_list.append(compact_map(d_full, args.depth_size))

        if (batch_start + args.batch_size) % (args.batch_size * 50) == 0 \
           or batch_start + args.batch_size >= len(pairs):
            done = batch_start + len(chunk)
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-6)
            eta = (len(pairs) - done) / max(rate, 1e-6)
            print(f'  [{done}/{len(pairs)}]  {rate:.1f} img/s  '
                  f'ETA {eta/60:.1f} min', flush=True)

    print(f'\nWriting outputs (prefix={args.prefix}) ...', flush=True)
    with open(keys_path, 'w') as f:
        json.dump(keys, f)
    np.save(stats_path, np.stack(stats_list).astype(np.float32))
    np.save(maps_path, np.stack(maps_list).astype(np.float16))
    print(f'  keys  : {keys_path}')
    print(f'  stats : {stats_path}  ({len(stats_list)}, 8) float32')
    print(f'  maps  : {maps_path}  ({len(maps_list)}, '
          f'{args.depth_size}, {args.depth_size}) float16')
    print(f'  total walltime: {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
