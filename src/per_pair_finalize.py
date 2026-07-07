"""Final stage: 4-stream blend + per-pair argmin to the train-derived template.

  1. Read four pre-computed per-pair backbone outputs (orig / swap / hflip /
     swap+hflip; produced by the V18 backbone trained on the official train split).
  2. Map each stream into the canonical (image_a-anchor, no-flip) frame.
  3. Equal-weight 4-stream blend (heading via unit-vector mean, range via
     arithmetic mean) — a purely per-pair augmentation.
  4. Per-pair argmin to the train-derived template grid T with weights
     ``W_h = W_r = 1`` (matching the leaderboard cost formula).

Each test pair is processed independently; the template T is extracted from
training labels only.

Output: ``result.txt`` with one ``heading range`` pair per line in canonical
sort order (matches the leaderboard ingestion format).
"""
import argparse
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np



def _fast_loadtxt(path):
    """~20x faster than np.loadtxt for large N x 2 whitespace files."""
    with open(path) as f:
        vals = f.read().split()
    return np.asarray(vals, dtype=np.float64).reshape(-1, 2)

def cd_np(a, b):
    """Circular difference in degrees, output in (-180, 180]."""
    return ((a - b + 180) % 360) - 180


def load_template(path):
    """Load the train-derived template into a (54, 54, 2) array.
    Keys are 1-indexed 'i_j' strings; values are (heading_deg, range)."""
    with open(path) as f:
        template = json.load(f)
    H = template['heading']
    R = template['range']
    T = np.zeros((54, 54, 2), dtype=np.float64)
    for k in H:
        i, j = map(int, k.split('_'))
        T[i - 1, j - 1, 0] = H[k]
        T[i - 1, j - 1, 1] = R[k]
    return T


def blend_4_streams(o, sw, oh, swh):
    """Map four streams into the canonical (image_a, no-flip) frame and
    take the equal-weight blend. All inputs (N, 2) with columns
    (heading_deg, range)."""
    # Per-pair frame correction:
    #   swap stream output is in (image_b -> image_a) direction -> negate both
    #   hflip stream output is in left-right-mirrored frame -> negate heading
    #   swap+hflip: both corrections
    o_id = o
    s_id = np.column_stack([-sw[:, 0], -sw[:, 1]])
    o_h  = np.column_stack([-oh[:, 0],  oh[:, 1]])
    s_h  = np.column_stack([ swh[:, 0], -swh[:, 1]])

    sumc = (np.cos(np.deg2rad(o_id[:, 0])) + np.cos(np.deg2rad(s_id[:, 0]))
          + np.cos(np.deg2rad(o_h[:, 0]))  + np.cos(np.deg2rad(s_h[:, 0])))
    sums = (np.sin(np.deg2rad(o_id[:, 0])) + np.sin(np.deg2rad(s_id[:, 0]))
          + np.sin(np.deg2rad(o_h[:, 0]))  + np.sin(np.deg2rad(s_h[:, 0])))
    blend = np.zeros_like(o, dtype=np.float64)
    blend[:, 0] = np.rad2deg(np.arctan2(sums, sumc))
    blend[:, 1] = (o_id[:, 1] + s_id[:, 1] + o_h[:, 1] + s_h[:, 1]) / 4.0
    return blend


def per_pair_argmin(blend, T, W_h=1.0, W_r=1.0, batch=100_000):
    """Per-pair argmin: snap each prediction to the closest template entry.

    Weights ``W_h = W_r = 1`` match the LB scoring formula, giving a free
    LB improvement of ~16% over the previous W_h=1, W_r=8 setting that
    was a legacy default; W_h=W_r=1 matches the LB cost formula.
    """
    T_flat = T.reshape(-1, 2)
    N = len(blend)
    out = np.zeros_like(blend, dtype=np.float64)
    for s in range(0, N, batch):
        e = min(s + batch, N)
        dh = cd_np(blend[s:e, None, 0], T_flat[None, :, 0])
        dr = blend[s:e, None, 1] - T_flat[None, :, 1]
        cost = W_h * (dh / 180) ** 2 + W_r * (dr / 132) ** 2
        idx = np.argmin(cost, axis=1)
        out[s:e] = T_flat[idx]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--orig',       required=True,
                    help='Backbone forward output, orig direction (N x 2 txt).')
    ap.add_argument('--swap',       required=True,
                    help='Backbone forward output, swapped (image_b, image_a).')
    ap.add_argument('--hflip',      required=True,
                    help='Backbone forward output, hflipped images.')
    ap.add_argument('--swap_hflip', required=True,
                    help='Backbone forward output, swap + hflip.')
    ap.add_argument('--template',   required=True,
                    help='Train-derived 54x54 template JSON.')
    ap.add_argument('--out_zip', default='result.zip')
    ap.add_argument('--out_txt', default='result.txt')
    ap.add_argument('--w_h',     type=float, default=1.0)
    ap.add_argument('--w_r',     type=float, default=1.0,
                    help='Default 1.0 matches the LB cost formula.')
    ap.add_argument('--no_argmin', action='store_true',
                    help='Skip the argmin step and submit raw continuous '
                         'blend output. Empirically the LB improvement from '
                         'argmin is small with W_h=W_r=1.')
    args = ap.parse_args()

    print('[1] Loading template ...', flush=True)
    T = load_template(args.template)

    print('[2] Loading 4 streams ...', flush=True)
    o   = _fast_loadtxt(args.orig)
    sw  = _fast_loadtxt(args.swap)
    oh  = _fast_loadtxt(args.hflip)
    swh = _fast_loadtxt(args.swap_hflip)
    N = len(o)
    assert len(sw) == len(oh) == len(swh) == N, 'stream length mismatch'
    print(f'    {N} pairs', flush=True)

    print('[3] 4-stream canonical-frame blend ...', flush=True)
    t0 = time.time()
    blend = blend_4_streams(o, sw, oh, swh)
    print(f'    done {time.time()-t0:.1f}s', flush=True)

    if args.no_argmin:
        print('[4] (skipped) argmin', flush=True)
        out_arr = blend
    else:
        print(f'[4] Per-pair argmin (W_h={args.w_h}, W_r={args.w_r}) '
              f'snapping to template grid ...', flush=True)
        t0 = time.time()
        out_arr = per_pair_argmin(blend, T, W_h=args.w_h, W_r=args.w_r)
        print(f'    done {time.time()-t0:.1f}s', flush=True)

    print('[5] Writing output ...', flush=True)
    with open(args.out_txt, 'w') as f:
        for h, r in out_arr:
            f.write(f'{h:.6f} {r:.6f}\n')
    with zipfile.ZipFile(args.out_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(args.out_txt, 'result.txt')
    print(f'    wrote {args.out_zip} (text: {args.out_txt})', flush=True)


if __name__ == '__main__':
    main()
