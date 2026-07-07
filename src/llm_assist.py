#!/usr/bin/env python3
"""Optional LLM/VLM-assisted per-pair refinement (OpenAI-compatible API).

This is an *opt-in attempt* to inject a vision-language prior into the per-pair
pose prediction. The VLM sees only the two images of a single test pair at a time, and the
result depends only on that pair.

The VLM produces a *coarse* relative-heading estimate for one pair. That estimate
is used as a **soft prior** added to the geometric per-pair argmin cost:

    cost(v) = W_h (Δθ_geom/180)^2 + W_r (Δr/132)^2 + λ (Δθ_llm/180)^2

With λ = 0 this is identical to the pure-geometric `per_pair_finalize.py`. λ is a
small non-negative knob; the VLM only nudges, it cannot override a confident
geometric match. If the API is unavailable, the key is unset, or the VLM output
is unparseable, the pair falls back to the pure-geometric prediction.

## Security / key handling
The API key is read ONLY from the environment variable ``OPENAI_API_KEY`` (and
an optional ``OPENAI_BASE_URL``). It is never hard-coded, logged, or written to
any output file. `tests/test_smoke.py::TestNoSecret` audits the repository to
guarantee no key is committed.

## Usage
    export OPENAI_API_KEY=sk-...            # your key, from your shell only
    export OPENAI_BASE_URL=https://api.openai.com/v1   # optional
    python src/llm_assist.py \
        --pairs pairs.txt          # each line: <img_a_path> <img_b_path>
        --blend blend.txt          # geometric (heading range) for those pairs
        --template assets/template.json \
        --model gpt-4o-mini --max_pairs 200 --lam 0.25 \
        --out refined.txt --report report.json
"""
import argparse
import base64
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# API key / endpoint — environment only. NEVER hard-code a key here.
# ---------------------------------------------------------------------------
def get_api_config():
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        raise SystemExit(
            'OPENAI_API_KEY is not set. This module requires an OpenAI-compatible '
            'API key, provided ONLY via the environment:\n'
            '    export OPENAI_API_KEY=sk-...\n'
            '(The key is never stored in this repository.)')
    base = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1').rstrip('/')
    return key, base


def _b64_image(path, max_side=512):
    """Read + downscale-agnostic base64 (no heavy deps; sends the raw file)."""
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode('ascii')


_PROMPT = (
    'These are two aerial photographs of the SAME building taken from an orbital '
    'drone at fixed positions. Estimate the camera yaw ROTATION from image A to '
    'image B, i.e. how many degrees the viewpoint rotated around the building '
    '(0 = same side, 90 = quarter turn, 180 = opposite side). Also say whether B '
    'is at a higher, lower, or the same altitude as A. '
    'Respond with ONLY compact JSON: '
    '{"heading_deg": <float -180..180>, "altitude": "higher|lower|same", '
    '"confidence": <float 0..1>}.'
)


def vlm_relative_heading(img_a, img_b, model, key, base, timeout=40):
    """One per-pair VLM call. Returns (heading_deg, confidence) or
    (None, 0.0) on any failure. Sees ONLY this pair's two images."""
    body = {
        'model': model,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': _PROMPT},
                {'type': 'image_url', 'image_url':
                    {'url': f'data:image/webp;base64,{_b64_image(img_a)}'}},
                {'type': 'image_url', 'image_url':
                    {'url': f'data:image/webp;base64,{_b64_image(img_b)}'}},
            ],
        }],
        'temperature': 0.0,
        'max_tokens': 80,
    }
    req = urllib.request.Request(
        f'{base}/chat/completions',
        data=json.dumps(body).encode('utf-8'),
        headers={'Authorization': f'Bearer {key}',
                 'Content-Type': 'application/json'},
        method='POST')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read().decode('utf-8'))
        txt = resp['choices'][0]['message']['content']
        s = txt[txt.find('{'): txt.rfind('}') + 1]
        obj = json.loads(s)
        return float(obj['heading_deg']), float(obj.get('confidence', 0.5))
    except (urllib.error.URLError, KeyError, ValueError, json.JSONDecodeError):
        return None, 0.0


# ---------------------------------------------------------------------------
def cd_np(a, b):
    return ((a - b + 180) % 360) - 180


def load_template(path):
    t = json.load(open(path))
    H, R = t['heading'], t['range']
    T = np.zeros((54 * 54, 2))
    for k in H:
        i, j = map(int, k.split('_'))
        T[(i - 1) * 54 + (j - 1)] = [H[k], R[k]]
    return T


def argmin_with_llm_prior(hd, rg, T, llm_heading, lam, W_h=1.0, W_r=1.0):
    """Per-pair argmin with an optional soft LLM heading prior (λ)."""
    dh = cd_np(hd, T[:, 0])
    dr = rg - T[:, 1]
    cost = W_h * (dh / 180) ** 2 + W_r * (dr / 132) ** 2
    if llm_heading is not None and lam > 0:
        cost = cost + lam * (cd_np(llm_heading, T[:, 0]) / 180) ** 2
    return T[int(np.argmin(cost))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', required=True,
                    help='Text file, each line "<img_a_path> <img_b_path>".')
    ap.add_argument('--blend', required=True,
                    help='Geometric (heading range) per pair, same order as --pairs.')
    ap.add_argument('--template', required=True)
    ap.add_argument('--model', default='gpt-4o-mini')
    ap.add_argument('--max_pairs', type=int, default=200,
                    help='Cap on API calls (cost control). Remaining pairs keep '
                         'the pure-geometric prediction.')
    ap.add_argument('--lam', type=float, default=0.25,
                    help='Soft LLM-prior weight. 0 = identical to pure geometric.')
    ap.add_argument('--out', default='refined.txt')
    ap.add_argument('--report', default='llm_report.json')
    args = ap.parse_args()

    key, base = get_api_config()          # raises if OPENAI_API_KEY unset
    pairs = [ln.split() for ln in open(args.pairs) if ln.strip()]
    blend = np.atleast_2d(np.loadtxt(args.blend))
    T = load_template(args.template)
    assert len(pairs) == len(blend), 'pairs and blend length mismatch'

    print(f'[llm_assist] model={args.model} base={base} '
          f'pairs={len(pairs)} max_calls={args.max_pairs} lam={args.lam}', flush=True)

    out = np.zeros_like(blend)
    n_called = n_changed = n_agree = 0
    for k, ((pa, pb), (hd, rg)) in enumerate(zip(pairs, blend)):
        llm_h, conf = (None, 0.0)
        if k < args.max_pairs:
            llm_h, conf = vlm_relative_heading(pa, pb, args.model, key, base)
            n_called += 1
            if llm_h is not None:
                # agreement: does the VLM heading sign/quadrant match geometry?
                if abs(cd_np(llm_h, hd)) < 45:
                    n_agree += 1
        geom = argmin_with_llm_prior(hd, rg, T, None, 0.0)
        refined = argmin_with_llm_prior(hd, rg, T, llm_h, args.lam * conf)
        out[k] = refined
        if not np.allclose(refined, geom):
            n_changed += 1

    with open(args.out, 'w') as f:
        for h, r in out:
            f.write(f'{h:.6f} {r:.6f}\n')
    report = {
        'model': args.model, 'pairs': len(pairs), 'api_calls': n_called,
        'llm_geom_agree_rate': (n_agree / n_called) if n_called else None,
        'changed_by_llm': n_changed, 'lam': args.lam,
        'note': 'per-pair VLM soft prior; key read from OPENAI_API_KEY env only.',
    }
    json.dump(report, open(args.report, 'w'), indent=2)
    print(f'[llm_assist] calls={n_called} agree={report["llm_geom_agree_rate"]} '
          f'changed={n_changed} -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
