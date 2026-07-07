#!/usr/bin/env python3
"""Inference smoke test: load V18 backbone, run 5 sample pairs, compare with cached.

This test exercises the local inference path end-to-end on a small subset
shipped under sample_inference/. It is opt-in (skipped by default) because
it requires:
  - DINOv2-L weights in the local torch hub cache (1.2 GB)
  - V18 checkpoint at $UAVM_CHECKPOINT (default: ../checkpoints/best_model_v16_L.pth)
  - depth_compact stats files at $UAVM_DEPTH_PREFIX
  - cached predictions at $UAVM_CACHED_PREDICTIONS

Trigger the test with:

    UAVM_RUN_INFERENCE=1 python -m pytest tests/test_inference.py

On Apple MPS (fp32) the predictions agree with the released CUDA bf16
output to ~0.5 deg / ~1.5 m, well within precision-conversion noise.
We tolerate up to 2 deg / 3 m.
"""
import os
import sys
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RELEASE_ROOT = HERE.parent
SRC = RELEASE_ROOT / 'src'
sys.path.insert(0, str(SRC))


def _resolved(env, default):
    p = os.environ.get(env, default)
    return Path(p) if p else None


SAMPLE_DIR = _resolved('UAVM_SAMPLE_DIR',
                       str(RELEASE_ROOT.parent.parent / 'sample_inference'))
WEIGHTS = _resolved('UAVM_CHECKPOINT',
                    str(RELEASE_ROOT.parent.parent / 'checkpoints' /
                        'best_model_v16_L.pth'))
DEPTH_PREFIX = _resolved('UAVM_DEPTH_PREFIX',
                         str(RELEASE_ROOT.parent.parent / 'raw_data' /
                             'depth_compact' / 'depth_compact_test'))
CACHED = _resolved('UAVM_CACHED_PREDICTIONS',
                   str(RELEASE_ROOT.parent.parent / 'raw_data' /
                       'test_predict_output_v18_baseline_4096.txt'))

# pair indices that the subset_test/0000/*.json files correspond to
DEFAULT_PAIR_INDICES = (902, 1789, 1795, 2195, 2627)
PAIR_INDICES = tuple(
    int(x) for x in
    os.environ.get('UAVM_SAMPLE_PAIR_INDICES',
                   ','.join(str(i) for i in DEFAULT_PAIR_INDICES)).split(',')
)


# --- The test only runs when explicitly requested ---
RUN_INFERENCE = os.environ.get('UAVM_RUN_INFERENCE', '0') == '1'
SKIP_REASON = (
    'set UAVM_RUN_INFERENCE=1 to enable the inference smoke test '
    '(requires DINOv2 weights, V18 checkpoint, depth stats, and a GPU/MPS).'
)

# Tolerances tuned for fp32-MPS vs bf16-CUDA precision drift. CUDA-vs-CUDA
# is bit-identical; cross-precision picks up small per-token noise that
# accumulates through cross-attention. 2 deg / 3 m comfortably covers this.
TOL_HEADING_DEG = 2.0
TOL_RANGE_M    = 3.0


@unittest.skipIf(not RUN_INFERENCE, SKIP_REASON)
class TestLocalInference(unittest.TestCase):
    """Run V18 forward on a 5-pair subset and compare against cached predictions."""

    @classmethod
    def setUpClass(cls):
        # All required artefacts must be present
        depth_keys = (DEPTH_PREFIX.parent / (DEPTH_PREFIX.name + '_keys.json')
                      if DEPTH_PREFIX else None)
        for label, p in [('SAMPLE_DIR', SAMPLE_DIR),
                         ('WEIGHTS', WEIGHTS),
                         ('DEPTH_PREFIX_keys', depth_keys),
                         ('CACHED', CACHED)]:
            if p is None or not Path(p).exists():
                raise unittest.SkipTest(
                    f'{label} not found at {p}; '
                    f'set UAVM_SAMPLE_DIR / UAVM_CHECKPOINT / UAVM_DEPTH_PREFIX / '
                    f'UAVM_CACHED_PREDICTIONS to override.')

        try:
            import torch  # noqa
            from train import (CrossAttentionPoseNetV15, DualImageDatasetV15,
                               safe_normalize, log_normalized_to_range)
        except ImportError as e:
            raise unittest.SkipTest(f'cannot import V18 architecture: {e}')

        cls.torch = torch
        cls.CrossAttentionPoseNetV15 = CrossAttentionPoseNetV15
        cls.DualImageDatasetV15 = DualImageDatasetV15
        # Wrap free functions in staticmethod-style holders so unittest's
        # instance-attribute access doesn't auto-bind them with self.
        cls._safe_normalize = staticmethod(safe_normalize)
        cls._log_normalized_to_range = staticmethod(log_normalized_to_range)

        # Pick device
        if torch.backends.mps.is_available():
            cls.device = torch.device('mps')
        elif torch.cuda.is_available():
            cls.device = torch.device('cuda')
        else:
            cls.device = torch.device('cpu')
        print(f'\n[test_inference] device = {cls.device}')

        # Build model and load weights
        model = cls.CrossAttentionPoseNetV15(
            dinov2_model='dinov2_vitl14',
            n_cross_layers=2, cross_heads=8, cross_dropout=0.1,
            use_spatial=False, max_correction_scale=0.02,
        ).to(cls.device)
        state = torch.load(WEIGHTS, weights_only=True, map_location=cls.device)
        # Strip DataParallel "module." prefix if present
        state = {k.replace('module.', ''): v for k, v in state.items()}
        res = model.load_state_dict(state, strict=False)
        if res.unexpected_keys:
            # The released checkpoint is a strict subset of the architecture; an
            # unexpected key here would mean a different model. Fail loudly.
            raise AssertionError(
                f'unexpected keys when loading checkpoint: {res.unexpected_keys[:3]}'
            )
        model.train(mode=False)
        cls.model = model

        # Build dataset on the 5-pair subset
        cls.dataset = cls.DualImageDatasetV15(
            str(SAMPLE_DIR / 'test_tour'),
            str(SAMPLE_DIR / 'subset_test'),
            str(SAMPLE_DIR / 'test_matches_lg_4096'),
            depth_compact_path=str(DEPTH_PREFIX),
            has_gt=False, force_image_ext='.webp', augment=False,
            depth_size=64, use_spatial=False,
        )
        if len(cls.dataset) != len(PAIR_INDICES):
            raise unittest.SkipTest(
                f'subset has {len(cls.dataset)} pairs but '
                f'{len(PAIR_INDICES)} indices configured; '
                f'set UAVM_SAMPLE_PAIR_INDICES to match.')

        cls.cached = np.loadtxt(CACHED)

    def test_predictions_match_cached(self):
        """Each subset pair's predicted (h, r) must match the cached
        prediction within TOL_HEADING_DEG / TOL_RANGE_M."""
        torch = self.torch
        max_h_err = 0.0
        max_r_err = 0.0
        with torch.no_grad():
            for i in range(len(self.dataset)):
                item = self.dataset[i]
                img_a, img_b, disp, ms, dp, dst, _jpath = item
                img_a = img_a.unsqueeze(0).to(self.device)
                img_b = img_b.unsqueeze(0).to(self.device)
                disp  = disp.unsqueeze(0).to(self.device)
                ms    = ms.unsqueeze(0).to(self.device)
                dp    = dp.unsqueeze(0).to(self.device)
                dst   = dst.unsqueeze(0).to(self.device)
                out = self.model(img_a, img_b, disp, ms, dp, dst)
                # use class-level holders to avoid auto-bound self
                safe_normalize = type(self)._safe_normalize
                log_normalized_to_range = type(self)._log_normalized_to_range
                xy = safe_normalize(out[:, :2].float(), dim=-1)
                h_pred = float(torch.rad2deg(
                    torch.atan2(xy[:, 1], xy[:, 0])).item())
                r_pred = float(log_normalized_to_range(
                    out[:, 2].float()).clamp(-132, 132).item())

                idx = PAIR_INDICES[i]
                h_cache, r_cache = self.cached[idx]
                # Circular distance for heading
                h_err = abs(((h_pred - h_cache + 180) % 360) - 180)
                r_err = abs(r_pred - r_cache)
                max_h_err = max(max_h_err, h_err)
                max_r_err = max(max_r_err, r_err)

                # Per-pair report (helpful when CI fails)
                print(f'  idx={idx:>5}  '
                      f'local=({h_pred:+8.3f}, {r_pred:+8.3f})  '
                      f'cached=({h_cache:+8.3f}, {r_cache:+8.3f})  '
                      f'delta=({h_err:6.3f} deg, {r_err:6.3f} m)')

        print(f'  max |delta-heading| = {max_h_err:.3f} deg  '
              f'max |delta-range| = {max_r_err:.3f} m')
        self.assertLess(max_h_err, TOL_HEADING_DEG,
                        f'heading drift {max_h_err:.3f} exceeds tolerance '
                        f'{TOL_HEADING_DEG}')
        self.assertLess(max_r_err, TOL_RANGE_M,
                        f'range drift {max_r_err:.3f} exceeds tolerance '
                        f'{TOL_RANGE_M}')


if __name__ == '__main__':
    unittest.main(verbosity=2)
