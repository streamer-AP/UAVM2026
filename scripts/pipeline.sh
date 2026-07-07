#!/usr/bin/env bash
# End-to-end pipeline for the UAVM 2026 PairUAV challenge submission.
# Reproduces the leaderboard submission (per-pair pipeline). No
# labelled validation file is
# used at any stage; all hyperparameters either follow from the paper's
# analysis or are baked into the released checkpoint.
#
# Prerequisites:
#   - Python 3.10+ with the packages listed in requirements.txt
#   - A GPU with at least 16 GB of memory
#   - The official UAVM 2026 PairUAV training and test splits laid out as:
#       $DATA_ROOT/train_tour/<group_id>/*.webp
#       $DATA_ROOT/test_tour/*.webp
#       $DATA_ROOT/test/<group_id>/*.json
#
# Usage:
#   DATA_ROOT=/path/to/pairUAV ./scripts/pipeline.sh

set -euo pipefail

# --- Make src/ importable from anywhere -------------------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$HERE/src:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the pairUAV dataset directory}"
WORKDIR="${WORKDIR:-$HERE/workdir}"
WEIGHTS="${WEIGHTS:-$HERE/checkpoints/v18_backbone.pth}"
TEMPLATE="${TEMPLATE:-$HERE/assets/template.json}"

mkdir -p "$WORKDIR" "$WORKDIR/matches_lg_4096"

# --- 1. Pre-extract LightGlue features for the test split (max_kp=4096). ----
if [ ! -f "$WORKDIR/matches_lg_4096/.done" ]; then
  echo "[1/4] Extracting LightGlue matches (test, 4096 kp)..."
  python -u "$HERE/src/extract_features.py" \
    --split test \
    --test_image_dir "$DATA_ROOT/test_tour" \
    --test_json_dir "$DATA_ROOT/test" \
    --out "$WORKDIR/matches_lg_4096" \
    --max_kp 4096
  touch "$WORKDIR/matches_lg_4096/.done"
fi

# --- 2. Pre-extract DepthAnythingV2 compact features (3 files). -------------
if [ ! -f "$WORKDIR/depth_compact_test_keys.json" ]; then
  echo "[2/4] Extracting depth compact features..."
  python -u "$HERE/src/extract_depth.py" \
    --image_dir "$DATA_ROOT/test_tour" \
    --out_dir "$WORKDIR" \
    --prefix depth_compact_test
fi

# --- 3. Run the four-stream geometric TTA inference and equal-weight blend. -
if [ ! -f "$WORKDIR/streams/pred_blended.txt" ]; then
  echo "[3/4] Running 4-stream inference + blend..."
  mkdir -p "$WORKDIR/streams"
  python -u "$HERE/src/infer_streams.py" \
    --weights "$WEIGHTS" \
    --test_image_dir "$DATA_ROOT/test_tour" \
    --test_json_dir "$DATA_ROOT/test" \
    --match_dir "$WORKDIR/matches_lg_4096" \
    --depth_compact "$WORKDIR/depth_compact_test" \
    --out_dir "$WORKDIR/streams"
fi

# --- 4. Per-pair finalisation → final submission. ----------------------------
echo "[4/4] Per-pair finalisation..."
python -u "$HERE/src/per_pair_finalize.py" \
  --orig       "$WORKDIR/streams/pred_orig.txt" \
  --swap       "$WORKDIR/streams/pred_swap.txt" \
  --hflip      "$WORKDIR/streams/pred_hflip.txt" \
  --swap_hflip "$WORKDIR/streams/pred_swap_hflip.txt" \
  --template   "$TEMPLATE" \
  --out_zip    "$WORKDIR/result.zip" \
  --out_txt    "$WORKDIR/result.txt" \
  --w_h 1.0 --w_r 1.0

echo
echo "Done. Submit:  $WORKDIR/result.zip"
