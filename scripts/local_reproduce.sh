#!/usr/bin/env bash
# Local reproduction of the submission, starting from the released 4-stream
# prediction text files (no GPU, no LightGlue, no DINOv2 inference):
#
#   1. Read the 4 stream prediction txts (V18 backbone outputs)
#   2. Equal-weight blend in the canonical (image_a anchor, no-flip) frame
#      — purely per-pair augmentation, no cross-pair info.
#   3. Per-pair argmin to the train-derived template grid (W_h = W_r = 1).
#
# Usage (from release/code/):
#   DATA_ROOT=/Volumes/easystore/UAVM2026_Final ./scripts/local_reproduce.sh
#

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$HERE/src:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-/Volumes/easystore/UAVM2026_Final}"
RAW="$DATA_ROOT/raw_data"
SUB="$DATA_ROOT/submissions"
WORK="${WORK:-/tmp/repro_uavm}"
mkdir -p "$WORK"

# 4-stream blend + per-pair argmin
python3 -u "$HERE/src/per_pair_finalize.py" \
  --orig       "$RAW/test_predict_output_v18_baseline_4096.txt" \
  --swap       "$RAW/test_predict_output_v18_swap.txt" \
  --hflip      "$RAW/test_predict_output_v18_hflip_4096.txt" \
  --swap_hflip "$RAW/test_predict_output_v18_swap_hflip_4096.txt" \
  --template   "$HERE/assets/template.json" \
  --out_zip    "$WORK/result.zip" \
  --out_txt    "$WORK/result.txt" \
  --w_h 1.0 --w_r 1.0

echo
echo "=== Submission ready ==="
echo "  $WORK/result.zip   ($(shasum -a 256 $WORK/result.zip | cut -d' ' -f1))"
echo "  Upload this to the leaderboard."

