#!/usr/bin/env bash
# Optional LLM/VLM-assisted per-pair refinement demo (OpenAI-compatible API).
#
# REQUIRES an API key, provided ONLY via the environment (never stored in-repo):
#   export OPENAI_API_KEY=sk-...
#   export OPENAI_BASE_URL=https://api.openai.com/v1   # optional; default shown
#
# Opt-in experiment: the VLM sees only the two images of a single test pair at
# a time. It does NOT alter the main geometric submission produced by
# per_pair_finalize.py.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY in your shell (never commit it).}"

# pairs.txt : each line "<img_a_path> <img_b_path>"
# blend.txt : geometric "heading range" per pair (same order); e.g. the
#             per_pair_finalize.py blended output for those pairs.
python3 "$HERE/src/llm_assist.py" \
  --pairs    "${PAIRS:?Set PAIRS=path/to/pairs.txt}" \
  --blend    "${BLEND:?Set BLEND=path/to/blend.txt}" \
  --template "$HERE/assets/template.json" \
  --model    "${MODEL:-gpt-4o-mini}" \
  --max_pairs "${MAX_PAIRS:-200}" \
  --lam       "${LAM:-0.25}" \
  --out      "${OUT:-refined.txt}" \
  --report   "${REPORT:-llm_report.json}"
