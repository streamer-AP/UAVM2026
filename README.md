# Multi-Modal Per-Pair Pose Estimation for UAVM 2026 PairUAV

Reference implementation of our UAVM 2026 PairUAV challenge submission. The
pipeline combines a frozen DINOv2-Large + LightGlue cross-attention backbone
with four-stream geometric test-time augmentation and a per-pair argmin onto a
train-derived template grid. Each test pair is processed independently.

The dataset's central observation serves as a train-time design prior: each
building is photographed from a fixed set of orbital positions (3 altitudes ×
18 yaws). The 54×54 template extracted from training data captures the
canonical relative poses, so a per-pair argmin onto this template is a cheap,
entirely local regularisation that snaps backbone predictions onto the
discrete pose grid.

> No labelled validation split is used at any stage — neither for training nor
> for hyperparameter selection. The argmin weights `W_h = W_r = 1` follow
> directly from the leaderboard cost formula (see below).

## Reproducing the submission

```bash
pip install -r requirements.txt

# Download the trained backbone checkpoint
mkdir -p checkpoints && curl -L <CHECKPOINT_URL> -o checkpoints/v18_backbone.pth

# Full pipeline (backbone inference ≈ 7h on one A100/RTX5090, then
# per_pair_finalize.py — a few minutes on CPU):
DATA_ROOT=/path/to/pairUAV ./scripts/pipeline.sh

# OR reproduce only the final stage from cached 4-stream prediction txts
# (no GPU, no inference):
DATA_ROOT=/path/to/pairUAV ./scripts/local_reproduce.sh

# Submit ./workdir/result.zip
```

## Repository layout

```
src/
  extract_features.py   — SuperPoint+LightGlue match extraction (max_kp=4096)
  extract_depth.py      — DepthAnythingV2 compact stat extraction
  train.py              — Backbone training (V18), train split only
  infer_streams.py      — 4-stream geom-TTA inference + equal-weight blend
  per_pair_finalize.py  — Final stage: 4-stream blend + per-pair argmin to the
                          train-derived template (W_h = W_r = 1)
  llm_assist.py         — Opt-in VLM soft-prior refinement (OpenAI API via
                          OPENAI_API_KEY env; not part of the main submission)
scripts/
  pipeline.sh           — End-to-end orchestrator
  local_reproduce.sh    — Final-stage reproduction from cached 4-stream txts (CPU)
  llm_assist_demo.sh    — Demo runner for the opt-in LLM-assist experiment
tests/
  test_smoke.py         — template sanity + secret-safety checks (CPU)
  test_inference.py     — V18 forward on a 5-pair subset (opt-in, GPU/MPS)
assets/
  template.json         — 54×54 train-derived (heading, range) template
checkpoints/            — trained backbone weights (downloaded separately)
```

## Tests

```bash
python -m unittest discover tests          # CPU-only
UAVM_RUN_INFERENCE=1 python -m unittest tests.test_inference   # opt-in forward pass
```

The inference test reads a 5-pair subset from `sample_inference/`; on Apple MPS
(fp32) it agrees with the released CUDA-bf16 predictions to ~0.5° / ~1.4 m.

## Optional: LLM/VLM-assisted refinement (opt-in experiment)

`src/llm_assist.py` adds a vision-language prior via an OpenAI-compatible API.
For each test pair it shows a VLM that pair's two images, obtains a coarse
relative-heading estimate, and applies it as a soft prior in that pair's argmin
(`cost += λ·(Δθ_llm/180)²`). With `--lam 0` it reduces exactly to the geometric
result; `--max_pairs` caps API cost. The API key is read only from the
`OPENAI_API_KEY` environment variable (endpoint from optional
`OPENAI_BASE_URL`, default `https://api.openai.com/v1`); no key is stored in
the repository, and `tests/test_smoke.py` enforces this.

```bash
export OPENAI_API_KEY=sk-...          # your key — shell only, never committed
PAIRS=pairs.txt BLEND=blend.txt MODEL=gpt-4o-mini LAM=0.25 \
    ./scripts/llm_assist_demo.sh
```

> Empirically the VLM prior is weak on this metric-geometry task (coarse
> orientation only), so the leaderboard submission stays pure-geometric; the
> module is included as a documented experiment.

## Method overview (two stages — all per-pair)

### Stage 1: Backbone (V18)

Pair-wise pose regressor. For an input pair $(I_a, I_b)$:

| Component          | Configuration                                   |
| ------------------ | ----------------------------------------------- |
| Visual encoder     | DINOv2-Large, frozen, 224 × 224 input           |
| Geometric features | SuperPoint+LightGlue at 640 × 480, max 4096 kp  |
| Depth features     | DepthAnythingV2-Large, 8-d per-image stats      |
| Cross-attention    | 2 layers, 8 heads, 0.1 dropout                  |
| Pose head          | MLP → (xy_2d, log_range)                         |
| Heading / Range    | $\theta=\mathrm{atan2}(y,x)$; bounded log-range |
| Loss               | $\ell_2$ on $(\cos\theta,\sin\theta)$ + log-range |

Trained once on the official training split.

### Stage 2: Four-stream geometric TTA + per-pair argmin

Two orthogonal symmetries of the task give four streams, each mapped back to
the canonical (image_a anchor, no-flip) frame and blended with equal weights:

| Stream      | Inputs                 | Map back to canonical |
| ----------- | ---------------------- | --------------------- |
| id          | $(I_a, I_b)$           | $(\theta, r)$         |
| swap        | $(I_b, I_a)$           | $(-\theta, -r)$       |
| hflip       | $(\bar I_a, \bar I_b)$ | $(-\theta, +r)$       |
| swap+hflip  | $(\bar I_b, \bar I_a)$ | $(+\theta, -r)$       |

Heading is blended via unit-vector mean, range via arithmetic mean — a purely
per-pair augmentation. The blended prediction is then snapped to the closest
entry of the train-derived 54×54 template by a per-pair argmin:

$$
\hat p_{(a,b)} = \arg\min_{v}\; W_h\!\left(\tfrac{\mathrm{cd}(\theta,\,\mathcal T[v].\theta)}{180}\right)^2
                 + W_r\!\left(\tfrac{r - \mathcal T[v].r}{132}\right)^2 .
$$

Each test pair is snapped independently; the template $\mathcal T$ is extracted
from training labels only.

### Hyperparameters without labels

The argmin weights match the leaderboard cost formula directly: $W_h = W_r = 1$
(the metric weights heading and range equally after normalisation). No labelled
validation is used to tune them. The `--no_argmin` flag additionally submits
the raw continuous blend; with $W_h=W_r=1$ both outputs score similarly.

## Cite

```
@inproceedings{anonymous2026pairuav,
  title  = {Multi-Modal Per-Pair Pose Estimation for Orbital UAV Image Sets},
  author = {Anonymous},
  booktitle = {CVPRW},
  year   = {2026}
}
```
