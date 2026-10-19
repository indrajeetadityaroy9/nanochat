#!/bin/bash
# The gradient-clipping ablation of the regime: d8 at 20 tokens per scaling parameter, seed 42, trained twice from the
# same initialization and data order with bitwise-deterministic gradients (model/gpt.py), so the arms differ by the
# clipping alone: train/optim.py GRAD_CLIP (DeepSeek LLM's 1.0) and none (inf). Each arm's final checkpoint is
# evaluated by base_eval; the comparison is written to results/ablation_clip.md (scripts/report.py). A rerun resumes
# each arm from its newest checkpoint.
#
#   runs/ablation_clip.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DEPTH=8 RATIO=20 SEED=42
read -r STEPS CLIP TAG_CLIP TAG_NONE < <(runs/container.sh python -c "from nanochat.config import run_tag; \
from nanochat.data.tokenizer import get_tokenizer; from nanochat.train.optim import GRAD_CLIP; \
from nanochat.train.schedule import plan; \
print(plan($DEPTH, $RATIO, get_tokenizer().vocab_size).num_iterations, GRAD_CLIP, \
run_tag($DEPTH, $RATIO, $SEED, GRAD_CLIP), run_tag($DEPTH, $RATIO, $SEED, float('inf')))" | tail -n 1)
for arm in "$CLIP $TAG_CLIP" "inf $TAG_NONE"; do
  read -r clip tag <<<"$arm"
  # one checkpoint, the final one (base_train always saves the last step)
  runs/container.sh python -m scripts.base_train --depth "$DEPTH" --target-param-data-ratio "$RATIO" --seed "$SEED" \
    --grad-clip "$clip" --save-every "$STEPS"
  runs/container.sh python -m scripts.base_eval --model-tag "$tag"
done
runs/container.sh python -m scripts.report --out results/ablation_clip.md "$TAG_CLIP" "$TAG_NONE"
