#!/bin/bash
# Run every arm of the ablation study (experiments/design.py) that has no result yet, one after another, then write the
# noise and effect tables to results/ablation_screen.md.
#
#   tmux new -d -s study 'cd ~/projects/nanochat && experiments/run.sh'
set -euo pipefail
cd "$(dirname "$0")/.."
DEPTH=$(python3 -c "from experiments.design import DEPTH; print(DEPTH)")
RATIO=$(python3 -c "from experiments.design import RATIO; print(RATIO)")
python3 -m experiments.design runs | while read -r tag seed data_seed variant; do
  [[ -s $HOME/nanochat-runs/ablation/$tag.json ]] && continue
  echo "$(date -Is) $tag $variant"
  runs/container.sh python -m experiments.arm "$tag" "$DEPTH" "$RATIO" "$seed" "$data_seed" "$variant" </dev/null
done
runs/container.sh python -m experiments.design analyze --out results/ablation_screen.md
