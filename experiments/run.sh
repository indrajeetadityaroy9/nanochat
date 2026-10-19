#!/bin/bash
# Run every arm of a phase of the ablation study (experiments/design.py: screen | regime) that has no result yet, one
# after another, then write its tables to results/ablation_<phase>.md.
#
#   tmux new -d -s study 'cd ~/projects/nanochat && experiments/run.sh screen && experiments/run.sh regime'
set -euo pipefail
cd "$(dirname "$0")/.."
PHASE=$1
DEPTH=$(python3 -c "from experiments.design import DEPTH; print(DEPTH)")
RATIO=$(python3 -c "from experiments.design import RATIO; print(RATIO)")
python3 -m experiments.design runs "$PHASE" | while read -r tag seed data_seed variant; do
  [[ -s $HOME/nanochat-runs/ablation/$tag.json ]] && continue
  echo "$(date -Is) $tag $variant"
  runs/container.sh python -m experiments.arm "$tag" "$DEPTH" "$RATIO" "$seed" "$data_seed" "$variant" </dev/null
done
runs/container.sh python -m experiments.design analyze "$PHASE" --out "results/ablation_$PHASE.md"
