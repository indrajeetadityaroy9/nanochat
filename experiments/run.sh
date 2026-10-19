#!/bin/bash
# Run the named studies of experiments/studies.py in order: every run not yet evaluated (experiments/train_eval.py),
# then the study's report (results/studies/<study>.md, .csv). "final": the production run's configuration every rule
# decides (results/studies/final.md). The code runs depend on must be committed (config.code_version).
#
#   tmux new -d -s studies 'cd ~/projects/nanochat-next && experiments/run.sh regime scale_up final'
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT=$(git rev-parse --short HEAD)
for study in "$@"; do
  if [[ $study != final ]]; then
    # the container's banner precedes the listing on stdout: the runs are the lines that begin with "run"
    runs/container.sh python -m experiments.studies runs "$study" | grep '^run ' | while read -r _ name _; do
      runs/container.sh python -m experiments.train_eval "$study" "$name" "$COMMIT" </dev/null
    done
  fi
  runs/container.sh python -m experiments.studies report "$study"
done
