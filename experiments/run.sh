#!/bin/bash
# Run the named studies of experiments/studies.py in order: every arm without an evaluation record, one after another
# (experiments/arm.py; "batch": the noise scale of every run listed without a record, experiments/noise_scale.py),
# then the study's analysis (results/studies/<study>.md, .csv); at the end the tidy tables of every study.
#
#   tmux new -d -s studies 'cd ~/projects/nanochat && experiments/run.sh noise screen long_context confirm lr_horizon optimizer batch'
set -euo pipefail
cd "$(dirname "$0")/.."
RUNS=$HOME/nanochat-runs
COMMIT=$(git rev-parse --short HEAD)$(git diff --quiet HEAD || echo -dirty)
for study in "$@"; do
  # the container's banner precedes the listing on stdout: the arms are the lines that begin with "arm"
  runs/container.sh python -m experiments.studies arms "$study" | grep '^arm ' | while read -r _ name tag; do
    if [[ $study == batch ]]; then
      [[ -s $RUNS/noise_scale/$tag.json ]] || runs/container.sh python -m experiments.noise_scale "$tag" </dev/null
    else
      [[ -s $RUNS/evals/$tag.json ]] || runs/container.sh python -m experiments.arm "$study" "$name" "$COMMIT" </dev/null
    fi
  done
  runs/container.sh python -m experiments.studies analyze "$study"
done
runs/container.sh python -m experiments.studies collect
