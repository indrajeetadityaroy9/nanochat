#!/bin/bash
# Every verification on the H100, on the data pipeline's rows, one GPU job after another, stopping at the first
# failure: the model against fp64 at both stages (the canonical model, each switch off, deletion equivalence), then the
# two-stage kill/resume smoke (tests/smoke.sh: its resume, and the native decoder on its trained model).
set -euo pipefail
cd "$(dirname "$0")/.."
run() { echo "== $*"; runs/container.sh python -m "$@"; }
run tests.model_check 8 file 2
run tests.model_check 8 repo 1
echo "== tests/smoke.sh"
tests/smoke.sh
