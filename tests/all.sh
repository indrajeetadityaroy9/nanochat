#!/bin/bash
# Every verification on the H100, one GPU job after another, stopping at the first failure: the ops and the model
# against fp64 (random init and the data pipeline's rows at both stages), bitwise determinism of the training step,
# the native decoder against the eager model, then the two-stage kill/resume smoke (tests/smoke.sh: its losses and
# its trained model's decoder).
set -euo pipefail
cd "$(dirname "$0")/.."
run() { echo "== $*"; runs/container.sh python -m "$@"; }
run tests.ops_check
run tests.model_check
run tests.model_check_data 8 file 2
run tests.model_check_data 8 repo 1
run tests.determinism
run tests.decoder_check 24 16 64 file
echo "== tests/smoke.sh"
tests/smoke.sh
