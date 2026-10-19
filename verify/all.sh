#!/bin/bash
# Every verification on the H100, one GPU job after another, stopping at the first failure: the ops and the model
# against fp64 (random init and the data pipeline's rows at both stages), bitwise determinism of the training step,
# the native decoder against the eager model, then the two-stage kill/resume smoke (verify/smoke.sh: its losses and
# its trained model's decoder).
set -euo pipefail
cd "$(dirname "$0")/.."
run() { echo "== $*"; runs/container.sh python -m "$@"; }
run verify.ops_check
run verify.model_check
run verify.model_check_data 8 file 2
run verify.model_check_data 8 repo 1
run verify.determinism
run verify.decoder_check 24 16 64 file
echo "== verify/smoke.sh"
verify/smoke.sh
