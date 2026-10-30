#!/bin/bash
# Upload a run's published files (full-state and weights-only checkpoints, metrics.jsonl and its diagnostics sidecar,
# evaluations and their sidecars, the analysis checkpoints' likelihoods, .hydra), the corpus tokenizer the checkpoints
# read, and results/ to a Hugging Face model repository in the checkout's layout.
# Usage: HF_TOKEN=<write token> scripts/release.sh <run_name> <repo>
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN=artifacts/runs/${1:?run_name}
INCLUDE=()
for pattern in "$RUN/checkpoint-*.pt" "$RUN/weights-*.pt" "$RUN/metrics.jsonl" "$RUN/diagnostics*" "$RUN/eval-*.json" \
               "$RUN/eval-*.samples.json.gz" "$RUN/likelihood-*.json" "$RUN/.hydra/*" "artifacts/corpus/tokenizer/*" \
               "results/*"; do
  INCLUDE+=(--include "$pattern")
done
"$ROOT"/scripts/container.sh hf upload "${2:?repo}" /workspace/nanochat . --repo-type model "${INCLUDE[@]}"
