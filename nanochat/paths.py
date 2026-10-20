"""
Every location the code reads or writes: ROOT, the repository (located from this file), and ARTIFACTS = ROOT/artifacts
(gitignored), whose directories `python -m nanochat.paths` (runs/setup.sh) creates once, with the tokenizer and the
compiled and packed data at the dataset revision config.py pins; code never creates them, so a missing one fails at
first use. RESULTS, the data the paper's tables and figures are drawn from, is tracked.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DATA = ARTIFACTS / "data"                # the data pipeline's stages (nanochat/data/__init__.py)
TOKENIZER = ARTIFACTS / "tokenizer"
CHECKPOINTS = ARTIFACTS / "checkpoints"  # <tag>.pt (resume), <tag>-step<NNNNNN>.pt (model-only snapshots)
METRICS = ARTIFACTS / "metrics"          # <tag>.jsonl (train/metrics.py)
BASE_EVAL = ARTIFACTS / "base_eval"      # <tag>[-step<NNNNNN>].json (scripts/base_eval.py)
PROBES = ARTIFACTS / "probes"            # <tag>[-step<NNNNNN>].json, .md (eval/probes.py)
LOGS = ARTIFACTS / "logs"                # scripts/production.py, ablations/harness.py
CACHE = ARTIFACTS / "cache"              # native libraries (native.py), compiler and dataset caches (docker/Dockerfile)
RESULTS = ROOT / "results"               # scripts/report.py, ablations/harness.py (tracked; results/.gitkeep)

if __name__ == "__main__":
    from huggingface_hub import snapshot_download

    from nanochat.config import DATASET, DATASET_REVISION

    for directory in (DATA, TOKENIZER, CHECKPOINTS, METRICS, BASE_EVAL, PROBES, LOGS, CACHE):
        directory.mkdir(parents=True, exist_ok=True)
    snapshot_download(DATASET, repo_type="dataset", revision=DATASET_REVISION, local_dir=ARTIFACTS)
