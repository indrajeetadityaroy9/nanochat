"""
Every location the code reads or writes: ROOT, the repository (located from this file), and ARTIFACTS = ROOT/artifacts
(gitignored), whose directories `python -m nanochat.paths` (runs/setup.sh) creates once, with the curated store and
the tokenizer at the dataset revision config.py pins (data/publish.py; compile.py and pack.py then build the compiled
and packed data on the host); code never creates them, so a missing one fails at first use. RESULTS, the data the
paper's tables and figures are drawn from, is tracked.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DATA = ARTIFACTS / "data"                # the data pipeline's stages (nanochat/data/__init__.py)
TOKENIZER = ARTIFACTS / "tokenizer"
# the tokenizer ablation (ablations/tokenizer.py): <arm>/tokenizer.*, and compiled/ every arm's compiled and packed data
TOKENIZER_ABLATION = ARTIFACTS / "tokenizer-ablation"
CHECKPOINTS = ARTIFACTS / "checkpoints"  # <tag>-step<step>.pt (train/checkpoint.py)
METRICS = ARTIFACTS / "metrics"          # <tag>.jsonl (train/metrics.py)
BASE_EVAL = ARTIFACTS / "base_eval"      # <tag>-step<step>.json (eval/suite.py evaluate)
LOGS = ARTIFACTS / "logs"                # <tag>.log per ablation run (ablations/harness.py), production.log (docs)
CACHE = ARTIFACTS / "cache"              # compiler and dataset caches (docker/Dockerfile)
NATIVE = CACHE / "native"                # native libraries and their build directories (native.py)
RESULTS = ROOT / "results"               # eval/suite.py export, ablations/harness.py (tracked; results/.gitkeep)

if __name__ == "__main__":
    from huggingface_hub import snapshot_download

    from nanochat.config import DATASET, DATASET_REVISION

    for directory in (DATA, TOKENIZER, CHECKPOINTS, METRICS, BASE_EVAL, LOGS, CACHE, NATIVE):
        directory.mkdir(parents=True, exist_ok=True)
    snapshot_download(DATASET, repo_type="dataset", revision=DATASET_REVISION, local_dir=ARTIFACTS)
