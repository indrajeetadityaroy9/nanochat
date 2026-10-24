"""
Every location the code reads or writes: ROOT, the repository (located from this file), and ARTIFACTS = ROOT/artifacts
(gitignored), whose directories `python -m nanochat.paths` (runs/setup.sh) creates once; code never creates them, so a
missing one fails at first use. The corpus comes from the published dataset (`python -m nanochat.corpus.publish pull`).
RESULTS, the data the paper's tables and figures are drawn from, is tracked.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
CORPUS = ARTIFACTS / "corpus"            # the corpus pipeline's stages (nanochat/corpus/__init__.py)
CHECKPOINTS = ARTIFACTS / "checkpoints"  # <tag>-step<step>.pt (train/checkpoint.py)
METRICS = ARTIFACTS / "metrics"          # <tag>.jsonl (train/metrics.py)
BASE_EVAL = ARTIFACTS / "base_eval"      # <tag>-step<step>.json (eval/suite.py evaluate)
LOGS = ARTIFACTS / "logs"                # <tag>.log per ablation run (ablations/harness.py), pretrain.log
CACHE = ARTIFACTS / "cache"              # compiler and dataset caches (docker/Dockerfile)
NATIVE = CACHE / "native"                # native libraries and their build directories (native.py)
RESULTS = ROOT / "results"               # eval/suite.py export, ablations/harness.py (tracked; results/.gitkeep)

if __name__ == "__main__":
    for directory in (CORPUS, CHECKPOINTS, METRICS, BASE_EVAL, LOGS, CACHE, NATIVE):
        directory.mkdir(parents=True, exist_ok=True)
