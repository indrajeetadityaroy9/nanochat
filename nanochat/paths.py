"""
Every location the code reads or writes: ROOT, the repository (located from this file), and ARTIFACTS = ROOT/artifacts
(gitignored), whose directories runs/setup.sh creates once; code never creates them, so a missing one fails at first
use. RESULTS, the tables the paper reports, is tracked.
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
LOGS = ARTIFACTS / "logs"                # runs/*.sh
CACHE = ARTIFACTS / "cache"              # native libraries (native.py), compiler and dataset caches (runs/container.sh)
RESULTS = ROOT / "results"               # reports: <tag>.md, <tag>-curves.md (tracked; results/.gitkeep)
