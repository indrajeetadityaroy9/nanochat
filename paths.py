"""
Every location the code reads or writes: ROOT, the repository (located from this file), and ARTIFACTS = ROOT/artifacts
(gitignored), whose directories `python -m paths` (scripts/setup.sh) creates once; code never creates them, so a
missing one fails at first use (Hydra creates RUNS/<key>, a run's directory, before its job starts). The corpus comes
from the published dataset (`python -m corpus.publish pull`). RESULTS, the data the paper's tables and figures
are drawn from, is tracked.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
CORPUS = ARTIFACTS / "corpus"    # the corpus pipeline's stages (corpus/__init__.py)
# RUNS/<key>/ per run (run.py): .hydra/, metrics.jsonl (metrics.py), step<step>.pt
# (train/checkpoint.py), eval-step<step>.json (eval/suite.py evaluate), run.log (its phases' output)
RUNS = ARTIFACTS / "runs"
# native libraries and their build directories (native.py), beside docker/Dockerfile's compiler caches in
# artifacts/cache
NATIVE = ARTIFACTS / "cache" / "native"
RESULTS = ROOT / "results"       # an experiment's records (ablations/records.py; tracked; results/.gitkeep)

if __name__ == "__main__":
    for directory in (CORPUS, RUNS, NATIVE):
        directory.mkdir(parents=True, exist_ok=True)
