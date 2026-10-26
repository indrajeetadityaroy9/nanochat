"""
Every location the code reads or writes. `python -m paths` (scripts/setup.sh) creates the artifact directories once;
code never creates them.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
CORPUS = ARTIFACTS / "corpus"
RUNS = ARTIFACTS / "runs"
# native libraries and their build directories (native.py)
NATIVE = ARTIFACTS / "cache" / "native"
RESULTS = ROOT / "results"       # tracked records (ablations/records.py)

if __name__ == "__main__":
    for directory in (CORPUS, RUNS, NATIVE):
        directory.mkdir(parents=True, exist_ok=True)
