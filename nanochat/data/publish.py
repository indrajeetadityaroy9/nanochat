"""
Publish the curated store and production's tokenizer to the Hugging Face dataset config.DATASET, what
`python -m nanochat.paths` (runs/setup.sh) pulls into ARTIFACTS on the training host (the GH200), which compiles and
packs them there (compile.py, pack.py) for the tokenizer it trains with, the tokenizer experiment's arms included:
- DATA/curated/<source>/{train,val}/<k>.parquet, every curated file;
- TOKENIZER/tokenizer.tiktoken and tokenizer.pattern;
- DATA/manifest.json: the code commit, the tokenizer, the sources' pins, the mixture and stages, every curated split's
  files, rows and content bytes, and what compile.py and pack.py make of them with this tokenizer (every compiled
  split's documents, tokens and packs, and the derivation of what was packed);
- DATA/curated.sha256 and DATA/compiled.sha256: the SHA-256 of every curated file and of every compiled and packed
  file this tokenizer makes, in sha256sum's format relative to ARTIFACTS, so `sha256sum -c` in ARTIFACTS checks a
  download and a rebuild on the host;
- ARTIFACTS/README.md, the dataset card, written from the manifest.
Each curated split is uploaded in a commit of its own, then the tokenizer, manifest, checksums and card in one; files on
the Hub outside this tree (an earlier commit's) are removed; each uploaded file's SHA-256 is checked against the local
file's; then the history is squashed to one commit, which training reads at main. The code is
the checkout's HEAD, committed (a modified tracked file stops publishing).

python -m nanochat.data.publish
"""

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import CommitOperationDelete, HfApi

from nanochat.config import DATASET, FILE_STAGE, REPO_SHARE, REPO_STAGE
from nanochat.data import pack
from nanochat.data.loader import mixture_corpora, packed, row_shards, shard_documents, split_path
from nanochat.data.sources import SOURCES, column_bytes, curated_dir, curated_files
from nanochat.data.tokenizer import Tokenizer, get_tokenizer
from nanochat.paths import ARTIFACTS, DATA, ROOT, TOKENIZER

MANIFEST = DATA / "manifest.json"
CARD = ARTIFACTS / "README.md"
CURATED_SUMS, COMPILED_SUMS = DATA / "curated.sha256", DATA / "compiled.sha256"


def commit() -> str:
    """HEAD of the checkout, whose tracked files are unmodified."""
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True,
                            text=True, check=True).stdout
    if status:
        raise RuntimeError(f"modified tracked files:\n{status}")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def curated_record(source: str, split: str) -> dict:
    """A curated split's files, rows and content bytes (uncompressed, from the Parquet footers)."""
    files = curated_files(source, split)
    return {"files": len(files), "rows": sum(pq.read_metadata(path).num_rows for path in files),
            "content_bytes": sum(column_bytes(path, "content") for path in files)}


def compiled_record(source: str, split: str, tokenizer: Tokenizer) -> dict:
    """A compiled split's curated files (compiled / curated), documents, tokens and packs (per row length: rows,
    pieces, FIM tokens, the curated files [first, last) its rows come from)."""
    documents = shard_documents(source, split, tokenizer)
    stages = [FILE_STAGE] + ([REPO_STAGE] if source in mixture_corpora(REPO_STAGE.mixture) else [])
    packs = {}
    for stage in stages:
        _, _, pieces, rows, fim = packed(source, split, tokenizer, stage.seq_len)
        shards = row_shards(source, split, tokenizer, stage.seq_len)
        packs[f"T{stage.seq_len}"] = {"rows": len(rows) - 1, "pieces": len(pieces),
                                      "fim_tokens": 0 if fim is None else len(fim),
                                      "files": [int(shards.min()), int(shards.max()) + 1]}
    return {"files": len(documents) - 1, "curated_files": len(curated_files(source, split)),
            "documents": int(documents[-1]), "tokens": pack.tokens(source, split, tokenizer), "packs": packs}


def manifest(tokenizer: Tokenizer) -> dict:
    compiled = {source: {split: compiled_record(source, split, tokenizer) for split in ("train", "val")}
                for source in mixture_corpora(FILE_STAGE.mixture)}
    first, last = compiled[mixture_corpora(REPO_STAGE.mixture)[0]]["train"]["packs"][f"T{REPO_STAGE.seq_len}"]["files"]
    return {
        "code": commit(),
        "tokenizer": {"data_key": tokenizer.data_key, "vocab_size": tokenizer.vocab_size, "pattern": tokenizer.pattern},
        "sources": {name: {"repo": s.repo, "files": s.files, "text": s.text,
                           "grammar": s.grammar} for name, s in SOURCES.items()},
        "curated": {source: {split: curated_record(source, split) for split in ("train", "val")} for source in SOURCES},
        "stages": {"file": {"seq_len": FILE_STAGE.seq_len, "mixture": FILE_STAGE.mixture},
                   "repo": {"seq_len": REPO_STAGE.seq_len, "mixture": REPO_STAGE.mixture, "share": REPO_SHARE}},
        "fim": {"rate": pack.FIM_RATE, "random": pack.RANDOM_SHARE, "single_node": pack.SINGLE_NODE_SHARE,
                "line": pack.LINE_SHARE, "spm": pack.SPM_SHARE},
        "packed": {"bound": pack.bound(tokenizer), "group_rows": pack.group_rows(tokenizer),
                   "repo_rows": pack.repo_rows(tokenizer), "repo_files": last - first},
        "compiled": compiled,
    }


def card(record: dict, licenses: dict[str, str | None]) -> str:
    """The dataset card: metadata, then the manifest in prose and tables."""
    file_seq, repo_seq = record["stages"]["file"]["seq_len"], record["stages"]["repo"]["seq_len"]
    rows = []
    for group, (share, corpora) in record["stages"]["file"]["mixture"].items():
        for source in corpora:
            train, val = record["curated"][source]["train"], record["curated"][source]["val"]
            built = record["compiled"][source]["train"]
            long = built["packs"].get(f"T{repo_seq}")
            rows.append(f"| {source} | {group} ({share:g}) | {train['files']} | {train['rows']:,} | "
                        f"{train['content_bytes'] / 1e9:.2f} GB | {val['rows']:,} | {built['files']} | "
                        f"{built['tokens'] / 1e9:.2f}B | {built['packs'][f'T{file_seq}']['rows']:,} | "
                        f"{'' if long is None else f'{long['rows']:,} (files [{long['files'][0]}, {long['files'][1]}))'} |")
    sources = [f"| {name} | `{s['repo']}` | {licenses[name] or 'not declared'} |"
               for name, s in record["sources"].items()]
    p = record["packed"]
    return "\n".join([
        "---", "license: other", "pretty_name: nanochat pretraining data", "viewer: false", "---", "",
        "# nanochat pretraining data", "",
        ("The curated pretraining documents of nanochat's Python and C++ code model and its tokenizer, pulled to the "
         "training host by `python -m nanochat.paths`, then compiled and "
         "packed there (`python -m nanochat.data.compile`, `python -m nanochat.data.pack`) for the tokenizer it "
         f"trains with. Published at commit `{record['code']}`; each step's output checked against that commit's "
         "code: `docs/h100-pretraining.md`, Data lineage. `data/manifest.json` holds every number below."), "",
        "## Layout", "",
        ("- `data/curated/<source>/{train,val}/<k>.parquet`: the curated documents, a repository source's files "
         "grouped into repository samples (`nanochat/data/schema.py` SAMPLE_FILE), any other source's in its schema; "
         "val holds 1% of each source's repositories, problems or documents."),
        (f"- `tokenizer/`: `tokenizer.tiktoken`, `tokenizer.pattern` (vocabulary {record['tokenizer']['vocab_size']:,}; "
         f"data key `{record['tokenizer']['data_key'][:12]}…`)."),
        ("- `data/curated.sha256`, `data/compiled.sha256`: the SHA-256 of every curated file and of every compiled and "
         "packed file this tokenizer makes, for `sha256sum -c` in `artifacts/`."), "",
        "## Corpora", "",
        (f"Compiled with this tokenizer, rows of {file_seq + 1:,} tokens (file-level stage) and {repo_seq + 1:,} "
         f"(repository-level stage). The code group's {p['bound']:,} train rows bound every run's draw; later groups "
         f"keep the leading curated files that hold their most drawn rows "
         f"({', '.join(f'{g} {n:,}' for g, n in p['group_rows'].items())}); the repository stage's packs hold the last "
         f"{p['repo_files']} curated files of each corpus, at least the {p['repo_rows']:,} rows a repository stage "
         "draws."), "",
        (f"| Corpus | Group (share) | Train files | Train rows | Train content | Val rows | Compiled train files | "
         f"Train tokens | T{file_seq} rows | T{repo_seq} rows |"),
        "|---|---|---|---|---|---|---|---|---|---|", *rows, "",
        "## Sources", "",
        "Licenses as each source's dataset card declares them.", "",
        "| Corpus | Hub dataset | License |", "|---|---|---|", *sources, "",
    ])


def sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def checksums(paths: list[Path]) -> tuple[str, dict[Path, str]]:
    """sha256sum's format for paths, relative to ARTIFACTS, and each path's digest."""
    with ThreadPoolExecutor() as pool:
        digests = dict(zip(paths, pool.map(sha256, paths), strict=True))
    return "".join(f"{d}  {p.relative_to(ARTIFACTS).as_posix()}\n" for p, d in digests.items()), digests


def main() -> None:
    tokenizer = get_tokenizer()
    record = manifest(tokenizer)
    api = HfApi()
    licenses = {name: getattr(api.dataset_info(s.repo).card_data, "license", None)
                for name, s in SOURCES.items()}
    MANIFEST.write_text(json.dumps(record, indent=1))
    CARD.write_text(card(record, licenses))
    curated = [path for source in SOURCES for split in ("train", "val") for path in curated_files(source, split)]
    compiled = sorted(path for source in mixture_corpora(FILE_STAGE.mixture)
                      for path in split_path(source, "train", tokenizer).parent.iterdir())
    text, hashes = checksums(curated)
    CURATED_SUMS.write_text(text)
    COMPILED_SUMS.write_text(checksums(compiled)[0])
    local = [*curated, TOKENIZER / "tokenizer.tiktoken", TOKENIZER / "tokenizer.pattern", MANIFEST, CARD,
             CURATED_SUMS, COMPILED_SUMS]
    published = {path.relative_to(ARTIFACTS).as_posix() for path in local}
    for source in SOURCES:
        for split in ("train", "val"):
            relative = curated_dir(source, split).relative_to(ARTIFACTS).as_posix()
            api.upload_folder(repo_id=DATASET, repo_type="dataset", folder_path=curated_dir(source, split),
                              path_in_repo=relative, commit_message=f"{relative} at {record['code'][:7]}")
    api.upload_folder(repo_id=DATASET, repo_type="dataset", folder_path=ARTIFACTS,
                      allow_patterns=sorted(published - {p.relative_to(ARTIFACTS).as_posix() for p in curated}),
                      commit_message="Tokenizer, manifest, checksums and card")
    remote = {f.path: f for f in api.list_repo_tree(DATASET, repo_type="dataset", recursive=True, expand=True)
              if hasattr(f, "size")}
    stale = sorted(set(remote) - published - {".gitattributes"})
    if stale:  # files of an earlier publication (another tokenizer's compiled splits)
        api.create_commit(DATASET, repo_type="dataset", commit_message="Files not in the published tree",
                          operations=[CommitOperationDelete(path_in_repo=path) for path in stale])
    hashes |= {path: sha256(path) for path in local if path not in hashes}
    for path in local:
        f = remote[path.relative_to(ARTIFACTS).as_posix()]
        remote_hash = f.lfs.sha256 if f.lfs else hashlib.sha256(
            Path(api.hf_hub_download(DATASET, f.path, repo_type="dataset")).read_bytes()).hexdigest()
        if remote_hash != hashes[path]:
            raise RuntimeError(f"{f.path}: the Hub's SHA-256 differs from the local file's")
    print(f"{len(local)} files on the Hub equal the local files; {len(stale)} others removed")
    api.super_squash_history(repo_id=DATASET, repo_type="dataset",
                             commit_message=f"nanochat pretraining data at {record['code'][:7]}")


if __name__ == "__main__":
    main()
