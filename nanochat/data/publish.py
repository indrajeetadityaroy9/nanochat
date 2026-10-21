"""
Publish the packed pretraining data to the Hugging Face dataset config.DATASET, what `python -m nanochat.paths`
(runs/setup.sh) pulls into ARTIFACTS on the H100: the tokenizer, every corpus's compiled splits and their packs
(pack.py), DATA/manifest.json (how they were made: the code, the tokenizer, the sources, the mixture and stages, every
split's documents, tokens and packs, and the derivation of what was packed) and the dataset card ARTIFACTS/README.md
written from it. Each corpus is uploaded in a commit of its own; files on the Hub outside this tree (an earlier
revision's) are removed; each uploaded file's SHA-256 is checked against the local file's; then the history is squashed
to one commit, the revision config.DATASET_REVISION pins (printed).

The code is the checkout's HEAD, committed (a modified tracked file stops publishing).

python -m nanochat.data.publish
"""

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import CommitOperationDelete, HfApi

from nanochat.config import DATASET, FILE_STAGE, REPO_SHARE, REPO_STAGE
from nanochat.data import pack
from nanochat.data.loader import mixture_corpora, packed, row_shards, shard_documents, split_path
from nanochat.data.sources import SOURCES, curated_files
from nanochat.data.tokenizer import Tokenizer, get_tokenizer
from nanochat.paths import ARTIFACTS, DATA, ROOT, TOKENIZER

MANIFEST = DATA / "manifest.json"
CARD = ARTIFACTS / "README.md"


def commit() -> str:
    """HEAD of the checkout, whose tracked files are unmodified."""
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True,
                            text=True, check=True).stdout
    if status:
        raise RuntimeError(f"modified tracked files:\n{status}")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def split_record(source: str, split: str, tokenizer: Tokenizer) -> dict:
    """A compiled split's curated files (published / curated), documents, tokens and packs (per row length: rows,
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
    corpora = mixture_corpora(FILE_STAGE.mixture)
    repo = mixture_corpora(REPO_STAGE.mixture)
    first, last = split_record(repo[0], "train", tokenizer)["packs"][f"T{REPO_STAGE.seq_len}"]["files"]
    return {
        "code": commit(),
        "tokenizer": {"data_key": tokenizer.data_key, "vocab_size": tokenizer.vocab_size, "pattern": tokenizer.pattern},
        "sources": {name: {"repo": s.repo, "revision": s.revision, "files": s.files, "text": s.text,
                           "grammar": s.grammar} for name, s in SOURCES.items()},
        "stages": {"file": {"seq_len": FILE_STAGE.seq_len, "mixture": FILE_STAGE.mixture},
                   "repo": {"seq_len": REPO_STAGE.seq_len, "mixture": REPO_STAGE.mixture, "share": REPO_SHARE}},
        "fim": {"rate": pack.FIM_RATE, "random": pack.RANDOM_SHARE, "single_node": pack.SINGLE_NODE_SHARE,
                "line": pack.LINE_SHARE, "spm": pack.SPM_SHARE},
        "packed": {"bound": pack.bound(tokenizer), "group_rows": pack.group_rows(tokenizer),
                   "repo_rows": pack.repo_rows(tokenizer), "repo_files": last - first},
        "corpora": {source: {split: split_record(source, split, tokenizer) for split in ("train", "val")}
                    for source in corpora},
    }


def card(record: dict, licenses: dict[str, str | None]) -> str:
    """The dataset card: metadata, then the manifest in prose and tables."""
    file_seq, repo_seq = record["stages"]["file"]["seq_len"], record["stages"]["repo"]["seq_len"]
    rows = []
    for group, (share, corpora) in record["stages"]["file"]["mixture"].items():
        for source in corpora:
            train = record["corpora"][source]["train"]
            long = train["packs"].get(f"T{repo_seq}")
            rows.append(f"| {source} | {group} ({share:g}) | {train['files']} / {train['curated_files']} | "
                        f"{train['tokens'] / 1e9:.2f}B | {train['packs'][f'T{file_seq}']['rows']:,} | "
                        f"{'' if long is None else f'{long['rows']:,} (files [{long['files'][0]}, {long['files'][1]}))'} |")
    sources = [f"| {name} | `{s['repo']}` @ `{s['revision'][:7]}` | {licenses[name] or 'not declared'} |"
               for name, s in record["sources"].items()]
    p = record["packed"]
    return "\n".join([
        "---", "license: other", "pretty_name: nanochat pretraining data", "viewer: false", "---", "",
        "# nanochat pretraining data", "",
        ("Tokenized and packed pretraining rows of nanochat's Python and C++ code model, pulled to the training host by "
         "`python -m nanochat.paths` at the revision `nanochat/config.py` pins. Packed by `nanochat/data/pack.py` at "
         f"commit `{record['code']}`; each earlier step's output checked against that commit's code: "
         "`docs/h100-pretraining.md`, Data lineage. `data/manifest.json` holds every number below."), "",
        "## Layout", "",
        (f"- `tokenizer/`: `tokenizer.tiktoken`, `tokenizer.pattern` (vocabulary {record['tokenizer']['vocab_size']:,}; "
         f"data key `{record['tokenizer']['data_key'][:12]}…`)."),
        ("- `data/compiled/<corpus>-<data key>/`: `{train,val}.bin` uint16 tokens of every document, no BOS; `.idx` "
         "int64 offset of each document's first token, then the total; `.json` documents per curated file; "
         "`{train,val}-T<seq_len>.pieces` int64 (token start, end, FIM start, end) per piece in row order, `.rows` "
         "int64 first piece of each row then the piece count, `.fim` uint16 tokens of the fill-in-the-middle "
         "examples."), "",
        "## Corpora", "",
        (f"Rows of {file_seq + 1:,} tokens (file-level stage) and {repo_seq + 1:,} (repository-level stage). The code "
         f"group's {p['bound']:,} train rows bound every run's draw; later groups keep the leading curated files that "
         f"hold their most drawn rows ({', '.join(f'{g} {n:,}' for g, n in p['group_rows'].items())}); the repository "
         f"stage's packs hold the last {p['repo_files']} curated files of each corpus, at least the "
         f"{p['repo_rows']:,} rows a repository stage draws."), "",
        f"| Corpus | Group (share) | Train files / curated | Train tokens | T{file_seq} rows | T{repo_seq} rows |",
        "|---|---|---|---|---|---|", *rows, "",
        "## Sources", "",
        "Licenses as each source's dataset card declares them.", "",
        "| Corpus | Hub dataset @ revision | License |", "|---|---|---|", *sources, "",
    ])


def sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main() -> None:
    tokenizer = get_tokenizer()
    record = manifest(tokenizer)
    api = HfApi()
    licenses = {name: getattr(api.dataset_info(s.repo, revision=s.revision).card_data, "license", None)
                for name, s in SOURCES.items()}
    MANIFEST.write_text(json.dumps(record, indent=1))
    CARD.write_text(card(record, licenses))
    corpora = mixture_corpora(FILE_STAGE.mixture)
    folders = [TOKENIZER] + [split_path(source, "train", tokenizer).parent for source in corpora]
    for folder in folders:
        relative = folder.relative_to(ARTIFACTS).as_posix()
        api.upload_folder(repo_id=DATASET, repo_type="dataset", folder_path=folder, path_in_repo=relative,
                          commit_message=f"{relative} at {record['code'][:7]}")
    api.upload_folder(repo_id=DATASET, repo_type="dataset", folder_path=ARTIFACTS, allow_patterns=[
        MANIFEST.relative_to(ARTIFACTS).as_posix(), CARD.name], commit_message="Manifest and card")
    local = [path for folder in folders for path in folder.iterdir()] + [MANIFEST, CARD]
    published = {path.relative_to(ARTIFACTS).as_posix() for path in local}
    remote = {f.path: f for f in api.list_repo_tree(DATASET, repo_type="dataset", recursive=True, expand=True)
              if hasattr(f, "size")}
    stale = sorted(set(remote) - published - {".gitattributes"})
    if stale:  # files of an earlier revision (another tokenizer's compiled splits)
        api.create_commit(DATASET, repo_type="dataset", commit_message="Files not in the published tree",
                          operations=[CommitOperationDelete(path_in_repo=path) for path in stale])
    with ThreadPoolExecutor() as pool:
        hashes = dict(zip(local, pool.map(sha256, local), strict=True))
    for path in local:
        f = remote[path.relative_to(ARTIFACTS).as_posix()]
        remote_hash = f.lfs.sha256 if f.lfs else hashlib.sha256(
            Path(api.hf_hub_download(DATASET, f.path, repo_type="dataset")).read_bytes()).hexdigest()
        if remote_hash != hashes[path]:
            raise RuntimeError(f"{f.path}: the Hub's SHA-256 differs from the local file's")
    print(f"{len(local)} files on the Hub equal the local files; {len(stale)} others removed")
    api.super_squash_history(repo_id=DATASET, repo_type="dataset",
                             commit_message=f"nanochat pretraining data at {record['code'][:7]}")
    print(f"DATASET_REVISION = {api.dataset_info(DATASET).sha}")


if __name__ == "__main__":
    main()
