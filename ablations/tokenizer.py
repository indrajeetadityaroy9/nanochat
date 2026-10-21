"""
The tokenizer. Production: SA-BPE merges (arXiv 2604.14053) within data/tokenizer.py's SPLIT_PATTERN, whose runs of
punctuation take no line break, 32,768 ids. Alternatives of the same size, the trained ones on production's sample
(data/tokenizer.py sample):
- sa-bpe-fused: SA-BPE within FUSED_PATTERN, whose runs of punctuation take the line breaks after them ("):\n" one
  chunk), the first tokenizer's (cl100k's clause; arXiv 2608.18062 Appendix F.3: MBPP pass@1 0.075 on average for
  vocabularies with such tokens against 0.226 without, at nanochat's d24 architecture);
- frequency-bpe: merges by frequency alone (BPE, the crate's score "frequency") within SPLIT_PATTERN: SA-BPE's merge
  score, which its paper tests on compression and under-trained tokens only (arXiv 2604.14053, Limitations);
- cl100k: GPT-4's tokenizer cut to this size, cl100k_base's first 32,764 ranks with its own pattern (digit runs of
  1 to 3, punctuation fused with line breaks): a vocabulary trained on other data.
Bits per byte are tokenizer-independent (eval/bpb.py), so the arms compare at equal tokens, the compute production
spends; a tokenizer that compresses better reads more bytes in it.

Settings: value_embeds.py's, d8 and d16 at Chinchilla's 20 tokens per parameter (arXiv 2203.15556 §3), 4 seeds.

Every arm, production's vocabulary ("sa-bpe") included, trains and is scored on the same documents, compiled and packed
by each into TOKENIZER_ABLATION/compiled (`data`): of each corpus's curated train files, the leading ones holding MARGIN
times the rows d16 at 20 draws at the file-level stage, and of each repository corpus the trailing ones holding MARGIN
times what its repository-level stage draws (the same count for every corpus, as data/pack.py repo_stage), rows
counted in production's packs; of its val split, the leading row groups whose production tokens fill at most
VAL_SHARE of eval/bpb.py's val sample at each row length, so each arm's sample is all of their rows but at most a
micro-batch's less one: the same documents.

python -m ablations.tokenizer data   (where the curated store and production's packs are: vocabularies, data, checks)
python -m ablations.tokenizer runs   (ablations/harness.py)
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from tiktoken_ext.openai_public import cl100k_base

from ablations.harness import main
from nanochat.config import FILE_STAGE, REPO_STAGE, Stage
from nanochat.data.compile import compile_split
from nanochat.data.loader import mixture_corpora, packed_rows, row_shards, split_path
from nanochat.data.pack import in_order, pack_split, write
from nanochat.data.schema import REPOSITORY_FILE
from nanochat.data.sources import SOURCES, curated_files
from nanochat.data.tokenizer import SPECIAL_TOKENS, SPLIT_PATTERN, Tokenizer, bpe, get_tokenizer, merges, sample, save
from nanochat.eval.bpb import VAL_TOKENS
from nanochat.model.gpt import GPT, gpt_config
from nanochat.paths import TOKENIZER_ABLATION
from nanochat.train.schedule import RunPlan, plan
from nanochat.train.stages import drawn_rows, readable_rows

# SPLIT_PATTERN with cl100k's clause for a run of punctuation, which takes the line breaks after it
FUSED_PATTERN = (r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]"""
                 r"""|\s+(?!\S)|\s+""")
ARMS = ("sa-bpe", "sa-bpe-fused", "frequency-bpe", "cl100k")
SETTINGS = ((8, 20), (16, 20))
SEEDS = 4
# Rows kept per drawn row: an arm compressing a corpus better than production packs its documents into fewer rows
# (production takes 9.6% more tokens than the first tokenizer on stack_edu_cpp's val split, the largest gap measured)
MARGIN = 1.25
# Val rows kept per row of bpb's sample: the same room for an arm compressing worse, whose rows must stay within it
VAL_SHARE = 1 / MARGIN


def run_plan(depth: int, ratio: float) -> RunPlan:
    with torch.device("meta"):
        return plan(model=GPT(gpt_config(depth=depth, stage=FILE_STAGE)), ratio=ratio)


def sample_rows(stage: Stage) -> int:
    """The rows of eval/bpb.py val_bits' sample of a corpus at a stage's rows."""
    return -(-VAL_TOKENS // (stage.batch_rows * stage.seq_len)) * stage.batch_rows


def vocabularies() -> None:
    """Each arm's tokenizer.tiktoken and tokenizer.pattern in TOKENIZER_ABLATION/<arm>."""
    for arm in ARMS:
        (TOKENIZER_ABLATION / arm).mkdir(parents=True)
    production = get_tokenizer()
    save(TOKENIZER_ABLATION / "sa-bpe", production.token_bytes[:production.bos_token_id], production.pattern)
    spec = cl100k_base()
    ranked = sorted(spec["mergeable_ranks"], key=spec["mergeable_ranks"].get)
    save(TOKENIZER_ABLATION / "cl100k", ranked[:production.vocab_size - len(SPECIAL_TOKENS)], spec["pat_str"])
    docs, sources = sample()
    trained = (("sa-bpe-fused", FUSED_PATTERN, "sa-bpe"), ("frequency-bpe", SPLIT_PATTERN, "frequency"))
    for arm, pattern, score in trained:
        save(TOKENIZER_ABLATION / arm, merges(docs, sources, pattern=pattern, score=score), pattern)
    for arm in ARMS:
        print(f"{arm}: data key {get_tokenizer(arm).data_key}")


def row_group_documents(source: str, path: Path) -> list[int]:
    """The documents of each row group of a curated file (compile.py's: a repository sample, else a row)."""
    shard = pq.ParquetFile(path)
    if SOURCES[source].schema is REPOSITORY_FILE:
        return [int(np.count_nonzero(np.diff(shard.read_row_group(g, columns=["sample"])["sample"].to_numpy(),
                                             prepend=-1))) for g in range(shard.num_row_groups)]
    return [shard.metadata.row_group(g).num_rows for g in range(shard.num_row_groups)]


def val_row_groups(corpus: str, production: Tokenizer) -> dict[Path, int]:
    """{val file: its leading row groups kept}: the leading row groups of the val split (a repository corpus has one
    val file) whose production tokens stay within VAL_SHARE of bpb's sample at every stage reading the corpus."""
    limit = min(VAL_SHARE * sample_rows(stage) * stage.seq_len for stage in (FILE_STAGE, REPO_STAGE)
                if corpus in mixture_corpora(stage.mixture))
    offsets = np.memmap(split_path(corpus, "val", production).with_suffix(".idx"), dtype=np.int64, mode="r")
    kept, document = {}, 0
    for path in curated_files(corpus, "val"):
        for g, count in enumerate(row_group_documents(corpus, path)):
            if offsets[document + count] > limit:
                return kept
            document += count
            kept[path] = g + 1
    return kept


def documents() -> dict[str, dict]:
    """{corpus: {"train": [file-level train files], "repo": [repository-level train files], "val": {val file: row
    groups}}} of the curated store, rows counted in production's packs."""
    production = get_tokenizer()
    largest = run_plan(*max(SETTINGS))
    chosen = {}
    for corpus, drawn in drawn_rows(production, stage=FILE_STAGE, run=largest).items():
        per_file = np.cumsum(np.bincount(row_shards(corpus, "train", production, FILE_STAGE.seq_len)))
        leading = int(np.searchsorted(per_file, math.ceil(MARGIN * drawn))) + 1
        chosen[corpus] = {"train": curated_files(corpus, "train")[:leading], "repo": [],
                          "val": val_row_groups(corpus, production)}
    repo = drawn_rows(production, stage=REPO_STAGE, run=largest)
    per_file = {corpus: np.bincount(row_shards(corpus, "train", production, REPO_STAGE.seq_len)) for corpus in repo}
    trailing = 1
    while sum(int(rows[-trailing:].sum()) for rows in per_file.values()) < MARGIN * sum(repo.values()):
        trailing += 1
    for corpus in repo:
        chosen[corpus]["repo"] = curated_files(corpus, "train")[-trailing:]
    return chosen


def data() -> None:
    """The arms' vocabularies, then each arm's compiled and packed documents (module docstring), then the checks: each
    setting's draws within each arm's readable rows, each arm's val rows within bpb's sample."""
    vocabularies()
    chosen = documents()
    bpe()  # built once, before the packer's threads call it
    for arm in ARMS:
        tokenizer = get_tokenizer(arm)
        for corpus, files in chosen.items():
            split_path(corpus, "val", tokenizer).parent.mkdir(parents=True)
            compile_split(corpus, "val", tokenizer, row_groups=files["val"])
            compile_split(corpus, "train", tokenizer, row_groups={file: pq.read_metadata(file).num_row_groups
                                                                  for file in files["train"] + files["repo"]})
            leading, trailing = len(files["train"]), len(files["repo"])
            stages = [(FILE_STAGE, range(leading))]
            if trailing:  # a repository corpus
                stages.append((REPO_STAGE, range(leading, leading + trailing)))
            for stage, shards in stages:
                pack_split(corpus, "val", tokenizer, seq_len=stage.seq_len)
                rows = write(corpus, "train", tokenizer, in_order(corpus, "train", tokenizer, shards,
                                                                  seq_len=stage.seq_len), seq_len=stage.seq_len)
                print(f"{arm} {corpus} T{stage.seq_len}: {len(shards)} train files, {rows:,} rows")
    for arm in ARMS:
        tokenizer = get_tokenizer(arm)
        for depth, ratio in SETTINGS:
            for stage in (FILE_STAGE, REPO_STAGE):
                run = run_plan(depth, ratio)
                readable = readable_rows(tokenizer, stage=stage, run=run)
                slack = {corpus: len(readable[corpus]) - drawn
                         for corpus, drawn in drawn_rows(tokenizer, stage=stage, run=run).items()}
                tight = min(slack, key=slack.get)
                print(f"{arm} d{depth} at {ratio} T{stage.seq_len}: least slack {slack[tight]:,} rows ({tight})")
        for stage in (FILE_STAGE, REPO_STAGE):
            fill = {corpus: packed_rows(corpus, "val", tokenizer, stage.seq_len) / sample_rows(stage)
                    for corpus in mixture_corpora(stage.mixture)}
            fullest = max(fill, key=fill.get)
            print(f"{arm} val T{stage.seq_len}: at most {fill[fullest]:.3f} of bpb's sample ({fullest})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The tokenizer ablation: its data, or its runs")
    parser.add_argument("step", choices=("data", "runs"))
    if parser.parse_args().step == "data":
        data()
    else:
        alternatives = {arm: {"tokenizer": arm} for arm in ARMS[1:]}
        main(settings=tuple((depth, ratio, range(SEEDS), alternatives) for depth, ratio in SETTINGS),
             production={"tokenizer": "sa-bpe"})
