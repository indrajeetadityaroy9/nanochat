"""
The tokenizer: the decision production's tokenizer waits on. Three vocabularies of 32,768 ids (32,764 mergeable tokens,
BOS and the 3 FIM sentinels), trained on the same sample (data/tokenizer.py sample):
- sa-bpe: production's, SA-BPE's F·ln R (arXiv 2604.14053 §4.1, its best monolingual criterion; R the authorship units
  of data/tokenizer.py sample: repositories as in the paper, web hosts and documents an extension it does not test)
  within SPLIT_PATTERN, whose runs of punctuation take no line break;
- sa-bpe-fused: SA-BPE within FUSED_PATTERN, whose runs take the line breaks after them (cl100k's clause, the first
  tokenizer's): arXiv 2608.18062's companion post deletes that clause with the tokenizer corpus held fixed and moves
  MBPP pass@1 from 0.086 to 0.230, 3 seeds each, ranges disjoint (Appendix F.3, footnote 15: 1.27B parameters, ~20B
  tokens, vocabularies of about 128K, healed prompts), its failures SyntaxErrors, while code bits per byte does not
  follow (its Table 18);
- bpe: merges by frequency alone (SA-BPE's paper's baseline; the crate's train score "frequency") within
  SPLIT_PATTERN. It is compiled, not trained on: the paper puts the two scores 0.55% apart in compression (3.623 and
  3.643, Table 3), a bits-per-byte difference below what a few seeds detect [inference], so the score is decided by
  the documents' statistics (describe).

Runs: sa-bpe against sa-bpe-fused at d16, 4 seeds, at equal bytes: each arm reads the document set once, as production
reads its data (train/stages.py max_ratio), so its tokens are the set's bytes over its bytes per token (arXiv
2402.01035 compares at equal tokens, its Appendix J at equal text); each run's plan is its own (schedule.plan at its
tokens), its context production's 8,192 and 32,768 tokens (eval/bpb.py's byte-offset buckets show the bytes of context
a better compressor sees). The set holds 20 tokens per scaling parameter of d16 for sa-bpe (Chinchilla, arXiv
2203.15556 §3): of each code corpus, the leading curated train documents whose sa-bpe tokens fill its file-level draw
(rows from production's packs: the same vocabulary, files and order), so the code group binds every arm's max_ratio;
of text and math the leading files holding MARGIN times their draw, and of each repository corpus the trailing files
holding MARGIN times the repository stage's draw (the same count for every corpus, as data/pack.py repo_stage), so
they never bind; of the val split a systematic sample of documents whose sa-bpe tokens fill at most 1 / MARGIN of
bpb's val sample at each row length, so every arm scores all of them (eval/bpb.py val_bits): the same documents.
RESULTS/tokenizer-data.json records each arm's bytes per token per corpus, its tokens holding a line break and fusing
punctuation with one, and how many vocabulary tokens its train documents hold never, under 10 and under 100 times.

Decisions, pre-registered (RESULTS/tokenizer.json "decision"):
- pattern: sa-bpe-fused is adopted iff its mean unparsed share (healed greedy HumanEval+ and MBPP+ completions that do
  not compile; ablations/harness.py evaluation) is not higher than sa-bpe's and its code bits per byte is lower with
  the 95% interval excluding zero; otherwise the non-fusing pattern;
- score, within the non-fusing pattern: bpe is adopted iff its documents leave fewer vocabulary tokens never read than
  sa-bpe's (SA-BPE's claimed benefit, fewer under-trained tokens: 640 against 72 verified, the paper's Table 3);
  otherwise SA-BPE.

python -m ablations.tokenizer vocabularies   (the arms into TOKENIZER_ABLATION/<arm>)
python -m ablations.tokenizer data           (every arm's document set, then the checks)
python -m ablations.tokenizer runs           (ablations/harness.py, then the decision)
"""
import argparse
import itertools
import json
import math

import numpy as np
import pyarrow.parquet as pq
import regex
import torch

from ablations.harness import main
from nanochat.config import FILE_STAGE, REPO_STAGE
from nanochat.data.compile import compile_split
from nanochat.data.loader import (
    TOKEN_DTYPE,
    mixture_corpora,
    packed_rows,
    row_shards,
    shard_documents,
    split_path,
)
from nanochat.data.pack import in_order, pack_split, write
from nanochat.data.sources import curated_files
from nanochat.data.tokenizer import SPLIT_PATTERN, Tokenizer, bpe, get_tokenizer, merges, sample, save
from nanochat.eval.bpb import sample_rows
from nanochat.model.gpt import GPT, gpt_config
from nanochat.paths import RESULTS, TOKENIZER_ABLATION
from nanochat.train.schedule import RunPlan, plan
from nanochat.train.stages import drawn_rows, max_ratio, readable_rows

# SPLIT_PATTERN with cl100k's clause for a run of punctuation, which takes the line breaks after it
FUSED_PATTERN = (r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]"""
                 r"""|\s+(?!\S)|\s+""")
# arm: (pattern, merge score)
ARMS = {"sa-bpe": (SPLIT_PATTERN, "sa-bpe"), "sa-bpe-fused": (FUSED_PATTERN, "sa-bpe"),
        "bpe": (SPLIT_PATTERN, "frequency")}
RUNS = ("sa-bpe", "sa-bpe-fused")  # the arms trained on; sa-bpe is the harness's production arm
DEPTH, SEEDS = 16, 4
# Text, math and repository-stage rows kept per drawn row: an arm compressing a corpus better than sa-bpe packs it into
# fewer rows (sa-bpe takes 9.21% more tokens than the first tokenizer, sa-bpe-fused's construction, on stack_edu_cpp's
# whole val split, the largest gap measured), so only the code group binds
MARGIN = 1.25


def run_plan(ratio: float) -> RunPlan:
    with torch.device("meta"):
        return plan(model=GPT(gpt_config(depth=DEPTH, stage=FILE_STAGE)), ratio=ratio)


def vocabularies() -> None:
    """Each arm's tokenizer.tiktoken and tokenizer.pattern in TOKENIZER_ABLATION/<arm>, on one load of the sample."""
    docs, sources = sample()
    for arm, (pattern, score) in ARMS.items():
        (TOKENIZER_ABLATION / arm).mkdir(parents=True)
        save(TOKENIZER_ABLATION / arm, merges(docs, sources, pattern=pattern, score=score), pattern)


def keep_documents(corpus: str, split: str, tokenizer: Tokenizer, keep: np.ndarray) -> None:
    """Rewrite a compiled split (data/compile.py) with the documents keep marks: its .bin, .idx and .json (each curated
    file's kept documents, so its packing shards stay the curated files)."""
    path = split_path(corpus, split, tokenizer)
    offsets = np.fromfile(path.with_suffix(".idx"), dtype=np.int64)
    lengths = np.diff(offsets)
    bounds = shard_documents(corpus, split, tokenizer)
    np.fromfile(path.with_suffix(".bin"), dtype=TOKEN_DTYPE)[np.repeat(keep, lengths)].tofile(path.with_suffix(".bin"))
    np.concatenate([[0], np.cumsum(lengths[keep])]).astype(np.int64).tofile(path.with_suffix(".idx"))
    path.with_suffix(".json").write_text(json.dumps({"files": [{"documents": int(keep[a:b].sum())}
                                                               for a, b in itertools.pairwise(bounds)]}))


def documents() -> dict[str, dict]:
    """{corpus: {"train": [file-level train files], "kept": their leading documents kept, "repo": [repository-level
    train files], "val_step": every how many val documents one is kept}}: the document set (module docstring), from
    production's compiled data and packs."""
    production = get_tokenizer()
    run = run_plan(20)
    code = set(next(iter(FILE_STAGE.mixture.values()))[1])
    chosen = {}
    for corpus, drawn in drawn_rows(production, stage=FILE_STAGE, run=run).items():
        rows = np.bincount(row_shards(corpus, "train", production, FILE_STAGE.seq_len))
        # the fewest leading files whose rows hold the draw (code) or MARGIN times it
        files = int(np.searchsorted(np.cumsum(rows), math.ceil(drawn if corpus in code else MARGIN * drawn))) + 1
        bounds = shard_documents(corpus, "train", production)
        offsets = np.memmap(split_path(corpus, "train", production).with_suffix(".idx"), dtype=np.int64, mode="r")
        tokens_per_row = offsets[bounds[files]] / rows[:files].sum()
        kept = int(np.searchsorted(offsets, drawn * tokens_per_row)) if corpus in code else int(bounds[files])
        limit = min(sample_rows(stage) * stage.seq_len / MARGIN for stage in (FILE_STAGE, REPO_STAGE)
                    if corpus in mixture_corpora(stage.mixture))
        val = np.memmap(split_path(corpus, "val", production).with_suffix(".idx"), dtype=np.int64, mode="r")
        chosen[corpus] = {"train": curated_files(corpus, "train")[:files], "kept": kept, "repo": [],
                          "val_step": math.ceil(val[-1] / limit)}
    repo = drawn_rows(production, stage=REPO_STAGE, run=run)
    rows = {corpus: np.bincount(row_shards(corpus, "train", production, REPO_STAGE.seq_len)) for corpus in repo}
    trailing = 1
    while sum(int(r[-trailing:].sum()) for r in rows.values()) < MARGIN * sum(repo.values()):
        trailing += 1
    for corpus in repo:
        chosen[corpus]["repo"] = curated_files(corpus, "train")[-trailing:]
    return chosen


def describe(tokenizer: Tokenizer, files: dict[str, dict]) -> dict:
    """An arm's intrinsic record on its document set: bytes per token of each corpus's train and val documents; its
    vocabulary tokens holding a line break and those fusing punctuation with one; how many mergeable tokens its train
    documents hold never, under 10 and under 100 times."""
    counts = np.zeros(tokenizer.bos_token_id, dtype=np.int64)
    lengths = np.array([len(b) for b in tokenizer.token_bytes[:tokenizer.bos_token_id]], dtype=np.int64)
    per_corpus = {}
    for corpus in files:
        per_corpus[corpus] = {}
        for split in ("train", "val"):
            tokens = np.memmap(split_path(corpus, split, tokenizer).with_suffix(".bin"), dtype=TOKEN_DTYPE, mode="r")
            held = np.bincount(tokens, minlength=tokenizer.bos_token_id)
            if split == "train":
                counts += held
            per_corpus[corpus][split] = float(held @ lengths / len(tokens))
    merged = tokenizer.token_bytes[256:tokenizer.bos_token_id]
    fused = regex.compile(r"[^\s\p{L}\p{N}][\r\n]")
    return {"data_key": tokenizer.data_key, "bytes_per_token": per_corpus,
            "line_break_tokens": sum(b"\n" in b or b"\r" in b for b in merged),
            "punctuation_line_break_tokens": sum(bool(fused.search(b.decode("utf-8", "ignore"))) for b in merged),
            "unread_tokens": {"never": int((counts == 0).sum()), "under 10": int((counts < 10).sum()),
                              "under 100": int((counts < 100).sum())}}


def data() -> None:
    """Every arm's compiled document set and RESULTS/tokenizer-data.json (describe); the trained-on arms' packs and
    their checks: max_ratio, every draw within its readable rows, every val split within bpb's sample."""
    chosen = documents()
    record = {}
    for arm in ARMS:
        tokenizer = get_tokenizer(arm)
        for corpus, files in chosen.items():
            split_path(corpus, "val", tokenizer).parent.mkdir(parents=True)
            for split, curated in (("val", curated_files(corpus, "val")), ("train", files["train"] + files["repo"])):
                compile_split(corpus, split, tokenizer,
                              row_groups={file: pq.read_metadata(file).num_row_groups for file in curated})
            val = np.arange(int(shard_documents(corpus, "val", tokenizer)[-1]))
            keep_documents(corpus, "val", tokenizer, val % files["val_step"] == 0)
            bounds = shard_documents(corpus, "train", tokenizer)
            train = np.arange(int(bounds[-1]))
            # the leading files' first kept documents and every document of the trailing (repository-stage) files
            keep_documents(corpus, "train", tokenizer, (train < files["kept"]) | (train >= bounds[len(files["train"])]))
        record[arm] = describe(tokenizer, chosen)
        print(f"{arm}: data key {tokenizer.data_key}")
    (RESULTS / "tokenizer-data.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    bpe()  # built once, before the packer's threads call it
    for arm in RUNS:
        tokenizer = get_tokenizer(arm)
        for corpus, files in chosen.items():
            first, trailing = len(files["train"]), len(files["repo"])
            stages = [(FILE_STAGE, range(first))]
            if trailing:  # a repository corpus
                stages.append((REPO_STAGE, range(first, first + trailing)))
            for stage, shards in stages:
                pack_split(corpus, "val", tokenizer, seq_len=stage.seq_len)
                rows = write(corpus, "train", tokenizer, in_order(corpus, "train", tokenizer, shards,
                                                                  seq_len=stage.seq_len), seq_len=stage.seq_len)
                print(f"{arm} {corpus} T{stage.seq_len}: {len(shards)} train files, {rows:,} rows")
        ratio = max_ratio(tokenizer, depth=DEPTH)
        run = run_plan(ratio)
        for stage in (FILE_STAGE, REPO_STAGE):
            readable = readable_rows(tokenizer, stage=stage, run=run)
            slack = {corpus: len(readable[corpus]) - drawn
                     for corpus, drawn in drawn_rows(tokenizer, stage=stage, run=run).items()}
            tight = min(slack, key=slack.get)
            fill = {corpus: packed_rows(corpus, "val", tokenizer, stage.seq_len) / sample_rows(stage)
                    for corpus in mixture_corpora(stage.mixture)}
            fullest = max(fill, key=fill.get)
            print(f"{arm} d{DEPTH} at {ratio:.2f} T{stage.seq_len}: least slack {slack[tight]:,} rows ({tight}); "
                  f"val at most {fill[fullest]:.3f} of bpb's sample ({fullest})")


def runs() -> None:
    """sa-bpe against sa-bpe-fused at d16, each at its max_ratio (ablations/harness.py main); then
    RESULTS/tokenizer.json gains the decision (module docstring)."""
    ratios = {arm: max_ratio(get_tokenizer(arm), depth=DEPTH) for arm in RUNS}
    main(settings=((DEPTH, ratios["sa-bpe"], range(SEEDS),
                    {"sa-bpe-fused": {"tokenizer": "sa-bpe-fused", "ratio": ratios["sa-bpe-fused"]}}),),
         production={"tokenizer": "sa-bpe"})
    summary = json.loads((RESULTS / "tokenizer.json").read_text(encoding="utf-8"))
    fused = summary["settings"][0]["alternatives"]["sa-bpe-fused"]
    never = {arm: r["unread_tokens"]["never"]
             for arm, r in json.loads((RESULTS / "tokenizer-data.json").read_text(encoding="utf-8")).items()}
    fusing = fused["difference"]["unparsed"] <= 0 and fused["ci_high"]["code"] < 0
    frequency = never["bpe"] < never["sa-bpe"]
    decision = {"fusing": fusing, "frequency": frequency,
                "tokenizer": "sa-bpe-fused" if fusing else "bpe" if frequency else "sa-bpe"}
    (RESULTS / "tokenizer.json").write_text(json.dumps(summary | {"decision": decision}, allow_nan=False),
                                            encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The tokenizer ablation: its vocabularies, its data, its runs")
    parser.add_argument("step", choices=("vocabularies", "data", "runs"))
    {"vocabularies": vocabularies, "data": data, "runs": runs}[parser.parse_args().step]()
