"""
The tokenizer: the decision production's tokenizer waits on. Four vocabularies of 32,768 ids (32,764 mergeable tokens,
BOS and the 3 FIM sentinels), trained on the same sample (data/tokenizer.py sample) by the same crate and encoded by
tiktoken, which leaves out what merge-rank encoding cannot run (PickyBPE's token removals, arXiv 2409.04599;
BoundlessBPE's supermerges, arXiv 2504.00178) and superword tokens, whose code result is a loss (SuperBPE, arXiv
2503.13423 Table 1: HumanEval pass@10 15.9 to 13.4):
- sa-bpe: production's, SA-BPE's F·ln R (arXiv 2604.14053 §4.1; R the authorship units of data/tokenizer.py sample:
  repositories as in the paper, web hosts and documents an extension it does not test) within SPLIT_PATTERN;
- sa-bpe-fused: within FUSED_PATTERN, whose runs of punctuation take the line breaks after them (cl100k's clause, the
  first tokenizer's), the positive control: arXiv 2608.18062's companion post deletes that clause with the tokenizer
  corpus held fixed and moves MBPP pass@1 from 0.086 to 0.230, 3 seeds each (Appendix F.3, footnote 15: 1.27B
  parameters, ~20B tokens, vocabularies of about 128K, healed prompts), its failures SyntaxErrors, while code bits per
  byte does not follow (its Table 18); a measure that misses it at d16 cannot decide the hybrid;
- hybrid: SA-BPE within HYBRID_PATTERN, the components of the others that resolve each other's limitations;
- bpe: merges by frequency alone (SA-BPE's baseline; the crate's score "frequency") within SPLIT_PATTERN, compiled, not
  trained on: the paper puts the two scores 0.55% apart in compression (3.623 and 3.643, Table 3), a bits-per-byte
  difference below what a few seeds detect [inference], so the score is decided by the documents' statistics.

Runs: sa-bpe, sa-bpe-fused and hybrid at d16, 4 seeds, at equal bytes: each arm reads the document set once, as
production reads its data (train/stages.py max_ratio), so its tokens are the set's bytes over its bytes per token (arXiv
2402.01035 compares at equal tokens, its Appendix J at equal text); each run's plan is its own (schedule.plan at its
tokens), its context production's 8,192 and 32,768 tokens (eval/bpb.py's byte-offset buckets show the bytes of context
a better compressor sees). The set holds 20 tokens per scaling parameter of d16 for sa-bpe (Chinchilla, arXiv
2203.15556 §3): of each code corpus, the leading curated train documents whose sa-bpe tokens fill its file-level draw
(rows from production's packs: the same vocabulary, files and order), so the code group binds every arm's max_ratio;
of text and math the leading files holding MARGIN times their draw, and of each repository corpus the trailing files
holding MARGIN times the repository stage's draw (the same count for every corpus, as data/pack.py repo_stage), so
they never bind; of the val split a systematic sample of documents whose sa-bpe tokens fill at most 1 / MARGIN of
bpb's val sample at each row length, so every arm scores all of them (eval/bpb.py val_bits): the same documents.

RESULTS/tokenizer-data.json holds each arm's intrinsic record (describe), on every corpus: bytes per token and Rényi
efficiency; its vocabulary tokens holding a line break (the best single predictor of MBPP in arXiv 2608.18062's
companion post, Spearman -0.72 over 58 tokenizers), fusing punctuation with one, and ending in a line break and
indentation (the post's MinGram pair: MBPP 0.016 against 0.216, 86.4% of the first's completions IndentationErrors);
the tokens its train documents hold never, under 10 and under 100 times; on the code corpora, TokEval's AST boundary
alignment, identifier fragmentation and Python indentation consistency (ablations/intrinsic.py). Not taken from
TokEval: its multilingual metrics (fertility, Gini, MorphScore), its digit metrics (every arm's pattern splits digit
runs into pairs alike) and operator isolation (no significant MBPP correlation; smallest adjusted p 0.058).

Decisions, pre-registered (RESULTS/tokenizer.json "decision"):
- pattern: a challenger (sa-bpe-fused, hybrid) qualifies iff its mean unparsed share (healed greedy HumanEval+ and MBPP+
  completions that do not compile; ablations/harness.py evaluation) is not higher than sa-bpe's, its code bits per
  byte is not higher with the 95% interval excluding zero, and one of the two is lower with the interval excluding
  zero; the qualifier with the lower unparsed share is adopted, else production's pattern; beside it, whether the
  control is detected: sa-bpe-fused's unparsed share higher with the interval excluding zero;
- score: bpe is adopted iff its documents leave fewer vocabulary tokens never read than sa-bpe's (SA-BPE's claimed
  benefit, fewer under-trained tokens: 640 against 72 verified, the paper's Table 3), carried to the adopted pattern
  [inference]; otherwise SA-BPE.

python -m ablations.tokenizer vocabularies   (the arms into TOKENIZER_ABLATION/<arm>)
python -m ablations.tokenizer data           (every arm's document set, its intrinsic record, then the checks)
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

from ablations import intrinsic
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
# The hybrid's pattern, from the arms' sources:
# - cl100k's letter run with one leading punctuation character (".append", "(self": arXiv 2402.01035 Table 1 gives the
#   GPT-4 regex 5% more code compression than its Punct regex, which drops it, at no measured pass@1 cost) or one
#   space, but no tab or line break, which cl100k's takes ("\tif");
# - no contraction clause (2402.01035's Punct; o200k joins contractions to the word): an apostrophe starts a letter run,
#   so English splits alike ("doesn" "'t") and a quoted code string no longer does (cl100k: "'s" "elf" "'" for 'self');
# - production's digit pairs and its run of punctuation without line breaks (arXiv 2608.18062's companion post);
# - whitespace by type, each run whole (the post's Claude-approximate pattern: MBPP 0.250, 0.234, 0.240 over 3 seeds,
#   0.3% of completions IndentationErrors; BoundlessBPE's whitespace rule, arXiv 2504.00178 App. A.4): line breaks,
#   tabs, spaces and other whitespace apart, so an indentation is one chunk and the statement after it starts its own,
#   the same at every depth ("    " "if" where cl100k's \s+(?!\S) gives "   " " if", and "if" at depth 0).
HYBRID_PATTERN = r"""(?:[^\s\p{L}\p{N}]| )?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++|[\r\n]+|\t+| +|[^\S\r\n\t ]+"""
# arm: (pattern, merge score)
ARMS = {"sa-bpe": (SPLIT_PATTERN, "sa-bpe"), "sa-bpe-fused": (FUSED_PATTERN, "sa-bpe"),
        "hybrid": (HYBRID_PATTERN, "sa-bpe"), "bpe": (SPLIT_PATTERN, "frequency")}
RUNS = ("sa-bpe", "sa-bpe-fused", "hybrid")  # the arms trained on; sa-bpe is the harness's production arm
CHALLENGERS = RUNS[1:]
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
    """An arm's intrinsic record on its document set: bytes per token of each corpus's train and val documents; the
    Rényi efficiency of each corpus's val documents and of all of them (ablations/intrinsic.py); its vocabulary tokens
    holding a line break, fusing punctuation with one and ending in a line break and indentation; how many mergeable
    tokens its train documents hold never, under 10 and under 100 times."""
    held = {split: np.zeros(tokenizer.bos_token_id, dtype=np.int64) for split in ("train", "val")}
    lengths = np.array([len(b) for b in tokenizer.token_bytes[:tokenizer.bos_token_id]], dtype=np.int64)
    bytes_per_token, renyi = {}, {}
    for corpus in files:
        bytes_per_token[corpus] = {}
        for split, total in held.items():
            tokens = np.memmap(split_path(corpus, split, tokenizer).with_suffix(".bin"), dtype=TOKEN_DTYPE, mode="r")
            counts = np.bincount(tokens, minlength=tokenizer.bos_token_id)
            total += counts
            bytes_per_token[corpus][split] = float(counts @ lengths / len(tokens))
            if split == "val":
                renyi[corpus] = intrinsic.renyi_efficiency(counts, tokenizer.vocab_size)
    renyi["all"] = intrinsic.renyi_efficiency(held["val"], tokenizer.vocab_size)
    merged = [b.decode("utf-8", "ignore") for b in tokenizer.token_bytes[256:tokenizer.bos_token_id]]
    # a line break and the indentation after it ending the token (\Z: $ also matches before a final line break)
    fused, indented = regex.compile(r"[^\s\p{L}\p{N}][\r\n]"), regex.compile(r"[\r\n][ \t]+\Z")
    never = held["train"]
    return {"data_key": tokenizer.data_key, "bytes_per_token": bytes_per_token, "renyi_efficiency": renyi,
            "line_break_tokens": sum("\n" in t or "\r" in t for t in merged),
            "punctuation_line_break_tokens": sum(bool(fused.search(t)) for t in merged),
            "line_break_indentation_tokens": sum(bool(indented.search(t)) for t in merged),
            "unread_tokens": {"never": int((never == 0).sum()), "under 10": int((never < 10).sum()),
                              "under 100": int((never < 100).sum())}}


def data() -> None:
    """Every arm's compiled document set and RESULTS/tokenizer-data.json (describe, and TokEval's code metrics of every
    code corpus: ablations/intrinsic.py code); the trained-on arms' packs and their checks: max_ratio, every draw within
    its readable rows, every val split within bpb's sample."""
    chosen = documents()
    tokenizers = [get_tokenizer(arm) for arm in ARMS]
    for tokenizer in tokenizers:
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
        print(f"{tokenizer.name}: data key {tokenizer.data_key}")
    record = {tokenizer.name: describe(tokenizer, chosen) for tokenizer in tokenizers}
    corpora = FILE_STAGE.mixture["code"][1]
    code = {corpus: intrinsic.code(corpus, tokenizers) for corpus in corpora}
    for arm in ARMS:
        record[arm]["code"] = {corpus: code[corpus][arm] for corpus in corpora}
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
    """sa-bpe against sa-bpe-fused and hybrid at d16, each at its max_ratio (ablations/harness.py main); then
    RESULTS/tokenizer.json gains the decision (module docstring)."""
    ratios = {arm: max_ratio(get_tokenizer(arm), depth=DEPTH) for arm in RUNS}
    main(settings=((DEPTH, ratios["sa-bpe"], range(SEEDS),
                    {arm: {"tokenizer": arm, "ratio": ratios[arm]} for arm in CHALLENGERS}),),
         production={"tokenizer": "sa-bpe"})
    summary = json.loads((RESULTS / "tokenizer.json").read_text(encoding="utf-8"))
    compared = summary["settings"][0]["alternatives"]
    qualified = [arm for arm, c in compared.items()
                 if c["difference"]["unparsed"] <= 0 and c["ci_low"]["code"] <= 0
                 and (c["ci_high"]["unparsed"] < 0 or c["ci_high"]["code"] < 0)]
    never = {arm: r["unread_tokens"]["never"]
             for arm, r in json.loads((RESULTS / "tokenizer-data.json").read_text(encoding="utf-8")).items()}
    decision = {"pattern": min(qualified, key=lambda arm: compared[arm]["difference"]["unparsed"], default="sa-bpe"),
                "score": "frequency" if never["bpe"] < never["sa-bpe"] else "sa-bpe",
                "control_detected": compared["sa-bpe-fused"]["ci_low"]["unparsed"] > 0}
    (RESULTS / "tokenizer.json").write_text(json.dumps(summary | {"decision": decision}, allow_nan=False),
                                            encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The tokenizer ablation: its vocabularies, its data, its runs")
    parser.add_argument("step", choices=("vocabularies", "data", "runs"))
    {"vocabularies": vocabularies, "data": data, "runs": runs}[parser.parse_args().step]()
