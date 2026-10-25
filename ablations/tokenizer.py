"""
The tokenizer ablation (experiment EXPERIMENT: pre-tokenizer split pattern x BPE merge score): it trains every
candidate vocabulary of the corpus's tokenizer, measures them on the corpus, decides the split pattern and merge score,
and adopts the decided vocabulary as the corpus's tokenizer (corpus/tokenizer.py TOKENIZER). Each candidate is a
vocabulary of 32,768 ids (32,764 mergeable tokens, BOS
and 3 FIM sentinels) trained on the same sample by the same trainer (train: the corpus crate's), encoded by tiktoken,
and trained on. The sample, the val splits and the packs come from the curated shards through corpus
(mixture.py, compile.py, pack.py).

Sample (sample): of each file-stage source's curated train split, its share of the stage's bytes (byte_shares: its
share of the stage's tokens, Stage.shares, at the reference tokenizer's bytes per token) as its share of SAMPLE_BYTES,
a systematic sample in shard order: tokenizer training data "representative of model training data"
(arXiv 2405.05417 §5); across tokenizer training mixes a subset's compression "only varies by 5 to 6 percentage
points", higher as the mix holds more of it (arXiv 2402.01035 §2.3); over languages, proportional tokenizer-training
weights give lower bits per byte than equal ones (arXiv 2610.12144 Table 6, §7). The sample's composition is not an
arm. An authorship unit (SA-BPE's R) is the repository, its name lowercased, for repository files and functions; any
other document is its own.

Arms (ARMS):
- sa-bpe: the reference tokenizer's pattern (SPLIT_PATTERN) and SA-BPE's score F·ln R (arXiv 2604.14053 §4.1: F the
  pair's frequency, R the authorship units holding it in the current segmentation, a pair of one unit scoring 0; the
  natural log as in pchizhov/sa-bpe scoring.py). The baseline: of Table 3's rows, F·log R has the highest compression
  and far fewer verified under-trained tokens than BPE. The repository as the unit is the paper's; the document as the
  unit of other text follows §6 ("documents, books, or document collections"), untested there. Not the paper's best
  F·log R·L (Table 4): for natural language "the language aspect must be avoided" (§6);
- sa-bpe-fused: FUSED_PATTERN, whose punctuation clause takes the line breaks after it, as cl100k's does. The positive
  control: in arXiv 2608.18062's companion post, deleting those trailing line breaks from GPT-4o's clause, the
  tokenizer corpus held fixed, raises MBPP pass@1 with non-overlapping seed ranges (three seeds each; healed prompts,
  the paper's §4.3 and App. C.1). If the parse test misses it at d16, it cannot decide the other arms;
- hybrid: HYBRID_PATTERN, whose statement chunks are the same at every indentation depth (its comment);
- bpe: frequency alone (Table 3's F row, SA-BPE's baseline) within SPLIT_PATTERN, close to F·log R in compression,
  so a tie in the runs is likely (assumed), which the val coverage breaks.
Not run: tiktoken encodes by merge rank alone, which has no form for PickyBPE's token removals (arXiv 2409.04599
Alg. 2) or BoundlessBPE's supermerges (arXiv 2504.00178 §3.2); SuperBPE moved HumanEval and MBPP pass@10 in opposite
directions, neither significantly (arXiv 2503.13423 Table 1).

vocabularies: every arm trained at once on one draw of the sample (every merge is an argmax with fixed tie-breaks, so
an arm does not depend on the others); ABLATION/<arm>: its vocabulary, pattern and merges;
RESULTS/tokenizer_split_merge-vocabularies.json: the sample's composition per source (target bytes, the fraction of
rows taken, the documents, bytes and units taken) and each arm's training time.

intrinsic: every arm's compiled val split of every source, and RESULTS/tokenizer_split_merge-intrinsic.json, only the
measures that plan the runs, gate their validity, test an arm's mechanism or break the decision's tie:
- planning: each source's bytes per token and the file stage's over its byte shares (an arm's tokens at equal bytes;
  the compile and pack budgets), and each group read in part's train tokens over its draw (below 1, the corpus's extent
  is too small for that arm: prepare.extent, mixture.SPREAD);
- validity, asserted: every arm holds the same bytes per document; on each source's first FIDELITY_DOCUMENTS val texts
  the pattern's chunks cover the text, tiktoken's encoding equals the arm's merges replayed in training order (Hugging
  Face tokenizers' BPE model), and every text decodes back;
- mechanism: the vocabulary entries holding a line break (the companion post's strongest MBPP correlate), fusing
  punctuation with one, and ending in a line break and indentation (the post's MinGram cb, whose MBPP completions
  mostly fail with IndentationError or TabError); depth invariance, the hybrid's clause (2): of each program source's
  first INDENTED_LINES indented val lines, those whose tokens are the indentation's, then the statement's as at depth 0;
- the tie-break: how many merged tokens the val documents read never, under 10 and under 100 times (SA-BPE's Coverage,
  arXiv 2604.14053 Table 3, graded).
Not measured: TokEval's intrinsic metrics, which no decision reads: of its 14 only AST alignment predicts MBPP, which
the paper attributes "to the vocabulary's line-break handling" (arXiv 2608.18062 §5), measured here directly, and
Rényi efficiency predicts neither code bits per byte nor MBPP (Table 2).

data: every arm's training set, the same documents (the set) compiled and packed under each arm (corpus compile.py and
pack.py, given the arm's directory), and RESULTS/tokenizer_split_merge-data.json: the set and, per arm, its packed rows
per stage, its d16 max_ratio and plan, and the share of each corpus's packed rows its run reads. The set holds a d16
run at RATIO tokens per scaling parameter under sa-bpe (arXiv 2203.15556 Table 3), each source at its share of the
base model's
stage (Stage.shares of the sources' train tokens), the tokens of a shard estimated from its text bytes at sa-bpe's val
bytes per token: of each code source the leading train shards that fill its file-stage draw, so code binds every arm's
max_ratio; of a "supply" source (notebooks, functions, traces), its draw, read whole once; of text and math the leading
shards of each stage, and of the repositories corpora the trailing partitions (compile.compile_stage, under sa-bpe),
holding MARGIN times their draws, so they never bind.

runs (conf/experiment/tokenizer_split_merge.yaml: `python run.py experiment=tokenizer_split_merge`): every arm at d16,
seeds 0 and 1, at equal bytes: each at its max_ratio (ratio null), so it reads the set's code documents once and its
tokens are their bytes over its bytes per token; text and math are drawn by rows. Each run's plan is its own
(schedule.plan at
its tokens), its contexts the corpus's 8,192 and 32,768 tokens (eval/bpb.py's byte-offset buckets: the bytes of
context a better compressor sees, arXiv 2503.13423 §3.1, App. B.1), trained and evaluated through run.py,
the records by ablations/records.py. RESULTS/tokenizer_split_merge.json "analysis" (decide): each challenger's
contrasts with sa-bpe, the baseline, and the decision.

Decisions, pre-registered (decide), each challenger (CHALLENGERS) against sa-bpe on the
seeds where both have an evaluation (a diverged run drops its seed from the arm's contrasts on both sides):
- parsing: per MBPP+ problem, the seeds whose healed greedy completion compiles under the challenger minus under
  sa-bpe; the exact sign test over the problems whose difference is not 0 (McNemar's exact test at one seed, the
  companion post's test, its App. D); better or worse at p < ALPHA, else equal. MBPP+ alone: its prompts leave the
  function header to the model, and "an evaluation that uses HumanEval alone does not expose this type of defect"
  (the post, which finds that code bits per byte "shows no association with MBPP at any level");
- code bits per byte: the difference of the arms' means, better or worse beyond its 1 - ALPHA two-sided normal bound
  1.96 CODE_SIGMA √(1 / n + 1 / n), n the contrast's seeds per arm, else equal;
- the positive control gates the decision: unless sa-bpe-fused parses worse, the corpus keeps sa-bpe's pattern and
  score. Else a challenger qualifies iff it is worse on neither and better on one; sa-bpe-fused, worse on parsing,
  never qualifies (the control detects, it is not a candidate). Pattern: the hybrid's iff it qualifies, else sa-bpe's.
  Score: bpe iff it qualifies, or is equal on both and the val documents read fewer of its merged tokens under 100
  times than of sa-bpe's (SA-BPE's claimed benefit, a higher coverage than BPE's), carried to the adopted pattern
  (assumed); otherwise SA-BPE.

adopt: the decided vocabulary (RESULTS/tokenizer_split_merge.json "analysis" "decision") as the corpus's tokenizer:
the arm of the decided pattern and score, or, bpe's score carried to another arm's pattern, that pair trained on the
sample.

python -m ablations.tokenizer vocabularies|intrinsic|data|adopt
"""
import datetime
import functools
import importlib.metadata
import itertools
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow.parquet as pq
import regex
from scipy.stats import binomtest, norm
from tokenizers import models

from corpus import mixture
from corpus.compile import compile_repositories, compile_source, compile_stage, train_parts
from corpus.layout import TOKEN_DTYPE, split_path, stage_rows
from corpus.mixture import weights
from corpus.pack import packed_shards, shard_count, write
from corpus.parallel import SPAWN
from corpus.schema import EXERCISE, FUNCTION, REPOSITORY_FILE, SYNTHETIC_FILE
from corpus.sources import SOURCES, curated_shards, text_bytes, texts
from corpus.tokenizer import ABLATION, SPECIAL_TOKENS, SPLIT_PATTERN, TOKENIZER, get_tokenizer, save
from model.config import VOCAB_SIZE
from native import corpus_crate
from paths import RESULTS, ROOT

# The experiment's name (conf/experiment/<EXPERIMENT>.yaml): its records and analysis, RESULTS/<EXPERIMENT>.jsonl and
# .json, and this module's results, RESULTS/<EXPERIMENT>-{vocabularies,intrinsic,data}.json
EXPERIMENT = "tokenizer_split_merge"

Score = Literal["sa-bpe", "frequency"]

# The sample's bytes: the reference tokenizer's sample's (its curated splits' Parquet footers; that data is no longer on
# the Hub), so an arm trains on as much text as the reference tokenizer did
SAMPLE_BYTES = 13.38e9
# karpathy/nanochat's split pattern (92d63d4): SPLIT_PATTERN with cl100k's punctuation clause, whose [\r\n]* takes the
# line breaks after a punctuation run ("):\n" one chunk)
FUSED_PATTERN = (r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]"""
                 r"""|\s+(?!\S)|\s+""")
# The hybrid's pattern. BPE merges, trained and applied within chunks (tiktoken encodes chunk by chunk), never cross a
# chunk's end, so every chunk end is a token end under every vocabulary trained with the pattern. No clause looks
# behind, so the chunks after a chunk end j are the chunks of the text from j alone. Its clauses:
# (1) \n+, \r+: a line break shares a chunk only with line breaks of its kind (the companion post's Claude
#     approximation's [\n]+, [\r]+; the forced break at line breaks of its MinGram cb_nl). No token joins a line break
#     to the punctuation before it (FUSED_PATTERN's "):\n"), so a prompt ending "):" ends at a token end: no
#     token-boundary bias for multi-line prompts without a terminal line break (arXiv 2412.19437 §4.1). Nor to the
#     indentation after it (the post's MinGram cb), nor to the whitespace before it (SPLIT_PATTERN's \s*[\r\n], the
#     post's gpt4onl and clean-multi);
# (2)  +, \t+, [^\S\r\n\t ]+: a run of spaces, of tabs or of other whitespace is one chunk (the Claude approximation's
#     [ ]+, [\t]+), and a letter run takes no tab or line break before it, so an indentation of d >= 2 spaces or d >= 1
#     tabs ends at a chunk end and chunks(indent + s) = (indent,) + chunks(s): one indentation chunk per depth, the
#     statement's chunks the same at every depth. SPLIT_PATTERN's \s+(?!\S) gives the indentation's last space to the
#     statement (" if" against "if" at depth 0) and its [^\r\n\p{L}\p{N}]?+ a tab ("\tif"). A single space before a
#     letter run stays its prefix (" foo"), as in prose;
# (3) no contraction clause (Punct drops cl100k's, arXiv 2402.01035 App. E, Fig. 3): the letter clause takes a whole
#     letter run, which '(?i:[sdmt]|ll|ve|re) cuts where an apostrophe starts a chunk ("'s" "elf" for 'self');
#     English contractions chunk as "doesn" "'t".
# Kept from cl100k: one leading punctuation character on a letter run (".append", "(self"), which Punct drops (it keeps
# the leading space, arXiv 2402.01035 Fig. 3), GPT-4's pattern giving "an additional 5% compression over Punct at no
# significant cost to performance" (App. E); a leading space (" foo"); SPLIT_PATTERN's digit pairs and punctuation runs.
# [^\S\r\n\t ]+ puts every character in some chunk (tiktoken drops what no clause matches). No lookahead: a long
# whitespace run costs fancy-regex no backtracking, where SPLIT_PATTERN's \s+(?!\S) beside \s*[\r\n] stops tiktoken past
# its backtrack limit
HYBRID_PATTERN = r"""(?:[^\s\p{L}\p{N}]| )?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++|\n+|\r+|\t+| +|[^\S\r\n\t ]+"""
# arm: (pattern, merge score); every arm is trained on, sa-bpe the runs' baseline (decide)
ARMS: dict[str, tuple[str, Score]] = {"sa-bpe": (SPLIT_PATTERN, "sa-bpe"), "sa-bpe-fused": (FUSED_PATTERN, "sa-bpe"),
                                      "hybrid": (HYBRID_PATTERN, "sa-bpe"), "bpe": (SPLIT_PATTERN, "frequency")}
CHALLENGERS = ("sa-bpe-fused", "hybrid", "bpe")
# The runs' depth, which the set is sized for (conf/experiment/tokenizer_split_merge.yaml)
DEPTH = 16
# The set's size: a d16 run at Chinchilla's 20 tokens per scaling parameter under sa-bpe (arXiv 2203.15556 Table 3)
RATIO = 20
# The set's margin over the draws of the corpora that must not bind (text, math, the repositories corpora): above the
# code shards' rounding up to whole shards and the arms' spread in compression (assumed)
MARGIN = 1.5
# The decisions' significance level, the conventional 0.05
ALPHA = 0.05
# The pooled seed standard deviation of a run's 8K code bits per byte (seed variances pooled over configurations): two
# d8 configurations at 20 tokens per parameter on the reference tokenizer's data, data seeds 0 to 2 (six runs),
# assumed to carry to d16 and to the corpus
CODE_SIGMA = 0.0034
# The benchmark of the parse test (Decisions)
PARSED = "mbpp_plus"
# The categories whose documents are whole program texts in their source's language (depth invariance)
PROGRAMS = (REPOSITORY_FILE, EXERCISE, SYNTHETIC_FILE, FUNCTION)
# The fidelity check's texts: the first of each source's val split; an engineering choice, bounding the check's cost
# (merges replayed in Python)
FIDELITY_DOCUMENTS = 200
# The depth-invariance check's lines: the first indented lines of each program source's val split; the rate's standard
# error at most √(1/(4n)) ≈ 0.0022
INDENTED_LINES = 50_000
# An indented line: its leading spaces and tabs, then its statement
INDENTED = regex.compile(r"([ \t]+)(\S.*)")


@functools.cache
def shard_bytes(source: str) -> list[int]:
    """The text bytes of each shard of the source's curated train split (Parquet footers)."""
    return [text_bytes(pq.read_metadata(path), "content") for path in curated_shards(source, "train")]


def byte_shares() -> dict[str, float]:
    """Each file-stage source's share of the stage's bytes: its share of the stage's tokens (Stage.shares, tokens
    counted at the reference tokenizer's bytes per token) at its bytes per token."""
    sources, per_token = mixture.FILE_STAGE.sources, mixture.BYTES_PER_TOKEN
    shares = mixture.FILE_STAGE.shares({s: sum(shard_bytes(s)) / per_token[s] for s in sources},
                                       mixture.FILE_STAGE.tokens)
    stage_bytes = sum(shares[s] * per_token[s] for s in sources)
    return {s: shares[s] * per_token[s] / stage_bytes for s in sources}


def sample_row_group(task: tuple[str, Path, int, int, float]) -> tuple[list[str], list[str], int]:
    """Row group g of a curated train shard: the texts (sources.texts), authorship units and total UTF-8 bytes of its
    rows that the source's systematic sample takes, `start` the source's rows before the group: row i when
    floor((i + 1) * fraction) exceeds floor(i * fraction). A row naming its repository (a repository file, a function)
    has that repository as its unit, its name lowercased (one unit across the sources that hold it); any other document
    is its own."""
    source, path, g, start, fraction = task
    group = pq.ParquetFile(path).read_row_group(g)
    rows = np.arange(start, start + group.num_rows)
    batch = group.take(np.flatnonzero(np.floor((rows + 1) * fraction) > np.floor(rows * fraction)))
    if "repo_name" in batch.column_names:
        units = ["repo:" + name.lower() for name in batch["repo_name"].to_pylist()]
    else:
        units = ["doc:" + id for id in batch["id"].to_pylist()]
    docs = texts(source, batch)
    return docs, units, sum(len(text.encode()) for text in docs)


def sample() -> tuple[list[str], list[int], dict[str, dict[str, float]]]:
    """The training sample (texts, unit ids) and its composition per source (its target bytes, the fraction of its rows
    taken, and the documents, bytes and authorship units taken), row groups read in parallel, the sample kept in
    order (module docstring)."""
    tasks, composition = [], {}
    for s, share in byte_shares().items():
        fraction, at = share * SAMPLE_BYTES / sum(shard_bytes(s)), 0
        composition[s] = {"target_bytes": share * SAMPLE_BYTES, "fraction": fraction, "documents": 0, "bytes": 0}
        for path in curated_shards(s, "train"):
            md = pq.read_metadata(path)
            for g in range(md.num_row_groups):
                tasks.append((s, path, g, at, fraction))
                at += md.row_group(g).num_rows
    docs, units, ids, held = [], [], {}, {s: set() for s in composition}
    with ProcessPoolExecutor(mp_context=SPAWN) as pool:
        for (s, *_), (batch, names, size) in zip(tasks, pool.map(sample_row_group, tasks), strict=True):
            docs += batch
            units += [ids.setdefault(name, len(ids)) for name in names]
            held[s].update(names)
            composition[s]["documents"] += len(batch)
            composition[s]["bytes"] += size
    for s, names in held.items():
        composition[s]["units"] = len(names)
    return docs, units, composition


def train(docs: list[str], units: list[int], *, pattern: str, score: Score) -> list[tuple[int, int]]:
    """VOCAB_SIZE - len(SPECIAL_TOKENS) - 256 merges (merge i makes token 256 + i) trained on the documents (each from
    its authorship unit) within the chunks of `pattern`, by SA-BPE's F·ln R or by frequency alone (the crate's
    train)."""
    return corpus_crate().train(docs, units, pattern=pattern, n_merges=VOCAB_SIZE - len(SPECIAL_TOKENS) - 256,
                                score=score)


def vocabulary(merges: list[tuple[int, int]]) -> list[bytes]:
    """The mergeable tokens of merges, bytes in rank order: the 256 bytes, then each merge's."""
    mergeable = [bytes([i]) for i in range(256)]
    for left, right in merges:
        mergeable.append(mergeable[left] + mergeable[right])
    return mergeable


def provenance() -> dict:
    """Where and when a step ran: the checkout's commit, the UTC time, the host, its cores and memory, and the versions
    of Python and of the packages that compute the measures."""
    commit = subprocess.run(["git", "-c", f"safe.directory={ROOT}", "-C", str(ROOT), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    return {"commit": commit, "utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "host": platform.node(), "machine": platform.machine(), "cpus": os.cpu_count(),
            "memory_gb": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30,
            "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name)
                         for name in ("tiktoken", "tokenizers", "numpy", "scipy", "pyarrow", "regex")}}


def vocabularies() -> None:
    """Every arm's tokenizer.tiktoken, tokenizer.pattern and merges.json in ABLATION/<arm>, trained at once on one draw
    of the sample, and RESULTS/<EXPERIMENT>-vocabularies.json (module docstring)."""
    for arm in ARMS:
        (ABLATION / arm).mkdir(parents=True)
    start, began = time.perf_counter(), provenance()
    docs, units, composition = sample()
    sampled = time.perf_counter() - start

    def timed(spec: tuple[str, Score]) -> tuple[list[tuple[int, int]], float]:
        begin = time.perf_counter()
        merges = train(docs, units, pattern=spec[0], score=spec[1])
        return merges, time.perf_counter() - begin

    arms = {}
    corpus_crate()  # built once, before the threads call it (pack.py's rule)
    with ThreadPoolExecutor(len(ARMS)) as pool:
        for (arm, (pattern, score)), (merges, seconds) in zip(ARMS.items(), pool.map(timed, ARMS.values()),
                                                              strict=True):
            save(ABLATION / arm, vocabulary(merges), pattern)
            (ABLATION / arm / "merges.json").write_text(json.dumps(merges))
            arms[arm] = {"pattern": pattern, "score": score, "merges": len(merges), "seconds": seconds,
                         "data_key": get_tokenizer(ABLATION / arm).data_key}
            print(f"{arm}: {len(merges):,} merges in {seconds:.0f} s, data key {arms[arm]['data_key']}", flush=True)
    record = {"sample": {"documents": len(docs), "bytes": sum(c["bytes"] for c in composition.values()),
                         "units": max(units) + 1, "seconds": sampled, "sources": composition},
              "arms": arms, "seconds": time.perf_counter() - start, "provenance": began}
    (RESULTS / f"{EXPERIMENT}-vocabularies.json").write_text(json.dumps(record, indent=1), encoding="utf-8")


def val_split(task: tuple[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """An arm's compiled val split of a source: each mergeable token's count and each document's bytes."""
    arm, corpus = task
    tokenizer = get_tokenizer(ABLATION / arm)
    path = split_path(corpus, "val", tokenizer)
    tokens = np.fromfile(path.with_suffix(".bin"), dtype=TOKEN_DTYPE)
    ends = np.concatenate([[0], np.cumsum(tokenizer.token_lengths[tokens])])
    return (np.bincount(tokens, minlength=tokenizer.bos_token_id),
            np.diff(ends[np.fromfile(path.with_suffix(".idx"), dtype=np.int64)]))


def describe(arm: str, counts: dict[str, np.ndarray]) -> dict:
    """An arm's planning, mechanism and tie-break measures from its token counts on each source's val split
    (val_split; module docstring)."""
    tokenizer = get_tokenizer(ABLATION / arm)
    lengths = tokenizer.token_lengths[:tokenizer.bos_token_id]
    held = np.zeros(tokenizer.bos_token_id, dtype=np.int64)
    sources = {}
    for corpus in SOURCES:
        held += counts[corpus]
        tokens = int(counts[corpus].sum())
        sources[corpus] = {"bytes": int(counts[corpus] @ lengths), "tokens": tokens,
                           "bytes_per_token": float(counts[corpus] @ lengths / tokens)}
    shares = byte_shares()

    def mixed(members: tuple[str, ...] | list[str]) -> float:
        """Bytes per token over the members' file-stage byte shares."""
        return sum(shares[s] for s in members) / sum(shares[s] / sources[s]["bytes_per_token"] for s in members)

    entries = [b.decode("utf-8", "ignore") for b in tokenizer.token_bytes[:tokenizer.bos_token_id]]
    # a line break and the indentation after it ending the token (\Z: $ also matches before a final line break)
    fused, indented = regex.compile(r"[^\s\p{L}\p{N}][\r\n]"), regex.compile(r"[\r\n][ \t]+\Z")
    merged = held[256:]
    return {"data_key": tokenizer.data_key, "sources": sources,
            "file_stage_bytes_per_token": {"all": mixed(mixture.FILE_STAGE.sources)}
            | {name: mixed(group.sources) for name, group in mixture.FILE_STAGE.groups.items()},
            "train_tokens_over_draw": {
                group: sum(sum(shard_bytes(s)) / sources[s]["bytes_per_token"] for s, g in mixture.PART.items()
                           if g == group) / mixture.draw(group) for group in mixture.IN_PART},
            "line_break_entries": {"line_break": sum("\n" in t or "\r" in t for t in entries),
                                   "punctuation_line_break": sum(bool(fused.search(t)) for t in entries),
                                   "line_break_indentation": sum(bool(indented.search(t)) for t in entries)},
            "merged_reads": {"never": int((merged == 0).sum()), "under 10": int((merged < 10).sum()),
                             "under 100": int((merged < 100).sum())}}


def indented_lines(source: str) -> list[tuple[str, str]]:
    """The first INDENTED_LINES indented lines of a source's val texts: (indentation, statement)."""
    def lines():
        for path in curated_shards(source, "val"):
            f = pq.ParquetFile(path)
            for g in range(f.num_row_groups):
                for doc in texts(source, f.read_row_group(g)):
                    for line in doc.split("\n"):
                        if m := INDENTED.fullmatch(line):
                            yield m.group(1), m.group(2)
    return list(itertools.islice(lines(), INDENTED_LINES))


def depth_invariance(source: str) -> dict[str, dict[str, float]]:
    """{arm: lines, invariant, rate} of a program source: its indented lines (indented_lines) whose tokens are the
    indentation's, then the statement's as at depth 0."""
    lines = indented_lines(source)
    result = {}
    for arm in ARMS:
        enc = get_tokenizer(ABLATION / arm).enc
        whole = enc.encode_ordinary_batch([indent + statement for indent, statement in lines])
        indents = enc.encode_ordinary_batch([indent for indent, _ in lines])
        statements = enc.encode_ordinary_batch([statement for _, statement in lines])
        invariant = sum(w == i + s for w, i, s in zip(whole, indents, statements, strict=True))
        result[arm] = {"lines": len(lines), "invariant": invariant, "rate": invariant / len(lines)}
    return result


def fidelity(task: tuple[str, str]) -> dict[str, int]:
    """An arm's encoding checked on a source's first FIDELITY_DOCUMENTS val texts (the documents and chunks checked):
    its pattern's chunks (Python regex, whose chunks fancy-regex's equal) cover every text, tiktoken's encoding (by the
    rank of the joined bytes) equals the training segmentation chunk by chunk, and every text decodes back. The
    training segmentation is Hugging Face tokenizers' BPE model over the arm's merges, each byte spelled as the
    Latin-1 character of its value: listed pairs merged by rank, each over every occurrence from the left (corpus/crate
    train.rs merge_pair), never a pair whose bytes another made."""
    arm, source = task
    tokenizer = get_tokenizer(ABLATION / arm)
    spelled = [entry.decode("latin-1") for entry in tokenizer.token_bytes[:tokenizer.bos_token_id]]
    merges = json.loads((ABLATION / arm / "merges.json").read_text())
    bpe = models.BPE(vocab={s: i for i, s in enumerate(spelled)}, merges=[(spelled[a], spelled[b]) for a, b in merges])
    pattern = regex.compile(tokenizer.pattern)
    first = pq.ParquetFile(curated_shards(source, "val")[0]).read_row_group(0)
    docs = texts(source, first.slice(0, FIDELITY_DOCUMENTS))
    chunks = 0
    for doc in docs:
        pieces = [m.group().encode() for m in pattern.finditer(doc)]
        if b"".join(pieces) != doc.encode():
            raise RuntimeError(f"{arm}: its pattern leaves text of a {source} document uncovered")
        if tokenizer.enc.decode(tokenizer.enc.encode_ordinary(doc)) != doc:
            raise RuntimeError(f"{arm}: a {source} document does not decode back")
        for piece in pieces:
            if tokenizer.enc._encode_single_piece(piece) != [t.id for t in bpe.tokenize(piece.decode("latin-1"))]:
                raise RuntimeError(f"{arm}: tiktoken encodes {piece!r} ({source}) unlike the trained merges")
        chunks += len(pieces)
    return {"documents": len(docs), "chunks": chunks}


def intrinsic() -> None:
    """Every arm's compiled val split of every source, the equal-bytes check, and RESULTS/<EXPERIMENT>-intrinsic.json
    (module docstring), all in one process pool: depth invariance per program source and fidelity per arm and source
    (which read no compiled data) beside the compiles, every split at once, each split's counts as it is compiled.
    "seconds": "compiled", the wall time to the last compile's end (the checks overlap the compiles), and "all"."""
    start, began = time.perf_counter(), provenance()
    programs = [s for s, spec in SOURCES.items() if spec.category in PROGRAMS]
    checks = [(arm, source) for arm in ARMS for source in SOURCES]
    with ProcessPoolExecutor(mp_context=SPAWN) as pool, ThreadPoolExecutor(len(checks)) as threads:
        for arm, corpus in checks:
            split_path(corpus, "val", get_tokenizer(ABLATION / arm)).parent.mkdir(parents=True)
        invariance = [pool.submit(depth_invariance, source) for source in programs]
        fidelities = [pool.submit(fidelity, check) for check in checks]

        def compiled(check: tuple[str, str]) -> Future:
            compile_source(check[1], "val", ABLATION / check[0], pool)
            return pool.submit(val_split, check)

        # a val split is a few row groups: one split at a time would leave the pool idle
        splits = dict(zip(checks, threads.map(compiled, checks), strict=True))
        compiled_seconds = time.perf_counter() - start
        counts, sizes = {}, {}
        for check, future in splits.items():
            counts[check], sizes[check] = future.result()
        for corpus in SOURCES:
            if not all(np.array_equal(sizes["sa-bpe", corpus], sizes[arm, corpus]) for arm in CHALLENGERS):
                raise RuntimeError(f"{corpus}: the arms' compiled val documents differ in bytes")
        record = {"arms": {arm: describe(arm, {corpus: counts[arm, corpus] for corpus in SOURCES})
                           | {"depth_invariance": {}, "fidelity": {}} for arm in ARMS}}
        for source, future in zip(programs, invariance, strict=True):
            for arm, result in future.result().items():
                record["arms"][arm]["depth_invariance"][source] = result
        for (arm, source), future in zip(checks, fidelities, strict=True):
            record["arms"][arm]["fidelity"][source] = future.result()
    record |= {"seconds": {"compiled": compiled_seconds, "all": time.perf_counter() - start}, "provenance": began}
    (RESULTS / f"{EXPERIMENT}-intrinsic.json").write_text(json.dumps(record, indent=1), encoding="utf-8")


def document_set(run) -> dict:
    """The set's shards (module docstring) for the d16 run `run` at RATIO under sa-bpe: {"file": {source: its leading
    train shards, the file stage's}, "repository": {source: its train shards after those, the repository stage's},
    "draws": {seq_len: {corpus: its tokens in the run}}}; the repositories corpora's partitions: data."""
    from train.stages import drawn

    measured = json.loads((RESULTS / f"{EXPERIMENT}-intrinsic.json").read_text(encoding="utf-8"))["arms"]["sa-bpe"]
    estimate = {s: np.array(shard_bytes(s)) / measured["sources"][s]["bytes_per_token"] for s in SOURCES}

    def shards(source: str, tokens: float, start: int) -> int:
        """The fewest train shards of a source from shard `start` whose estimated sa-bpe tokens hold `tokens`."""
        held = np.cumsum(estimate[source][start:])
        if held[-1] < tokens:
            raise RuntimeError(f"{source}: its train shards from {start} hold {held[-1]:,.0f} of {tokens:,.0f} tokens")
        return int(np.searchsorted(held, tokens)) + 1

    def draws(stage: mixture.Stage) -> dict[str, float]:
        """Each corpus's tokens in the run: its share of the base model's stage (Stage.shares of the sources' train
        tokens; the repositories corpora, unestimated, alike within their group) at the run's tokens in the stage."""
        shares = stage.shares({s: estimate[s].sum() if s in SOURCES else 1.0 for s in stage.sources}, stage.tokens)
        return {s: share * drawn(stage, run) * stage.seq_len for s, share in shares.items()}

    file_draws, repo_draws = draws(mixture.FILE_STAGE), draws(mixture.REPO_STAGE)
    file = {s: shards(s, tokens * (1 if s in mixture.FILE_STAGE.code else MARGIN), 0)
            for s, tokens in file_draws.items()}
    repository = {s: shards(s, tokens * (1 if s in mixture.REPO_STAGE.supply else MARGIN), file.get(s, 0))
                  for s, tokens in repo_draws.items() if s in SOURCES}
    return {"file": file, "repository": repository, "draws": {"8192": file_draws, "32768": repo_draws}}


def data() -> None:
    """Every arm's training set, compiled and packed, and RESULTS/<EXPERIMENT>-data.json (module docstring): each corpus
    of each arm compiled, then packed, as its own job, every job into one process pool; compile_stage, beside the other
    sources' compiles, chooses the partitions and compiles sa-bpe's repositories corpora, then the set check, then the
    repository sources' jobs and the other arms' repositories corpora, one arm after another (each fills the cores).
    "seconds": "compiled", the wall time to the last compile's end (packs overlap the compiles), and "all"."""
    # the training stack (the model's kernels) here alone: the measures' spawned workers import this module
    from run import base_plan
    from train.stages import drawn, max_ratio

    start, began = time.perf_counter(), provenance()
    run = base_plan(depth=DEPTH, ratio=RATIO)
    shards = document_set(run)
    stages = (mixture.FILE_STAGE, mixture.REPO_STAGE)
    leading = {s: shards["file"].get(s, 0) + shards["repository"].get(s, 0) for s in SOURCES}
    compiled = []  # each compile's end

    def pack(corpus: str, directory: Path) -> None:
        """A compiled corpus's packs in each stage that reads it: its train split's shards in the set, its val split
        whole."""
        for stage in stages:
            if corpus not in stage.sources:
                continue
            if corpus in mixture.REPOSITORIES.values():
                train = range(shard_count(corpus, "train", directory))
            elif stage is mixture.FILE_STAGE:
                train = range(shards["file"][corpus])
            else:
                first = shards["file"].get(corpus, 0)
                train = range(first, first + shards["repository"][corpus])
            for split, listed in (("train", train), ("val", range(shard_count(corpus, "val", directory)))):
                write(corpus, split, directory, packed_shards(corpus, split, directory, listed, stage.seq_len),
                      stage.seq_len)

    def source(arm: str, corpus: str) -> None:
        compile_source(corpus, "train", ABLATION / arm, pool, leading=leading[corpus])
        compiled.append(time.perf_counter())
        pack(corpus, ABLATION / arm)

    def repositories(parts: list[int]) -> None:
        for arm in ARMS:
            if arm != "sa-bpe":  # compile_stage compiled sa-bpe's as it chose the partitions
                compile_repositories(ABLATION / arm, parts, pool)
            compiled.append(time.perf_counter())
            for corpus in mixture.REPOSITORIES.values():
                pack(corpus, ABLATION / arm)

    corpus_crate()  # built once, before the threads call it (pack.py's rule)
    with (ProcessPoolExecutor(mp_context=SPAWN) as pool,
          ThreadPoolExecutor(len(ARMS) * len(SOURCES) + 2) as threads):  # the source jobs, repositories, compile_stage
        partitions = threads.submit(compile_stage, ABLATION / "sa-bpe", pool,
                                    MARGIN * sum(shards["draws"]["32768"][c] for c in mixture.REPOSITORIES.values()))
        # the sources the set check does not gate compile while compile_stage chooses the partitions
        jobs = [threads.submit(source, arm, corpus) for arm in ARMS for corpus in SOURCES
                if corpus not in mixture.REPOSITORY_SOURCES]
        parts = partitions.result()
        if max(shards["file"][s] for s in mixture.REPOSITORY_SOURCES) > train_parts().index(parts[0]):
            raise RuntimeError("the file stage's repository shards reach the repository stage's partitions")
        jobs += [threads.submit(source, arm, corpus) for arm in ARMS for corpus in mixture.REPOSITORY_SOURCES]
        jobs.append(threads.submit(repositories, parts))
        for job in jobs:
            job.result()
    chosen = {"file": shards["file"], "repository": shards["repository"], "partitions": parts,
              "draws": shards["draws"]}
    compiled_seconds = max(compiled) - start
    arms = {}
    for arm in ARMS:
        tokenizer = get_tokenizer(ABLATION / arm)
        ratio = max_ratio(tokenizer, depth=DEPTH)
        arm_run = base_plan(depth=DEPTH, ratio=ratio)
        read = {}
        for stage in stages:
            rows, count = stage_rows(tokenizer, stage), drawn(stage, arm_run)
            shares, draws, _ = weights(stage, rows, count)
            read[str(stage.seq_len)] = {n: {"rows": rows[n], "weight": shares[n], "read": d / rows[n]}
                                        for n, d in draws.items()}
        arms[arm] = {"data_key": tokenizer.data_key, "ratio": ratio, "tokens_per_param": arm_run.tokens_per_param,
                     "num_iterations": arm_run.num_iterations, "total_batch_size": arm_run.total_batch_size,
                     "repo_start": arm_run.repo_start, "lr": arm_run.lr, "stages": read}
        print(f"{arm}: ratio {ratio:.2f}, {arm_run.num_iterations:,} steps of {arm_run.total_batch_size:,} tokens",
              flush=True)
    record = {"run": {"depth": DEPTH, "ratio": RATIO, "num_iterations": run.num_iterations,
                      "total_batch_size": run.total_batch_size, "repo_start": run.repo_start},
              "set": chosen, "arms": arms,
              "seconds": {"compiled": compiled_seconds, "all": time.perf_counter() - start},
              "provenance": began}
    (RESULTS / f"{EXPERIMENT}-data.json").write_text(json.dumps(record, indent=1), encoding="utf-8")


def evaluated(lines: list[dict], field: str) -> dict[tuple, tuple[float, list[bool]]]:
    """{(the run's `field`, seed): (code bits per byte, per PARSED problem in the benchmark's order whether its healed
    greedy completion compiles)} of every evaluated run in an experiment's records (a diverged run has none)."""
    return {(line["config"][field], line["config"]["seed"]): (
        line["measures"]["code"], [o == "parses" for o in line["evaluations"][-1]["parse"][PARSED]["outcomes"]])
        for line in lines if line["measures"] is not None}


def contrast(runs: dict[tuple, tuple[float, list[bool]]], arm: object, baseline: object) -> dict:
    """An arm against the baseline (module docstring, Decisions; evaluated's runs) over the seeds where both have an
    evaluation: parsing by the exact sign test, code bits per byte by the difference of the means beyond its bound."""
    seeds = sorted(seed for value, seed in runs if value == arm and (baseline, seed) in runs)
    if not seeds:
        raise RuntimeError(f"no seed evaluated under both {arm} and {baseline}")
    code = {v: statistics.fmean(runs[v, seed][0] for seed in seeds) for v in (arm, baseline)}
    # per problem, the seeds whose completion parses under the arm minus under the baseline
    seeds_parsing = (np.sum([runs[arm, seed][1] for seed in seeds], axis=0)
                     - np.sum([runs[baseline, seed][1] for seed in seeds], axis=0))
    here, there = int((seeds_parsing > 0).sum()), int((seeds_parsing < 0).sum())
    p = binomtest(here, here + there).pvalue if here + there else 1.0
    difference = code[arm] - code[baseline]
    margin = norm.ppf(1 - ALPHA / 2) * CODE_SIGMA * math.sqrt(2 / len(seeds))
    return {"seeds": seeds, "problems_parsing_more_here": here, "problems_parsing_more_in_baseline": there, "p": p,
            "parsing": "equal" if p >= ALPHA else "better" if here > there else "worse",
            "code_difference": difference, "code_margin": margin,
            "code": "better" if difference < -margin else "worse" if difference > margin else "equal"}


def decide(lines: list[dict]) -> dict:
    """The runs' analysis (RESULTS/tokenizer.json "analysis", ablations/records.py): each challenger's contrast with
    sa-bpe, with the difference of the arms' MBPP+ unparsed shares, and the decision (module docstring), over
    RESULTS/tokenizer.jsonl's records."""
    runs = evaluated(lines, "tokenizer")
    contrasts = {}
    for arm in CHALLENGERS:
        c = contrast(runs, arm, "sa-bpe")
        unparsed = {v: 1 - statistics.fmean(itertools.chain.from_iterable(runs[v, seed][1] for seed in c["seeds"]))
                    for v in (arm, "sa-bpe")}
        contrasts[arm] = c | {"unparsed_difference": unparsed[arm] - unparsed["sa-bpe"]}
    verdicts = {arm: (c["parsing"], c["code"]) for arm, c in contrasts.items()}
    qualified = {arm for arm, v in verdicts.items() if "worse" not in v and "better" in v}
    record = json.loads((RESULTS / f"{EXPERIMENT}-intrinsic.json").read_text(encoding="utf-8"))
    rare = {arm: measures["merged_reads"]["under 100"] for arm, measures in record["arms"].items()}
    frequency = "bpe" in qualified or (verdicts["bpe"] == ("equal", "equal") and rare["bpe"] < rare["sa-bpe"])
    # the positive control gates the decision: undetected, the corpus keeps sa-bpe's pattern and score; detected, it
    # parses worse, so it never qualifies and the hybrid is the only pattern that can
    control = verdicts["sa-bpe-fused"][0] == "worse"
    decision = {"pattern": "hybrid" if control and "hybrid" in qualified else "sa-bpe",
                "score": "frequency" if control and frequency else "sa-bpe",
                "control_detected": control}
    return {"contrasts": contrasts, "decision": decision}


def adopt() -> None:
    """The decided vocabulary (RESULTS/<EXPERIMENT>.json "analysis" "decision") as the corpus's tokenizer, TOKENIZER:
    the arm of the decided pattern and score, copied, or that pair trained on the sample when no arm holds it. It writes
    over the provisional tokenizer (the sa-bpe arm's copy) in place: compiled and packed data are keyed by a
    vocabulary's data key, so adopting sa-bpe reuses the provisional corpus and any other vocabulary compiles into new
    directories (run.py's stale guard rejects a run recorded under the replaced one)."""
    decision = json.loads((RESULTS / f"{EXPERIMENT}.json").read_text(encoding="utf-8"))["analysis"]["decision"]
    pattern, score = ARMS[decision["pattern"]][0], decision["score"]
    TOKENIZER.mkdir(parents=True, exist_ok=True)
    arm = next((arm for arm, spec in ARMS.items() if spec == (pattern, score)), None)
    if arm is None:
        docs, units, _ = sample()
        save(TOKENIZER, vocabulary(train(docs, units, pattern=pattern, score=score)), pattern)
    else:
        for name in ("tokenizer.tiktoken", "tokenizer.pattern"):
            shutil.copyfile(ABLATION / arm / name, TOKENIZER / name)
    print(f"the corpus's tokenizer: {decision['pattern']}'s pattern, score {score}, data key "
          f"{get_tokenizer(TOKENIZER).data_key}", flush=True)


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["vocabularies"]:
            vocabularies()
        case ["intrinsic"]:
            intrinsic()
        case ["data"]:
            data()
        case ["adopt"]:
            adopt()
        case _:
            raise SystemExit("usage: python -m ablations.tokenizer vocabularies|intrinsic|data|adopt")
