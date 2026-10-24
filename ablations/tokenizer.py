"""
The tokenizer ablation: it trains every candidate vocabulary of the corpus's tokenizer, measures them on the corpus,
decides the split pattern and merge score, and adopts the decided vocabulary as the corpus's tokenizer
(nanochat/corpus/tokenizer.py TOKENIZER). Each candidate is a vocabulary of 32,768 ids (32,764 mergeable tokens, BOS
and 3 FIM sentinels) trained on the same sample by the same trainer (train: the data crate's), encoded by tiktoken,
and trained on. The sample, the val splits and the packs come from the curated shards through nanochat/corpus
(mixture.py, compile.py, pack.py).

Sample (sample): of each file-stage source's curated train split, its share of the stage's bytes (byte_shares: its
share of the stage's tokens, Stage.shares, at the reference tokenizer's bytes per token) as its share of SAMPLE_BYTES,
a systematic sample in shard order: tokenizer training data "representative of model training data"
(arXiv 2405.05417 §5); across tokenizer training mixes a subset's compression "only varies by 5 to 6 percentage
points", higher as the mix holds more of it (arXiv 2402.01035 §2.3); over languages, proportional tokenizer-training
weights give lower bits per byte than equal ones (arXiv 2610.12144 Table 6, §7). The sample's composition is not an arm
(normalization_doc.md §9). An authorship unit (SA-BPE's R) is the repository, its name lowercased, for repository files
and functions; any other document is its own.

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
  so a tie in the runs is likely [inference], which the val coverage breaks.
Not run: tiktoken encodes by merge rank alone, which has no form for PickyBPE's token removals (arXiv 2409.04599
Alg. 2) or BoundlessBPE's supermerges (arXiv 2504.00178 §3.2); SuperBPE moved HumanEval and MBPP pass@10 in opposite
directions, neither significantly (arXiv 2503.13423 Table 1).

vocabularies: every arm trained at once on one draw of the sample (every merge is an argmax with fixed tie-breaks, so
an arm does not depend on the others); ABLATION/<arm>: its vocabulary, pattern and merges;
RESULTS/tokenizer-vocabularies.json: the sample's composition per source (target bytes, the fraction of rows taken,
the documents, bytes and units taken) and each arm's training time.

intrinsic: every arm's compiled val split of every source, and RESULTS/tokenizer-intrinsic.json, only the measures
that plan the runs, gate their validity, test an arm's mechanism or break the decision's tie:
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
pack.py, given the arm's directory), and RESULTS/tokenizer-data.json: the set and, per arm, its packed rows per stage,
its d16 max_ratio and plan, and the share of each corpus's packed rows its run reads. The set holds a d16 run at RATIO
tokens per scaling parameter under sa-bpe (arXiv 2203.15556 Table 3), each source at its share of the base model's
stage (Stage.shares of the sources' train tokens), the tokens of a shard estimated from its text bytes at sa-bpe's val
bytes per token: of each code source the leading train shards that fill its file-stage draw, so code binds every arm's
max_ratio; of a "supply" source (notebooks, functions, traces), its draw, read whole once; of text and math the leading
shards of each stage, and of the repositories corpora the trailing partitions (compile.stage_parts, under sa-bpe),
holding MARGIN times their draws, so they never bind.

runs (conf/experiment/tokenizer.yaml: `python -m nanochat.run experiment=tokenizer`): every arm at d16, SEEDS seeds
each, at equal bytes: each at its max_ratio (ratio null), so it reads the set's code documents once and its tokens are
their bytes over its bytes per token; text and math are drawn by rows. Each run's plan is its own (schedule.plan at
its tokens), its contexts the corpus's 8,192 and 32,768 tokens (eval/bpb.py's byte-offset buckets: the bytes of
context a better compressor sees, arXiv 2503.13423 §3.1, App. B.1), trained and evaluated through nanochat/run.py,
the records by ablations/records.py. RESULTS/tokenizer.json "analysis" (decide): each challenger's contrasts with
sa-bpe, the baseline, and the decision.

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
  score. Else a challenger qualifies iff it is worse on neither and better on one. Pattern: of sa-bpe-fused and the
  hybrid, the qualifier with the lower MBPP+ unparsed share, else sa-bpe's. Score: bpe iff it qualifies, or is equal
  on both and the val documents read fewer of its merged tokens under 100 times than of sa-bpe's (SA-BPE's claimed
  benefit, a higher coverage than BPE's), carried to the adopted pattern [inference]; otherwise SA-BPE.

adopt: the decided vocabulary (RESULTS/tokenizer.json "analysis" "decision") as the corpus's tokenizer: the arm of
the decided pattern and score, or, bpe's score carried to another arm's pattern, that pair trained on the sample.

The comparisons, their measures and the recorded results: ablations/tokenizer.md.

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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow.parquet as pq
import regex
from scipy.stats import binomtest, norm
from tokenizers import models

from nanochat.config import VOCAB_SIZE
from nanochat.corpus import mixture
from nanochat.corpus.compile import compile_repositories, compile_source, stage_parts
from nanochat.corpus.loader import TOKEN_DTYPE, mixture_rows, split_path, weights
from nanochat.corpus.pack import packed_shards, shard_count, write
from nanochat.corpus.parallel import SPAWN
from nanochat.corpus.schema import EXERCISE, FUNCTION, REPOSITORY_FILE, SYNTHETIC_FILE
from nanochat.corpus.sources import SOURCES, curated_shards, text_bytes, texts
from nanochat.corpus.tokenizer import ABLATION, SPECIAL_TOKENS, SPLIT_PATTERN, TOKENIZER, get_tokenizer, save
from nanochat.native import data_crate
from nanochat.paths import RESULTS, ROOT

Score = Literal["sa-bpe", "frequency"]

# The sample's bytes: the reference tokenizer's sample's (its curated splits' Parquet footers)
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
# The runs' depth, which the set is sized for, and seeds per arm (conf/experiment/tokenizer.yaml)
DEPTH, SEEDS = 16, 2
# The set's size: a d16 run at Chinchilla's 20 tokens per scaling parameter under sa-bpe (arXiv 2203.15556 Table 3)
RATIO = 20
# The set's margin over the draws of the corpora that must not bind (text, math, the repositories corpora): above the
# code shards' rounding up to whole shards and the arms' spread in compression [inference]
MARGIN = 1.5
# The decisions' significance level, the conventional 0.05
ALPHA = 0.05
# The pooled seed standard deviation of a run's 8K code bits per byte (seed variances pooled over configurations): two
# d8 configurations at 20 tokens per parameter on the reference tokenizer's data, data seeds 0 to 2 (six runs; no
# results/ file holds them), carried to d16 and to the corpus [inference]
CODE_SIGMA = 0.0034
# The benchmark of the parse test (Decisions)
PARSED = "mbpp_plus"
# The categories whose documents are whole program texts in their source's language (depth invariance)
PROGRAMS = (REPOSITORY_FILE, EXERCISE, SYNTHETIC_FILE, FUNCTION)
# The fidelity check's texts: the first of each source's val split
FIDELITY_DOCUMENTS = 200
# The depth-invariance check's lines: the first indented lines of each program source's val split
INDENTED_LINES = 50_000
# An indented line: its leading spaces and tabs, then its statement
INDENTED = regex.compile(r"([ \t]+)(\S.*)")


@functools.cache
def shard_bytes(source: str) -> list[int]:
    """The text bytes of each shard of the source's curated train split (Parquet footers)."""
    return [text_bytes(pq.read_metadata(path), "content") for path in curated_shards(source, "train")]


def train_bytes(source: str) -> int:
    """The text bytes of the source's curated train split."""
    return sum(shard_bytes(source))


def byte_shares() -> dict[str, float]:
    """Each file-stage source's share of the stage's bytes: its share of the stage's tokens (Stage.shares, tokens
    counted at the reference tokenizer's bytes per token) at its bytes per token."""
    sources, per_token = mixture.FILE_STAGE.sources, mixture.BYTES_PER_TOKEN
    shares = mixture.FILE_STAGE.shares({s: train_bytes(s) / per_token[s] for s in sources}, mixture.FILE_STAGE.tokens)
    stage_bytes = sum(shares[s] * per_token[s] for s in sources)
    return {s: shares[s] * per_token[s] / stage_bytes for s in sources}


def sample_row_group(task: tuple[str, Path, int, int, float]) -> tuple[list[str], list[str]]:
    """Row group g of a curated train shard: the texts (sources.texts) and authorship units of its rows that the
    source's systematic sample takes, `start` the source's rows before the group: row i when floor((i + 1) * fraction)
    exceeds floor(i * fraction). A row naming its repository (a repository file, a function) has that repository as its
    unit, its name lowercased (one unit across the sources that hold it); any other document is its own."""
    source, path, g, start, fraction = task
    group = pq.ParquetFile(path).read_row_group(g)
    rows = np.arange(start, start + group.num_rows)
    batch = group.take(np.flatnonzero(np.floor((rows + 1) * fraction) > np.floor(rows * fraction)))
    if "repo_name" in batch.column_names:
        units = ["repo:" + name.lower() for name in batch["repo_name"].to_pylist()]
    else:
        units = ["doc:" + id for id in batch["id"].to_pylist()]
    return texts(source, batch), units


def sample() -> tuple[list[str], list[int], dict[str, dict[str, float]]]:
    """The training sample (texts, unit ids) and its composition per source (its target bytes, the fraction of its rows
    taken, and the documents, bytes and authorship units taken), row groups read in parallel, the sample kept in
    order (module docstring)."""
    tasks, composition = [], {}
    for s, share in byte_shares().items():
        fraction, at = share * SAMPLE_BYTES / train_bytes(s), 0
        composition[s] = {"target_bytes": share * SAMPLE_BYTES, "fraction": fraction, "documents": 0, "bytes": 0}
        for path in curated_shards(s, "train"):
            md = pq.read_metadata(path)
            for g in range(md.num_row_groups):
                tasks.append((s, path, g, at, fraction))
                at += md.row_group(g).num_rows
    docs, units, ids, held = [], [], {}, {s: set() for s in composition}
    with ProcessPoolExecutor(mp_context=SPAWN) as pool:
        for (s, *_), (batch, names) in zip(tasks, pool.map(sample_row_group, tasks), strict=True):
            docs += batch
            units += [ids.setdefault(name, len(ids)) for name in names]
            held[s].update(names)
            composition[s]["documents"] += len(batch)
            composition[s]["bytes"] += sum(len(text.encode()) for text in batch)
    for s, names in held.items():
        composition[s]["units"] = len(names)
    return docs, units, composition


def train(docs: list[str], units: list[int], *, pattern: str, score: Score) -> list[tuple[int, int]]:
    """VOCAB_SIZE - len(SPECIAL_TOKENS) - 256 merges (merge i makes token 256 + i) trained on the documents (each from
    its authorship unit) within the chunks of `pattern`, by SA-BPE's F·ln R or by frequency alone (the crate's
    train)."""
    return data_crate().train(docs, units, pattern=pattern, n_merges=VOCAB_SIZE - len(SPECIAL_TOKENS) - 256,
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
    of the sample, and RESULTS/tokenizer-vocabularies.json (module docstring)."""
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
    data_crate()  # built once, before the threads call it (pack.py's rule)
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
    (RESULTS / "tokenizer-vocabularies.json").write_text(json.dumps(record, indent=1), encoding="utf-8")


def describe(arm: str) -> dict:
    """An arm's planning, mechanism and tie-break measures on the corpus's val splits (module docstring)."""
    tokenizer = get_tokenizer(ABLATION / arm)
    lengths = tokenizer.token_lengths[:tokenizer.bos_token_id]
    held = np.zeros(tokenizer.bos_token_id, dtype=np.int64)
    sources = {}
    for corpus in SOURCES:
        path = split_path(corpus, "val", tokenizer)
        tokens = np.memmap(path.with_suffix(".bin"), dtype=TOKEN_DTYPE, mode="r")
        counts = np.bincount(tokens, minlength=tokenizer.bos_token_id)
        held += counts
        sources[corpus] = {"bytes": int(counts @ lengths), "tokens": len(tokens),
                           "bytes_per_token": float(counts @ lengths / len(tokens))}
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
                group: sum(train_bytes(s) / sources[s]["bytes_per_token"] for s, g in mixture.PART.items()
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
    Latin-1 character of its value: listed pairs merged by rank, each over every occurrence from the left (data/crate
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
    """Every arm's compiled val split of every source, the equal-bytes check, and RESULTS/tokenizer-intrinsic.json
    (module docstring): the compiles, every split at once, then each arm's measures, depth invariance per program
    source and fidelity per arm and source, all in one process pool."""
    seconds, start, began = {}, time.perf_counter(), provenance()
    programs = [s for s, spec in SOURCES.items() if spec.category in PROGRAMS]
    checks = [(arm, source) for arm in ARMS for source in SOURCES]
    with ProcessPoolExecutor(mp_context=SPAWN) as pool, ThreadPoolExecutor(len(checks)) as threads:
        for arm, corpus in checks:
            split_path(corpus, "val", get_tokenizer(ABLATION / arm)).parent.mkdir(parents=True)
        # a val split is a few row groups: one split at a time would leave the pool idle
        list(threads.map(lambda check: compile_source(check[1], "val", ABLATION / check[0], pool), checks))
        seconds["compile"] = time.perf_counter() - start
        for corpus in SOURCES:
            sizes = []
            for arm in ARMS:
                tokenizer = get_tokenizer(ABLATION / arm)
                path = split_path(corpus, "val", tokenizer)
                tokens = np.fromfile(path.with_suffix(".bin"), dtype=TOKEN_DTYPE)
                ends = np.concatenate([[0], np.cumsum(tokenizer.token_lengths[tokens])])
                sizes.append(np.diff(ends[np.fromfile(path.with_suffix(".idx"), dtype=np.int64)]))
            if not all(np.array_equal(sizes[0], other) for other in sizes[1:]):
                raise RuntimeError(f"{corpus}: the arms' compiled val documents differ in bytes")
        begin = time.perf_counter()
        described = pool.map(describe, ARMS)  # every map submits at once
        invariance = pool.map(depth_invariance, programs)
        fidelities = pool.map(fidelity, checks)
        record = {"arms": {arm: result | {"depth_invariance": {}, "fidelity": {}}
                           for arm, result in zip(ARMS, described, strict=True)}}
        for source, by_arm in zip(programs, invariance, strict=True):
            for arm, result in by_arm.items():
                record["arms"][arm]["depth_invariance"][source] = result
        for (arm, source), result in zip(checks, fidelities, strict=True):
            record["arms"][arm]["fidelity"][source] = result
        seconds["measures"] = time.perf_counter() - begin
    record |= {"seconds": seconds | {"all": time.perf_counter() - start}, "provenance": began}
    (RESULTS / "tokenizer-intrinsic.json").write_text(json.dumps(record, indent=1), encoding="utf-8")


def document_set(pool: ProcessPoolExecutor, run) -> dict:
    """The set (module docstring) for the d16 run `run` at RATIO under sa-bpe: {"file": {source: its leading train
    shards, the file stage's}, "repository": {source: its train shards after those, the repository stage's},
    "partitions": the repositories corpora's train partitions, "draws": {seq_len: {corpus: its tokens in the run}}}."""
    from nanochat.train.stages import drawn

    measured = json.loads((RESULTS / "tokenizer-intrinsic.json").read_text(encoding="utf-8"))["arms"]["sa-bpe"]
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
    partitions = stage_parts(ABLATION / "sa-bpe", pool,
                             MARGIN * sum(repo_draws[c] for c in mixture.REPOSITORIES.values()))
    parts = [int(path.stem) for path in curated_shards(mixture.REPOSITORY_SOURCES[0], "train")]
    if max(file[s] for s in mixture.REPOSITORY_SOURCES) > parts.index(partitions[0]):
        raise RuntimeError("the file stage's repository shards reach the repository stage's partitions")
    return {"file": file, "repository": repository, "partitions": partitions,
            "draws": {"8192": file_draws, "32768": repo_draws}}


def data() -> None:
    """Every arm's training set, compiled and packed, and RESULTS/tokenizer-data.json (module docstring): every arm's
    train splits at once into one process pool, then their repositories corpora, then every pack at once."""
    # the training stack (the model's kernels) here alone: the measures' spawned workers import this module
    import torch

    from nanochat.model.gpt import GPT, gpt_config
    from nanochat.train.schedule import plan
    from nanochat.train.stages import drawn, max_ratio, stage_rows

    start, began = time.perf_counter(), provenance()
    with torch.device("meta"):
        model = GPT(gpt_config(depth=DEPTH, stage=mixture.FILE_STAGE))
    run = plan(model=model, ratio=RATIO)
    stages = (mixture.FILE_STAGE, mixture.REPO_STAGE)
    data_crate()  # built once, before the threads call it (pack.py's rule)
    with ProcessPoolExecutor(mp_context=SPAWN) as pool, ThreadPoolExecutor(os.cpu_count()) as threads:
        chosen = document_set(pool, run)
        leading = {s: chosen["file"].get(s, 0) + chosen["repository"].get(s, 0) for s in SOURCES}
        list(threads.map(lambda job: compile_source(job[1], "train", ABLATION / job[0], pool, leading=leading[job[1]]),
                         [(arm, s) for arm in ARMS for s in SOURCES]))
        list(threads.map(lambda arm: compile_repositories(ABLATION / arm, chosen["partitions"], pool), ARMS))
        compiled = time.perf_counter() - start
        packs = []
        for arm, stage in itertools.product(ARMS, stages):
            directory = ABLATION / arm
            for corpus in stage.sources:
                if corpus in mixture.REPOSITORIES.values():
                    train = range(shard_count(corpus, "train", directory))
                elif stage is mixture.FILE_STAGE:
                    train = range(chosen["file"][corpus])
                else:
                    first = chosen["file"].get(corpus, 0)
                    train = range(first, first + chosen["repository"][corpus])
                packs += [(corpus, "train", directory, train, stage.seq_len),
                          (corpus, "val", directory, range(shard_count(corpus, "val", directory)), stage.seq_len)]
        list(threads.map(lambda job: write(job[0], job[1], job[2], packed_shards(*job), job[4]), packs))
    packed = time.perf_counter() - start - compiled
    arms = {}
    for arm in ARMS:
        tokenizer = get_tokenizer(ABLATION / arm)
        ratio = max_ratio(tokenizer, depth=DEPTH)
        arm_run = plan(model=model, ratio=ratio)
        read = {}
        for stage in stages:
            rows, count = stage_rows(tokenizer, stage), drawn(stage, arm_run)
            shares, _ = weights(stage, rows, count)
            names = list(shares)
            draws = np.bincount(mixture_rows(np.array([shares[n] for n in names]), start=0, count=count),
                                minlength=len(names))
            read[str(stage.seq_len)] = {n: {"rows": rows[n], "weight": shares[n], "read": int(d) / rows[n]}
                                        for n, d in zip(names, draws, strict=True)}
        arms[arm] = {"data_key": tokenizer.data_key, "ratio": ratio, "tokens_per_param": arm_run.tokens_per_param,
                     "num_iterations": arm_run.num_iterations, "total_batch_size": arm_run.total_batch_size,
                     "repo_start": arm_run.repo_start, "lr": arm_run.lr, "stages": read}
        print(f"{arm}: ratio {ratio:.2f}, {arm_run.num_iterations:,} steps of {arm_run.total_batch_size:,} tokens",
              flush=True)
    record = {"run": {"depth": DEPTH, "ratio": RATIO, "num_iterations": run.num_iterations,
                      "total_batch_size": run.total_batch_size, "repo_start": run.repo_start},
              "set": chosen, "arms": arms,
              "seconds": {"compile": compiled, "pack": packed, "all": time.perf_counter() - start}, "provenance": began}
    (RESULTS / "tokenizer-data.json").write_text(json.dumps(record, indent=1), encoding="utf-8")


def decide(lines: list[dict]) -> dict:
    """The runs' analysis (RESULTS/tokenizer.json "analysis", ablations/records.py): each challenger's contrast with
    sa-bpe and the decision (module docstring), over RESULTS/tokenizer.jsonl's records."""
    # (arm, seed) of every evaluated run (a diverged run has none): its measures and, per MBPP+ problem in the
    # benchmark's order, whether its completion compiles
    evaluated = {}
    for run in lines:
        if run["measures"] is not None:
            outcomes = run["evaluations"][-1]["parse"][PARSED]["outcomes"]
            evaluated[run["config"]["tokenizer"], run["config"]["seed"]] = (run["measures"],
                                                                            [o == "parses" for o in outcomes])
    contrasts = {}
    for arm in CHALLENGERS:
        seeds = [seed for seed in range(SEEDS) if (arm, seed) in evaluated and ("sa-bpe", seed) in evaluated]
        if not seeds:
            raise RuntimeError(f"no seed evaluated under both {arm} and sa-bpe")
        both = {name: [evaluated[name, seed] for seed in seeds] for name in (arm, "sa-bpe")}
        code = {name: statistics.fmean(measures["code"] for measures, _ in runs_) for name, runs_ in both.items()}
        unparsed = {name: 1 - statistics.fmean(itertools.chain.from_iterable(parsed for _, parsed in runs_))
                    for name, runs_ in both.items()}
        # per problem, the seeds whose completion parses under the challenger minus under sa-bpe
        seeds_parsing = np.sum([parsed for _, parsed in both[arm]], axis=0) - np.sum(
            [parsed for _, parsed in both["sa-bpe"]], axis=0)
        here, there = int((seeds_parsing > 0).sum()), int((seeds_parsing < 0).sum())
        p = binomtest(here, here + there).pvalue if here + there else 1.0
        difference = code[arm] - code["sa-bpe"]
        margin = norm.ppf(1 - ALPHA / 2) * CODE_SIGMA * math.sqrt(2 / len(seeds))
        contrasts[arm] = {"seeds": seeds, "problems_parsing_more_here": here, "problems_parsing_more_in_sa-bpe": there,
                          "p": p, "parsing": "equal" if p >= ALPHA else "better" if here > there else "worse",
                          "code_difference": difference, "code_margin": margin,
                          "code": "better" if difference < -margin else "worse" if difference > margin else "equal",
                          "unparsed_difference": unparsed[arm] - unparsed["sa-bpe"]}
    verdicts = {arm: (c["parsing"], c["code"]) for arm, c in contrasts.items()}
    qualified = {arm for arm, v in verdicts.items() if "worse" not in v and "better" in v}
    record = json.loads((RESULTS / "tokenizer-intrinsic.json").read_text(encoding="utf-8"))
    rare = {arm: measures["merged_reads"]["under 100"] for arm, measures in record["arms"].items()}
    frequency = "bpe" in qualified or (verdicts["bpe"] == ("equal", "equal") and rare["bpe"] < rare["sa-bpe"])
    # the positive control gates the decision: undetected, the corpus keeps sa-bpe's pattern and score
    control = verdicts["sa-bpe-fused"][0] == "worse"
    pattern = min(qualified & {"sa-bpe-fused", "hybrid"}, key=lambda arm: contrasts[arm]["unparsed_difference"],
                  default="sa-bpe")
    decision = {"pattern": pattern if control else "sa-bpe",
                "score": "frequency" if control and frequency else "sa-bpe",
                "control_detected": control}
    return {"contrasts": contrasts, "decision": decision}


def adopt() -> None:
    """The decided vocabulary (RESULTS/tokenizer.json "analysis" "decision") as the corpus's tokenizer, TOKENIZER: the
    arm of the decided pattern and score, copied, or that pair trained on the sample when no arm holds it."""
    decision = json.loads((RESULTS / "tokenizer.json").read_text(encoding="utf-8"))["analysis"]["decision"]
    pattern, score = ARMS[decision["pattern"]][0], decision["score"]
    TOKENIZER.mkdir(parents=True)
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
