"""
The tokenizer ablation (experiment EXPERIMENT: pre-tokenizer split pattern x BPE merge score): it trains every
candidate vocabulary of the corpus's tokenizer (corpus/train_tokenizer.py: one sample, one trainer, VOCAB_SIZE ids),
measures them on the corpus, decides the split pattern and merge score, and adopts the decided vocabulary as the
corpus's tokenizer (corpus/tokenizer.py TOKENIZER). The val splits and the packs come from the curated shards through
corpus (mixture.py, tokenize.py, pack.py).

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

intrinsic: every arm's tokenized val split of every source, and RESULTS/tokenizer_split_merge-intrinsic.json, only the
measures that plan the runs, gate their validity, test an arm's mechanism or break the decision's tie:
- planning: each source's bytes per token and the file stage's over its byte shares (an arm's tokens at equal bytes;
  the tokenize and pack budgets), and each group read in part's train tokens over its draw (below 1, the corpus's extent
  is too small for that arm: prepare.extent, mixture.SPREAD);
- validity, asserted: every arm holds the same bytes per document, and on each source's first val texts its encoding
  matches its merges (corpus/train_tokenizer.py fidelity);
- mechanism: the vocabulary entries holding a line break (the companion post's strongest MBPP correlate), fusing
  punctuation with one, and ending in a line break and indentation (the post's MinGram cb, whose MBPP completions
  mostly fail with IndentationError or TabError); depth invariance, the hybrid's clause (2): of each program source's
  first INDENTED_LINES indented val lines, those whose tokens are the indentation's, then the statement's as at depth 0;
- the tie-break: how many merged tokens the val documents read never, under 10 and under 100 times (SA-BPE's Coverage,
  arXiv 2604.14053 Table 3, graded);
- TokEval (arXiv 2608.18062, metrics.tokeval), recorded, read by no decision: each arm's global compression, fertility,
  Rényi efficiency, bigram and trigram entropy, UTF-8 integrity and character splits, AST boundary alignment and its
  per-category rates, identifier fragmentation, indentation consistency, vocabulary utilization and reconstruction, on
  each source's leading val documents up to TOKEVAL_BYTES, grouped as TokEval languages (tokeval_group). Of its 14
  metrics only AST alignment predicts MBPP, which the paper attributes "to the vocabulary's line-break handling" (§5),
  measured above directly, and Rényi efficiency predicts neither code bits per byte nor MBPP (Table 2).

data: every arm's training set, the same documents (the set) tokenized and packed under each arm (corpus tokenize.py
and pack.py, given the arm's directory), and RESULTS/tokenizer_split_merge-data.json: the set and, per arm, its packed
rows per stage, its d16 max_tokens_per_param and plan, and the share of each corpus's packed rows its run reads. The set
holds a d16 run at TOKENS_PER_PARAM tokens per non-embedding parameter under sa-bpe (arXiv 2203.15556 Table 3), each
source at its share of the base model's stage (Stage.shares of the sources' train tokens), the tokens of a shard
estimated from its text bytes at sa-bpe's val bytes per token: of each code source the leading train shards that fill
its file-stage draw, so code binds every arm's max_tokens_per_param; of a "supply" source (notebooks, functions,
traces), its draw, read whole once; of text and math the leading shards of each stage, and of the repositories corpora
the trailing partitions (tokenize.tokenize_repo_level_stage, under sa-bpe), holding MARGIN times their draws, so they
never bind.

runs (conf/experiment/tokenizer_split_merge.yaml: `python run.py experiment=tokenizer_split_merge`): every arm at d16,
seeds 0 and 1, at equal bytes: each at its max_tokens_per_param (tokens_per_param null), so it reads the set's code
documents once and its tokens are their bytes over its bytes per token; text and math are drawn by rows. Each run's
plan is its own (schedule.plan at its tokens), its contexts the corpus's 8,192 and 32,768 tokens (eval/bpb.py's
byte-offset buckets: the bytes of
context a better compressor sees, arXiv 2503.13423 §3.1, App. B.1), trained and evaluated through run.py,
the records by ablations/records.py. RESULTS/tokenizer_split_merge.json "analysis" (analysis): each challenger's
contrasts with sa-bpe, the baseline, and the decision.

Decisions, pre-registered (analysis), each challenger (CHALLENGERS) against sa-bpe by ablations/contrast.py's tests
(parsing: the exact sign test on MBPP+; code bits per byte: the normal bound):
- the positive control gates the decision: unless sa-bpe-fused parses worse, the corpus keeps sa-bpe's pattern and
  score. Else a challenger qualifies iff it is worse on neither and better on one; sa-bpe-fused, worse on parsing,
  never qualifies (the control detects, it is not a candidate). Pattern: the hybrid's iff it qualifies, else sa-bpe's.
  Score: bpe iff it qualifies, or is equal on both and the val documents read fewer of its merged tokens under 100
  times than of sa-bpe's (SA-BPE's claimed benefit, a higher coverage than BPE's), carried to the adopted pattern
  (assumed); otherwise SA-BPE.

repack: every arm's set packed again from its tokenized splits, the set data recorded, after a change to the packer
(corpus/pack.py, the corpus crate's pack.rs); the data record's arms rewritten.

adopt: the decided vocabulary (RESULTS/tokenizer_split_merge.json "analysis" "decision") as the corpus's tokenizer:
the arm of the decided pattern and score, or, bpe's score carried to another arm's pattern, that pair trained on the
sample.

python -m ablations.tokenizer_split_merge vocabularies|intrinsic|data|repack|adopt
"""
import argparse
import datetime
import importlib.metadata
import itertools
import json
import os
import platform
import shutil
import statistics
import subprocess
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import regex
from scipy.stats import hmean

import metrics
from ablations.contrast import contrast, evaluated
from corpus import mixture
from corpus.layout import TOKEN_DTYPE, split_path, stage_rows
from corpus.mixture import weights
from corpus.pack import packed_shards, shard_count, write
from corpus.parallel import SPAWN, memory
from corpus.schema import EXERCISE, FUNCTION, REPOSITORY_FILE, SYNTHETIC_FILE, TEXT
from corpus.sources import SOURCES, curated_shards, sample_starts, texts
from corpus.tokenize import tokenize_repo_level_stage, tokenize_repositories, tokenize_source, train_parts
from corpus.tokenizer import ABLATION, SPLIT_PATTERN, TOKENIZER, get_tokenizer, save
from corpus.train_tokenizer import Score, byte_shares, fidelity, sample, shard_bytes, train, vocabulary
from native import corpus_crate
from paths import RESULTS, ROOT

# The experiment's name (conf/experiment/<EXPERIMENT>.yaml): its records and analysis, RESULTS/<EXPERIMENT>.jsonl and
# .json, and this module's results, RESULTS/<EXPERIMENT>-{vocabularies,intrinsic,data}.json
EXPERIMENT = "tokenizer_split_merge"

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
# arm: (pattern, merge score); every arm is trained on, sa-bpe the runs' baseline (analysis)
ARMS: dict[str, tuple[str, Score]] = {"sa-bpe": (SPLIT_PATTERN, "sa-bpe"), "sa-bpe-fused": (FUSED_PATTERN, "sa-bpe"),
                                      "hybrid": (HYBRID_PATTERN, "sa-bpe"), "bpe": (SPLIT_PATTERN, "frequency")}
CHALLENGERS = ("sa-bpe-fused", "hybrid", "bpe")
# The runs' num_hidden_layers, which the set is sized for (conf/experiment/tokenizer_split_merge.yaml)
NUM_HIDDEN_LAYERS = 16
# The set's size: a d16 run at Chinchilla's 20 tokens per non-embedding parameter under sa-bpe (arXiv 2203.15556
# Table 3)
TOKENS_PER_PARAM = 20
# The set's margin over the draws of the corpora that must not bind (text, math, the repositories corpora): above the
# code shards' rounding up to whole shards and the arms' spread in compression (assumed)
MARGIN = 1.5
# The categories whose documents are whole program texts in their source's language (depth invariance)
PROGRAMS = (REPOSITORY_FILE, EXERCISE, SYNTHETIC_FILE, FUNCTION)
# The depth-invariance check's lines: the first indented lines of each program source's val split; the rate's standard
# error at most √(1/(4n)) ≈ 0.0022
INDENTED_LINES = 50_000
# An indented line: its leading spaces and tabs, then its statement
INDENTED = regex.compile(r"([ \t]+)(\S.*)")
# TokEval's sample: each source's leading val documents until their UTF-8 bytes reach TOKEVAL_BYTES (the document
# crossing it included), as TokEval languages in TOKEVAL_GROUPS order (tokeval_group)
TOKEVAL_BYTES = 2_000_000
TOKEVAL_GROUPS = ("python", "cpp", "text", "math", "mixed")
# The code languages' TokEval names (tree-sitter's)
TOKEVAL_LANGUAGES = {"Python": "python", "C++": "cpp"}


def provenance() -> dict:
    """Where and when a step ran: the checkout's commit, the UTC time, the host, its cores and memory, and the versions
    of Python and of the packages that compute the measures."""
    commit = subprocess.run(["git", "-c", f"safe.directory={ROOT}", "-C", str(ROOT), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    return {"commit": commit, "utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "host": platform.node(), "machine": platform.machine(), "cpus": os.cpu_count(),
            "memory_gb": memory() / 2**30, "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name)
                         for name in ("tiktoken", "tokenizers", "numpy", "scipy", "pyarrow", "regex",
                                      "tokenizer-intrinsic-evals", "tree-sitter-language-pack")}}


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
    """An arm's tokenized val split of a source: each mergeable token's count and each document's bytes."""
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
        """Bytes per token over the members' file-stage byte shares: their weighted harmonic mean."""
        return float(hmean([sources[s]["bytes_per_token"] for s in members], weights=[shares[s] for s in members]))

    entries = [b.decode("utf-8", "ignore") for b in tokenizer.token_bytes[:tokenizer.bos_token_id]]
    # a line break and the indentation after it ending the token (\Z: $ also matches before a final line break)
    fused, indented = regex.compile(r"[^\s\p{L}\p{N}][\r\n]"), regex.compile(r"[\r\n][ \t]+\Z")
    merged = held[256:]
    return {"data_key": tokenizer.data_key, "sources": sources,
            "file_stage_bytes_per_token": {"all": mixed(mixture.FILE_LEVEL_STAGE.sources)}
            | {name: mixed(group.sources) for name, group in mixture.FILE_LEVEL_STAGE.groups.items()},
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


def tokeval_group(source: str) -> str:
    """A source's TokEval language: a program source's language, math (mixture.MATH_SOURCES), text (other web
    documents) or mixed (prose with fenced code: Codeforces reasoning traces, Kaggle notebooks)."""
    spec = SOURCES[source]
    if spec.category in PROGRAMS:
        return TOKEVAL_LANGUAGES[spec.language]
    if source in mixture.MATH_SOURCES:
        return "math"
    return "text" if spec.category == TEXT else "mixed"


def tokeval_documents(source: str) -> tuple[list[str], bool]:
    """A source's leading val documents (corpus/tokenize.py row_group_documents' documents: a repository sample's files
    concatenated) until their UTF-8 bytes reach TOKEVAL_BYTES, and whether the split ran out first."""
    taken, size = [], 0
    for path in curated_shards(source, "val"):
        f = pq.ParquetFile(path)
        for g in range(f.num_row_groups):
            table = f.read_row_group(g)
            documents = texts(source, table)
            if SOURCES[source].category == REPOSITORY_FILE:
                starts = sample_starts(table["sample"].to_numpy())
                documents = ["".join(documents[a:b]) for a, b in itertools.pairwise([*starts, len(documents)])]
            for doc in documents:
                taken.append(doc)
                size += len(doc.encode("utf-8"))
                if size >= TOKEVAL_BYTES:
                    return taken, False
    return taken, True


def tokeval(arm: str) -> dict:
    """An arm's TokEval record (module docstring): "sample", per source its group, documents, bytes and whether its
    split ran out; "metrics", TokEval's global values per metric, the per-category AST alignment and the character
    splits. Prose languages are the groups suffixed "_val" (TokEval reports the code and math corpora under the domain
    keys python, cpp and math); the code corpus is the python and cpp groups, the math corpus the math group."""
    sample = {source: tokeval_documents(source) for source in SOURCES}
    grouped = {group: [d for s, (docs, _) in sample.items() if tokeval_group(s) == group for d in docs]
               for group in TOKEVAL_GROUPS}
    results = metrics.tokeval({arm: get_tokenizer(ABLATION / arm)}, {f"{g}_val": d for g, d in grouped.items()},
                              {language: grouped[language] for language in TOKEVAL_LANGUAGES.values()},
                              grouped["math"])
    measures = {name: results[name]["per_tokenizer"][arm] for name in (
        "compression_rate", "fertility", "renyi_efficiency", "bigram_entropy", "trigram_entropy",
        "utf8_token_integrity", "ast_boundary_alignment", "identifier_fragmentation", "indentation_consistency",
        "vocabulary_utilization", "reconstruction_fidelity")}
    return {"sample": {s: {"group": tokeval_group(s), "documents": len(docs),
                           "bytes": sum(len(d.encode("utf-8")) for d in docs), "split_exhausted": exhausted}
                       for s, (docs, exhausted) in sample.items()},
            "metrics": {name: m["global"] for name, m in measures.items()}
            | {"ast_boundary_alignment_per_category": measures["ast_boundary_alignment"]["per_category"],
               "utf8_char_split": measures["utf8_token_integrity"]["char_split"]["global"]}}


def intrinsic() -> None:
    """Every arm's tokenized val split of every source, the equal-bytes check, and RESULTS/<EXPERIMENT>-intrinsic.json
    (module docstring), all in one process pool: depth invariance per program source, fidelity per arm and source and
    TokEval per arm (which read no tokenized data) beside the tokenize jobs, every split at once, each split's counts as
    it is tokenized. "tokeval_sample": TokEval's documents per source (tokeval). "seconds": "tokenized", the wall time
    to the last tokenize job's end (the checks overlap them), and "all"."""
    start, began = time.perf_counter(), provenance()
    programs = [s for s, spec in SOURCES.items() if spec.category in PROGRAMS]
    checks = [(arm, source) for arm in ARMS for source in SOURCES]
    with ProcessPoolExecutor(mp_context=SPAWN) as pool, ThreadPoolExecutor(len(checks)) as threads:
        for arm, corpus in checks:
            split_path(corpus, "val", get_tokenizer(ABLATION / arm)).parent.mkdir(parents=True)
        invariance = [pool.submit(depth_invariance, source) for source in programs]
        fidelities = [pool.submit(fidelity, (ABLATION / arm, source)) for arm, source in checks]
        tokevals = {arm: pool.submit(tokeval, arm) for arm in ARMS}

        def tokenized(check: tuple[str, str]) -> Future:
            tokenize_source(check[1], "val", ABLATION / check[0], pool)
            return pool.submit(val_split, check)

        # a val split is a few row groups: one split at a time would leave the pool idle
        splits = dict(zip(checks, threads.map(tokenized, checks), strict=True))
        tokenized_seconds = time.perf_counter() - start
        counts, sizes = {}, {}
        for check, future in splits.items():
            counts[check], sizes[check] = future.result()
        for corpus in SOURCES:
            if not all(np.array_equal(sizes["sa-bpe", corpus], sizes[arm, corpus]) for arm in CHALLENGERS):
                raise RuntimeError(f"{corpus}: the arms' tokenized val documents differ in bytes")
        record = {"arms": {arm: describe(arm, {corpus: counts[arm, corpus] for corpus in SOURCES})
                           | {"depth_invariance": {}, "fidelity": {}} for arm in ARMS}}
        for source, future in zip(programs, invariance, strict=True):
            for arm, result in future.result().items():
                record["arms"][arm]["depth_invariance"][source] = result
        for (arm, source), future in zip(checks, fidelities, strict=True):
            record["arms"][arm]["fidelity"][source] = future.result()
        for arm, future in tokevals.items():
            arm_tokeval = future.result()
            record["tokeval_sample"] = arm_tokeval["sample"]
            record["arms"][arm]["tokeval"] = arm_tokeval["metrics"]
    record |= {"seconds": {"tokenized": tokenized_seconds, "all": time.perf_counter() - start}, "provenance": began}
    (RESULTS / f"{EXPERIMENT}-intrinsic.json").write_text(json.dumps(record, indent=1), encoding="utf-8")


def document_set(run) -> dict:
    """The set's shards (module docstring) for the d16 run `run` at TOKENS_PER_PARAM under sa-bpe: {"file": {source:
    its leading train shards, the file stage's}, "repository": {source: its train shards after those, the repository
    stage's},
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

    file_draws, repo_draws = draws(mixture.FILE_LEVEL_STAGE), draws(mixture.REPO_LEVEL_STAGE)
    file = {s: shards(s, tokens * (1 if s in mixture.FILE_LEVEL_STAGE.code else MARGIN), 0)
            for s, tokens in file_draws.items()}
    repository = {s: shards(s, tokens * (1 if s in mixture.REPO_LEVEL_STAGE.supply else MARGIN), file.get(s, 0))
                  for s, tokens in repo_draws.items() if s in SOURCES}
    return {"file": file, "repository": repository, "draws": {"8192": file_draws, "32768": repo_draws}}


def pack_corpus(corpus: str, directory: Path, shards: dict) -> None:
    """A tokenized corpus's packs in each stage that reads it (shards: the set, data's "set" record): its train split's
    shards in the set, its val split whole."""
    for stage in (mixture.FILE_LEVEL_STAGE, mixture.REPO_LEVEL_STAGE):
        if corpus not in stage.sources:
            continue
        if corpus in mixture.REPOSITORIES.values():
            train = range(shard_count(corpus, "train", directory))
        elif stage is mixture.FILE_LEVEL_STAGE:
            train = range(shards["file"][corpus])
        else:
            first = shards["file"].get(corpus, 0)
            train = range(first, first + shards["repository"][corpus])
        for split, listed in (("train", train), ("val", range(shard_count(corpus, "val", directory)))):
            write(corpus, split, directory, packed_shards(corpus, split, directory, listed, stage.seq_len),
                  stage.seq_len)


def arm_record() -> dict:
    """Each arm's run and data at its max_tokens_per_param: data key, plan, and per stage each corpus's packed rows,
    weight and the share of its rows the run reads."""
    from run import base_plan
    from train.stages import drawn, max_tokens_per_param

    arms = {}
    for arm in ARMS:
        tokenizer = get_tokenizer(ABLATION / arm)
        tokens_per_param = max_tokens_per_param(tokenizer, num_hidden_layers=NUM_HIDDEN_LAYERS)
        arm_run = base_plan(num_hidden_layers=NUM_HIDDEN_LAYERS, tokens_per_param=tokens_per_param)
        read = {}
        for stage in (mixture.FILE_LEVEL_STAGE, mixture.REPO_LEVEL_STAGE):
            rows, count = stage_rows(tokenizer, stage), drawn(stage, arm_run)
            shares, draws, _ = weights(stage, rows, count)
            read[str(stage.seq_len)] = {n: {"rows": rows[n], "weight": shares[n], "read": d / rows[n]}
                                        for n, d in draws.items()}
        arms[arm] = {"data_key": tokenizer.data_key, "tokens_per_param": arm_run.tokens_per_param,
                     "max_steps": arm_run.max_steps, "batch_size": arm_run.batch_size,
                     "repo_level_start": arm_run.repo_level_start, "learning_rate": arm_run.learning_rate,
                     "stages": read}
        print(f"{arm}: tokens_per_param {tokens_per_param:.2f}, {arm_run.max_steps:,} steps of "
              f"{arm_run.batch_size:,} tokens", flush=True)
    return arms


def data() -> None:
    """Every arm's training set, tokenized and packed, and RESULTS/<EXPERIMENT>-data.json (module docstring): each
    corpus of each arm tokenized, then packed, as its own job, every job into one process pool;
    tokenize_repo_level_stage, beside the other sources' tokenize jobs, chooses the partitions and tokenizes sa-bpe's
    repositories corpora, then the set check, then the repository sources' jobs and the other arms' repositories
    corpora, one arm after another (each fills the cores). "seconds": "tokenized", the wall time to the last tokenize
    job's end (packs overlap them), and "all"."""
    # the training stack (the model's kernels) here alone: the measures' spawned workers import this module
    from run import base_plan

    start, began = time.perf_counter(), provenance()
    run = base_plan(num_hidden_layers=NUM_HIDDEN_LAYERS, tokens_per_param=TOKENS_PER_PARAM)
    shards = document_set(run)
    leading = {s: shards["file"].get(s, 0) + shards["repository"].get(s, 0) for s in SOURCES}
    tokenized = []  # each tokenize job's end

    def source(arm: str, corpus: str) -> None:
        tokenize_source(corpus, "train", ABLATION / arm, pool, leading=leading[corpus])
        tokenized.append(time.perf_counter())
        pack_corpus(corpus, ABLATION / arm, shards)

    def repositories(parts: list[int]) -> None:
        for arm in ARMS:
            if arm != "sa-bpe":  # tokenize_repo_level_stage tokenized sa-bpe's as it chose the partitions
                tokenize_repositories(ABLATION / arm, parts, pool)
            tokenized.append(time.perf_counter())
            for corpus in mixture.REPOSITORIES.values():
                pack_corpus(corpus, ABLATION / arm, shards)

    corpus_crate()  # built once, before the threads call it (pack.py's rule)
    with (ProcessPoolExecutor(mp_context=SPAWN) as pool,
          ThreadPoolExecutor(len(ARMS) * len(SOURCES) + 2) as threads):  # source jobs, repositories, the 32K stage's
        partitions = threads.submit(tokenize_repo_level_stage, ABLATION / "sa-bpe", pool,
                                    MARGIN * sum(shards["draws"]["32768"][c] for c in mixture.REPOSITORIES.values()))
        # the sources the set check does not gate tokenize while tokenize_repo_level_stage chooses the partitions
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
    tokenized_seconds = max(tokenized) - start
    record = {"run": {"num_hidden_layers": NUM_HIDDEN_LAYERS, "tokens_per_param": TOKENS_PER_PARAM,
                      "max_steps": run.max_steps, "batch_size": run.batch_size,
                      "repo_level_start": run.repo_level_start},
              "set": chosen, "arms": arm_record(),
              "seconds": {"tokenized": tokenized_seconds, "all": time.perf_counter() - start},
              "provenance": began}
    (RESULTS / f"{EXPERIMENT}-data.json").write_text(json.dumps(record, indent=1), encoding="utf-8")


def repack() -> None:
    """Every arm's set packed again from its tokenized splits (the set RESULTS/<EXPERIMENT>-data.json records), after a
    change to the packer (corpus/pack.py, the corpus crate's pack.rs), each corpus of each arm a job (the crate releases
    the GIL); the record's arms and its "seconds" "repacked" rewritten."""
    start = time.perf_counter()
    path = RESULTS / f"{EXPERIMENT}-data.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    corpora = sorted({c for stage in (mixture.FILE_LEVEL_STAGE, mixture.REPO_LEVEL_STAGE) for c in stage.sources})
    corpus_crate()  # built once, before the threads call it
    with ThreadPoolExecutor(len(ARMS) * len(corpora)) as threads:
        for job in [threads.submit(pack_corpus, corpus, ABLATION / arm, record["set"])
                    for arm in ARMS for corpus in corpora]:
            job.result()
    record["arms"] = arm_record()
    record["seconds"]["repacked"] = time.perf_counter() - start
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")


def analysis(lines: list[dict]) -> dict:
    """The runs' analysis (RESULTS/<EXPERIMENT>.json "analysis", ablations/records.py): each challenger's contrast with
    sa-bpe, with the difference of the arms' MBPP+ unparsed shares, and the decision (module docstring), over
    RESULTS/<EXPERIMENT>.jsonl's records."""
    swept = {line["config"]["tokenizer"] for line in lines}
    if swept != set(ARMS):
        raise RuntimeError(f"the experiment swept {sorted(swept)}, not the arms {sorted(ARMS)}")
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
    over the provisional tokenizer (the sa-bpe arm's copy) in place: tokenized and packed data are keyed by a
    vocabulary's data key, so adopting sa-bpe reuses the provisional corpus and any other vocabulary tokenizes into new
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
    steps = {"vocabularies": vocabularies, "intrinsic": intrinsic, "data": data, "repack": repack, "adopt": adopt}
    parser = argparse.ArgumentParser(prog="python -m ablations.tokenizer_split_merge",
                                     description="A step of the tokenizer ablation.")
    parser.add_argument("step", choices=steps)
    steps[parser.parse_args().step]()
