"""
The tokenizer ablation: it decides the corpus tokenizer's split pattern and merge score (nanochat/corpus/tokenizer.py
PATTERN, SCORE), the tokenizer every later step of pretraining reads. Each candidate is a vocabulary of 32,768 ids
(32,764 mergeable tokens, BOS and 3 FIM sentinels) trained on the same sample (corpus/tokenizer.py sample: the curated
train splits at the file stage's shares of bytes) by the same trainer (corpus/tokenizer.py train, the data crate's) and
encoded by tiktoken, and each is trained on, so every candidate the corpus keeps or dismisses has runs of its own.

Arms (ARMS):
- sa-bpe: production's pattern (SPLIT_PATTERN) and score, SA-BPE's F·log R (arXiv 2604.14053 §4.1: F the pair's
  frequency, R the authorship units holding it in the current segmentation, a pair of one unit scoring 0; the paper's
  code takes the natural log, pchizhov/sa-bpe scoring.py). Table 3: the highest compression of its rows and far fewer
  tokens verified under-trained than BPE's (one tokenizer each, no seeds). A unit is the repository for repository
  files and functions, as in the paper; the document otherwise, as §6 suggests for natural language ("documents,
  books, or document collections"), untested there. Not the paper's recommended F·log R·L (§6, Table 4): §6 says the
  language aspect "must be avoided" for natural language, five of the sample's 17 sources, and with Python, C++ and
  English L takes three values;
- sa-bpe-fused: FUSED_PATTERN, whose punctuation clause takes the line breaks after it, as cl100k's does. The positive
  control: arXiv 2608.18062's companion post deletes those trailing line breaks from GPT-4o's clause, the tokenizer
  corpus held fixed, and MBPP pass@1 rises with non-overlapping seed ranges (three seeds each; healed prompts, the
  paper's §4.3 and App. C.1), while code bits per byte shows no association with MBPP over its tokenizers; if the
  parse test misses it at d16, it cannot decide the other arms;
- hybrid: HYBRID_PATTERN, whose statement chunks are the same at every indentation depth (its comment);
- bpe: frequency alone (Table 3's F row, SA-BPE's baseline) within SPLIT_PATTERN, close to F·log R in compression,
  so a tie in the runs is likely [inference], which the val coverage breaks.
The sample's composition is not an arm (normalization_doc.md §9). Not run: tiktoken encodes by merge rank alone, which
has no form for PickyBPE's token removals (arXiv 2409.04599 Alg. 2) or BoundlessBPE's supermerges (arXiv 2504.00178
§3.2); SuperBPE's code results moved HumanEval and MBPP pass@10 in opposite directions, neither significantly
(arXiv 2503.13423 Table 1).

vocabularies: every arm on one draw of the sample, at once (the crate releases the GIL and each training's merge loop
is mostly serial; every merge is an argmax with fixed tie-breaks, so an arm does not depend on the others); its
vocabulary and merges in CORPUS/tokenizer-ablation/<arm>; RESULTS/tokenizer-vocabularies.json: the sample's
composition per source (target bytes, the fraction of rows taken, the documents, bytes and units taken: the sample is
what corpus/tokenizer.py trains on) and each arm's training time.

intrinsic: every arm's compiled val splits of the 18 sources, and RESULTS/tokenizer-intrinsic.json, only the measures
that plan the runs, gate their validity, test an arm's mechanism or break the decision's tie:
- planning: each source's bytes per token and the file stage's over its byte shares (an arm's tokens at equal bytes;
  production's compile and pack budgets), and each group read in part's train tokens over its draw (below 1, the
  corpus's extent is too small for that arm: prepare.extent, mixture.SPREAD);
- validity, asserted: every arm holds the same bytes per document; on each source's first FIDELITY_DOCUMENTS val texts
  the pattern's chunks cover the text, tiktoken's encoding equals the arm's merges replayed in training order (Hugging
  Face tokenizers' BPE model), and every text decodes back;
- mechanism: the vocabulary entries holding a line break (the companion post's strongest MBPP correlate), fusing
  punctuation with one, and ending in a line break and indentation (its MinGram pair: most of the fusing tokenizer's
  completions IndentationError or TabError); depth invariance, the hybrid's clause (2): of each program source's first
  INDENTED_LINES indented val lines, those whose tokens are the indentation's, then the statement's as at depth 0;
- the tie-break: how many merged tokens the val documents read never, under 10 and under 100 times (SA-BPE's Coverage,
  Table 3, graded).
Not measured: TokEval's intrinsic metrics, which no decision reads; of its 14 only AST alignment predicts MBPP, which
it attributes "to the vocabulary's line-break handling" (arXiv 2608.18062 §5), measured here directly, and Rényi
efficiency predicts neither code bits per byte nor MBPP (Table 2).

runs: every arm at d16, SEEDS seeds each, at equal bytes: each reads the code documents of the set once, as production
reads its data (train/stages.py max_ratio), so its tokens are their bytes over its bytes per token; text and math are
drawn by rows. Each run's plan is its own (schedule.plan at its tokens), its contexts the corpus's 8,192 and 32,768
tokens (eval/bpb.py's byte-offset buckets show the bytes of context a better compressor sees, arXiv 2503.13423 §3.1).
The set holds 20 tokens per scaling parameter of d16 for sa-bpe (arXiv 2203.15556 Table 3): of each code corpus the
leading train documents whose sa-bpe tokens fill its file-level draw, so the code group binds every arm's max_ratio;
of text and math the leading shards, and of each repository corpus the trailing partitions, holding a margin over
their draws, so they never bind. The runs train through ablations/harness.py, whose stack (nanochat/train,
nanochat/eval) reads nanochat/data until training reads the corpus; that cutover builds the set and every arm's packs
(corpus compile.py and pack.py take the arm's tokenizer; nanochat/corpus/gh200.md).

Decisions, pre-registered (RESULTS/tokenizer.json "decision"), each challenger (CHALLENGERS) against sa-bpe on the
seeds where both have an evaluation (a diverged run drops its seed from the arm's contrasts on both sides):
- parsing: per MBPP+ problem, the seeds whose healed greedy completion compiles under the challenger minus under
  sa-bpe; the exact sign test over the problems whose difference is not 0 (McNemar's exact test at one seed, the
  companion post's test); better or worse at p < ALPHA, else equal. MBPP+ alone: its prompts leave the function header
  to the model, where the post's failures arise, and an evaluation "that uses HumanEval alone does not expose this
  type of defect" (the post);
- code bits per byte: the difference of the arms' means, better or worse beyond its 1 - ALPHA two-sided normal bound
  1.96 CODE_SIGMA √(1 / n + 1 / n), n the contrast's seeds per arm, else equal;
- the positive control gates the decision: unless sa-bpe-fused parses worse, the corpus keeps its pattern and score.
  Else a challenger qualifies iff it is worse on neither and better on one. Pattern: of sa-bpe-fused and the hybrid,
  the qualifier with the lower MBPP+ unparsed share, else sa-bpe's. Score: bpe iff it qualifies, or is equal on both
  and the val documents read fewer of its merged tokens under 100 times than of sa-bpe's (SA-BPE's claimed benefit, a
  higher coverage than BPE's), carried to the adopted pattern [inference]; otherwise SA-BPE.

python -m ablations.tokenizer vocabularies|intrinsic|runs
"""
import datetime
import importlib.metadata
import itertools
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq
import regex
from scipy.stats import binomtest, norm
from tokenizers import models

from ablations.harness import Setting, main
from nanochat.corpus import mixture
from nanochat.corpus.compile import compile_source
from nanochat.corpus.loader import TOKEN_DTYPE, split_path
from nanochat.corpus.parallel import SPAWN
from nanochat.corpus.schema import EXERCISE, FUNCTION, REPOSITORY_FILE, SYNTHETIC_FILE
from nanochat.corpus.sources import SOURCES, curated_shards, texts
from nanochat.corpus.tokenizer import (
    ABLATION,
    Score,
    byte_shares,
    get_tokenizer,
    sample,
    train,
    train_bytes,
    vocabulary,
)
from nanochat.data.tokenizer import SPLIT_PATTERN, save
from nanochat.native import data_crate
from nanochat.paths import RESULTS, ROOT
from nanochat.train.stages import max_ratio

# karpathy/nanochat's split pattern (92d63d4), the first tokenizer's: SPLIT_PATTERN with cl100k's punctuation clause,
# whose [\r\n]* takes the line breaks after a run of punctuation ("):\n" one chunk)
FUSED_PATTERN = (r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]"""
                 r"""|\s+(?!\S)|\s+""")
# The hybrid's pattern. A pattern cuts a text into chunks, its matches from left to right; BPE merges, trained and
# applied within chunks (tiktoken encodes chunk by chunk), never cross a chunk's end, so every chunk end is a token end
# under every vocabulary trained with the pattern: a property of the chunks holds whatever the merges. No clause looks
# behind, so the chunks after a chunk end j are the chunks of the text from j alone. Its clauses give:
# (1) a line break shares a chunk only with line breaks of its kind (\n+, \r+: the post's Claude-approximate [\n]+,
#     [\r]+; no other clause matches one), the forced break at line breaks of the post's MinGram cb_nl. So no token
#     joins a line break to the punctuation before it, where FUSED_PATTERN's [\r\n]* makes "):\n" one chunk and a
#     prompt ending "):" ends inside the token training saw (the token-boundary bias DeepSeek-V3 names for multi-line
#     prompts without a terminal line break, arXiv 2412.19437 §4.1); production's pattern, the post's gpt4onl edit
#     (the punctuation clause without its trailing line breaks), shares this. Neither joins a line break to the next
#     line's indentation, as the post's MinGram cb does; production's \s*[\r\n] joins it to the whitespace before it
#     (trailing spaces, a whitespace-only line "\n    \n"), the convention of the post's best-scoring tokenizers
#     (gpt4onl, clean-multi), which the hybrid keeps apart;
# (2) a run of spaces, of tabs or of other whitespace is one chunk ( +, \t+: the Claude-approximate [ ]+, [\t]+), and a
#     letter run takes no tab or line break before it: an indentation of d >= 2 spaces or d >= 1 tabs ends at a chunk
#     end, so, no clause looking behind, the statement s after it is cut as at depth 0: chunks(indent + s) =
#     (indent,) + chunks(s), one indentation chunk per depth and the statement's chunks the same at every depth.
#     Production's \s+(?!\S) gives the indentation's last space to the statement's first chunk (" if", " (") or to a
#     chunk of its own (before a digit): chunks = (" " * (d - 1), " " + w, ...), the depth carried by d - 1 spaces and
#     a space inside a statement chunk that differs from its depth-0 form ("if" against " if"); its letter clause
#     [^\r\n\p{L}\p{N}]?+ admits a tab, so a tab-indented statement starts "\tif". A single space before a letter run
#     stays its prefix (" foo"), as in prose. The Claude-approximate pattern, an early reconstruction in the post,
#     scored MBPP on par with gpt4onl over seeds, few of its completions IndentationErrors; it differs from the hybrid
#     beyond whitespace (a letter run takes only a space before it, contractions stay, digits come in threes), so the
#     hybrid as a whole is untested;
# (3) no contraction clause (2402.01035's Punct drops cl100k's, its App. E and Fig. 3; o200k folds contractions into
#     the letter run): the letter clause takes a whole run of letters, which '(?i:[sdmt]|ll|ve|re) cuts where an
#     apostrophe starts a chunk ("'s" "elf" for a line's 'self'); English contractions chunk as before ("doesn" "'t").
# Kept from cl100k: one leading punctuation character on a letter run (".append", "(self"), which Punct drops (it keeps
# the leading space, " ?\p{L}+", 2402.01035 Fig. 3), the paper recommending GPT-4's pattern "since it provides an
# additional 5% compression over Punct at no significant cost to performance" (App. E), and a leading space (" foo");
# production's digit pairs (karpathy/nanochat: "2 is the sweet spot for vocab size of 32K") and punctuation runs. And
# [^\S\r\n\t ]+ takes every other whitespace character, so every character falls in some chunk: tiktoken drops what no
# clause matches. Without a lookahead, a long whitespace run costs fancy-regex no backtracking, where production's and
# FUSED_PATTERN's \s+(?!\S) beside \s*[\r\n] stop tiktoken past its backtrack limit (no corpus document comes near it)
HYBRID_PATTERN = r"""(?:[^\s\p{L}\p{N}]| )?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++|\n+|\r+|\t+| +|[^\S\r\n\t ]+"""
# arm: (pattern, merge score); every arm is trained on, sa-bpe as the harness's production arm
ARMS: dict[str, tuple[str, Score]] = {"sa-bpe": (SPLIT_PATTERN, "sa-bpe"), "sa-bpe-fused": (FUSED_PATTERN, "sa-bpe"),
                                      "hybrid": (HYBRID_PATTERN, "sa-bpe"), "bpe": (SPLIT_PATTERN, "frequency")}
CHALLENGERS = ("sa-bpe-fused", "hybrid", "bpe")
DEPTH, SEEDS = 16, 2
# The decisions' significance level, the conventional one ("Statistical Methods for Research Workers" (1925), §20)
ALPHA = 0.05
# The spread of a run's 8K code bits per byte over seeds (pooled within each configuration): the six runs at d8 at 20 of
# the standard-regime comparison (60138de's baseline and c00d2f6's default, data seeds 0 to 2), on production's data,
# carried to d16 and to the corpus [inference]
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
                         "data_key": get_tokenizer(arm).data_key}
            print(f"{arm}: {len(merges):,} merges in {seconds:.0f} s, data key {arms[arm]['data_key']}", flush=True)
    record = {"sample": {"documents": len(docs), "bytes": sum(c["bytes"] for c in composition.values()),
                         "units": max(units) + 1, "seconds": sampled, "sources": composition},
              "arms": arms, "seconds": time.perf_counter() - start, "provenance": began}
    (RESULTS / "tokenizer-vocabularies.json").write_text(json.dumps(record, indent=1), encoding="utf-8")


def describe(arm: str) -> dict:
    """An arm's planning, mechanism and tie-break measures on the corpus's val splits (module docstring)."""
    tokenizer = get_tokenizer(arm)
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
        enc = get_tokenizer(arm).enc
        whole = enc.encode_ordinary_batch([indent + statement for indent, statement in lines])
        indents = enc.encode_ordinary_batch([indent for indent, _ in lines])
        statements = enc.encode_ordinary_batch([statement for _, statement in lines])
        invariant = sum(w == i + s for w, i, s in zip(whole, indents, statements, strict=True))
        result[arm] = {"lines": len(lines), "invariant": invariant, "rate": invariant / len(lines)}
    return result


def byte_alphabet() -> list[str]:
    """GPT-2's reversible map of the 256 bytes to printable characters, the byte-level alphabet of Hugging Face
    tokenizers' ByteLevel: a printable byte stands for itself, every other takes the next code point from 256."""
    printable = {*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)}
    beyond = iter(range(256, 512))
    return [chr(b) if b in printable else chr(next(beyond)) for b in range(256)]


def fidelity(task: tuple[str, str]) -> dict[str, int]:
    """An arm's encoding checked on a source's first FIDELITY_DOCUMENTS val texts (the documents and chunks checked):
    its pattern's chunks (Python regex, whose chunks fancy-regex's equal) cover every text, tiktoken's encoding (by the
    rank of the joined bytes) equals the training segmentation chunk by chunk, and every text decodes back. The
    training segmentation is Hugging Face tokenizers' BPE model over the arm's merges: listed pairs merged by rank,
    each over every occurrence from the left (data/crate train.rs merge_pair), never a pair whose bytes another made."""
    arm, source = task
    tokenizer = get_tokenizer(arm)
    alphabet = byte_alphabet()
    spelled = ["".join(alphabet[b] for b in entry) for entry in tokenizer.token_bytes[:tokenizer.bos_token_id]]
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
            if tokenizer.enc._encode_single_piece(piece) != [t.id for t in bpe.tokenize(
                    "".join(alphabet[b] for b in piece))]:
                raise RuntimeError(f"{arm}: tiktoken encodes {piece!r} ({source}) unlike the trained merges")
        chunks += len(pieces)
    return {"documents": len(docs), "chunks": chunks}


def intrinsic() -> None:
    """Every arm's compiled val split of every source, and RESULTS/tokenizer-intrinsic.json (module docstring): the
    equal-bytes check, then each arm's measures, depth invariance per program source and fidelity per arm and source in
    one process pool."""
    seconds, start, began = {}, time.perf_counter(), provenance()
    tokenizers = [get_tokenizer(arm) for arm in ARMS]
    for tokenizer in tokenizers:
        begin = time.perf_counter()
        for corpus in SOURCES:
            split_path(corpus, "val", tokenizer).parent.mkdir(parents=True)
            compile_source(corpus, "val", tokenizer)
        seconds[f"compile {tokenizer.name}"] = time.perf_counter() - begin
    for corpus in SOURCES:
        sizes = []
        for tokenizer in tokenizers:
            path = split_path(corpus, "val", tokenizer)
            tokens = np.fromfile(path.with_suffix(".bin"), dtype=TOKEN_DTYPE)
            ends = np.concatenate([[0], np.cumsum(tokenizer.token_lengths[tokens])])
            sizes.append(np.diff(ends[np.fromfile(path.with_suffix(".idx"), dtype=np.int64)]))
        if not all(np.array_equal(sizes[0], other) for other in sizes[1:]):
            raise RuntimeError(f"{corpus}: the arms' compiled val documents differ in bytes")
    programs = [s for s, spec in SOURCES.items() if spec.category in PROGRAMS]
    checks = [(arm, source) for arm in ARMS for source in SOURCES]
    begin = time.perf_counter()
    with ProcessPoolExecutor(mp_context=SPAWN) as pool:  # every map submits at once
        described = pool.map(describe, ARMS)
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


def runs() -> None:
    """Every arm at d16, each at its max_ratio (ablations/harness.py main); then RESULTS/tokenizer.json gains each
    challenger's contrast with sa-bpe and the decision (module docstring)."""
    ratios = {arm: max_ratio(get_tokenizer(arm), depth=DEPTH) for arm in ARMS}
    main(experiment="tokenizer",
         settings=[Setting(depth=DEPTH, ratio=ratios["sa-bpe"], seeds=range(SEEDS),
                           alternatives={arm: {"tokenizer": arm, "ratio": ratios[arm]} for arm in CHALLENGERS})],
         production={"tokenizer": "sa-bpe"})
    # (arm, seed) of every evaluated run (a diverged run has none): its measures and, per MBPP+ problem in the
    # benchmark's order, whether its completion compiles
    evaluated = {}
    for line in (RESULTS / "tokenizer.jsonl").read_text(encoding="utf-8").splitlines():
        run = json.loads(line)
        if run["measures"] is not None:
            outcomes = run["evaluations"][-1]["parse"][PARSED]["outcomes"]
            evaluated[run["arm"], run["fields"]["seed"]] = (run["measures"], [o == "parses" for o in outcomes])
    summary = json.loads((RESULTS / "tokenizer.json").read_text(encoding="utf-8"))
    contrasts = {}
    for arm in CHALLENGERS:
        seeds = [seed for seed in range(SEEDS) if (arm, seed) in evaluated and ("production", seed) in evaluated]
        if not seeds:
            raise RuntimeError(f"no seed evaluated under both {arm} and sa-bpe")
        both = {name: [evaluated[name, seed] for seed in seeds] for name in (arm, "production")}
        code = {name: statistics.fmean(measures["code"] for measures, _ in runs_) for name, runs_ in both.items()}
        unparsed = {name: 1 - statistics.fmean(itertools.chain.from_iterable(parsed for _, parsed in runs_))
                    for name, runs_ in both.items()}
        # per problem, the seeds whose completion parses under the challenger minus under sa-bpe
        seeds_parsing = np.sum([parsed for _, parsed in both[arm]], axis=0) - np.sum(
            [parsed for _, parsed in both["production"]], axis=0)
        here, there = int((seeds_parsing > 0).sum()), int((seeds_parsing < 0).sum())
        p = binomtest(here, here + there).pvalue if here + there else 1.0
        difference = code[arm] - code["production"]
        margin = norm.ppf(1 - ALPHA / 2) * CODE_SIGMA * math.sqrt(2 / len(seeds))
        contrasts[arm] = {"seeds": seeds, "problems_parsing_more_here": here, "problems_parsing_more_in_sa-bpe": there,
                          "p": p, "parsing": "equal" if p >= ALPHA else "better" if here > there else "worse",
                          "code_difference": difference, "code_margin": margin,
                          "code": "better" if difference < -margin else "worse" if difference > margin else "equal",
                          "unparsed_difference": unparsed[arm] - unparsed["production"]}
    verdicts = {arm: (c["parsing"], c["code"]) for arm, c in contrasts.items()}
    qualified = {arm for arm, v in verdicts.items() if "worse" not in v and "better" in v}
    record = json.loads((RESULTS / "tokenizer-intrinsic.json").read_text(encoding="utf-8"))
    rare = {arm: measures["merged_reads"]["under 100"] for arm, measures in record["arms"].items()}
    frequency = "bpe" in qualified or (verdicts["bpe"] == ("equal", "equal") and rare["bpe"] < rare["sa-bpe"])
    # the positive control gates the decision: undetected, the corpus keeps its pattern and score
    control = verdicts["sa-bpe-fused"][0] == "worse"
    pattern = min(qualified & {"sa-bpe-fused", "hybrid"}, key=lambda arm: contrasts[arm]["unparsed_difference"],
                  default="sa-bpe")
    decision = {"pattern": pattern if control else "sa-bpe",
                "score": "frequency" if control and frequency else "sa-bpe",
                "control_detected": control}
    (RESULTS / "tokenizer.json").write_text(
        json.dumps(summary | {"contrasts": contrasts, "decision": decision}, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["vocabularies"]:
            vocabularies()
        case ["intrinsic"]:
            intrinsic()
        case ["runs"]:
            runs()
        case _:
            raise SystemExit("usage: python -m ablations.tokenizer vocabularies|intrinsic|runs")
