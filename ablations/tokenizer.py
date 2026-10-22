"""
The tokenizer: the decision production's tokenizer waits on. Four vocabularies of 32,768 ids (32,764 mergeable tokens,
BOS and the 3 FIM sentinels), trained on the same sample (data/tokenizer.py sample) by the same crate and encoded by
tiktoken, each trained on, so that every candidate production keeps or dismisses has a run of its own:
- sa-bpe: production's, SA-BPE's priority F·log R (arXiv 2604.14053 §4.1: F the pair's frequency, R the repositories
  holding it in the current segmentation, a pair of one repository scoring 0; Table 3's F·log R row: compression
  above BPE's and far fewer tokens verified under-trained) within SPLIT_PATTERN. R counts the authorship units of
  data/tokenizer.py sample: repositories as in the paper, web hosts and documents as §6 suggests for natural language
  ("documents, books, or document collections"), which its Limitations leave untested. Not the paper's best and
  recommended default F·log R·L (Table 4: "our best model"; §6: "a good default choice"), L the languages holding the
  pair, "prioritizing language-universal merges" (§4.1) among programming languages of natural shares, where it
  supports the lower-resource ones (§6, App. G): on Python and C++, this sample's languages, it compresses marginally
  less (Table 10) and uses more of the vocabulary (Table 12); it leaves fewer under-trained tokens (Table 3; one
  tokenizer each, no seeds or repeats reported); §6 says the language aspect "must be avoided" for natural language, a
  third of this sample's sources; and nearly all of production's merged tokens occur in the val documents of the
  Python and C++ repository corpora and of the English (climbmix and both math corpora) alike, so L would be 3 for all
  but the vocabulary's tail [inference: val presence standing for the training sample's];
- sa-bpe-fused: within FUSED_PATTERN (karpathy/nanochat's, the first tokenizer's), whose punctuation clause takes the
  line breaks after it, as cl100k's does; also the positive control: arXiv 2608.18062's companion post deletes the
  trailing [\r\n/]* of GPT-4o's punctuation clause with the tokenizer corpus held fixed and raises MBPP pass@1 far
  beyond its seed spread (healed prompts: the paper's §4.1-4.3), while code bits per byte shows no association with
  MBPP over the companion post's tokenizers (Spearman); if the parse measure misses it at d16, it cannot decide the
  others (Decisions);
- hybrid: SA-BPE within HYBRID_PATTERN, whose clauses resolve the limitations of the others' (at HYBRID_PATTERN);
- bpe: merges by frequency alone (Table 3's F row, SA-BPE's baseline; the crate's score "frequency") within
  SPLIT_PATTERN, close to F·log R in compression, so a tie in the runs is likely [inference], which the documents'
  statistics break.
Described, not trained on: sa-bpe-mixture (COMPOSITION), sa-bpe on the sample at the training mixture's shares
(data/tokenizer.py sample "mixture"; the arms' sample is production's, SAMPLE_SHARES "equal"): of its bytes more code,
less math and more of the Python corpora (the three Python repository corpora and kaggle_notebooks) than the equal
sample's. The evidence is on compression and language modelling, none on code accuracy. More code in a tokenizer's
data compresses code better (arXiv 2402.01035 §2.3: "tokenizers should be trained on the data mix that they are
expected to see"; the NSL of a subset varies little over its mixes); a tokenizer "trained on similar data as the
model" leaves the fewest under-trained tokens (arXiv 2405.05417 §5, across released models); at nanochat's d24 equal
tokenizer-training weights over many languages raised the model's mean FLORES+ bits per byte above weights
proportional to its training data in every pair, most in the languages of the smallest training shares, and the
authors recommend "proportional tokenizer-training weights as the default" (arXiv 2610.12144 Table 6, §7). On code,
the companion post's tokenizer corpora of varying code shares (by documents, one run each, at a fixed token budget)
leave MBPP unordered with math and code bits per byte slightly lower at the largest share; code holding most of the
training data, as the largest languages there, the contrast would move code bits per byte by less than the runs
resolve (1.96 √(2 / SEEDS) CODE_SIGMA = 0.0067) [inference]. Its intrinsic record holds the measures that literature
names: bytes per token and the tokens the documents hold rarely.
Not run: tiktoken encodes by merge rank alone, which has no form for PickyBPE's token removals (its encoder replays
merges and removals in training order, arXiv 2409.04599 Alg. 2; its code compression no better than BPE's in
arXiv 2604.14053 Table 4), BoundlessBPE's supermerges (rule dictionaries of their own, arXiv 2504.00178 §3.2) or
UnigramLM; SuperBPE's code results: HumanEval and MBPP pass@10 moved in opposite directions, neither significantly
(arXiv 2503.13423 Table 1, McNemar).

Runs: every arm at d16, SEEDS seeds each (the seed's spread is the measures' largest: the post's fusing tokenizer's
MBPP varied widely over seeds), at equal bytes: each arm reads the code documents of the set once, as production reads
its data (train/stages.py max_ratio), so its tokens are their bytes over its bytes per token; text and math are drawn
by rows, so their bytes vary slightly with the arm's compression [inference]; arXiv 2402.01035 compares at equal
tokens, its Appendix J at equal characters. Each run's plan is its
own (schedule.plan at its tokens), its context production's 8,192 and 32,768 tokens (eval/bpb.py's byte-offset
buckets show the bytes of context a better compressor sees; arXiv 2503.13423 §3.1 fixes context in bytes). The set
holds 20 tokens per scaling parameter of d16
for sa-bpe (Chinchilla, arXiv 2203.15556 Table 3): of each code corpus, the leading
curated train documents whose sa-bpe tokens fill its file-level draw
(rows from production's packs: the same vocabulary, files and order), so the code group binds every arm's max_ratio;
of text and math the leading files holding MARGIN times their draw, and of each repository corpus the trailing files
holding MARGIN times the repository stage's draw (the same count for every corpus, as data/pack.py repo_stage), so
they never bind; of the val split a systematic sample of documents whose sa-bpe tokens fill at most 1 / MARGIN of
bpb's val sample at each row length, so every arm scores all of them (eval/bpb.py val_bits): the same documents.

RESULTS/tokenizer-data.json holds each vocabulary's intrinsic record (describe), per corpus: bytes per token and Rényi
efficiency; its vocabulary entries holding a line break (the count correlating most strongly with MBPP in the
companion post), fusing punctuation with one, and ending in a line break and indentation (the post's MinGram pair:
the first's MBPP far lower, most of its completions IndentationError or TabError); the merged tokens its train
documents hold never, under 10 and under 100 times (a proxy for SA-BPE's
verified under-trained tokens); TokEval's AST boundary alignment, identifier fragmentation and Python indentation
consistency on the code corpora, its operator isolation on the code and math corpora (ablations/tokeval.py). Not taken
from TokEval: its multilingual metrics (fertility, Gini, MorphScore) and its digit metrics (every arm's pattern cuts a
digit run into the same chunks of at most two digits).

Decisions, pre-registered (RESULTS/tokenizer.json "decision"), each challenger (sa-bpe-fused, hybrid, bpe) against
sa-bpe on the seeds where both have an evaluation (a diverged run, ablations/harness.py, drops its seed from its
arm's contrasts on both sides):
- parsing: per HumanEval+ and MBPP+ problem, the number of seeds whose healed greedy completion compiles under the
  challenger minus under sa-bpe (ablations/harness.py evaluation); the exact sign test over the problems whose
  difference is not 0, the problems the units (McNemar's exact test at one seed, the companion post's test of single
  runs on its MinGram pair); better or worse at p < ALPHA, else equal;
- code bits per byte: the difference of the arms' means over those seeds, better or worse beyond its 1 - ALPHA
  two-sided normal bound, 1.96 CODE_SIGMA √(1 / n + 1 / n), n the contrast's seeds in each arm (1.96 √(2 / SEEDS)
  CODE_SIGMA = 0.0067 with no divergence), else equal;
- the positive control gates the decision: unless sa-bpe-fused parses worse (control_detected), production's pattern
  and score are kept. Else a challenger qualifies iff it is worse on neither and better on one. Pattern: of
  sa-bpe-fused and the hybrid, the qualifier with the lower unparsed share is adopted, else production's pattern.
  Score: bpe is adopted iff it qualifies, or is equal on both and its documents leave fewer merged tokens never read
  than sa-bpe's (a proxy for SA-BPE's claimed benefit, fewer verified under-trained tokens than F, Table 3), carried
  to the adopted pattern [inference]; otherwise SA-BPE.

python -m ablations.tokenizer vocabularies   (the arms and COMPOSITION into TOKENIZER_ABLATION/<name>)
python -m ablations.tokenizer data           (every vocabulary's document set, its intrinsic record, then the checks)
python -m ablations.tokenizer runs           (ablations/harness.py, then the decision)
"""
import argparse
import itertools
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import regex
from scipy.stats import binomtest, norm

from ablations import tokeval
from ablations.harness import ALPHA, Setting, main, production_plan
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
from nanochat.data.tokenizer import SAMPLE_SHARES, SPLIT_PATTERN, Tokenizer, get_tokenizer, merges, sample, save
from nanochat.eval.bpb import sample_rows
from nanochat.native import data_crate
from nanochat.paths import RESULTS, TOKENIZER_ABLATION
from nanochat.train.stages import drawn_rows, max_ratio, readable_rows

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
#     (the punctuation clause without its trailing line breaks), shares this. And no token joins a line break to
#     whitespace, where production's \s*[\r\n] takes the spaces before a break and a whitespace-only line
#     ("\n    \n"), whose merges make tokens of a line break and indentation: such entries of the post's MinGram cb,
#     whose completions failed mostly by IndentationError or TabError (MBPP far below cb_nl's);
# (2) a run of spaces, of tabs or of other whitespace is one chunk ( +, \t+: the Claude-approximate [ ]+, [\t]+), and a
#     letter run takes no tab or line break before it: an indentation of d >= 2 spaces or d >= 1 tabs ends at a chunk
#     end, so, no clause looking behind, the statement s after it is cut as at depth 0: chunks(indent + s) =
#     (indent,) + chunks(s), one indentation chunk per depth and the statement's chunks the same at every depth.
#     Production's \s+(?!\S) gives the indentation's last space to the statement's first chunk (" if", " (") or to a
#     chunk of its own (before a digit): chunks = (" " * (d - 1), " " + w, ...), the depth carried by d - 1 spaces and
#     a space inside a statement chunk that differs from its depth-0 form ("if" against " if"); its letter clause
#     [^\r\n\p{L}\p{N}]?+ admits a tab, so a tab-indented statement starts "\tif". A single space before a letter run
#     stays its prefix (" foo"), as in prose. The Claude-approximate pattern, an early reconstruction in the post,
#     scored MBPP on par with gpt4onl over seeds, few of its completions IndentationErrors;
# (3) no contraction clause (2402.01035's Punct drops cl100k's, its App. E and Fig. 3; o200k folds contractions into
#     the letter run): the letter clause takes a whole run of letters, which '(?i:[sdmt]|ll|ve|re) cuts where an
#     apostrophe starts a chunk ("'s" "elf" for a line's 'self'); English contractions chunk as before ("doesn" "'t").
# Kept from cl100k: one leading punctuation character on a letter run (".append", "(self"), which Punct drops (it keeps
# the leading space, " ?\p{L}+", 2402.01035 Fig. 3; Table 1, code: compresses better than Punct, App. E; pass@1 higher
# for Punct at the smaller model and vocabulary, lower at the larger, untested for significance), and a leading space
# (" foo");
# production's digit pairs (karpathy/nanochat: "2 is the sweet spot for vocab size of 32K") and punctuation runs. And
# [^\S\r\n\t ]+ takes every other whitespace character, so every character falls in some chunk: tiktoken drops what no
# clause matches.
HYBRID_PATTERN = r"""(?:[^\s\p{L}\p{N}]| )?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++|\n+|\r+|\t+| +|[^\S\r\n\t ]+"""
# arm: (pattern, merge score); every arm is trained on, sa-bpe as the harness's production arm
ARMS = {"sa-bpe": (SPLIT_PATTERN, "sa-bpe"), "sa-bpe-fused": (FUSED_PATTERN, "sa-bpe"),
        "hybrid": (HYBRID_PATTERN, "sa-bpe"), "bpe": (SPLIT_PATTERN, "frequency")}
CHALLENGERS = ("sa-bpe-fused", "hybrid", "bpe")
# The sample-composition vocabulary (module docstring): sa-bpe's pattern and score on the sample at the training
# mixture's shares, described beside the arms, not trained on
COMPOSITION = "sa-bpe-mixture"
DEPTH, SEEDS = 16, 2
# The spread of a run's 8K code bits per byte over seeds (pooled within each configuration): the six runs at d8 at 20 of
# the standard-regime comparison (60138de's baseline and c00d2f6's default, data seeds 0 to 2), carried to d16
# [inference]
CODE_SIGMA = 0.0034
# Text, math and repository-stage rows kept per drawn row: an arm compressing a corpus better than sa-bpe packs it into
# fewer rows (sa-bpe takes more tokens than the first tokenizer, sa-bpe-fused's construction, most on stack_edu_cpp's
# val split), so only the code group binds
MARGIN = 1.25


def vocabularies() -> None:
    """Each arm's tokenizer.tiktoken and tokenizer.pattern in TOKENIZER_ABLATION/<arm>, on one load of the sample, then
    COMPOSITION's, on the sample at the mixture's shares."""
    for shares, arms in ((SAMPLE_SHARES, ARMS), ("mixture", {COMPOSITION: ARMS["sa-bpe"]})):
        docs, sources = sample(shares=shares)
        for arm, (pattern, score) in arms.items():
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


@dataclass(frozen=True, kw_only=True)
class Documents:
    """A corpus's part of the document set: its leading curated train files (`train`), of which the first `kept`
    documents are kept; its trailing train files the repository stage reads (`repo`: a repository corpus's, else none);
    every `val_step`-th document of its val split."""
    train: list[Path]
    kept: int
    repo: list[Path]
    val_step: int


def documents() -> dict[str, Documents]:
    """The document set of every corpus (module docstring), from production's compiled data and packs, which count
    sa-bpe's tokens: sa-bpe is production's vocabulary, retrained."""
    production = get_tokenizer()
    assert get_tokenizer("sa-bpe").data_key == production.data_key
    run = production_plan(depth=DEPTH, ratio=20)
    # the repository stage: the fewest trailing files, as many of every repository corpus, holding MARGIN times its draw
    repo = drawn_rows(production, stage=REPO_STAGE, run=run)
    repo_rows = {corpus: np.bincount(row_shards(corpus, "train", production, REPO_STAGE.seq_len)) for corpus in repo}
    trailing = 1
    while sum(int(rows[-trailing:].sum()) for rows in repo_rows.values()) < MARGIN * sum(repo.values()):
        assert trailing < min(len(rows) for rows in repo_rows.values()), "the repository corpora hold too few rows"
        trailing += 1
    chosen = {}
    for corpus, drawn in drawn_rows(production, stage=FILE_STAGE, run=run).items():
        code = corpus in FILE_STAGE.mixture["code"][1]
        rows = np.bincount(row_shards(corpus, "train", production, FILE_STAGE.seq_len))
        # the fewest leading files whose rows hold the draw (code) or MARGIN times it
        files = int(np.searchsorted(np.cumsum(rows), math.ceil(drawn if code else MARGIN * drawn))) + 1
        bounds = shard_documents(corpus, "train", production)
        offsets = np.memmap(split_path(corpus, "train", production).with_suffix(".idx"), dtype=np.int64, mode="r")
        tokens_per_row = offsets[bounds[files]] / rows[:files].sum()
        limit = min(sample_rows(stage) * stage.seq_len / MARGIN for stage in (FILE_STAGE, REPO_STAGE)
                    if corpus in mixture_corpora(stage.mixture))
        val = np.memmap(split_path(corpus, "val", production).with_suffix(".idx"), dtype=np.int64, mode="r")
        train = curated_files(corpus, "train")
        assert files + (trailing if corpus in repo else 0) <= len(train), corpus
        chosen[corpus] = Documents(
            train=train[:files],
            kept=int(np.searchsorted(offsets, drawn * tokens_per_row)) if code else int(bounds[files]),
            repo=train[-trailing:] if corpus in repo else [],
            val_step=math.ceil(val[-1] / limit))
    return chosen


def describe(tokenizer: Tokenizer, chosen: dict[str, Documents]) -> dict:
    """An arm's intrinsic record on its document set: bytes per token of each corpus's train and val documents; the
    Rényi efficiency of each corpus's val documents and of all of them (tokeval.py); its vocabulary entries below BOS
    (the 256 bytes included, as the companion post counts vocabulary entries) holding a line break, fusing punctuation
    with one and ending in a line break and indentation; how many merged tokens (ids 256 to BOS: of the byte ids, 13
    cannot occur in UTF-8) its train documents hold never, under 10 and under 100 times, a proxy for SA-BPE's verified
    under-trained tokens (arXiv 2604.14053 Table 3)."""
    held = {split: np.zeros(tokenizer.bos_token_id, dtype=np.int64) for split in ("train", "val")}
    lengths = tokenizer.token_lengths[:tokenizer.bos_token_id]
    bytes_per_token, renyi = {}, {}
    for corpus in chosen:
        bytes_per_token[corpus] = {}
        for split, total in held.items():
            tokens = np.memmap(split_path(corpus, split, tokenizer).with_suffix(".bin"), dtype=TOKEN_DTYPE, mode="r")
            counts = np.bincount(tokens, minlength=tokenizer.bos_token_id)
            total += counts
            bytes_per_token[corpus][split] = float(counts @ lengths / len(tokens))
            if split == "val":
                renyi[corpus] = tokeval.renyi_efficiency(counts, tokenizer.vocab_size)
    renyi["all"] = tokeval.renyi_efficiency(held["val"], tokenizer.vocab_size)
    entries = [b.decode("utf-8", "ignore") for b in tokenizer.token_bytes[:tokenizer.bos_token_id]]
    # a line break and the indentation after it ending the token (\Z: $ also matches before a final line break)
    fused, indented = regex.compile(r"[^\s\p{L}\p{N}][\r\n]"), regex.compile(r"[\r\n][ \t]+\Z")
    never = held["train"][256:]
    return {"data_key": tokenizer.data_key, "bytes_per_token": bytes_per_token, "renyi_efficiency": renyi,
            "line_break_tokens": sum("\n" in t or "\r" in t for t in entries),
            "punctuation_line_break_tokens": sum(bool(fused.search(t)) for t in entries),
            "line_break_indentation_tokens": sum(bool(indented.search(t)) for t in entries),
            "unread_tokens": {"never": int((never == 0).sum()), "under 10": int((never < 10).sum()),
                              "under 100": int((never < 100).sum())}}


def data() -> None:
    """Every arm's and COMPOSITION's compiled document set, checked to hold the same bytes per document in each, and
    RESULTS/tokenizer-data.json (describe, and TokEval's metrics: tokeval.code of every code corpus,
    tokeval.operator_isolation of every code and math corpus); every arm's packs (COMPOSITION is not trained on) and
    their checks: max_ratio, every draw within its readable rows, every val split within bpb's sample."""
    chosen = documents()
    tokenizers = [get_tokenizer(name) for name in (*ARMS, COMPOSITION)]
    for tokenizer in tokenizers:
        for corpus, part in chosen.items():
            split_path(corpus, "val", tokenizer).parent.mkdir(parents=True)
            compile_split(corpus, "val", tokenizer, files=curated_files(corpus, "val"))
            compile_split(corpus, "train", tokenizer, files=part.train + part.repo)
            val = np.arange(int(shard_documents(corpus, "val", tokenizer)[-1]))
            keep_documents(corpus, "val", tokenizer, val % part.val_step == 0)
            bounds = shard_documents(corpus, "train", tokenizer)
            train = np.arange(int(bounds[-1]))
            # the leading files' first kept documents and every document of the trailing (repository-stage) files
            keep_documents(corpus, "train", tokenizer, (train < part.kept) | (train >= bounds[len(part.train)]))
        print(f"{tokenizer.name}: data key {tokenizer.data_key}")
    for corpus, split in itertools.product(chosen, ("train", "val")):
        sizes = []
        for tokenizer in tokenizers:
            path = split_path(corpus, split, tokenizer)
            tokens = np.fromfile(path.with_suffix(".bin"), dtype=TOKEN_DTYPE)
            ends = np.concatenate([[0], np.cumsum(tokenizer.token_lengths[tokens])])
            sizes.append(np.diff(ends[np.fromfile(path.with_suffix(".idx"), dtype=np.int64)]))
        assert all(np.array_equal(sizes[0], other) for other in sizes[1:]), (corpus, split)
    record = {tokenizer.name: describe(tokenizer, chosen) | {"code": {}, "operator_isolation": {}}
              for tokenizer in tokenizers}
    code, math_ = FILE_STAGE.mixture["code"][1], FILE_STAGE.mixture["math"][1]
    for corpus in code + math_:
        tokens, docs = tokeval.val_split(corpus, tokenizers)
        operators = tokeval.operator_isolation(tokens, docs, tokenizers)
        metrics = tokeval.code(corpus, tokens, docs, tokenizers) if corpus in code else None
        for arm in record:
            record[arm]["operator_isolation"][corpus] = operators[arm]
            if metrics is not None:
                record[arm]["code"][corpus] = metrics[arm]
    (RESULTS / "tokenizer-data.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    data_crate()  # built once, before the packer's threads call it
    for arm in ARMS:
        tokenizer = get_tokenizer(arm)
        for corpus, part in chosen.items():
            first, trailing = len(part.train), len(part.repo)
            stages = [(FILE_STAGE, range(first))]
            if trailing:  # a repository corpus
                stages.append((REPO_STAGE, range(first, first + trailing)))
            for stage, shards in stages:
                pack_split(corpus, "val", tokenizer, seq_len=stage.seq_len)
                rows = write(corpus, "train", tokenizer, in_order(corpus, "train", tokenizer, shards,
                                                                  seq_len=stage.seq_len), seq_len=stage.seq_len)
                print(f"{arm} {corpus} T{stage.seq_len}: {len(shards)} train files, {rows:,} rows")
        ratio = max_ratio(tokenizer, depth=DEPTH)
        run = production_plan(depth=DEPTH, ratio=ratio)
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
            assert slack[tight] >= 0 and fill[fullest] <= 1, (arm, stage.seq_len)


def runs() -> None:
    """Every arm at d16, each at its max_ratio (ablations/harness.py main); then RESULTS/tokenizer.json gains each
    challenger's contrast with sa-bpe and the decision (module docstring)."""
    ratios = {arm: max_ratio(get_tokenizer(arm), depth=DEPTH) for arm in ARMS}
    main(experiment="tokenizer",
         settings=[Setting(depth=DEPTH, ratio=ratios["sa-bpe"], seeds=range(SEEDS),
                           alternatives={arm: {"tokenizer": arm, "ratio": ratios[arm]} for arm in CHALLENGERS})],
         production={"tokenizer": "sa-bpe"})
    # (arm, seed) of every evaluated run (a diverged run has none): its measures and, per problem in the benchmarks'
    # order, whether its completion compiles
    evaluated = {}
    for line in (RESULTS / "tokenizer.jsonl").read_text(encoding="utf-8").splitlines():
        run = json.loads(line)
        if run["measures"] is not None:
            outcomes = run["evaluations"][-1]["parse"].values()
            evaluated[run["arm"], run["fields"]["seed"]] = (
                run["measures"], [o == "parses" for b in outcomes for o in b["outcomes"]])
    summary = json.loads((RESULTS / "tokenizer.json").read_text(encoding="utf-8"))
    contrasts = {}
    for arm in CHALLENGERS:
        seeds = [seed for seed in range(SEEDS) if (arm, seed) in evaluated and ("production", seed) in evaluated]
        assert seeds, f"no seed evaluated under both {arm} and sa-bpe"
        both = {name: [evaluated[name, seed] for seed in seeds] for name in (arm, "production")}
        mean = {name: {m: statistics.fmean(measures[m] for measures, _ in runs_) for m in ("code", "unparsed")}
                for name, runs_ in both.items()}
        # per problem, the seeds whose completion parses under the challenger minus under sa-bpe
        seeds_parsing = np.sum([parsed for _, parsed in both[arm]], axis=0) - np.sum(
            [parsed for _, parsed in both["production"]], axis=0)
        here, there = int((seeds_parsing > 0).sum()), int((seeds_parsing < 0).sum())
        p = binomtest(here, here + there).pvalue if here + there else 1.0
        code = mean[arm]["code"] - mean["production"]["code"]
        margin = norm.ppf(1 - ALPHA / 2) * CODE_SIGMA * math.sqrt(2 / len(seeds))
        contrasts[arm] = {"seeds": seeds, "problems_parsing_more_here": here, "problems_parsing_more_in_sa-bpe": there,
                          "p": p, "parsing": "equal" if p >= ALPHA else "better" if here > there else "worse",
                          "code_difference": code, "code_margin": margin,
                          "code": "better" if code < -margin else "worse" if code > margin else "equal",
                          "unparsed_difference": mean[arm]["unparsed"] - mean["production"]["unparsed"]}
    verdicts = {arm: (c["parsing"], c["code"]) for arm, c in contrasts.items()}
    qualified = {arm for arm, v in verdicts.items() if "worse" not in v and "better" in v}
    never = {arm: r["unread_tokens"]["never"]
             for arm, r in json.loads((RESULTS / "tokenizer-data.json").read_text(encoding="utf-8")).items()}
    frequency = "bpe" in qualified or (verdicts["bpe"] == ("equal", "equal") and never["bpe"] < never["sa-bpe"])
    # the positive control gates the decision: undetected, production's pattern and score are kept
    control = verdicts["sa-bpe-fused"][0] == "worse"
    pattern = min(qualified & {"sa-bpe-fused", "hybrid"}, key=lambda arm: contrasts[arm]["unparsed_difference"],
                  default="sa-bpe")
    decision = {"pattern": pattern if control else "sa-bpe",
                "score": "frequency" if control and frequency else "sa-bpe",
                "control_detected": control}
    (RESULTS / "tokenizer.json").write_text(
        json.dumps(summary | {"contrasts": contrasts, "decision": decision}, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python -m ablations.tokenizer",
                                     description="The tokenizer experiment's steps")
    parser.add_argument("step", choices=("vocabularies", "data", "runs"))
    {"vocabularies": vocabularies, "data": data, "runs": runs}[parser.parse_args().step]()
