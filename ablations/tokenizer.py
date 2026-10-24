"""
The tokenizer ablation: the decision the corpus tokenizer's split pattern and merge score wait on
(nanochat/corpus/tokenizer.py PATTERN, SCORE). Four vocabularies of 32,768 ids (32,764 mergeable tokens, BOS and the 3
FIM sentinels), trained on the same sample (corpus/tokenizer.py sample: the corpus's curated train splits at the file
stage's shares of bytes) by the same crate (corpus/tokenizer.py train) and encoded by tiktoken, each trained on, so that
every candidate the corpus keeps or dismisses has a run of its own:
- sa-bpe: production's pattern and score, the corpus tokenizer's until this decides: SA-BPE's priority F·log R
  (arXiv 2604.14053 §4.1: F the pair's frequency, R the repositories holding it in the current segmentation, a pair of
  one repository scoring 0; the paper's code takes the natural log, pchizhov/sa-bpe scoring.py; Table 3's F·log R row:
  the highest compression of its rows and far fewer tokens verified under-trained than BPE's) within SPLIT_PATTERN. R
  counts the sample's authorship units: the repository for repository files and functions, as in the paper; the
  document otherwise (synthetic code, notebooks, web text and mathematics: their datasets name no author), as §6
  suggests for natural language in the multilingual case ("documents, books, or document collections"), which its
  Limitations leave untested. Not the paper's best and recommended default F·log R·L (Table 4: "our best model"; §6: "a
  good default choice"), L the languages holding the pair, "prioritizing language-universal merges" (§4.1) among
  programming languages of natural shares, where it supports the lower-resource ones (§6, App. G): on Python and C++,
  this sample's languages, it compresses marginally less (Table 10) and uses more of the vocabulary (Table 12); it
  leaves fewer under-trained tokens (Table 3; one tokenizer each, no seeds or repeats reported); §6 says the language
  aspect "must be avoided" for natural language, five of the sample's 17 sources; and nearly all of production's
  merged tokens occur in the val documents of the Python and C++ repository corpora and of the English alike, so L
  would be 3 for all but the vocabulary's tail [inference: val presence standing for the training sample's];
- sa-bpe-fused: within FUSED_PATTERN (karpathy/nanochat's, the first tokenizer's), whose punctuation clause takes the
  line breaks after it, as cl100k's does; also the positive control: arXiv 2608.18062's companion post deletes the
  trailing [\r\n/]* of GPT-4o's punctuation clause with the tokenizer corpus held fixed and raises MBPP pass@1, the
  seed ranges not overlapping (three seeds each; prompts healed: the paper's §4.3, App. C.1), while code bits per byte
  shows no association with MBPP over the post's tokenizers (Spearman); if the parse measure misses it at d16, it
  cannot decide the others (Decisions);
- hybrid: SA-BPE within HYBRID_PATTERN, whose statement chunks are the same at every indentation depth (at
  HYBRID_PATTERN);
- bpe: merges by frequency alone (Table 3's F row, SA-BPE's baseline; the crate's score "frequency") within
  SPLIT_PATTERN, close to F·log R in compression, so a tie in the runs is likely [inference], which the val documents'
  coverage breaks.
The sample's composition is not an arm: the corpus tokenizer's sample follows the training mixture (its evidence in
normalization_doc.md §9), and every arm trains on it.
Not run: tiktoken encodes by merge rank alone, which has no form for PickyBPE's token removals (its encoder replays
merges and removals in training order, arXiv 2409.04599 Alg. 2; its code compression no better than BPE's in
arXiv 2604.14053 Table 4), BoundlessBPE's supermerges (rule dictionaries of their own, arXiv 2504.00178 §3.2) or
UnigramLM; SuperBPE's code results: HumanEval and MBPP pass@10 moved in opposite directions, neither significantly
(arXiv 2503.13423 Table 1, McNemar).

Runs: every arm at d16, SEEDS seeds each (the seed's spread is the measures' largest: the post's fusing tokenizer's
MBPP varied widely over seeds), at equal bytes: each arm reads the code documents of the set once, as production reads
its data (train/stages.py max_ratio), so its tokens are their bytes over its bytes per token; text and math are drawn
by rows, so their bytes vary slightly with the arm's compression [inference]; arXiv 2402.01035 compares at equal
tokens, its Appendix J at equal characters. Each run's plan is its own (schedule.plan at its tokens), its context the
corpus's 8,192 and 32,768 tokens (eval/bpb.py's byte-offset buckets show the bytes of context a better compressor sees;
arXiv 2503.13423 §3.1 fixes context in bytes). The set holds 20 tokens per scaling parameter of d16 for sa-bpe
(Chinchilla, arXiv 2203.15556 Table 3): of each code corpus, the leading curated train documents whose sa-bpe tokens
fill its file-level draw, so the code group binds every arm's max_ratio; of text and math the leading shards holding a
margin over their draw, and of each repository corpus the trailing partitions holding a margin over the repository
stage's draw, so they never bind; of the val split a systematic sample of documents every arm scores whole. The runs
train through ablations/harness.py, whose stack (nanochat/train, nanochat/eval) reads nanochat/data until training
reads the corpus (nanochat/corpus/gh200.md); that cutover builds the set and every arm's packs (corpus compile.py and
pack.py, each taking the arm's tokenizer).

RESULTS/tokenizer-data.json holds each vocabulary's intrinsic record on the corpus's val splits (describe), per
source: bytes per token and Rényi efficiency; its vocabulary entries holding a line break (the count correlating most
strongly with MBPP in the companion post), fusing punctuation with one, and ending in a line break and indentation (the
post's MinGram pair: the first's MBPP far lower, most of its completions IndentationError or TabError); the merged
tokens the val documents read never, under 10 and under 100 times (SA-BPE's Coverage, Table 3, graded), those of them
not UTF-8 on their own and the 243 byte ids UTF-8 can hold counted apart (Magikarp, arXiv 2405.05417 §3.1, names both
as sources of untrained tokens); the entries whose text does not encode back to them (Magikarp §5's check; a token
the pattern forms only in context, as a line break and indentation inside a whitespace-only line, counts too);
TokEval's AST boundary alignment, identifier fragmentation and Python indentation consistency on the code corpora, its
operator isolation on the code and math corpora (ablations/tokeval.py). Not taken from TokEval: its multilingual
metrics (fertility, Gini, MorphScore) and its digit metrics (every arm's pattern cuts a digit run into the same chunks
of at most two digits).

Decisions, pre-registered (RESULTS/tokenizer.json "decision"), each challenger (sa-bpe-fused, hybrid, bpe) against
sa-bpe on the seeds where both have an evaluation (a diverged run, ablations/harness.py, drops its seed from its
arm's contrasts on both sides):
- parsing: per MBPP+ problem, the number of seeds whose healed greedy completion compiles under the challenger minus
  under sa-bpe (ablations/harness.py evaluation); the exact sign test over the problems whose difference is not 0, the
  problems the units (McNemar's exact test at one seed, the companion post's test of single runs on its MinGram pair);
  better or worse at p < ALPHA, else equal. MBPP+ alone: its prompts leave the function header to the model, where
  the post's failures arise, and an evaluation "that uses HumanEval alone does not expose this type of defect" (the
  post); HumanEval+'s outcomes stay in the runs' record;
- code bits per byte: the difference of the arms' means over those seeds, better or worse beyond its 1 - ALPHA
  two-sided normal bound, 1.96 CODE_SIGMA √(1 / n + 1 / n), n the contrast's seeds in each arm (1.96 √(2 / SEEDS)
  CODE_SIGMA = 0.0067 with no divergence), else equal;
- the positive control gates the decision: unless sa-bpe-fused parses worse (control_detected), the corpus keeps its
  pattern and score. Else a challenger qualifies iff it is worse on neither and better on one. Pattern: of
  sa-bpe-fused and the hybrid, the qualifier with the lower MBPP+ unparsed share is adopted, else sa-bpe's. Score: bpe
  is adopted iff it qualifies, or is equal on both and the val documents read fewer of its merged tokens under 100
  times than of sa-bpe's (SA-BPE's claimed benefit, a higher coverage than BPE's, Table 3), carried to the adopted
  pattern [inference]; otherwise SA-BPE.

python -m ablations.tokenizer vocabularies   (the arms into CORPUS/tokenizer-ablation/<arm>)
python -m ablations.tokenizer data           (every arm's compiled val splits, its intrinsic record)
python -m ablations.tokenizer runs           (ablations/harness.py, then the decision; once training reads the corpus)
"""
import argparse
import itertools
import json
import math
import statistics
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import regex
from scipy.stats import binomtest, norm

from ablations import tokeval
from ablations.harness import ALPHA, Setting, main
from nanochat.corpus import mixture
from nanochat.corpus.compile import compile_source
from nanochat.corpus.loader import TOKEN_DTYPE, split_path
from nanochat.corpus.schema import TEXT
from nanochat.corpus.sources import SOURCES
from nanochat.corpus.tokenizer import ABLATION, Score, get_tokenizer, sample, train
from nanochat.data.tokenizer import SPLIT_PATTERN, Tokenizer, save
from nanochat.paths import RESULTS
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
# The spread of a run's 8K code bits per byte over seeds (pooled within each configuration): the six runs at d8 at 20 of
# the standard-regime comparison (60138de's baseline and c00d2f6's default, data seeds 0 to 2), on production's data,
# carried to d16 and to the corpus [inference]
CODE_SIGMA = 0.0034
# The byte ids that cannot occur in UTF-8 (RFC 3629 §4: C0, C1 and F5 to FF)
NOT_UTF8 = frozenset([0xC0, 0xC1, *range(0xF5, 0x100)])
# The benchmark of the parse test (Decisions)
PARSED = "mbpp_plus"


def vocabularies() -> None:
    """Each arm's tokenizer.tiktoken and tokenizer.pattern in ABLATION/<arm>, on one draw of the corpus tokenizer's
    sample. The arms train at once: the crate releases the GIL, and each training's merge loop is mostly serial (one heap
    pop and one pass over the merged pair's words per merge), so the others take the idle cores; every merge is an argmax
    with fixed tie-breaks, so an arm's vocabulary does not depend on the others."""
    for arm in ARMS:
        (ABLATION / arm).mkdir(parents=True)
    docs, units = sample()
    with ThreadPoolExecutor(len(ARMS)) as pool:
        trained = pool.map(lambda spec: train(docs, units, pattern=spec[0], score=spec[1]), ARMS.values())
        for (arm, (pattern, _)), mergeable in zip(ARMS.items(), trained, strict=True):
            save(ABLATION / arm, mergeable, pattern)
            print(f"{arm}: data key {get_tokenizer(arm).data_key}", flush=True)


def reads(counts: np.ndarray) -> dict[str, int]:
    """How many of the tokens were read never, under 10 and under 100 times."""
    return {"never": int((counts == 0).sum()), "under 10": int((counts < 10).sum()),
            "under 100": int((counts < 100).sum())}


def describe(tokenizer: Tokenizer) -> dict:
    """An arm's intrinsic record on the corpus's val splits (module docstring): bytes per token and Rényi efficiency
    (tokeval.py) of each source and of all of them; its vocabulary entries below BOS holding a line break, fusing
    punctuation with one and ending in a line break and indentation; how many of its merged tokens (ids 256 to BOS) the
    val documents read never, under 10 and under 100 times, of them those not UTF-8 on their own, and the same of the
    243 byte ids UTF-8 can hold; the entries whose text does not encode back to them."""
    held = np.zeros(tokenizer.bos_token_id, dtype=np.int64)
    lengths = tokenizer.token_lengths[:tokenizer.bos_token_id]
    bytes_per_token, renyi = {}, {}
    for corpus in SOURCES:
        tokens = np.memmap(split_path(corpus, "val", tokenizer).with_suffix(".bin"), dtype=TOKEN_DTYPE, mode="r")
        counts = np.bincount(tokens, minlength=tokenizer.bos_token_id)
        held += counts
        bytes_per_token[corpus] = float(counts @ lengths / len(tokens))
        renyi[corpus] = tokeval.renyi_efficiency(counts, tokenizer.vocab_size)
    renyi["all"] = tokeval.renyi_efficiency(held, tokenizer.vocab_size)
    entries = tokenizer.token_bytes[:tokenizer.bos_token_id]
    texts = [b.decode("utf-8", "ignore") for b in entries]
    utf8 = np.array([b.decode("utf-8", "ignore").encode() == b for b in entries])
    # a line break and the indentation after it ending the token (\Z: $ also matches before a final line break)
    fused, indented = regex.compile(r"[^\s\p{L}\p{N}][\r\n]"), regex.compile(r"[\r\n][ \t]+\Z")
    merged = held[256:]
    return {"data_key": tokenizer.data_key, "bytes_per_token": bytes_per_token, "renyi_efficiency": renyi,
            "line_break_tokens": sum("\n" in t or "\r" in t for t in texts),
            "punctuation_line_break_tokens": sum(bool(fused.search(t)) for t in texts),
            "line_break_indentation_tokens": sum(bool(indented.search(t)) for t in texts),
            "val_reads": {"merged": reads(merged), "merged_not_utf8": reads(merged[~utf8[256:]]),
                          "bytes": reads(held[[b for b in range(256) if b not in NOT_UTF8]])},
            "not_encoding_back": sum(tokenizer.enc.encode_ordinary(t) != [i]
                                     for i, t in enumerate(texts) if utf8[i] and i >= 256)}


def data() -> None:
    """Every arm's compiled val split of every source, checked to hold the same bytes per document in each, and
    RESULTS/tokenizer-data.json: each arm's intrinsic record (describe) and TokEval's metrics (tokeval.code of every
    code corpus, tokeval.operator_isolation of every code and math corpus)."""
    tokenizers = [get_tokenizer(arm) for arm in ARMS]
    for tokenizer in tokenizers:
        for corpus in SOURCES:
            split_path(corpus, "val", tokenizer).parent.mkdir(parents=True)
            compile_source(corpus, "val", tokenizer)
    for corpus in SOURCES:
        sizes = []
        for tokenizer in tokenizers:
            path = split_path(corpus, "val", tokenizer)
            tokens = np.fromfile(path.with_suffix(".bin"), dtype=TOKEN_DTYPE)
            ends = np.concatenate([[0], np.cumsum(tokenizer.token_lengths[tokens])])
            sizes.append(np.diff(ends[np.fromfile(path.with_suffix(".idx"), dtype=np.int64)]))
        assert all(np.array_equal(sizes[0], other) for other in sizes[1:]), corpus
    record = {tokenizer.name: describe(tokenizer) | {"code": {}, "operator_isolation": {}} for tokenizer in tokenizers}
    code = [s for s, spec in SOURCES.items() if spec.category != TEXT]
    for corpus in code + list(mixture.FILE_STAGE.groups["math"].sources):
        tokens, docs = tokeval.val_split(corpus, tokenizers)
        operators = tokeval.operator_isolation(tokens, docs, tokenizers)
        metrics = tokeval.code(corpus, tokens, docs, tokenizers) if corpus in code else None
        for arm in record:
            record[arm]["operator_isolation"][corpus] = operators[arm]
            if metrics is not None:
                record[arm]["code"][corpus] = metrics[arm]
    (RESULTS / "tokenizer-data.json").write_text(json.dumps(record, indent=1), encoding="utf-8")


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
        assert seeds, f"no seed evaluated under both {arm} and sa-bpe"
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
    rare = {arm: r["val_reads"]["merged"]["under 100"]
            for arm, r in json.loads((RESULTS / "tokenizer-data.json").read_text(encoding="utf-8")).items()}
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
    parser = argparse.ArgumentParser(prog="python -m ablations.tokenizer",
                                     description="The tokenizer experiment's steps")
    parser.add_argument("step", choices=("vocabularies", "data", "runs"))
    {"vocabularies": vocabularies, "data": data, "runs": runs}[parser.parse_args().step]()
