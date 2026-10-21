"""
The tokenizer: the decision production's tokenizer waits on. Five vocabularies of 32,768 ids (32,764 mergeable tokens,
BOS and the 3 FIM sentinels), the four trained ones on the same sample (data/tokenizer.py sample), a 2 x 2 of
- merge score: SA-BPE's F·ln R (arXiv 2604.14053 §4.1, its best monolingual criterion; R the authorship units of
  data/tokenizer.py sample, repositories as in the paper, web hosts and documents an extension it does not test) or
  frequency alone, BPE (the paper's baseline, rustbpe's order; the crate's train score "frequency"); and
- pattern: data/tokenizer.py's SPLIT_PATTERN, whose runs of punctuation take no line break, or FUSED_PATTERN, whose
  runs take the line breaks after them (cl100k's clause): arXiv 2608.18062's companion post deletes that clause with
  the tokenizer corpus held fixed and moves MBPP pass@1 from 0.086 to 0.230, 3 seeds each, ranges disjoint (Appendix
  F.3, footnote 15: 1.27B parameters, ~20B tokens, about 128K-token vocabularies, healed prompts), its failures
  SyntaxErrors, while code bits per byte does not follow (its Table 18);
and cl100k, GPT-4's tokenizer cut to its first 32,764 ranks with its own pattern (digit runs of 1 to 3, punctuation
fused with line breaks, other training data, frequency merges): a reference with no decision attached (the cut is
prefix-closed: ranks 0-255 are the bytes and every kept token encodes to itself, checked in `data`).

Equal bytes: each arm reads its document set once, as production reads its data (train/stages.py max_ratio), so its
tokens are the set's bytes over its bytes per token; at equal tokens a better compressor would read more data
(arXiv 2402.01035 compares at equal tokens, its Appendix J at equal text). Each run's plan is its own (schedule.plan
at its tokens), its context production's 8,192 and 32,768 tokens; eval/bpb.py's byte-offset buckets show the bytes of
context that buys. Settings: d16 on the document set worth 20 tokens per scaling parameter to sa-bpe (Chinchilla,
arXiv 2203.15556 §3) and d8 on the same documents (~91), whether rankings move with training length (arXiv
2503.13423 §4.3), 4 seeds each; then d24 on its own document set (20 for sa-bpe), sa-bpe against one challenger
(--challenger), 2 seeds each, before production adopts a tokenizer (arXiv 2608.18062 §E.3: d16-d24 Kendall τ 0.752 on
bits per byte; small sweeps "a screening tool").

A document set (`data --depth D`) holds, of each code corpus, the leading curated train files whose rows under
sa-bpe hold what d D at 20 tokens per parameter draws at the file-level stage (its rows are production's packs' rows
of those files: the same vocabulary, files and order), so the code group binds every arm's max_ratio; of text and
math, MARGIN times that draw, so they never bind; of each repository corpus, the trailing files holding MARGIN times
the repository stage's draw (the same count for every corpus, as data/pack.py repo_stage); of its val split, the
leading row groups whose tokens under sa-bpe fill at most 1 / MARGIN of bpb's val sample at each row length, so every
arm scores all of them (eval/bpb.py val_bits): the same documents. RESULTS/tokenizer-data.json records each arm's
bytes per token per corpus, its line-break tokens and how many vocabulary tokens its train documents never hold.

Decisions, pre-registered (ablations/harness.py measures; per setting, alternatives against sa-bpe):
- pattern: the fusing level is adopted iff, at d16, its mean unparsed share (HumanEval+ and MBPP+ healed greedy
  completions that do not compile) is not higher than the non-fusing level's and its code bits per byte is lower with
  the 95% interval excluding zero; otherwise non-fusing;
- score: frequency BPE is adopted iff, at d16, its code bits per byte is lower with the interval excluding zero, or the
  interval lies within ±EQUIVALENCE and its document set leaves fewer vocabulary tokens unread; otherwise SA-BPE;
- each from the 2 x 2's 16 d16 runs (8 against 8 per level), the score within the chosen pattern if the interaction
  is significant; every other comparison descriptive (Benjamini-Hochberg within each setting and measure, as arXiv
  2608.18062 §C.2); cl100k reported only. The challenger at d24 is the decided arm, or, if that is sa-bpe, the arm with
  the next lowest d16 code bits per byte.

python -m ablations.tokenizer vocabularies              (the five arms into TOKENIZER_ABLATION/<arm>)
python -m ablations.tokenizer data --depth 16           (every arm's d16 document set, then the checks)
python -m ablations.tokenizer runs                      (d16 and d8; ablations/harness.py)
python -m ablations.tokenizer data --depth 24 --arms sa-bpe <challenger>
python -m ablations.tokenizer runs --challenger <challenger>   (adds d24)
"""
import argparse
import itertools
import json
import math
import statistics

import numpy as np
import pyarrow.parquet as pq
import regex
import torch
from scipy.stats import false_discovery_control, ttest_ind
from scipy.stats import t as student_t
from tiktoken_ext.openai_public import cl100k_base

from ablations.harness import ALPHA, main
from nanochat.config import FILE_STAGE, REPO_STAGE, VOCAB_SIZE, Stage
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
from nanochat.data.tokenizer import (
    SPECIAL_TOKENS,
    SPLIT_PATTERN,
    Tokenizer,
    bpe,
    get_tokenizer,
    merges,
    sample,
    save,
)
from nanochat.eval.bpb import VAL_TOKENS
from nanochat.model.gpt import GPT, gpt_config
from nanochat.paths import RESULTS, TOKENIZER_ABLATION
from nanochat.train.schedule import RunPlan, plan
from nanochat.train.stages import drawn_rows, max_ratio, readable_rows

# SPLIT_PATTERN with cl100k's clause for a run of punctuation, which takes the line breaks after it
FUSED_PATTERN = (r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]"""
                 r"""|\s+(?!\S)|\s+""")
# arm: (pattern, merge score) of the trained arms; cl100k's are its own
TRAINED = {"sa-bpe": (SPLIT_PATTERN, "sa-bpe"), "sa-bpe-fused": (FUSED_PATTERN, "sa-bpe"),
           "bpe": (SPLIT_PATTERN, "frequency"), "bpe-fused": (FUSED_PATTERN, "frequency")}
ARMS = (*TRAINED, "cl100k")
REFERENCE = "sa-bpe"  # the harness's production arm: production's current vocabulary
SEEDS, CHALLENGER_SEEDS = 4, 2
# Text, math and repository-stage rows kept per drawn row: an arm compressing a corpus better than sa-bpe packs it into
# fewer rows (sa-bpe takes 9.21% more tokens than the first tokenizer, sa-bpe-fused's construction, on stack_edu_cpp's
# whole val split, the largest gap measured), so only the code group binds
MARGIN = 1.25
# The equivalence margin of the score decision: about one per-run σ of code bits per byte (0.0034 at d8, the paired
# comparison of Against the standard regime) and three seed SDs of arXiv 2608.18062 (§6: 0.00109 at 1.27B)
EQUIVALENCE = 0.003


def documents_name(depth: int) -> str:
    return f"d{depth}"


def arm_tokenizer(arm: str, depth: int) -> Tokenizer:
    return get_tokenizer(f"{arm}@{documents_name(depth)}")


def run_plan(depth: int, ratio: float) -> RunPlan:
    with torch.device("meta"):
        return plan(model=GPT(gpt_config(depth=depth, stage=FILE_STAGE)), ratio=ratio)


def sample_rows(stage: Stage) -> int:
    """The rows of eval/bpb.py val_bits' sample of a corpus at a stage's rows."""
    return -(-VAL_TOKENS // (stage.batch_rows * stage.seq_len)) * stage.batch_rows


def vocabularies() -> None:
    """Each arm's tokenizer.tiktoken and tokenizer.pattern in TOKENIZER_ABLATION/<arm>: the trained arms on one load
    of the sample, cl100k cut."""
    for arm in ARMS:
        (TOKENIZER_ABLATION / arm).mkdir(parents=True)
    docs, sources = sample()
    for arm, (pattern, score) in TRAINED.items():
        save(TOKENIZER_ABLATION / arm, merges(docs, sources, pattern=pattern, score=score), pattern)
    spec = cl100k_base()
    ranked = sorted(spec["mergeable_ranks"], key=spec["mergeable_ranks"].get)
    save(TOKENIZER_ABLATION / "cl100k", ranked[:VOCAB_SIZE - len(SPECIAL_TOKENS)], spec["pat_str"])


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


def leading(rows: np.ndarray, need: float) -> int:
    """The fewest leading files whose rows (per file) hold need."""
    return int(np.searchsorted(np.cumsum(rows), math.ceil(need))) + 1


def documents(depth: int) -> dict[str, dict]:
    """{corpus: {"train": [file-level train files], "kept": their leading documents kept, "repo": [repository-level
    train files], "val_step": every how many val documents one is kept}} of the document set for d<depth> at 20 tokens
    per parameter of the reference vocabulary, in production's compiled data and packs (the same vocabulary): of a
    code corpus the leading documents whose tokens fill its file-level draw's rows at its leading files' tokens per row,
    of text and math the leading files holding MARGIN times theirs; of the val split a systematic sample of documents
    whose tokens fill at most 1 / MARGIN of bpb's sample at every stage reading the corpus."""
    production = get_tokenizer()
    run = run_plan(depth, 20)
    code = set(next(iter(FILE_STAGE.mixture.values()))[1])
    chosen = {}
    for corpus, drawn in drawn_rows(production, stage=FILE_STAGE, run=run).items():
        rows = np.bincount(row_shards(corpus, "train", production, FILE_STAGE.seq_len))
        files = leading(rows, drawn if corpus in code else MARGIN * drawn)
        bounds = shard_documents(corpus, "train", production)
        offsets = np.memmap(split_path(corpus, "train", production).with_suffix(".idx"), dtype=np.int64, mode="r")
        tokens_per_row = offsets[bounds[files]] / rows[:files].sum()
        kept = (int(np.searchsorted(offsets, drawn * tokens_per_row)) if corpus in code else int(bounds[files]))
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


def data(depth: int, arms: list[str]) -> None:
    """The arms' compiled and packed document set for d<depth> (module docstring), then the checks: the cl100k cut
    prefix-closed, the trained arms sa-bpe and sa-bpe-fused the vocabularies of production and of the first tokenizer
    (data keys printed), each arm's max_ratio at the set's settings, every draw within its readable rows, every val
    split within bpb's sample; RESULTS/tokenizer-data.json gains the arms' records (describe)."""
    chosen = documents(depth)
    bpe()  # built once, before the packer's threads call it
    cut = arm_tokenizer("cl100k", depth)
    print("cl100k cut prefix-closed:", cut.token_bytes[:256] == [bytes([i]) for i in range(256)]
          and all(cut.enc.encode_single_token(b) == i for i, b in enumerate(cut.token_bytes[:cut.bos_token_id])))
    record_path = RESULTS / "tokenizer-data.json"
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}
    for arm in arms:
        tokenizer = arm_tokenizer(arm, depth)
        for corpus, files in chosen.items():
            split_path(corpus, "val", tokenizer).parent.mkdir(parents=True)
            for split, curated in (("val", curated_files(corpus, "val")), ("train", files["train"] + files["repo"])):
                compile_split(corpus, split, tokenizer,
                              row_groups={file: pq.read_metadata(file).num_row_groups for file in curated})
            val = np.arange(int(shard_documents(corpus, "val", tokenizer)[-1]))
            keep_documents(corpus, "val", tokenizer, val % files["val_step"] == 0)
            train = np.arange(int(shard_documents(corpus, "train", tokenizer)[-1]))
            repository = int(shard_documents(corpus, "train", tokenizer)[len(files["train"])])  # trailing files' first
            keep_documents(corpus, "train", tokenizer, (train < files["kept"]) | (train >= repository))
            first, trailing = len(files["train"]), len(files["repo"])
            stages = [(FILE_STAGE, range(first))]
            if trailing:  # a repository corpus
                stages.append((REPO_STAGE, range(first, first + trailing)))
            for stage, shards in stages:
                pack_split(corpus, "val", tokenizer, seq_len=stage.seq_len)
                rows = write(corpus, "train", tokenizer, in_order(corpus, "train", tokenizer, shards,
                                                                  seq_len=stage.seq_len), seq_len=stage.seq_len)
                print(f"{arm} {corpus} T{stage.seq_len}: {len(shards)} train files, {rows:,} rows")
        record.setdefault(documents_name(depth), {})[arm] = describe(tokenizer, chosen)
        print(f"{arm}: data key {tokenizer.data_key}")
        for setting_depth in {16: (16, 8), 24: (24,)}[depth]:  # the runs on this document set
            ratio = max_ratio(tokenizer, depth=setting_depth)
            run = run_plan(setting_depth, ratio)
            for stage in (FILE_STAGE, REPO_STAGE):
                readable = readable_rows(tokenizer, stage=stage, run=run)
                slack = {corpus: len(readable[corpus]) - drawn
                         for corpus, drawn in drawn_rows(tokenizer, stage=stage, run=run).items()}
                tight = min(slack, key=slack.get)
                print(f"{arm} d{setting_depth} at {ratio:.2f} T{stage.seq_len}: least slack {slack[tight]:,} rows "
                      f"({tight})")
        for stage in (FILE_STAGE, REPO_STAGE):
            fill = {corpus: packed_rows(corpus, "val", tokenizer, stage.seq_len) / sample_rows(stage)
                    for corpus in mixture_corpora(stage.mixture)}
            fullest = max(fill, key=fill.get)
            print(f"{arm} val T{stage.seq_len}: at most {fill[fullest]:.3f} of bpb's sample ({fullest})")
    record_path.write_text(json.dumps(record, indent=1), encoding="utf-8")


def contrast(a: list[float], b: list[float]) -> dict:
    """a's mean minus b's, the two-sided pooled t-test's p and the difference's 1 - ALPHA interval (harness.compare's
    statistics)."""
    test = ttest_ind(a, b)
    interval = test.confidence_interval(confidence_level=1 - ALPHA)
    return {"difference": statistics.fmean(a) - statistics.fmean(b), "p": float(test.pvalue),
            "interval": [float(interval.low), float(interval.high)]}


def analysis(lines: list[dict], summaries: dict, record: dict) -> dict:
    """The pre-registered decisions (module docstring) from RESULTS/tokenizer.jsonl's records (lines), the harness's
    per-setting comparisons (summaries: RESULTS/tokenizer.json) and RESULTS/tokenizer-data.json (record)."""
    def runs_of(arm: str, measure: str) -> list[float]:
        name = "production" if arm == REFERENCE else arm
        return [line["measures"][measure] for line in lines
                if line["arm"] == name and line["setting"]["depth"] == 16 and line["measures"]]

    def level(arms: tuple[str, ...], measure: str) -> list[float]:
        return [value for arm in arms for value in runs_of(arm, measure)]

    # score x pattern interaction on code at d16: the contrast of contrasts over the four cells, pooled variance
    cells = [runs_of(arm, "code") for arm in ("sa-bpe", "sa-bpe-fused", "bpe", "bpe-fused")]
    means, df = [statistics.fmean(c) for c in cells], sum(len(c) - 1 for c in cells)
    pooled = sum((len(c) - 1) * statistics.variance(c) for c in cells) / df
    interaction = (means[0] - means[1]) - (means[2] - means[3])
    t = interaction / math.sqrt(pooled * sum(1 / len(c) for c in cells))
    interacts = 2 * student_t.sf(abs(t), df) < ALPHA
    pattern = {"code": contrast(level(("sa-bpe-fused", "bpe-fused"), "code"), level(("sa-bpe", "bpe"), "code")),
               "unparsed": statistics.fmean(level(("sa-bpe-fused", "bpe-fused"), "unparsed"))
               - statistics.fmean(level(("sa-bpe", "bpe"), "unparsed"))}
    fusing = pattern["unparsed"] <= 0 and pattern["code"]["interval"][1] < 0
    suffix = "-fused" if fusing else ""
    score = (contrast(runs_of("bpe" + suffix, "code"), runs_of("sa-bpe" + suffix, "code")) if interacts
             else contrast(level(("bpe", "bpe-fused"), "code"), level(("sa-bpe", "sa-bpe-fused"), "code")))
    never = {arm: record["d16"][arm]["unread_tokens"]["never"] for arm in TRAINED}
    low, high = score["interval"]
    frequency = high < 0 or (-EQUIVALENCE < low <= high < EQUIVALENCE
                             and never["bpe" + suffix] < never["sa-bpe" + suffix])
    decided = ("bpe" if frequency else "sa-bpe") + suffix
    code = {arm: statistics.fmean(runs_of(arm, "code")) for arm in TRAINED}
    challenger = decided if decided != REFERENCE else min((a for a in TRAINED if a != REFERENCE), key=code.get)
    # Benjamini-Hochberg over the alternatives of each setting and measure (the descriptive comparisons)
    adjusted = {}
    for setting in summaries["settings"]:
        alternatives = setting["alternatives"]
        for measure in next(iter(alternatives.values())).get("p", {}):
            ps = {arm: s["p"][measure] for arm, s in alternatives.items() if s.get("p", {}).get(measure) is not None}
            if ps:
                bh = false_discovery_control(list(ps.values()), method="bh")
                adjusted.setdefault(f"d{setting['depth']}", {})[measure] = dict(zip(ps, bh.tolist(), strict=True))
    return {"interaction": {"contrast": interaction, "t": t, "df": df, "significant": bool(interacts)},
            "pattern": pattern | {"fusing": bool(fusing)}, "score": score | {"never_read": never,
                                                                            "frequency": bool(frequency)},
            "decided": decided, "challenger": challenger, "bh_adjusted_p": adjusted}


def runs(challenger: str | None) -> None:
    """Every arm at d16 and d8 on the d16 document set (and sa-bpe against the challenger at d24 on its own), each at
    its own max_ratio: the harness's experiment; then RESULTS/tokenizer.json gains the analysis."""
    plans = [(16, 16, ARMS, SEEDS), (8, 16, ARMS, SEEDS)]
    if challenger:
        plans.append((24, 24, (REFERENCE, challenger), CHALLENGER_SEEDS))
    settings = []
    for depth, documents_depth, arms, seeds in plans:
        names = {arm: f"{arm}@{documents_name(documents_depth)}" for arm in arms}
        ratios = {arm: max_ratio(get_tokenizer(name), depth=depth) for arm, name in names.items()}
        settings.append((depth, ratios[REFERENCE], range(seeds), {"tokenizer": names[REFERENCE]},
                         {arm: {"tokenizer": names[arm], "ratio": ratios[arm]} for arm in arms[1:]}))
    main(settings=tuple(settings))
    lines = [json.loads(line) for line in (RESULTS / "tokenizer.jsonl").read_text(encoding="utf-8").splitlines()]
    summaries = json.loads((RESULTS / "tokenizer.json").read_text(encoding="utf-8"))
    record = json.loads((RESULTS / "tokenizer-data.json").read_text(encoding="utf-8"))
    (RESULTS / "tokenizer.json").write_text(json.dumps(summaries | {"analysis": analysis(lines, summaries, record)},
                                                       allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The tokenizer ablation: its vocabularies, its data, its runs")
    parser.add_argument("step", choices=("vocabularies", "data", "runs"))
    parser.add_argument("--depth", type=int, choices=(16, 24), default=16)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--challenger", choices=ARMS[1:])
    args = parser.parse_args()
    match args.step:
        case "vocabularies":
            vocabularies()
        case "data":
            data(args.depth, args.arms)
        case "runs":
            runs(args.challenger)
