"""Tokenizer ablation: pre-tokenizer split pattern x BPE merge score (ARMS); trains, measures and packs each arm.
Scores: SA-BPE's F·ln R (arXiv 2604.14053 §4.1) and frequency F (Table 3); measures include TokEval (arXiv 2608.18062).

python -m ablations.tokenizer_split_merge vocabularies|intrinsic|data|repack
"""
import argparse
import datetime
import importlib.metadata
import itertools
import json
import os
import platform
import subprocess
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import regex
from scipy.stats import hmean

import metrics
from corpus import mixture
from corpus.layout import TOKEN_DTYPE, split_path, stage_rows
from corpus.pack import packed_shards, shard_count, write
from corpus.parallel import SPAWN, memory
from corpus.schema import EXERCISE, FUNCTION, REPOSITORY_FILE, SYNTHETIC_FILE, TEXT
from corpus.sources import SOURCES, curated_documents, curated_shards, texts
from corpus.tokenize import tokenize_repo_level_stage, tokenize_repositories, tokenize_source, train_parts
from corpus.tokenizer import ABLATION, SPLIT_PATTERN, get_tokenizer, save
from corpus.train_tokenizer import Score, byte_shares, fidelity, sample, shard_bytes, train, vocabulary
from native import corpus_crate
from paths import RESULTS, ROOT

# Records: RESULTS/<EXPERIMENT>-{vocabularies,intrinsic,data}.json
EXPERIMENT = "tokenizer_split_merge"

# karpathy/nanochat 92d63d4 split pattern: cl100k's punctuation clause [\r\n]* takes the line breaks after it ("):\n")
FUSED_PATTERN = (r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]"""
                 r"""|\s+(?!\S)|\s+""")
# Line breaks, space runs and tab runs are chunks of their own, so chunks(indent + s) = (indent,) + chunks(s) for an
# indent of >= 2 spaces or >= 1 tab; no contraction clause (arXiv 2402.01035 App. E); no lookahead
HYBRID_PATTERN = r"""(?:[^\s\p{L}\p{N}]| )?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++|\n+|\r+|\t+| +|[^\S\r\n\t ]+"""
ARMS: dict[str, tuple[str, Score]] = {"sa-bpe": (SPLIT_PATTERN, "sa-bpe"), "sa-bpe-fused": (FUSED_PATTERN, "sa-bpe"),
                                      "hybrid": (HYBRID_PATTERN, "sa-bpe"), "bpe": (SPLIT_PATTERN, "frequency")}
NUM_HIDDEN_LAYERS = 16
# Chinchilla tokens per non-embedding parameter (arXiv 2203.15556 Table 3), d16 under sa-bpe
TOKENS_PER_PARAM = 20
# The set's margin over the draws of the corpora that must not bind (text, math, repositories); assumed
MARGIN = 1.5
# Categories whose documents are whole program texts in their source's language
PROGRAMS = (REPOSITORY_FILE, EXERCISE, SYNTHETIC_FILE, FUNCTION)
# Rate standard error <= √(1/(4n)) ≈ 0.0022
INDENTED_LINES = 50_000
INDENTED = regex.compile(r"([ \t]+)(\S.*)")
TOKEVAL_BYTES = 2_000_000
TOKEVAL_GROUPS = ("python", "cpp", "text", "math", "mixed")
TOKEVAL_LANGUAGES = {"Python": "python", "C++": "cpp"}


def provenance() -> dict:
    """Commit, UTC time, host, cores, memory and the measuring packages' versions of a step."""
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    return {"commit": commit, "utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "host": platform.node(), "machine": platform.machine(), "cpus": os.cpu_count(),
            "memory_gb": memory() / 2**30, "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name)
                         for name in ("tiktoken", "tokenizers", "numpy", "scipy", "pyarrow", "regex",
                                      "tokenizer-intrinsic-evals", "tree-sitter-language-pack")}}


def vocabularies() -> None:
    """Train every arm on one draw of the sample; write ABLATION/<arm> and the vocabularies record."""
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
    """An arm's planning, mechanism and tie-break measures from its val token counts per source."""
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
        """Weighted harmonic mean of the members' bytes per token over their file-stage byte shares."""
        return float(hmean([sources[s]["bytes_per_token"] for s in members], weights=[shares[s] for s in members]))

    entries = [b.decode("utf-8", "ignore") for b in tokenizer.mergeable]
    # \Z: $ also matches before a final line break
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
    """{arm: lines, invariant, rate}: indented lines tokenized as the indentation's then the statement's at depth 0."""
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
    """A source's TokEval language: its code language, math, text or mixed (prose with fenced code)."""
    spec = SOURCES[source]
    if spec.category in PROGRAMS:
        return TOKEVAL_LANGUAGES[spec.language]
    if source in mixture.MATH_SOURCES:
        return "math"
    return "text" if spec.category == TEXT else "mixed"


def tokeval_documents(source: str) -> tuple[list[str], bool]:
    """A source's leading val documents up to TOKEVAL_BYTES, and whether the split ran out first."""
    taken, size = [], 0
    for path in curated_shards(source, "val"):
        f = pq.ParquetFile(path)
        for g in range(f.num_row_groups):
            for doc in curated_documents(source, f.read_row_group(g)):
                taken.append(doc)
                size += len(doc.encode("utf-8"))
                if size >= TOKEVAL_BYTES:
                    return taken, False
    return taken, True


def tokeval(arm: str) -> dict:
    """An arm's TokEval record: "sample" per source and "metrics" (global values, AST per category, char splits)."""
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
    """Tokenize every arm's val splits, check equal bytes, run the measures in one pool; write the intrinsic record."""
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

        splits = dict(zip(checks, threads.map(tokenized, checks), strict=True))
        tokenized_seconds = time.perf_counter() - start
        counts, sizes = {}, {}
        for check, future in splits.items():
            counts[check], sizes[check] = future.result()
        for corpus in SOURCES:
            if not all(np.array_equal(sizes["sa-bpe", corpus], sizes[arm, corpus]) for arm in ARMS):
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
    """The set's shards for the d16 run `run` under sa-bpe: {"file", "repository", "draws"}."""
    from train.stages import drawn

    measured = json.loads((RESULTS / f"{EXPERIMENT}-intrinsic.json").read_text(encoding="utf-8"))["arms"]["sa-bpe"]
    estimate = {s: np.array(shard_bytes(s)) / measured["sources"][s]["bytes_per_token"] for s in SOURCES}

    def shards(source: str, tokens: float, start: int) -> int:
        """The fewest train shards of a source from `start` whose estimated sa-bpe tokens hold `tokens`."""
        held = np.cumsum(estimate[source][start:])
        if held[-1] < tokens:
            raise RuntimeError(f"{source}: its train shards from {start} hold {held[-1]:,.0f} of {tokens:,.0f} tokens")
        return int(np.searchsorted(held, tokens)) + 1

    def draws(stage: mixture.Stage) -> dict[str, float]:
        """Each corpus's tokens in the run at its share of the base model's stage."""
        shares = stage.shares({s: estimate[s].sum() if s in SOURCES else 1.0 for s in stage.sources}, stage.tokens)
        return {s: share * drawn(stage, run) * stage.seq_len for s, share in shares.items()}

    file_draws, repo_draws = draws(mixture.FILE_LEVEL_STAGE), draws(mixture.REPO_LEVEL_STAGE)
    file = {s: shards(s, tokens * (1 if s in mixture.FILE_LEVEL_STAGE.code else MARGIN), 0)
            for s, tokens in file_draws.items()}
    repository = {s: shards(s, tokens * (1 if s in mixture.REPO_LEVEL_STAGE.supply else MARGIN), file.get(s, 0))
                  for s, tokens in repo_draws.items() if s in SOURCES}
    return {"file": file, "repository": repository, "draws": {"8192": file_draws, "32768": repo_draws}}


def pack_corpus(corpus: str, directory: Path, shards: dict) -> None:
    """Pack a tokenized corpus in each stage that reads it: its train shards in the set, its val split whole."""
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
    """Each arm's run at its max_tokens_per_param: data key, plan, per stage each corpus's rows, weight and read share."""
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
            shares, draws, _ = mixture.weights(stage, rows, count)
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
    """Tokenize and pack every arm's set, each corpus a job in one process pool; write the data record."""
    # the training stack (the model's kernels) here alone: the measures' spawned workers import this module
    from run import base_plan

    start, began = time.perf_counter(), provenance()
    run = base_plan(num_hidden_layers=NUM_HIDDEN_LAYERS, tokens_per_param=TOKENS_PER_PARAM)
    shards = document_set(run)
    leading = {s: shards["file"].get(s, 0) + shards["repository"].get(s, 0) for s in SOURCES}
    tokenized = []

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
    """Repack every arm's set from its tokenized splits after a packer change; rewrite the data record's arms."""
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


if __name__ == "__main__":
    steps = {"vocabularies": vocabularies, "intrinsic": intrinsic, "data": data, "repack": repack}
    parser = argparse.ArgumentParser(prog="python -m ablations.tokenizer_split_merge",
                                     description="A step of the tokenizer ablation.")
    parser.add_argument("step", choices=steps)
    steps[parser.parse_args().step]()
