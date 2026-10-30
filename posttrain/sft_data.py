"""
The SFT corpus (docs/post-training.md §11.2): the reasoning traces of Nemotron-Cascade's SFT stages 1 and 2 (code)
and of Nemotron-SFT-Competitive-Programming-v2 (Python, C++, Exercism), rendered in §11.1's format (render), filtered,
at most CAP traces per prompt in each group, split by problem (problem) into train and val, train less the problems
near a val one, tokenized and packed whole into 32,768-token rows by the base's packer (corpus/pack.py), one attention
segment per trace. Each document's context (the prompt and THINK) is written as <split>.context, whose labels the
loader ignores (corpus/loader.py).

python -m posttrain.sft_data
"""

import hashlib
import json
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from corpus import decontaminate, pack
from corpus.layout import TOKEN_DTYPE, split_path
from corpus.parallel import SPAWN
from corpus.tokenize import tokenize_split
from corpus.tokenizer import get_tokenizer
from native import corpus_crate
from paths import ARTIFACTS
from posttrain.sft import END_THINK, NAME, SFT_STAGE, THINK

SOURCE = ARTIFACTS / "posttrain" / "nvidia"
# Each group's files (globs under SOURCE) and row format, in the order duplicates are resolved (the first kept)
GROUPS = {
    "cascade_stage1": ("Nemotron-Cascade-SFT-Stage-1/code/*.jsonl", "cascade"),
    "cascade_stage2": ("Nemotron-Cascade-SFT-Stage-2/code/*.jsonl", "cascade"),
    "cp_python": ("Nemotron-SFT-Competitive-Programming-v2/data/competitive_programming_python_*.jsonl", "competitive"),
    "cp_cpp": ("Nemotron-SFT-Competitive-Programming-v2/data/competitive_programming_cpp_*.jsonl", "competitive"),
    "cp_exercism": ("Nemotron-SFT-Competitive-Programming-v2/data/exercism.jsonl", "exercism"),
}
CAP = 4                 # traces per prompt in each group, drawn at random
VAL_SHARE = 0.01        # problems held out
MIN_REASONING = 20      # reasoning tokens
# A train problem sharing a run of OVERLAP_N words with a held-out problem is dropped: one past the longest common
# stock phrases measured (input conventions of Codeforces and CodeChef statements, standard definitions: 45 words);
# near-duplicate problems share runs from 40 words up (docs/post-training.md §11.2)
OVERLAP_N = 46
SEED = 0
CHUNK_BYTES = 2**28     # a worker task: one byte range of a source file
SHARD_DOCUMENTS = 4096  # documents per packed shard (one crate pack call)
# Drop reasons, in the order a row is tested
DROPS = ("malformed", "placeholder", "no_reasoning", "no_answer", "short_reasoning", "over_length", "contaminated",
         "duplicate", "capped", "val_overlap")
# Aider's file listing (CP-v2 Exercism): a filename line, then a fenced block
LISTING = re.compile(r"^(\S+\.\w+)\n```[^\n]*\n(.*?)\n```", re.S | re.M)
# Aider's editing instructions after an Exercism problem statement
AIDER_REQUEST = "\n####\n\nUse the above instructions to modify the supplied files:"
# The sources' stock instructions around a problem: CP-v2's header, Nemotron-Cascade's language and format requests
WRAPPERS = re.compile("|".join((
    r"You are a helpful and harmless assistant\. You should think step-by-step before responding to the instruction "
    r"below\.",
    r"Please use \S+ programming language only\.",
    r"You must use ```\w+ for just the final solution code block with the following format:\s*```\w+\s*\S+ Your code "
    r"here\s*```",
    r"Write (?:Python|C\+\+) code to solve the problem\.",
    r"Solve the problem starting with the provided function header\.\s*Function header:",
    r"Please place the solution code in the following format:\s*```\w+\s*# Your solution code here\s*```")))

Parsed = tuple[str, str, str] | str  # (prompt, reasoning, answer) or a drop reason


def cascade(row: dict) -> Parsed:
    """Nemotron-Cascade: the user message; the reply's text between its leading <think> and </think>, then the rest."""
    messages = row["messages"]
    if [m["role"] for m in messages] != ["user", "assistant"]:
        return "malformed"
    head, closed, answer = messages[1]["content"].lstrip().partition("</think>")
    if not closed or not head.startswith("<think>"):
        return "no_reasoning"
    return messages[0]["content"], head.removeprefix("<think>"), answer


def competitive(row: dict) -> Parsed:
    """CP-v2 Python and C++: the user message; the reply's reasoning_content, then its content."""
    messages = row["messages"]
    if [m["role"] for m in messages] != ["user", "assistant"]:
        return "malformed"
    return messages[0]["content"], messages[1].get("reasoning_content") or "", messages[1]["content"] or ""


def exercism(row: dict) -> Parsed:
    """CP-v2 Exercism, Aider's 9 messages (system prompt and an example edit first): the stub's file listings
    (message 5 without Aider's preamble) and the problem statement (message 7 without AIDER_REQUEST's instructions);
    the reply's prose before its first listing, then that listing's code as a Python block."""
    messages = row["messages"]
    if len(messages) != 9 or (stub := LISTING.search(messages[5]["content"])) is None:
        return "malformed"
    reply = messages[8]["content"]
    if (listing := LISTING.search(reply)) is None:
        return "no_answer"
    statement = messages[7]["content"].partition(AIDER_REQUEST)[0]
    return (messages[5]["content"][stub.start():].strip() + "\n\n" + statement.strip(), reply[:listing.start()],
            f"```python\n{listing.group(2)}\n```")


FORMATS = {"cascade": cascade, "competitive": competitive, "exercism": exercism}


def parse(row: dict, kind: str) -> Parsed:
    """A row's stripped (prompt, reasoning, answer), or the first drop reason before tokenization."""
    parsed = FORMATS[kind](row)
    if isinstance(parsed, str):
        return parsed
    prompt, reasoning, answer = (text.strip() for text in parsed)
    if prompt in ("", "-"):
        return "placeholder"
    if not reasoning:
        return "no_reasoning"
    if not answer:
        return "no_answer"
    return prompt, reasoning, answer


def render(prompt: str, reasoning: str, answer: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """§11.1's document without its BOS tokens: (context, reasoning, the rest) tokens, each encoded alone as ordinary
    text (the context as inference encodes a prompt)."""
    enc = get_tokenizer().enc
    return tuple(enc.encode_to_numpy(text, disallowed_special=()).astype(TOKEN_DTYPE)
                 for text in (prompt + THINK, reasoning, END_THINK + answer))


def digest(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")


def problem(prompt: str) -> str:
    """A prompt's problem, the split's unit: the prompt without WRAPPERS, lowercased, its words (runs of letters and
    digits) joined by single spaces, so one problem under the sources' different wrappers, whitespace and markup is
    one text."""
    return " ".join(re.findall(r"[^\W_]+", WRAPPERS.sub(" ", prompt).lower()))


def read_chunk(task: tuple[str, Path, int, int, Path]) -> dict:
    """The rows starting in bytes [start, end) of a source file: each row's drop reason ("reasons", an index into
    DROPS, -1 for a candidate); per candidate the digests of its prompt, its problem and its text, its context and
    document tokens; {problem digest: problem} ("texts"); the candidates' tokens appended to `part`. Runs in a
    decontaminate.worker_init worker (decontaminate.GRAMS)."""
    kind, path, start, end, part = task
    reasons, prompts, problems, keys, contexts, lengths, texts = [], [], [], [], [], [], {}
    with open(path, "rb") as f, open(part, "wb") as out:
        f.seek(max(start - 1, 0))
        if start > 0 and f.read(1) != b"\n":  # a row starting before `start` is the previous task's
            f.readline()
        while f.tell() < end and (line := f.readline()):
            parsed = parse(json.loads(line), kind)
            if isinstance(parsed, str):
                reasons.append(DROPS.index(parsed))
                continue
            prompt, reasoning, answer = parsed
            context, thought, rest = render(prompt, reasoning, answer)
            length = len(context) + len(thought) + len(rest)
            if len(thought) < MIN_REASONING:
                reasons.append(DROPS.index("short_reasoning"))
            elif length + 2 > SFT_STAGE.seq_len:  # BOS, the document and BOS
                reasons.append(DROPS.index("over_length"))
            elif decontaminate.GRAMS.matches([f"{prompt}\n{reasoning}\n{answer}"])[0]:
                reasons.append(DROPS.index("contaminated"))
            else:
                reasons.append(-1)
                text = problem(prompt)
                texts[digest(text)] = text
                prompts.append(digest(prompt))
                problems.append(digest(text))
                keys.append(digest(prompt + THINK + reasoning + END_THINK + answer))
                contexts.append(len(context))
                lengths.append(length)
                out.write(np.concatenate([context, thought, rest]))
    return {"reasons": np.array(reasons, np.int64), "prompts": np.array(prompts, np.uint64),
            "problems": np.array(problems, np.uint64), "keys": np.array(keys, np.uint64),
            "contexts": np.array(contexts, np.int64), "lengths": np.array(lengths, np.int64), "texts": texts}


def ranks(values: np.ndarray) -> np.ndarray:
    """Each element's position among the equal values before it."""
    order = np.argsort(values, kind="stable")
    starts = np.r_[True, values[order][1:] != values[order][:-1]]
    rank = np.empty(len(values), np.int64)
    rank[order] = np.arange(len(values)) - np.flatnonzero(starts)[np.cumsum(starts) - 1]
    return rank


def split_stats(lengths: np.ndarray, contexts: np.ndarray, prompts: np.ndarray, problems: np.ndarray
                ) -> dict[str, int]:
    """Rows, prompts, problems, tokens (BOS, document, BOS) and loss tokens (the response and the closing BOS)."""
    return {"rows": len(lengths), "prompts": len(np.unique(prompts)), "problems": len(np.unique(problems)),
            "tokens": int((lengths + 2).sum()), "loss_tokens": int((lengths - contexts + 1).sum())}


def main() -> None:
    tokenizer, rng = get_tokenizer(), np.random.default_rng(SEED)
    directory = split_path(NAME, "train", tokenizer).parent
    directory.mkdir(parents=True)  # no resume: a new directory per tokenizer
    parts_dir = directory / "parts"
    parts_dir.mkdir()
    tasks, owners = [], []
    for g, (pattern, kind) in enumerate(GROUPS.values()):
        for path in sorted(SOURCE.glob(pattern)):
            size = path.stat().st_size
            for start in range(0, size, CHUNK_BYTES):
                tasks.append((kind, path, start, min(start + CHUNK_BYTES, size), parts_dir / f"{len(tasks)}.bin"))
                owners.append(g)
    texts = list(decontaminate.benchmark_texts().values())
    corpus_crate()  # built once here, before the workers load it
    with ProcessPoolExecutor(mp_context=SPAWN, initializer=decontaminate.worker_init, initargs=(texts,)) as pool:
        chunks = list(pool.map(read_chunk, tasks))

    # the candidates, in task order: their group, part file and offset in it
    columns = {k: np.concatenate([c[k] for c in chunks])
               for k in ("prompts", "problems", "keys", "contexts", "lengths")}
    texts = {k: t for c in chunks for k, t in c["texts"].items()}
    counts = [len(c["lengths"]) for c in chunks]
    group = np.repeat(owners, counts)
    part = np.repeat(np.arange(len(chunks)), counts)
    offset = np.concatenate([np.cumsum(c["lengths"]) - c["lengths"] for c in chunks])
    prompts, problems = columns["prompts"], columns["problems"]
    contexts, lengths = columns["contexts"], columns["lengths"]
    duplicate = np.ones(len(lengths), bool)
    duplicate[np.unique(columns["keys"], return_index=True)[1]] = False
    kept, capped = [], np.zeros(len(lengths), bool)
    for g in range(len(GROUPS)):
        drawn = rng.permutation(np.flatnonzero((group == g) & ~duplicate))
        rank = ranks(prompts[drawn])
        capped[drawn[rank >= CAP]] = True
        kept.append(drawn[rank < CAP])
    kept = np.concatenate(kept)
    # val: VAL_SHARE of the problems; train: the rest, less every problem sharing OVERLAP_N words with a val one
    distinct = np.unique(problems[kept])
    held = rng.choice(distinct, size=round(VAL_SHARE * len(distinct)), replace=False)
    val = np.isin(problems[kept], held)
    grams = corpus_crate().Grams([texts[k] for k in held.tolist()], n=OVERLAP_N)
    candidates = np.unique(problems[kept[~val]])
    near = candidates[[bool(m) for m in grams.matches([texts[k] for k in candidates.tolist()])]]
    overlap = np.zeros(len(lengths), bool)
    overlap[kept[~val & np.isin(problems[kept], near)]] = True
    splits = {"train": rng.permutation(kept[~val & ~overlap[kept]]), "val": rng.permutation(kept[val])}

    parts = {i: np.memmap(task[4], dtype=TOKEN_DTYPE, mode="r") for i, task in enumerate(tasks)
             if task[4].stat().st_size}
    for split, rows in splits.items():
        shards = [rows[i:i + SHARD_DOCUMENTS] for i in range(0, len(rows), SHARD_DOCUMENTS)]
        tokenize_split(NAME, split, [1] * len(shards), (
            (np.concatenate([parts[part[i]][offset[i]:offset[i] + lengths[i]] for i in shard]), lengths[shard])
            for shard in shards))
        contexts[rows].tofile(split_path(NAME, split, tokenizer).with_suffix(".context"))
    del parts
    shutil.rmtree(parts_dir)

    stats = {}
    for g, name in enumerate(GROUPS):
        reasons = np.concatenate([c["reasons"] for c, owner in zip(chunks, owners, strict=True) if owner == g])
        mine = group == g
        dropped = dict(zip(DROPS, np.bincount(reasons[reasons >= 0], minlength=len(DROPS)).tolist(), strict=True))
        dropped |= {"duplicate": int((mine & duplicate).sum()), "capped": int((mine & capped).sum()),
                    "val_overlap": int((mine & overlap).sum())}
        stats[name] = {"read": len(reasons), "dropped": dropped} | {
            split: split_stats(lengths[r], contexts[r], prompts[r], problems[r]) for split, rows in splits.items()
            for r in [rows[group[rows] == g]]}
    stats["all"] = {split: split_stats(lengths[rows], contexts[rows], prompts[rows], problems[rows])
                    for split, rows in splits.items()}
    stats["all"]["overlap_n"] = OVERLAP_N
    for split in splits:
        packed = pack.write(NAME, split, pack.packed_shards(NAME, split, range(pack.shard_count(NAME, split)),
                                                            SFT_STAGE.seq_len), SFT_STAGE.seq_len)
        # input positions in use: each document after its BOS
        stats["all"][split] |= {"packed_rows": packed,
                                "fill": int((lengths[splits[split]] + 1).sum()) / (packed * SFT_STAGE.seq_len)}
    (directory / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1), flush=True)


if __name__ == "__main__":
    main()
