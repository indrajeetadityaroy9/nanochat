"""
Train a vocabulary of the corpus: byte-level BPE merges by SA-BPE's F·ln R (arXiv 2604.14053 §4.1) or frequency,
on a systematic sample of the file stage's curated train splits; fidelity checks the encoding against the merges.
"""
import functools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow.parquet as pq
import regex
from tokenizers import models

from corpus import mixture
from corpus.parallel import SPAWN
from corpus.sources import curated_shards, text_bytes, texts
from corpus.tokenizer import SPECIAL_TOKENS, get_tokenizer
from model.config import VOCAB_SIZE
from native import corpus_crate

Score = Literal["sa-bpe", "frequency"]

# The reference tokenizer's sample size, so every vocabulary trains on as much text
SAMPLE_BYTES = 13.38e9


@functools.cache
def shard_bytes(source: str) -> list[int]:
    """The text bytes of each shard of the source's curated train split."""
    return [text_bytes(pq.read_metadata(path), "content") for path in curated_shards(source, "train")]


def byte_shares() -> dict[str, float]:
    """Each file-stage source's share of the stage's bytes: its token share (Stage.shares) at its bytes per token."""
    sources, per_token = mixture.FILE_LEVEL_STAGE.sources, mixture.BYTES_PER_TOKEN
    shares = mixture.FILE_LEVEL_STAGE.shares({s: sum(shard_bytes(s)) / per_token[s] for s in sources},
                                             mixture.FILE_LEVEL_STAGE.tokens)
    stage_bytes = sum(shares[s] * per_token[s] for s in sources)
    return {s: shares[s] * per_token[s] / stage_bytes for s in sources}


def sample_row_group(task: tuple[str, Path, int, int, float]) -> tuple[list[str], list[str], int]:
    """Row group g of a curated train shard: (texts, authorship units, UTF-8 bytes) of the rows the systematic sample
    takes, row i when floor((i + 1) * fraction) > floor(i * fraction), `start` the source's rows before the group."""
    source, path, g, start, fraction = task
    group = pq.ParquetFile(path).read_row_group(g)
    rows = np.arange(start, start + group.num_rows)
    batch = group.take(np.flatnonzero(np.floor((rows + 1) * fraction) > np.floor(rows * fraction)))
    if "repo_name" in batch.column_names:
        units = [name.lower() for name in batch["repo_name"].to_pylist()]
    else:
        units = batch["id"].to_pylist()
    docs = texts(source, batch)
    return docs, units, sum(len(text.encode()) for text in docs)


def sample() -> tuple[list[str], list[int], dict[str, dict[str, float]]]:
    """The training sample (texts, unit ids) and its composition per source (target bytes, fraction, documents, bytes,
    units)."""
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
    """VOCAB_SIZE - len(SPECIAL_TOKENS) - 256 merges (merge i makes token 256 + i) by the crate's train."""
    return corpus_crate().train(docs, units, pattern=pattern, n_merges=VOCAB_SIZE - len(SPECIAL_TOKENS) - 256,
                                score=score)


def vocabulary(merges: list[tuple[int, int]]) -> list[bytes]:
    """The mergeable tokens of merges in rank order: the 256 bytes, then each merge's."""
    mergeable = [bytes([i]) for i in range(256)]
    for left, right in merges:
        mergeable.append(mergeable[left] + mergeable[right])
    return mergeable


def fidelity(task: tuple[Path, str]) -> dict[str, int]:
    """The vocabulary in `directory` checked on the first val row group of a source: chunks cover every text, tiktoken
    equals the merges replayed (Hugging Face tokenizers' BPE over Latin-1-spelled bytes), every text decodes back.
    Raises RuntimeError on a mismatch."""
    directory, source = task
    tokenizer = get_tokenizer(directory)
    spelled = [entry.decode("latin-1") for entry in tokenizer.token_bytes[:tokenizer.bos_token_id]]
    merges = json.loads((directory / "merges.json").read_text())
    bpe = models.BPE(vocab={s: i for i, s in enumerate(spelled)}, merges=[(spelled[a], spelled[b]) for a, b in merges])
    pattern = regex.compile(tokenizer.pattern)
    docs = texts(source, pq.ParquetFile(curated_shards(source, "val")[0]).read_row_group(0))
    chunks = 0
    for doc in docs:
        pieces = [m.group().encode() for m in pattern.finditer(doc)]
        if b"".join(pieces) != doc.encode():
            raise RuntimeError(f"{directory.name}: its pattern leaves text of a {source} document uncovered")
        if tokenizer.enc.decode(tokenizer.enc.encode_ordinary(doc)) != doc:
            raise RuntimeError(f"{directory.name}: a {source} document does not decode back")
        for piece in pieces:
            if tokenizer.enc._encode_single_piece(piece) != [t.id for t in bpe.tokenize(piece.decode("latin-1"))]:
                raise RuntimeError(f"{directory.name}: tiktoken encodes {piece!r} ({source}) unlike the trained merges")
        chunks += len(pieces)
    return {"documents": len(docs), "chunks": chunks}
