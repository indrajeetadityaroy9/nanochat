"""
BPE Tokenizer in the style of GPT-4: train with rustbpe, inference with tiktoken.

Train on pretraining corpora, an equal share of --max-chars from each (writes <base_dir>/tokenizer/tokenizer.pkl and
tokenizer_stats.json):
python -m nanochat.tokenizer
python -m nanochat.tokenizer --dataset=stack_edu,climbmix
"""

import os
import time
import pickle
import argparse
import hashlib

import torch
import rustbpe
import tiktoken

from nanochat.common import get_base_dir

BOS = "<|bos|>"  # the only special token: every document begins with it, so it delimits documents in packed rows

# NOTE: this split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
# I did this because I didn't want to "waste" too many tokens on numbers for smaller vocab sizes.
# I verified that 2 is the sweet spot for vocab size of 32K. 1 is a bit worse, 3 was worse still.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# Tokenizer based on rustbpe + tiktoken combo

class RustBPETokenizer:
    """Light wrapper around tiktoken (for efficient inference) but train with rustbpe"""

    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # 1) train using rustbpe; BOS takes the id after the merges
        tokenizer = rustbpe.Tokenizer()
        assert vocab_size - 1 >= 256, f"vocab_size must be at least 257 (256 bytes and BOS), got {vocab_size}"
        tokenizer.train_from_iterator(text_iterator, vocab_size - 1, pattern=SPLIT_PATTERN)
        # 2) construct the associated tiktoken encoding for inference
        mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}  # token bytes -> merge priority rank
        enc = tiktoken.Encoding(name="rustbpe", pat_str=tokenizer.get_pattern(), mergeable_ranks=mergeable_ranks,
                                special_tokens={BOS: len(mergeable_ranks)})
        return cls(enc)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        """Token ids of a string, or of each string of a list (encoded on num_threads threads); prepend: a token id put
        first."""
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            return ids if prepend is None else [prepend, *ids]
        ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
        return ids if prepend is None else [[prepend, *row] for row in ids]

    def decode(self, ids):
        return self.enc.decode(ids)

    def get_token_bytes(self, device="cpu"):
        """
        Byte length of every token id, as an int32 tensor of shape (vocab_size,), used for
        vocab-size-invariant bits per byte. BOS counts 0 bytes. Uses the raw token bytes: decoding to str first corrupts
        tokens that are not valid standalone UTF-8.
        """
        token_bytes = self.enc.decode_tokens_bytes(list(range(self.get_vocab_size())))
        lengths = torch.tensor([len(b) for b in token_bytes], dtype=torch.int32)
        lengths[self.bos_token_id] = 0
        return lengths.to(device)

    def fingerprint(self):
        """Stable 16-hex identity of the vocabulary (split pattern, every token's bytes, special tokens), keying compiled data."""
        digest = hashlib.sha256(self.enc._pat_str.encode())
        for token in self.enc.decode_tokens_bytes(list(range(self.get_vocab_size()))):
            digest.update(len(token).to_bytes(4, "little") + token)
        return digest.hexdigest()[:16]

    def save(self, tokenizer_dir):
        # save the encoding object to disk
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")


# -----------------------------------------------------------------------------
# nanochat-specific convenience functions

def get_tokenizer_dir():
    return os.path.join(get_base_dir(), "tokenizer")

def get_tokenizer():
    return RustBPETokenizer.from_directory(get_tokenizer_dir())

# -----------------------------------------------------------------------------
# Train the tokenizer on the pretraining data

if __name__ == "__main__":
    import json
    import pyarrow.parquet as pq
    from nanochat.data.pretrain.sources import DATASETS, DEFAULT_DATASET, CodeCorpus, list_raw_files, require_raw_files, fetch_command

    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on the pretraining data")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help=f"pretraining corpus, or comma-separated corpora, each an equal share of --max-chars (nanochat/data/pretrain/sources.py) (default: {DEFAULT_DATASET})")
    parser.add_argument("--max-chars", type=int, default=2_000_000_000, help="Maximum characters to train on (default: 2B)")
    parser.add_argument("--doc-cap", type=int, default=10_000, help="Maximum characters per document (default: 10,000)")
    parser.add_argument("--vocab-size", type=int, default=32768, help="Vocabulary size (default: 32768 = 2^15)")
    args = parser.parse_args()
    print(f"dataset: {args.dataset} | max_chars: {args.max_chars:,} | doc_cap: {args.doc_cap:,} | vocab_size: {args.vocab_size:,}")
    datasets = args.dataset.split(",")
    share = args.max_chars // len(datasets)

    def iter_batches(name, paths):
        """(texts, languages) per row group of the corpus's files, in order; a code corpus also reads its language column."""
        code = isinstance(DATASETS[name], CodeCorpus)
        for path in paths:
            pf = pq.ParquetFile(path)
            for i in range(pf.num_row_groups):
                table = pf.read_row_group(i, columns=["text", "language"] if code else ["text"])
                texts = table.column("text").to_pylist()
                yield texts, table.column("language").to_pylist() if code else [None] * len(texts)

    # the training sample of each corpus (per language for code)
    composition = {name: {"chars": 0, "documents": 0} | ({"languages": {}} if isinstance(DATASETS[name], CodeCorpus) else {}) for name in datasets}

    def corpus_texts(name):
        """Train documents of a corpus in file order, each cropped to doc_cap characters, until its share is seen."""
        counts = composition[name]
        for texts, languages in iter_batches(name, list_raw_files(name, "train")):
            for doc, language in zip(texts, languages):
                doc = doc[:args.doc_cap]
                counts["chars"] += len(doc)
                counts["documents"] += 1
                if language is not None:
                    per_language = counts["languages"].setdefault(language, {"chars": 0, "documents": 0})
                    per_language["chars"] += len(doc)
                    per_language["documents"] += 1
                yield doc
                if counts["chars"] > share:
                    return
        raise FileNotFoundError(f"'{name}' train: the completed files hold {counts['chars']:,} characters, short of its share "
                                f"{share:,}: fetch more files, raising --max-files: `{fetch_command(name)}`")

    t0 = time.time()
    tokenizer = RustBPETokenizer.train_from_iterator((doc for name in datasets for doc in corpus_texts(name)), args.vocab_size)
    print(f"Training time: {time.time() - t0:.2f}s")
    tokenizer.save(get_tokenizer_dir())

    # sanity check: round-trip ASCII, numbers, contractions, punctuation and multi-byte unicode
    test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
    assert tokenizer.decode(tokenizer.encode(test_text)) == test_text

    print("Training sample composition:")
    for name, counts in composition.items():
        print(f"  {name}: {counts['chars']:,} chars, {counts['documents']:,} documents")
        for language, per_language in sorted(counts.get("languages", {}).items(), key=lambda kv: -kv[1]["chars"]):
            print(f"    {language:<24} {per_language['chars']:>15,} chars {per_language['documents']:>12,} documents")

    # Held-out categories: prose for a text corpus, code by Linguist language family; unlisted languages are other code
    CODE_CATEGORIES = {"Python": "Python", "C": "C/C++", "C++": "C/C++", "Java": "Java", "JavaScript": "JavaScript/TypeScript",
                       "TypeScript": "JavaScript/TypeScript", "Go": "Go", "Rust": "Rust"}
    CATEGORIES = ["general prose", *dict.fromkeys(CODE_CATEGORIES.values()), "other code"]
    totals = {}  # category -> chars, bytes, tokens over each corpus's first val file
    for name in datasets:
        code = isinstance(DATASETS[name], CodeCorpus)
        for texts, languages in iter_batches(name, require_raw_files(name, "val")[:1]):
            for text, language, ids in zip(texts, languages, tokenizer.encode(texts)):
                total = totals.setdefault(CODE_CATEGORIES.get(language, "other code") if code else "general prose", {"chars": 0, "bytes": 0, "tokens": 0})
                total["chars"] += len(text)
                total["bytes"] += len(text.encode("utf-8"))
                total["tokens"] += len(ids)
    heldout = {c: totals[c] | {"bytes_per_token": totals[c]["bytes"] / totals[c]["tokens"], "chars_per_token": totals[c]["chars"] / totals[c]["tokens"]}
               for c in CATEGORIES if c in totals}
    print("Held-out statistics (first val file of each corpus):")
    print(f"  {'category':<24} {'chars':>15} {'bytes':>15} {'tokens':>15} {'bytes/token':>12} {'chars/token':>12}")
    for category, s in heldout.items():
        print(f"  {category:<24} {s['chars']:>15,} {s['bytes']:>15,} {s['tokens']:>15,} {s['bytes_per_token']:>12.3f} {s['chars_per_token']:>12.3f}")

    stats_path = os.path.join(get_tokenizer_dir(), "tokenizer_stats.json")
    with open(stats_path, "w") as f:
        json.dump({"dataset": args.dataset, "max_chars": args.max_chars, "doc_cap": args.doc_cap, "vocab_size": args.vocab_size,
                   "fingerprint": tokenizer.fingerprint(), "composition": composition, "heldout": heldout}, f, indent=2)
    print(f"Saved tokenizer statistics to {stats_path}")
    print(f"Tokenizer fingerprint: {tokenizer.fingerprint()} (keys compiled data: python -m nanochat.data.pretrain.compile)")
