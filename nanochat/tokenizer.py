"""
BPE Tokenizer in the style of GPT-4: train with rustbpe, inference with tiktoken.

Train on the pretraining data, a corpus or a mixture (writes <base_dir>/tokenizer/tokenizer.pkl and tokenizer_stats.json):
python -m nanochat.tokenizer
python -m nanochat.tokenizer --dataset=stack_edu:0.3,smol_pdfedu_dclm_fwedu:0.7
"""

import os
import copy
import time
import pickle
import argparse
import hashlib

import torch
import rustbpe
import tiktoken

from nanochat.common import get_base_dir

SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>", # user messages
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>",
    "<|python_start|>", # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>", # python REPL outputs back to assistant
    "<|output_end|>",
]

# NOTE: this split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
# I did this because I didn't want to "waste" too many tokens on numbers for smaller vocab sizes.
# I verified that 2 is the sweet spot for vocab size of 32K. 1 is a bit worse, 3 was worse still.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# Tokenizer based on rustbpe + tiktoken combo

class RustBPETokenizer:
    """Light wrapper around tiktoken (for efficient inference) but train with rustbpe"""

    def __init__(self, enc, bos_token):
        self.enc = enc
        self.special_ids = {name: enc.encode_single_token(name) for name in enc.special_tokens_set}
        self.bos_token_id = self.special_ids[bos_token]

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # 1) train using rustbpe
        tokenizer = rustbpe.Tokenizer()
        # the special tokens are inserted later in __init__, we don't train them here
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        # 2) construct the associated tiktoken encoding for inference
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks, # dict[bytes, int] (token bytes -> merge priority rank)
            special_tokens=special_tokens, # dict[str, int] (special token name -> token id)
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    def get_vocab_size(self):
        return self.enc.n_vocab

    def encode_special(self, text):
        return self.special_ids[text]

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        # text can be either a string or a list of strings

        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id) # TODO: slightly inefficient here? :( hmm
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id) # TODO: same
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids

    def decode(self, ids):
        return self.enc.decode(ids)

    def get_token_bytes(self, device="cpu"):
        """
        Byte length of every token id, as an int32 tensor of shape (vocab_size,), used for
        vocab-size-invariant bits per byte. Special tokens count 0 bytes. Uses the raw token
        bytes: decoding to str first corrupts tokens that are not valid standalone UTF-8.
        """
        token_bytes = self.enc.decode_tokens_bytes(list(range(self.get_vocab_size())))
        lengths = torch.tensor([len(b) for b in token_bytes], dtype=torch.int32)
        lengths[list(self.special_ids.values())] = 0
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

    def render_conversation(self, conversation, max_tokens=2048):
        """
        Tokenize a single Chat conversation (which we call a "doc" or "document" here).
        Returns:
        - ids: list[int] is a list of token ids of this rendered conversation
        - mask: list[int] of same length, mask = 1 for tokens that the Assistant is expected to train on.
        """
        # ids, masks that we will return and a helper function to help build them up.
        ids, mask = [], []
        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # sometimes the first message is a system message...
        # => just merge it with the second (user) message
        if conversation["messages"][0]["role"] == "system":
            # some conversation surgery is necessary here for now...
            conversation = copy.deepcopy(conversation) # avoid mutating the original
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", "System message must be followed by a user message"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        # fetch all the special tokens we need
        bos = self.get_bos_token_id()
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

        # now we can tokenize the conversation
        add_tokens(bos, 0)
        for i, message in enumerate(messages):

            # some sanity checking here around assumptions, to prevent footguns
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, f"Message {i} is from {message['role']} but should be from {must_be_from}"

            # content can be either a simple string or a list of parts (e.g. containing tool calls)
            content = message["content"]

            if message["role"] == "user":
                assert isinstance(content, str), "User messages are simply expected to be strings"
                value_ids = self.encode(content)
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    # simple string => simply add the tokens
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            # string part => simply add the tokens
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            # python tool call => add the tokens inside <|python_start|> and <|python_end|>
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            # python output => add the tokens inside <|output_start|> and <|output_end|>
                            # none of these tokens are supervised because the tokens come from Python at test time
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                add_tokens(assistant_end, 1)

        # truncate to max_tokens tokens MAX (helps prevent OOMs)
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def render_for_completion(self, conversation):
        """
        Used during Reinforcement Learning. In that setting, we want to
        render the conversation priming the Assistant for a completion.
        Unlike the Chat SFT case, we don't need to return the mask.
        """
        # We have some surgery to do: we need to pop the last message (of the Assistant)
        conversation = copy.deepcopy(conversation) # avoid mutating the original
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", "Last message must be from the Assistant"
        messages.pop() # remove the last message (of the Assistant) inplace

        # Now tokenize the conversation
        ids, mask = self.render_conversation(conversation)

        # Finally, to prime the Assistant for a completion, append the Assistant start token
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids

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
    from nanochat.data.pretrain.sources import (DATASETS, DEFAULT_DATASET, CodeCorpus, parse_mixture, mixture_without,
                                                list_raw_files, require_raw_files, read_training_documents, fetch_command)

    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on the pretraining data")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help=f"pretraining corpus or mixture, e.g. stack_edu:0.3,smol_pdfedu_dclm_fwedu:0.7 (default: {DEFAULT_DATASET})")
    parser.add_argument("--max-chars", type=int, default=2_000_000_000, help="Maximum characters to train on, split across the mixture by weight (default: 2B)")
    parser.add_argument("--doc-cap", type=int, default=10_000, help="Maximum characters per document (default: 10,000)")
    parser.add_argument("--vocab-size", type=int, default=32768, help="Vocabulary size (default: 32768 = 2^15)")
    args = parser.parse_args()
    print(f"dataset: {args.dataset} | max_chars: {args.max_chars:,} | doc_cap: {args.doc_cap:,} | vocab_size: {args.vocab_size:,}")
    mixture = parse_mixture(args.dataset)

    def iter_batches(name, paths):
        """(texts, languages) per row group of a corpus's files, in order, of the documents training reads (as compile
        selects them: a corpus mixed with one it overlaps skips the documents it shares with it); code corpora also read
        their language column."""
        code = isinstance(DATASETS[name], CodeCorpus)
        without = mixture_without(name, [n for n, _ in mixture])
        for path in paths:
            for table, _ in read_training_documents(name, path, ["text", "language"] if code else ["text"], without):
                texts = table.column("text").to_pylist()
                yield texts, table.column("language").to_pylist() if code else [None] * len(texts)

    composition = {}  # name -> characters and documents of the training sample (per language for code corpora)

    def component_documents(name, weight):
        """Train documents of one component in file order, each cropped to doc_cap characters, until
        weight * max_chars characters have been seen, tallied into composition."""
        code = isinstance(DATASETS[name], CodeCorpus)
        budget = weight * args.max_chars
        tally = composition[name] = {"weight": weight, "chars": 0, "documents": 0} | ({"languages": {}} if code else {})
        for texts, languages in iter_batches(name, list_raw_files(name, "train")):
            for doc, language in zip(texts, languages):
                doc = doc[:args.doc_cap]
                tally["chars"] += len(doc)
                tally["documents"] += 1
                if code:
                    per_language = tally["languages"].setdefault(language, {"chars": 0, "documents": 0})
                    per_language["chars"] += len(doc)
                    per_language["documents"] += 1
                yield doc
                if tally["chars"] > budget:
                    return
        raise FileNotFoundError(f"'{name}' train: the completed files hold {tally['chars']:,} characters, short of its "
                                f"{budget:,.0f}: fetch more files, raising --max-files: `{fetch_command(name)}`")

    def text_iterator():
        """The training sample: each mixture component's share of max_chars, one component after another."""
        for name, weight in mixture:
            yield from component_documents(name, weight)

    t0 = time.time()
    tokenizer = RustBPETokenizer.train_from_iterator(text_iterator(), args.vocab_size)
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
    for name, tally in composition.items():
        print(f"  {name} (weight {tally['weight']:.3f}): {tally['chars']:,} chars, {tally['documents']:,} documents")
        for language, counts in sorted(tally.get("languages", {}).items(), key=lambda kv: -kv[1]["chars"]):
            print(f"    {language:<24} {counts['chars']:>15,} chars {counts['documents']:>12,} documents")

    # Held-out categories: prose from the general-domain corpora, code by Linguist language family; unlisted languages are other code
    CODE_CATEGORIES = {"Python": "Python", "C": "C/C++", "C++": "C/C++", "Java": "Java", "JavaScript": "JavaScript/TypeScript",
                       "TypeScript": "JavaScript/TypeScript", "Go": "Go", "Rust": "Rust"}
    CATEGORIES = ["general prose", *dict.fromkeys(CODE_CATEGORIES.values()), "other code"]
    totals = {}  # category -> chars, bytes, tokens over the first val file of every component
    for name, _ in mixture:
        code = isinstance(DATASETS[name], CodeCorpus)
        for texts, languages in iter_batches(name, require_raw_files(name, "val")[:1]):
            for text, language, ids in zip(texts, languages, tokenizer.encode(texts)):
                total = totals.setdefault(CODE_CATEGORIES.get(language, "other code") if code else "general prose", {"chars": 0, "bytes": 0, "tokens": 0})
                total["chars"] += len(text)
                total["bytes"] += len(text.encode("utf-8"))
                total["tokens"] += len(ids)
    heldout = {c: totals[c] | {"bytes_per_token": totals[c]["bytes"] / totals[c]["tokens"], "chars_per_token": totals[c]["chars"] / totals[c]["tokens"]}
               for c in CATEGORIES if c in totals}
    print("Held-out statistics (first val file of each component):")
    print(f"  {'category':<24} {'chars':>15} {'bytes':>15} {'tokens':>15} {'bytes/token':>12} {'chars/token':>12}")
    for category, s in heldout.items():
        print(f"  {category:<24} {s['chars']:>15,} {s['bytes']:>15,} {s['tokens']:>15,} {s['bytes_per_token']:>12.3f} {s['chars_per_token']:>12.3f}")

    stats_path = os.path.join(get_tokenizer_dir(), "tokenizer_stats.json")
    with open(stats_path, "w") as f:
        json.dump({"dataset": args.dataset, "max_chars": args.max_chars, "doc_cap": args.doc_cap, "vocab_size": args.vocab_size,
                   "fingerprint": tokenizer.fingerprint(), "composition": composition, "heldout": heldout}, f, indent=2)
    print(f"Saved tokenizer statistics to {stats_path}")
    print(f"Tokenizer fingerprint: {tokenizer.fingerprint()} (keys compiled data: python -m nanochat.data.pretrain.compile)")
