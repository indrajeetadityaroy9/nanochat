"""
Test the SFT dataloader: BOS-aligned best-fit packing with padding, target masking, over-long
conversations, rank sharding and pass tracking. Hermetic: a tiny in-process tokenizer and toy tasks.

python -m pytest tests/test_data_sft.py -v
"""

import pytest
import torch
from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS
from nanochat.data.task import Task
from nanochat.data.posttrain.sft import SFTLoader


@pytest.fixture(scope="module")
def tokenizer():
    # no digits in the corpus: every digit encodes to its own token, so "question 007" and
    # "question 123" render to the same length
    corpus = ["the user asks a question and the assistant answers with an answer"] * 8
    return RustBPETokenizer.train_from_iterator(iter(corpus), 256 + len(SPECIAL_TOKENS) + 20)


class ToyChat(Task):
    """Conversation i: the user asks question i, the assistant answers with reply_words[i] words.
    The questions of long_questions are longer than any row, so truncation leaves them no assistant token."""

    def __init__(self, reply_words, long_questions=(), **kwargs):
        super().__init__(**kwargs)
        self.reply_words = reply_words
        self.long_questions = set(long_questions)

    def num_examples(self):
        return len(self.reply_words)

    def get_example(self, index):
        question = f"question {index:03d}"
        if index in self.long_questions:
            question += " and" * 200
        return {"messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": " ".join(["answer"] * self.reply_words[index])},
        ]}


def make_loader(dataset, tokenizer, **kwargs):
    kwargs = {"batch_size": 4, "max_seq_len": 64, "device": "cpu", "rank": 0, "world_size": 1, **kwargs}
    return SFTLoader(dataset, tokenizer, **kwargs)


def one_pass(loader, max_batches=100):
    batches = []
    while not loader.last_step:
        assert len(batches) < max_batches, "the loader never completed the pass"
        batches.append(next(loader))
    return batches


def unpack(tokenizer, renders, inputs_row, targets_row):
    """Split a row into the whole conversations packed in it (by index) and the padding after them."""
    bos = tokenizer.get_bos_token_id()
    last = targets_row[-1] # the last token of a row only appears as a target: padding if ignored
    row = inputs_row + [last if last >= 0 else bos]
    convs, pos = [], 0
    while pos < len(row):
        match = [i for i, (ids, _) in enumerate(renders) if row[pos:pos + len(ids)] == ids]
        if not match:
            break
        convs.append(match[0])
        pos += len(renders[match[0]][0])
    assert row[pos:] == [bos] * (len(row) - pos), "a row must hold whole conversations, then BOS padding"
    return row, convs, pos


def rows_of(batches):
    for inputs, targets in batches:
        yield from zip(inputs.tolist(), targets.tolist())


def test_rows_start_with_bos_and_hold_whole_conversations(tokenizer):
    dataset = ToyChat([(i * 7) % 13 + 1 for i in range(40)])
    renders = [tokenizer.render_conversation(dataset[i]) for i in range(len(dataset))]
    batches = one_pass(make_loader(dataset, tokenizer))
    for inputs, targets in batches:
        assert inputs.shape == targets.shape == (4, 64)
        assert inputs.dtype == torch.int32 and targets.dtype == torch.int64
    for inputs_row, targets_row in rows_of(batches):
        assert inputs_row[0] == tokenizer.get_bos_token_id()
        _, convs, _ = unpack(tokenizer, renders, inputs_row, targets_row)
        assert convs, "every row starts with a whole conversation"


def test_targets_follow_render_mask_and_padding_is_ignored(tokenizer):
    dataset = ToyChat([(i * 7) % 13 + 1 for i in range(40)])
    renders = [tokenizer.render_conversation(dataset[i]) for i in range(len(dataset))]
    padded_rows = 0
    for inputs_row, targets_row in rows_of(one_pass(make_loader(dataset, tokenizer))):
        row, convs, content_len = unpack(tokenizer, renders, inputs_row, targets_row)
        mask = [m for i in convs for m in renders[i][1]] + [0] * (len(row) - content_len)
        assert targets_row == [row[j + 1] if mask[j + 1] else -1 for j in range(len(inputs_row))]
        if content_len < len(row):
            padded_rows += 1
            assert all(t == -1 for t in targets_row[content_len - 1:])
    assert padded_rows > 0, "the data must exercise padding"


def test_overlong_conversations_are_truncated_to_one_row(tokenizer):
    # both conversations render far longer than a row: each must still be emitted, cut to the row
    dataset = ToyChat([100, 80])
    max_seq_len = 32
    batches = one_pass(make_loader(dataset, tokenizer, batch_size=1, max_seq_len=max_seq_len))
    full = [tokenizer.render_conversation(dataset[i], max_tokens=10_000) for i in range(len(dataset))]
    emitted = []
    for inputs_row, targets_row in rows_of(batches):
        match = [i for i, (ids, _) in enumerate(full) if inputs_row == ids[:max_seq_len]]
        assert len(match) == 1
        ids, mask = full[match[0]]
        assert len(ids) > max_seq_len + 1
        assert targets_row == [ids[j + 1] if mask[j + 1] else -1 for j in range(max_seq_len)]
        emitted.append(match[0])
    assert sorted(emitted) == [0, 1]


def test_conversations_with_nothing_to_train_on_are_skipped(tokenizer):
    # questions 2 and 5 fill a row by themselves: truncated, they keep no assistant token
    dataset = ToyChat([3] * 8, long_questions={2, 5})
    renders = [tokenizer.render_conversation(dataset[i]) for i in range(len(dataset))]
    loader = make_loader(dataset, tokenizer, batch_size=1, max_seq_len=32)
    emitted = set()
    for inputs_row, targets_row in rows_of(one_pass(loader)):
        assert any(t != -1 for t in targets_row), "every row has targets to train on"
        emitted |= set(unpack(tokenizer, renders, inputs_row, targets_row)[1])
    assert emitted == {0, 1, 3, 4, 6, 7}
    # the skipped conversations still count toward the pass, which ends exactly at its end
    assert loader.progress == 1.0 and loader.epoch == 1
    with pytest.raises(ValueError, match="to train on"):
        next(make_loader(ToyChat([3] * 4, long_questions=range(4)), tokenizer, max_seq_len=32))


def test_ranks_see_disjoint_conversations(tokenizer):
    dataset = ToyChat([3] * 24)
    renders = [tokenizer.render_conversation(dataset[i]) for i in range(len(dataset))]
    assert len({len(ids) for ids, _ in renders}) == 1 # equal lengths: packed in fetch order
    seen = []
    for rank in range(2):
        loader = make_loader(dataset, tokenizer, batch_size=2, rank=rank, world_size=2)
        seen.append({i for row in rows_of(one_pass(loader)) for i in unpack(tokenizer, renders, *row)[1]})
    assert seen[0] and seen[1]
    assert seen[0].isdisjoint(seen[1])
    assert seen[0] | seen[1] == set(range(len(dataset)))


def test_pass_tracking(tokenizer):
    dataset = ToyChat([3] * 18)
    conv_len = len(tokenizer.render_conversation(dataset[0])[0])
    # rows of exactly 3 conversations, batches of 2 rows: one pass is 3 batches
    loader = make_loader(dataset, tokenizer, batch_size=2, max_seq_len=3 * conv_len - 1)
    for step in range(1, 4):
        assert not loader.last_step
        next(loader)
        assert loader.progress == pytest.approx(step / 3)
        assert loader.epoch == 1
    assert loader.last_step
    next(loader) # past the end of the pass, rows wrap into the next one and progress stays at 1
    assert loader.epoch == 2 and loader.progress == 1.0
    # with num_batches, the run ends after that many batches
    loader = make_loader(dataset, tokenizer, batch_size=2, max_seq_len=3 * conv_len - 1, num_batches=2)
    next(loader)
    assert not loader.last_step and loader.progress == pytest.approx(0.5)
    next(loader)
    assert loader.last_step and loader.progress == pytest.approx(1.0)
    next(loader)
    assert loader.progress == 1.0
