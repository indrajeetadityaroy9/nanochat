"""
Batched decoding of a CausalLM (model/modeling.py) by model/kernels/decode.cu: a request's prompt is prefilled (its
positions run as rows of one batched pass, with any other new requests' prompts), then captured CUDA graphs advance
every active slot a token per step (the greedy argmax or a seeded sample, optionally from its nucleus, within its
constraint's allowed tokens) until BOS, the token budget or the caller's stopping criteria; a freed slot takes the next
request (continuous batching). The decoder snapshots the model's weights, context, windows and rotary tables at
construction (a stage change needs a new decoder). fim_prompt builds FIM prompts; heal_tokens builds the healing
constraint.
"""

import bisect
import ctypes
import functools
import time
import weakref
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch

from corpus.tokenizer import FIM_BEGIN, FIM_END, FIM_HOLE
from inference import syntax_constraint
from inference.syntax_constraint import SyntaxConstraint
from model.config import VOCAB_SIZE
from model.modeling import CausalLM
from native import cuda_library

type Constraint = Reemit | SyntaxConstraint


def fim_prompt(tokenizer, prefix: str, suffix: str, *, mode: Literal["psm", "spm"]) -> list[int]:
    """The token ids of a fill-in-the-middle prompt for the text between prefix and suffix (strings, each encoded on its
    own), in the layout the packer trains (corpus/pack.py, loader.py; arXiv 2207.14255 §3, App. D):
        psm: BOS <fim_begin> prefix <fim_hole> suffix <fim_end>
        spm: BOS <fim_begin> <fim_hole> suffix <fim_end> prefix
    The model continues with the middle and ends it with BOS (NativeDecoder.generate stops there)."""
    special = tokenizer.special_tokens
    pre, suf, mid = special[FIM_BEGIN], special[FIM_HOLE], special[FIM_END]
    p, s = tokenizer.enc.encode_ordinary(prefix), tokenizer.enc.encode_ordinary(suffix)
    match mode:
        case "psm":
            return [tokenizer.bos_token_id, pre, *p, suf, *s, mid]
        case "spm":
            return [tokenizer.bos_token_id, pre, suf, *s, mid, *p]


@dataclass(frozen=True, kw_only=True)
class Request:
    """A continuation of input_ids (BOS and at least one more token) by at most max_new_tokens (at least 1) tokens.
    stopping_criteria: None or a predicate of the generated ids, true once the sequence is done. temperature 0: the
    greedy argmax (ties to the lowest id); > 0: a sample of softmax(logits / temperature), logits capped by
    ModelConfig.final_logit_softcapping if set, its draw at
    generated position i a function of (seed, i) only (decode.cu gumbel); top_p < 1 samples the fewest most likely
    tokens holding top_p of the probability (nucleus sampling, arXiv 1904.09751 §3.1; decode.cu k_nucleus).
    constraint: the tokens allowed at each step."""
    input_ids: list[int]
    max_new_tokens: int
    stopping_criteria: Callable[[list[int]], bool] | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    constraint: Constraint | None = None


@functools.cache
def prefix_index(tokenizer) -> tuple[list[bytes], np.ndarray]:
    """The mergeable tokens' bytes sorted, and their ids in that order: the tokens whose bytes start with a given string
    are one contiguous run of it (Reemit)."""
    ids = np.array(sorted(range(tokenizer.bos_token_id), key=tokenizer.token_bytes.__getitem__))
    return [tokenizer.token_bytes[i] for i in ids], ids


class Reemit:
    """heal_tokens's constraint: while bytes of the dropped token are owed, each token's bytes are a non-empty proper
    prefix of them or start with them (BOS excluded); once they are re-emitted, no restriction. (A healed Python
    prompt's SyntaxConstraint holds it: syntax_constraint.masks intersects its masks with this one's.)"""

    def __init__(self, tokenizer, owed: bytes):
        self.tokenizer, self.owed = tokenizer, owed

    def mask(self) -> np.ndarray | None:
        if not self.owed:
            return None
        keys, ids = prefix_index(self.tokenizer)
        lo = bisect.bisect_left(keys, self.owed)
        hi = bisect.bisect_right(keys, self.owed, lo=lo, key=lambda k: k[:len(self.owed)])
        allowed = np.zeros(self.tokenizer.vocab_size, dtype=bool)
        allowed[ids[lo:hi]] = True
        for k in range(1, len(self.owed)):  # the owed bytes' proper prefixes that are tokens
            i = bisect.bisect_left(keys, self.owed[:k])
            if i < len(keys) and keys[i] == self.owed[:k]:
                allowed[ids[i]] = True
        # the decoder's mask layout: bit t % 32 of word t // 32 set iff token t is allowed
        return np.packbits(allowed, bitorder="little").view("<u4")

    def push(self, token: int) -> None:
        b = self.tokenizer.token_bytes[token]
        self.owed = b"" if b.startswith(self.owed) else self.owed[len(b):]


def heal_tokens(tokenizer, input_ids: list[int]) -> tuple[list[int], Reemit, bytes]:
    """Token healing (Guidance, "The Art of Prompt Design: Prompt Boundaries and Token Healing", 2023), constrained
    until the dropped token's bytes are re-emitted (TokEval, arXiv 2608.18062 App. C; Guidance constrains one token,
    which forbids splitting those bytes over two): the prompt without its last token, its Reemit constraint, and the
    dropped bytes. input_ids: BOS and at least two more tokens."""
    dropped = tokenizer.token_bytes[input_ids[-1]]
    return input_ids[:-1], Reemit(tokenizer, dropped), dropped


def step_masks(out: np.ndarray, constrained: list[tuple[int, Constraint, bool]]) -> tuple[list[int], list[int]]:
    """Writes each (slot, constraint, eager)'s next mask into the slot's row of out (the decoder's mask rows): the
    syntax constraints' by one syntax_constraint.masks call (in parallel, without the GIL, straight into the rows), a
    BOS candidate decided there (syntax_constraint.ends) only if eager, else by the caller once BOS is chosen. The
    slots whose rows restrict their choice, and those whose rows allow no token."""
    syntax = [(b, c, eager) for b, c, eager in constrained if isinstance(c, SyntaxConstraint)]
    masked, empty = [], []
    if syntax:
        verdicts = syntax_constraint.masks([c for _, c, _ in syntax], out, [b for b, _, _ in syntax])
        masked += [b for b, _, _ in syntax]
        empty += [b for (b, _, _), ok in zip(syntax, verdicts, strict=True) if not ok]
        candidates = [(b, c) for b, c, eager in syntax if eager and out[b, c.bos // 32] >> (c.bos % 32) & 1]
        if candidates:
            for (b, c), ok in zip(candidates, syntax_constraint.ends([c for _, c in candidates]), strict=True):
                if not ok:
                    out[b, c.bos // 32] &= ~np.uint32(1 << (c.bos % 32))
                    if not out[b].any():
                        empty.append(b)
    for b, c, _ in constrained:
        if isinstance(c, SyntaxConstraint):
            continue
        mask = c.mask()
        if mask is None:
            continue
        out[b] = mask
        masked.append(b)
        if not mask.any():
            empty.append(b)
    return masked, empty


# the prompt store's page in positions: a stored prompt takes whole pages anywhere in the store (vLLM's PagedAttention,
# arXiv 2309.06180, pages a KV cache likewise; 64 keeps a code prompt to a few copies, at most 63 positions unused)
PAGE = 64


@dataclass
class Stored:
    """A prompt in the store: its pages, and the run (NativeDecoder.runs, generate calls) that last used it."""
    pages: list[int]
    used: int


@dataclass
class Slot:
    """A request in a decoder slot: its index, the request, when it took the slot and when its first generated token
    reached the host (time.perf_counter), the generated tokens."""
    index: int
    request: Request
    start: float
    first: float | None = None
    out: list = field(default_factory=list)


class NativeDecoder:
    """Decoding by decode.cu of up to `batch` sequences at once (slots); a sequence's tokens do not depend on the batch
    (decode.cu computes each slot and each prefill row alone, in a fixed order) and a sample's noise depends only on its
    request's seed and the generated position. The KV cache is e4m3 with a scale per position and head (decode.cu), so
    a sequence's tokens can differ from the bf16 forward's where two logits nearly tie.
    batch: dec_create's slot count, at most decode.cu's MAX_BATCH, bounded by free device memory.
    max_length: a slot's cache capacity in positions, at most the model's max_position_embeddings: the longest
    len(input_ids) + max_new_tokens - 1 the caller will run (a slot's cache is 2 num_hidden_layers max_length
    (hidden_size + 4 num_attention_heads) bytes, so a smaller max_length fits more slots; the tokens do not depend on
    it).
    prompt_store: positions requested for decode.cu's prompt store (at most the device memory left after the slots, 2
    num_hidden_layers (hidden_size + 4 num_attention_heads) bytes a position). A prompt's K and V at all but its last
    position are kept once prefilled; a later request with the same prompt copies them instead of prefilling, its
    tokens unchanged (K and V depend on the prompt alone). To free pages, prompts unused in this run and the previous
    one are dropped, least recently used first, so a search's rounds keep the prompts they reuse and a too-small store
    keeps a fixed part rather than cycling.
    memory_fraction: the most device memory the decoder takes (slots, prefill rows, graphs and prompt store), as a
    fraction of the device's total, capped by its free memory (vLLM's gpu_memory_utilization, but counting this
    decoder alone); None: all free memory."""

    @torch.no_grad()
    def __init__(self, model: CausalLM, *, bos_token_id: int, max_length: int, prompt_store: int,
                 memory_fraction: float | None = None):
        lib = cuda_library("decode")
        lib.dec_create.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        lib.dec_create.restype = ctypes.c_void_p
        lib.dec_batch.argtypes = [ctypes.c_void_p]
        lib.dec_set_prompt.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int32, ctypes.c_int32, ctypes.c_float,
                                       ctypes.c_float, ctypes.c_ulonglong]
        lib.dec_prefill_rows.argtypes = [ctypes.c_void_p]
        lib.dec_prefill.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        for copy in (lib.dec_save_prompt, lib.dec_load_prompt):
            copy.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.dec_store.argtypes = [ctypes.c_void_p]
        lib.dec_masks.argtypes = [ctypes.c_void_p]
        lib.dec_masks.restype = ctypes.POINTER(ctypes.c_uint32)
        lib.dec_set_active.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.dec_launch.argtypes = [ctypes.c_void_p]
        lib.dec_launch_body.argtypes = [ctypes.c_void_p]
        lib.dec_launch_head.argtypes = [ctypes.c_void_p]
        lib.dec_destroy.argtypes = [ctypes.c_void_p]
        lib.dec_tokens.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.dec_take_alt.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib = lib
        cfg = model.config
        L, C, T = cfg.num_hidden_layers, cfg.hidden_size, cfg.max_position_embeddings
        if not 1 < max_length <= T:
            raise ValueError(f"max_length {max_length} outside (1, max_position_embeddings {T}]")
        self.max_length = max_length
        # bf16 copies of the matrices and tables (the forward's rounding at use); the fp32 embedding table (the compiled
        # forward's x0 is the fp32 row's norm, decode.cu k_norm); fp32 q/k norm weights, gate weights and rotary table,
        # as modeling.py uses them
        weights = [model.embed_tokens.weight.clone(), model.cos, model.sin, model.lm_head.weight.bfloat16()]
        for i, layer in enumerate(model.layers):
            attn, mlp = layer.self_attn, layer.mlp
            ve = str(i) in model.value_embeds
            weights += [torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight]).bfloat16(),
                        attn.o_proj.weight.bfloat16(), mlp.up_proj.weight.bfloat16(), mlp.down_proj.weight.bfloat16(),
                        model.value_embeds[str(i)].weight.bfloat16() if ve else None,
                        attn.ve_gate.weight.clone() if ve else None, attn.q_norm_weight.clone(),
                        attn.k_norm_weight.clone()]
        self.weights = weights  # referenced while the decoder reads them through pointers
        # per layer (resid, x0), fp32 as the compiled forward computes the mix (decode.cu k_norm); then the output-logit
        # cap (0: none)
        lambdas = torch.stack([model.resid_lambdas, model.x0_lambdas], 1).flatten().cpu()
        scal = np.append(lambdas.numpy(), np.float32(cfg.final_logit_softcapping or 0.0))
        # F: up_proj is (F, C); the lazy row: BOS, a syntax constraint's end, confirmed once chosen (generate); the
        # memory budget in MiB (0: none)
        device = model.embed_tokens.weight.device
        budget = 0 if memory_fraction is None else max(1, int(
            memory_fraction * torch.cuda.get_device_properties(device).total_memory) >> 20)
        config = np.array([L, C, model.layers[0].mlp.up_proj.weight.size(0), VOCAB_SIZE, max_length, T, prompt_store,
                           bos_token_id, budget], dtype=np.int32)
        # the prompt store: {prompt: its pages}, the free pages (once the decoder exists), the runs started
        self.stored: dict[tuple[int, ...], Stored] = {}
        self.runs = 0
        self.timings: dict[int, tuple[float, float, float]] = {}  # generate's request timestamps
        self.bos = bos_token_id
        # dec_destroy (the finalizer) releases every native resource; the buffers it reads are this object's tensors
        self.stream = torch.cuda.Stream(model.embed_tokens.weight.device)
        ptrs = np.array([0 if t is None else t.data_ptr() for t in weights], dtype=np.uint64)
        window = np.array(model.window_sizes, dtype=np.int32)
        torch.cuda.synchronize()  # the weight copies above are on the current stream; the decoder uses its own
        self.handle = lib.dec_create(config.ctypes.data, ptrs.ctypes.data, scal.ctypes.data, window.ctypes.data,
                                     self.stream.cuda_stream)
        weakref.finalize(self, lib.dec_destroy, self.handle)
        self.batch = lib.dec_batch(self.handle)
        self.tokens = np.zeros(2 * self.batch, dtype=np.int32)  # dec_tokens: the choices, then their alternatives
        self.free = list(range(lib.dec_store(self.handle) // PAGE))
        # the pinned host masks decode.cu copies to the masked slots (dec_masks), a Constraint's mask per slot
        self.masks = np.ctypeslib.as_array(lib.dec_masks(self.handle),
                                           shape=(self.batch, VOCAB_SIZE // np.iinfo(np.uint32).bits))
        self.prefill_rows = lib.dec_prefill_rows(self.handle)
        self.pending: list[tuple[int, list[int]]] = []  # started slots whose prompts await prefill

    def start(self, b: int, index: int, request: Request) -> Slot:
        """Start request `index` in slot b at its prompt's last token: the positions before it loaded from the prompt
        store, else prefilled (NativeDecoder.prefill) before the next step."""
        n = len(request.input_ids) - 1  # the positions before the prompt's last token
        stored = self.stored.get(tuple(request.input_ids))
        if stored is not None:
            stored.used = self.runs
            for i, page in enumerate(stored.pages):
                self.lib.dec_load_prompt(self.handle, b, i * PAGE, page * PAGE, min(PAGE, n - i * PAGE))
        else:
            self.pending.append((b, request.input_ids))
        self.lib.dec_set_prompt(self.handle, b, request.input_ids[-1], n, request.temperature, request.top_p,
                                request.seed)
        return Slot(index, request, start=time.perf_counter())

    def prefill(self) -> None:
        """The pending slots' prompts but their last tokens through decode.cu's prefill (dec_prefill: every position a
        row, prefill_rows rows a launch, a prompt's positions in order), each prompt then kept in the store (save)."""
        if not self.pending:
            return
        rows = np.concatenate([np.array([ids[:-1], range(len(ids) - 1), [b] * (len(ids) - 1)], dtype=np.int32)
                               for b, ids in self.pending], axis=1)
        for i in range(0, rows.shape[1], self.prefill_rows):
            chunk = np.ascontiguousarray(rows[:, i:i + self.prefill_rows])
            self.lib.dec_prefill(self.handle, chunk.shape[1], chunk.ctypes.data)
        for b, ids in self.pending:
            self.save(b, ids)
        self.pending = []

    def save(self, b: int, input_ids: list[int]) -> None:
        """Keep slot b's prompt K and V (all but the last position, written by the launched prefill) in free pages,
        dropping prompts unused in this run and the previous one (least recently used first) if pages are short; not
        kept if still short, or if a request with the same prompt kept it meanwhile."""
        key, n = tuple(input_ids), len(input_ids) - 1
        need = -(-n // PAGE)
        if key in self.stored:
            return
        for old in sorted((k for k, s in self.stored.items() if s.used < self.runs - 1),
                          key=lambda k: self.stored[k].used):
            if len(self.free) >= need:
                break
            self.free += self.stored.pop(old).pages
        if len(self.free) < need:
            return
        pages = [self.free.pop() for _ in range(need)]
        for i, page in enumerate(pages):
            self.lib.dec_save_prompt(self.handle, b, i * PAGE, page * PAGE, min(PAGE, n - i * PAGE))
        self.stored[key] = Stored(pages, self.runs)

    def confirm(self, slots: list[Slot | None], chosen: list[int], *, eager: set[int]) -> None:
        """Among chosen (syntax-constrained slots, their step's tokens in self.tokens), a chosen BOS candidate that does
        not end the program (syntax_constraint.ends; the eager slots' decided before the step) gives way to the step's
        alternative (dec_take_alt: the same choice as BOS masked). Raises ValueError when such a slot's mask allows no
        other token."""
        B, bos = self.batch, self.bos
        lazy = [b for b in chosen if self.tokens[b] == bos and b not in eager]
        if lazy:
            for b, ok in zip(lazy, syntax_constraint.ends([slots[b].request.constraint for b in lazy]), strict=True):
                if ok:
                    continue
                if self.tokens[B + b] < 0:
                    raise ValueError(f"request {slots[b].index}: its constraint allows no token at position "
                                     f"{len(slots[b].out)}")
                self.tokens[b] = self.tokens[B + b]
                self.lib.dec_take_alt(self.handle, b)

    @torch.no_grad()
    def generate(self, requests: Iterable[Request]) -> Iterator[tuple[int, list[int]]]:
        """Continuations of requests, consumed lazily. Yields (request index, generated ids) in completion order: at BOS
        (excluded), after max_new_tokens, or once stopping_criteria holds (that token included). The prompts of the
        requests that took a slot since the last step are prefilled before the next. A step's stopping criteria are
        evaluated while the next step runs; a stopped slot's extra token is dropped. A step with a sampled or
        constrained slot launches decode.cu's body, writes the constraints' masks meanwhile (step_masks: a mask
        follows from the previous token; the syntax constraints' masks in one parallel batch), then its head, the
        choice by mask and temperature; other steps launch the whole greedy step. A syntax constraint's ends are
        confirmed only once chosen (confirm): a rejected BOS gives way to the step's best other allowed token
        (decode.cu's alternative: the same choice as BOS masked; a nucleus-sampled slot's BOS is decided before the
        step).
        Raises ValueError when len(input_ids) + max_new_tokens - 1 > max_length or a constraint allows no token (not
        even BOS).
        timings: {request index: (start, first, end)} of this call's requests yielded so far (time.perf_counter
        seconds, read on the host once each step's tokens arrive, no further device synchronization): when the request
        took its slot, when its first generated token arrived (end for a request without one), and the step that
        finished it."""
        lib, handle, B, bos, max_length = self.lib, self.handle, self.batch, self.bos, self.max_length
        self.runs += 1
        self.timings = {}
        queue, slots = enumerate(requests), [None] * B
        stopping = []  # slots whose last token awaits its stopping criteria
        # row 0: the active slots, row 1: the masked slots; dec_set_active takes each row as 64-bit words, slot b at
        # bit b % 64 of word b // 64
        word = np.iinfo(np.uint64).bits
        flags = np.zeros((2, -(-B // word) * word), dtype=bool)

        def set_active():
            words = np.packbits(flags, axis=1, bitorder="little").view("<u8")
            lib.dec_set_active(handle, words.ctypes.data)

        def fill(b):
            """Slot b takes the next request (NativeDecoder.start), or idles once none is left."""
            for i, request in queue:
                if request.max_new_tokens < 1:  # its finish test would never fire: decoding past max_length
                    raise ValueError(f"request {i}: max_new_tokens {request.max_new_tokens} < 1")
                if (end := len(request.input_ids) + request.max_new_tokens - 1) > max_length:
                    raise ValueError(f"request {i}: len(input_ids) {len(request.input_ids)} + max_new_tokens "
                                     f"{request.max_new_tokens} - 1 = {end} > max_length {max_length}")
                slots[b] = self.start(b, i, request)
                flags[0, b] = True
                return
            slots[b] = None
            flags[0, b] = False

        def nucleus(request: Request) -> bool:
            return request.temperature > 0 and request.top_p < 1

        for b in range(B):
            fill(b)
        set_active()
        while flags[0].any():
            self.prefill()
            active = [b for b, s in enumerate(slots) if s is not None]
            choosing = any(slots[b].request.temperature > 0 or slots[b].request.constraint is not None
                           for b in active)
            if choosing:
                lib.dec_launch_body(handle)
            else:
                lib.dec_launch(handle)
            finished = sorted(b for b in stopping if slots[b].request.stopping_criteria(slots[b].out))  # meanwhile
            stopping = []
            if choosing:
                # a stopped slot's next token is dropped
                masked, empty = step_masks(self.masks, [(b, r.constraint, nucleus(r)) for b in active
                                                        if (r := slots[b].request).constraint is not None
                                                        and b not in finished])
                for b in empty:
                    raise ValueError(f"request {slots[b].index}: its constraint allows no token at position "
                                     f"{len(slots[b].out)}")
                flags[1, masked] = True
                set_active()
                flags[1] = False
                lib.dec_launch_head(handle)
            lib.dec_tokens(handle, self.tokens.ctypes.data)
            if choosing:
                syntax = [b for b in active if b not in finished
                          and isinstance(slots[b].request.constraint, SyntaxConstraint)]
                self.confirm(slots, syntax, eager={b for b in syntax if nucleus(slots[b].request)})
            now = time.perf_counter()
            for b, s in enumerate(slots):
                if s is None or b in finished:
                    continue
                token = int(self.tokens[b])
                constraint = s.request.constraint
                if token == bos:
                    finished.append(b)
                    continue
                if not s.out:
                    s.first = now
                s.out.append(token)
                if constraint is not None:
                    constraint.push(token)
                if len(s.out) == s.request.max_new_tokens:
                    finished.append(b)
                elif s.request.stopping_criteria is not None:
                    stopping.append(b)
            if finished:
                for b in finished:
                    done = slots[b]
                    fill(b)
                    self.timings[done.index] = (done.start, now if done.first is None else done.first, now)
                    yield done.index, done.out
                set_active()  # each choosing step sets its masks anew
