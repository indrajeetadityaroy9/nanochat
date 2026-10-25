"""
Batched decoding of a CausalLM (model/modeling.py) by model/kernels/decode.cu: captured CUDA graphs advance every
active slot a token per step, through its prompt and then its generation (the greedy argmax or a seeded sample,
optionally from its nucleus, within its constraint's allowed tokens) until BOS, the token budget or the caller's
stopping criteria; a freed slot
takes the next request (continuous batching), so one slot's prompt runs in the same steps as the others' generations.
The decoder snapshots the model's weights, context, windows and rotary tables at construction (a stage change needs a
new decoder). fim_prompt builds FIM prompts; heal_tokens and AllOf build constraints.
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
from model.config import VOCAB_SIZE
from model.modeling import CausalLM
from native import cuda_library

type Constraint = Reemit | AllOf | syntax_constraint.SyntaxConstraint


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
    """A continuation of input_ids (BOS and at least one more token) by at most max_new_tokens tokens.
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
    prefix of them or start with them (BOS excluded); once they are re-emitted, no restriction."""

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


class AllOf:
    """The intersection of constraints: each step allows the tokens every constraint allows (step_masks)."""

    def __init__(self, *constraints: Reemit | syntax_constraint.SyntaxConstraint):
        self.constraints = constraints

    def push(self, token: int) -> None:
        for c in self.constraints:
            c.push(token)


def step_masks(constraints: list[Constraint]) -> list[np.ndarray | None]:
    """Each constraint's next mask (Constraint.mask), those of the syntax constraints among them (AllOf's members too)
    computed by one syntax_constraint.masks call: in parallel, without the GIL."""
    members = [c.constraints if isinstance(c, AllOf) else (c,) for c in constraints]
    syntax = [m for group in members for m in group if isinstance(m, syntax_constraint.SyntaxConstraint)]
    batched = dict(zip(map(id, syntax), syntax_constraint.masks(syntax), strict=True)) if syntax else {}
    out = []
    for group in members:
        group_masks = [m for c in group if (m := batched[id(c)] if id(c) in batched else c.mask()) is not None]
        out.append(functools.reduce(np.bitwise_and, group_masks) if group_masks else None)
    return out


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
    """A request in a decoder slot: its index, the request, the steps left in its prompt before the step that chooses
    its first token, whether its prompt's K and V go to the prompt store once written, when it took the slot and when
    its first generated token reached the host (time.perf_counter), the generated tokens."""
    index: int
    request: Request
    prompt_steps: int
    save: bool
    start: float
    first: float | None = None
    out: list = field(default_factory=list)


class NativeDecoder:
    """Decoding by decode.cu of up to `batch` sequences at once (slots); a sequence's tokens do not depend on the batch
    (decode.cu computes each slot alone in a fixed order, its prompt included, and draws a sample's noise from its
    request's seed and the generated position).
    batch: dec_create's slot count, at most decode.cu's MAX_BATCH, bounded by free device memory.
    max_length: a slot's cache capacity in positions, at most the model's max_position_embeddings: the longest
    len(input_ids) + max_new_tokens - 1 the caller will run (a slot's cache is 2 num_hidden_layers max_length
    hidden_size bf16 values, so a smaller max_length fits more slots). The tokens do not depend on it: decode.cu splits
    attention by the model's max_position_embeddings.
    prompt_store: positions requested for decode.cu's prompt store (at most the device memory left after the slots, 2
    num_hidden_layers hidden_size bf16 values a position). A prompt's K and V at all but its last position are kept
    once written; a later request with the same prompt starts at its last token, its tokens unchanged (K and V depend
    on the prompt
    alone). To free pages, prompts unused in this run and the previous one are dropped, least recently used first, so a
    search's rounds keep the prompts they reuse and a too-small store keeps a fixed part rather than cycling."""

    @torch.no_grad()
    def __init__(self, model: CausalLM, *, bos_token_id: int, max_length: int, prompt_store: int):
        lib = cuda_library("decode")
        lib.dec_create.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        lib.dec_create.restype = ctypes.c_void_p
        lib.dec_batch.argtypes = [ctypes.c_void_p]
        lib.dec_set_prompt.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
                                       ctypes.c_float, ctypes.c_float, ctypes.c_ulonglong]
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
        self.lib = lib
        cfg = model.config
        L, C, T = cfg.num_hidden_layers, cfg.hidden_size, cfg.max_position_embeddings
        if not 1 < max_length <= T:
            raise ValueError(f"max_length {max_length} outside (1, max_position_embeddings {T}]")
        self.max_length = max_length
        # bf16 copies of the matrices and tables (the forward's rounding at use); fp32 q/k norm weights, gate weights
        # and rotary table, as modeling.py uses them
        weights = [model.embed_tokens.weight.bfloat16(), model.cos, model.sin, model.lm_head.weight.bfloat16()]
        for i, layer in enumerate(model.layers):
            attn, mlp = layer.self_attn, layer.mlp
            ve = str(i) in model.value_embeds
            weights += [torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight]).bfloat16(),
                        attn.o_proj.weight.bfloat16(), mlp.up_proj.weight.bfloat16(), mlp.down_proj.weight.bfloat16(),
                        model.value_embeds[str(i)].weight.bfloat16() if ve else None,
                        attn.ve_gate.weight.clone() if ve else None, attn.q_norm_weight.clone(),
                        attn.k_norm_weight.clone()]
        self.weights = weights  # referenced while the decoder reads them through pointers
        # per layer (resid, x0), rounded to bf16: a 0-dim fp32 tensor times a bf16 tensor computes in bf16; then the
        # output-logit cap (0: none)
        lambdas = torch.stack([model.resid_lambdas, model.x0_lambdas], 1).to(torch.bfloat16).float().flatten().cpu()
        scal = np.append(lambdas.numpy(), np.float32(cfg.final_logit_softcapping or 0.0))
        # F: up_proj is (F, C)
        config = np.array([L, C, model.layers[0].mlp.up_proj.weight.size(0), VOCAB_SIZE, max_length, T, prompt_store],
                          dtype=np.int32)
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
        self.tokens = np.zeros(self.batch, dtype=np.int32)
        self.free = list(range(lib.dec_store(self.handle) // PAGE))
        # the pinned host masks decode.cu copies to the masked slots (dec_masks), a Constraint.mask per slot
        self.masks = np.ctypeslib.as_array(lib.dec_masks(self.handle),
                                           shape=(self.batch, VOCAB_SIZE // np.iinfo(np.uint32).bits))

    def start(self, b: int, index: int, request: Request) -> Slot:
        """Start request `index` in slot b: from its prompt's stored K and V at its last token, else at position 0."""
        prompt = np.array(request.input_ids, dtype=np.int32)
        n = len(prompt) - 1  # the positions before the prompt's last token
        stored = self.stored.get(tuple(request.input_ids))
        if stored is not None:
            stored.used = self.runs
            for i, page in enumerate(stored.pages):
                self.lib.dec_load_prompt(self.handle, b, i * PAGE, page * PAGE, min(PAGE, n - i * PAGE))
        start = 0 if stored is None else n
        self.lib.dec_set_prompt(self.handle, b, prompt.ctypes.data, len(prompt), start, request.temperature,
                                request.top_p, request.seed)
        return Slot(index, request, prompt_steps=n - start, save=stored is None, start=time.perf_counter())

    def save(self, b: int, input_ids: list[int]) -> None:
        """Keep slot b's prompt K and V (all but the last position, written by the launched steps) in free pages,
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

    @torch.no_grad()
    def generate(self, requests: Iterable[Request]) -> Iterator[tuple[int, list[int]]]:
        """Continuations of requests, consumed lazily. Yields (request index, generated ids) in completion order: at BOS
        (excluded), after max_new_tokens, or once stopping_criteria holds (that token included). A step's stopping
        criteria are evaluated while the next step runs; a stopped slot's extra token is dropped. A step with a sampled
        or constrained slot launches decode.cu's body, takes the constraints' masks meanwhile (step_masks: a mask
        follows from the previous token;
        the syntax constraints' masks in one parallel batch), then its head, the choice by mask and temperature; other
        steps launch the whole greedy step. Raises ValueError when len(input_ids) + max_new_tokens - 1 > max_length or a
        constraint's mask allows no token (not even BOS).
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
                if (end := len(request.input_ids) + request.max_new_tokens - 1) > max_length:
                    raise ValueError(f"request {i}: len(input_ids) {len(request.input_ids)} + max_new_tokens "
                                     f"{request.max_new_tokens} - 1 = {end} > max_length {max_length}")
                slots[b] = self.start(b, i, request)
                flags[0, b] = True
                return
            slots[b] = None
            flags[0, b] = False

        for b in range(B):
            fill(b)
        set_active()
        while flags[0].any():
            # slots whose step chooses a token: past their prompt (a prompt step's choice is replaced by the prompt)
            choosers = [b for b, s in enumerate(slots) if s is not None and s.prompt_steps == 0]
            choosing = any(slots[b].request.temperature > 0 or slots[b].request.constraint is not None
                           for b in choosers)
            if choosing:
                lib.dec_launch_body(handle)
            else:
                lib.dec_launch(handle)
            finished = sorted(b for b in stopping if slots[b].request.stopping_criteria(slots[b].out))  # meanwhile
            stopping = []
            if choosing:
                # a stopped slot's next token is dropped
                constrained = [b for b in choosers if slots[b].request.constraint is not None and b not in finished]
                for b, mask in zip(constrained, step_masks([slots[b].request.constraint for b in constrained]),
                                   strict=True):
                    if mask is None:
                        continue
                    if not mask.any():
                        raise ValueError(f"request {slots[b].index}: its constraint allows no token at position "
                                         f"{len(slots[b].out)}")
                    self.masks[b] = mask
                    flags[1, b] = True
                set_active()
                flags[1] = False
                lib.dec_launch_head(handle)
            lib.dec_tokens(handle, self.tokens.ctypes.data)
            now = time.perf_counter()
            for b, s in enumerate(slots):
                if s is None or b in finished:
                    continue
                if s.prompt_steps:  # the step's next input is the prompt's next token
                    s.prompt_steps -= 1
                    if not s.prompt_steps and s.save:  # its K and V before the last prompt token are written
                        self.save(b, s.request.input_ids)
                    continue
                token = int(self.tokens[b])
                if token == bos:
                    finished.append(b)
                    continue
                if not s.out:
                    s.first = now
                s.out.append(token)
                if s.request.constraint is not None:
                    s.request.constraint.push(token)
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
