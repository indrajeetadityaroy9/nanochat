"""
Batched decoding of a GPT (nanochat/model/gpt.py) by csrc/decode.cu: the model prefills each prompt but its last
token, then captured CUDA graphs advance every active slot a token per step (the greedy argmax or a seeded sample,
within its constraint's allowed tokens) until BOS, the token budget or the caller's stop rule; a freed slot takes the
next request (continuous batching). The decoder snapshots the model's weights, context, windows and rotary tables at
construction (a stage change needs a new decoder). fim_prompt builds FIM prompts; heal and AllOf build constraints.
"""

import bisect
import ctypes
import functools
import weakref
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
import torch

from nanochat.data.tokenizer import FIM_MIDDLE, FIM_PREFIX, FIM_SUFFIX
from nanochat.inference import syntax_constraint
from nanochat.native import CSRC, cuda_library


def fim_prompt(tokenizer, prefix, suffix, *, mode: Literal["psm", "spm"]):
    """The token ids of a fill-in-the-middle prompt for the text between prefix and suffix (strings, each encoded on its
    own), in the layout the packer trains (nanochat/data/pack.py, loader.py; Bavarian et al. arXiv 2207.14255 §3, App. D):
        psm: BOS <fim_prefix> prefix <fim_suffix> suffix <fim_middle>
        spm: BOS <fim_prefix> <fim_suffix> suffix <fim_middle> prefix
    The model continues with the middle and ends it with BOS (NativeDecoder.run stops there)."""
    special = tokenizer.special_tokens
    pre, suf, mid = special[FIM_PREFIX], special[FIM_SUFFIX], special[FIM_MIDDLE]
    p, s = tokenizer.enc.encode_ordinary(prefix), tokenizer.enc.encode_ordinary(suffix)
    match mode:
        case "psm":
            return [tokenizer.bos_token_id, pre, *p, suf, *s, mid]
        case "spm":
            return [tokenizer.bos_token_id, pre, suf, *s, mid, *p]


class Constraint(Protocol):
    """A restriction of a request's generated tokens, advanced token by token (NativeDecoder.run)."""

    def mask(self) -> np.ndarray | None:
        """The tokens allowed next: uint32[vocab_size // 32], bit t % 32 of word t // 32 set iff token t may come next
        (BOS ends the generation; token_mask packs one); None: no restriction at this step."""

    def push(self, token: int) -> None:
        """Advance by the token chosen."""


@dataclass(frozen=True, kw_only=True)
class Request:
    """A continuation of prompt (BOS and at least one more token) by at most max_tokens tokens. stop: None or a
    predicate of the generated ids, true once the sequence is done. temperature 0: the greedy argmax (ties to the lowest
    id); > 0: a sample of softmax(logits / temperature) whose draw at generated position i depends only on (seed, i),
    not on the slot or the batch (decode.cu gumbel). constraint: the tokens allowed at each step."""
    prompt: list[int]
    max_tokens: int
    stop: Callable[[list[int]], bool] | None = None
    temperature: float = 0.0
    seed: int = 0
    constraint: Constraint | None = None


def token_mask(allowed: np.ndarray) -> np.ndarray:
    """Constraint.mask's layout of a boolean array over the vocabulary (its length a multiple of 32)."""
    return np.packbits(allowed, bitorder="little").view("<u4")


@functools.cache
def prefix_index(tokenizer) -> tuple[list[bytes], np.ndarray]:
    """The mergeable tokens' bytes sorted, and their ids in that order: the tokens whose bytes start with a given string
    are one contiguous run of it (heal)."""
    ids = np.array(sorted(range(tokenizer.bos_token_id), key=tokenizer.token_bytes.__getitem__))
    return [tokenizer.token_bytes[i] for i in ids], ids


class FirstToken:
    """heal's constraint: the first token within a mask, then no restriction."""

    def __init__(self, first: np.ndarray):
        self.first = first

    def mask(self) -> np.ndarray | None:
        return self.first

    def push(self, token: int) -> None:
        self.first = None


def heal(tokenizer, prompt: list[int]) -> tuple[list[int], Constraint]:
    """Token healing (Lundberg, "The Art of Prompt Design: Prompt Boundaries and Token Healing", 2023; Guidance): the
    prompt without its last token, and a constraint whose first token's bytes start with that token's bytes (BOS
    excluded), so the model picks the token across the prompt's boundary; the text of the healed prompt and the
    generated ids extends the prompt's text. prompt: BOS and at least two more tokens."""
    keys, ids = prefix_index(tokenizer)
    dropped = tokenizer.token_bytes[prompt[-1]]
    lo = bisect.bisect_left(keys, dropped)
    hi = bisect.bisect_right(keys, dropped, lo=lo, key=lambda k: k[:len(dropped)])
    allowed = np.zeros(tokenizer.vocab_size, dtype=bool)
    allowed[ids[lo:hi]] = True
    return prompt[:-1], FirstToken(token_mask(allowed))


class AllOf:
    """The intersection of constraints: each step allows the tokens every constraint allows."""

    def __init__(self, *constraints: Constraint):
        self.constraints = constraints

    def mask(self) -> np.ndarray | None:
        masks = [m for c in self.constraints if (m := c.mask()) is not None]
        return functools.reduce(np.bitwise_and, masks) if masks else None

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


class KVCache:
    """The model's prefill cache (gpt.py's kv_cache) for one row of seq_len tokens from position 0: keys and values per
    layer in FlashAttention's (B, T, H, D) layout, filled in place by the attention layers."""

    def __init__(self, *, num_layers, seq_len, num_heads, head_dim, device):
        self.k_cache = torch.empty(num_layers, 1, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
        self.v_cache = torch.empty_like(self.k_cache)


@dataclass
class Slot:
    """A request in a decoder slot: its index, the request, the generated tokens."""
    index: int
    request: Request
    out: list = field(default_factory=list)


class NativeDecoder:
    """Decoding by decode.cu of up to `batch` sequences at once (slots); a sequence's tokens do not depend on the batch
    (decode.cu computes each slot alone in a fixed order and draws a sample's noise from its request's seed and the
    generated position; each prompt is prefilled at batch 1).
    batch: dec_create's slot count, at most the active mask's bits, bounded by free device memory.
    max_len: a slot's cache capacity in positions, at most the model's sequence_len: the longest len(prompt) +
    max_tokens - 1 the caller will run (a slot's cache is 2 n_layer max_len n_embd bf16 values, so a smaller max_len
    fits more slots). The tokens do not depend on it: decode.cu splits attention by the model's sequence_len."""

    @torch.no_grad()
    def __init__(self, model, *, bos_token_id, max_len):
        lib = cuda_library(CSRC / "decode.cu", headers=(CSRC / "qk_row.cuh",), defines=())
        lib.dec_create.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        lib.dec_create.restype = ctypes.c_void_p
        lib.dec_batch.argtypes = [ctypes.c_void_p]
        lib.dec_set_prefilled.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p,
                                          ctypes.c_void_p, ctypes.c_longlong, ctypes.c_float, ctypes.c_ulonglong]
        lib.dec_masks.argtypes = [ctypes.c_void_p]
        lib.dec_masks.restype = ctypes.POINTER(ctypes.c_uint32)
        lib.dec_set_active.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_ulonglong]
        lib.dec_launch.argtypes = [ctypes.c_void_p]
        lib.dec_launch_body.argtypes = [ctypes.c_void_p]
        lib.dec_launch_head.argtypes = [ctypes.c_void_p]
        lib.dec_destroy.argtypes = [ctypes.c_void_p]
        lib.dec_tokens.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib, self.model = lib, model
        cfg = model.config
        L, C, T = cfg.n_layer, cfg.n_embd, cfg.sequence_len
        if not 1 < max_len <= T:
            raise ValueError(f"max_len {max_len} outside (1, sequence_len {T}]")
        self.max_len = max_len
        self.device = model.wte.weight.device
        # bf16 copies of the matrices and tables (the forward's rounding at use); fp32 q/k gains, gate weights and
        # rotary table, as gpt.py uses them
        weights = [model.wte.weight.bfloat16(), model.cos, model.sin, model.lm_head.weight.bfloat16()]
        for i, block in enumerate(model.blocks):
            attn, mlp = block.attn, block.mlp
            ve = str(i) in model.value_embeds
            weights += [torch.cat([attn.c_q.weight, attn.c_k.weight, attn.c_v.weight]).bfloat16(),
                        attn.c_o.weight.bfloat16(), mlp.c_fc.weight.bfloat16(), mlp.c_proj.weight.bfloat16(),
                        model.value_embeds[str(i)].weight.bfloat16() if ve else None,
                        attn.ve_gate.weight.clone() if ve else None, attn.q_gain.clone(), attn.k_gain.clone()]
        self.weights = weights  # referenced while the decoder reads them through pointers
        # per layer (resid, x0), rounded to bf16: a 0-dim fp32 tensor times a bf16 tensor computes in bf16
        scal = torch.stack([model.resid_lambdas, model.x0_lambdas], 1).to(torch.bfloat16).float().cpu().numpy()
        # F: c_fc is (F, C)
        config = np.array([L, C, model.blocks[0].mlp.c_fc.weight.size(0), cfg.vocab_size, max_len, T], dtype=np.int32)
        self.bos = bos_token_id
        # dec_destroy (the finalizer) releases every native resource; the buffers it reads are this object's tensors
        self.stream = torch.cuda.Stream(self.device)
        ptrs = np.array([0 if t is None else t.data_ptr() for t in weights], dtype=np.uint64)
        window = np.array(model.window_sizes, dtype=np.int32)
        torch.cuda.synchronize()  # the weight copies above are on the current stream; the decoder uses its own
        self.handle = lib.dec_create(config.ctypes.data, ptrs.ctypes.data, scal.ctypes.data, window.ctypes.data,
                                     self.stream.cuda_stream)
        weakref.finalize(self, lib.dec_destroy, self.handle)
        self.batch = lib.dec_batch(self.handle)
        self.tokens = np.zeros(self.batch, dtype=np.int32)
        # the pinned host masks decode.cu copies to the masked slots (dec_masks), a Constraint.mask per slot
        self.masks = np.ctypeslib.as_array(lib.dec_masks(self.handle),
                                           shape=(self.batch, cfg.vocab_size // np.iinfo(np.uint32).bits))

    @torch.no_grad()
    def prefill(self, b: int, request: Request) -> None:
        """Start a request in slot b: the model's prefill of all but the prompt's last token fills the slot's cache."""
        cfg, tokens = self.model.config, request.prompt
        n = len(tokens) - 1
        cache = KVCache(num_layers=cfg.n_layer, seq_len=n, num_heads=cfg.n_head, head_dim=cfg.n_embd // cfg.n_head,
                        device=self.device)
        self.model.prefill(torch.tensor([tokens[:n]], dtype=torch.int32, device=self.device), cache)
        torch.cuda.synchronize()  # the decoder copies the cache on its own stream
        self.lib.dec_set_prefilled(self.handle, b, int(tokens[n]), n, cache.k_cache.data_ptr(), cache.v_cache.data_ptr(),
                                   cache.k_cache.stride(0), request.temperature, request.seed)
        torch.cuda.synchronize()  # the copies read the cache tensors, freed on return

    @torch.no_grad()
    def run(self, requests: Iterable[Request]) -> Iterator[tuple[int, list[int]]]:
        """Continuations of requests, consumed lazily. Yields (request index, generated ids) in completion order: at BOS
        (excluded), after max_tokens, or once stop holds (that token included). A step's stop rule is evaluated while
        the next step runs; a stopped slot's extra token is dropped. A step with a sampled or constrained slot launches
        decode.cu's body, takes the constraints' masks meanwhile (step_masks: a mask follows from the previous token;
        the syntax constraints' masks in one parallel batch), then its head, the choice by mask and temperature; other
        steps launch the whole greedy step. Raises ValueError when len(prompt) + max_tokens - 1 > max_len or a
        constraint's mask allows no token (not even BOS)."""
        lib, handle, B, bos, max_len = self.lib, self.handle, self.batch, self.bos, self.max_len
        queue, slots, active = enumerate(requests), [None] * B, 0
        stopping = []  # slots whose last token awaits its stop rule

        def start(b):
            nonlocal active
            for i, request in queue:
                if (end := len(request.prompt) + request.max_tokens - 1) > max_len:
                    raise ValueError(f"request {i}: len(prompt) {len(request.prompt)} + max_tokens "
                                     f"{request.max_tokens} - 1 = {end} > max_len {max_len}")
                self.prefill(b, request)
                slots[b] = Slot(i, request)
                active |= 1 << b
                return
            slots[b] = None
            active &= ~(1 << b)

        def finish(b):
            s = slots[b]
            start(b)
            return s.index, s.out

        for b in range(B):
            start(b)
        lib.dec_set_active(handle, active, 0)
        while active:
            choosing = any(s is not None and (s.request.temperature > 0 or s.request.constraint is not None)
                           for s in slots)
            if choosing:
                lib.dec_launch_body(handle)
            else:
                lib.dec_launch(handle)
            finished = sorted(b for b in stopping if slots[b].request.stop(slots[b].out))  # meanwhile the step runs
            stopping = []
            if choosing:
                masked = 0
                # a stopped slot's next token is dropped
                constrained = [b for b, s in enumerate(slots)
                               if s is not None and s.request.constraint is not None and b not in finished]
                for b, mask in zip(constrained, step_masks([slots[b].request.constraint for b in constrained]),
                                   strict=True):
                    if mask is None:
                        continue
                    if not mask.any():
                        raise ValueError(f"request {slots[b].index}: its constraint allows no token at position "
                                         f"{len(slots[b].out)}")
                    self.masks[b] = mask
                    masked |= 1 << b
                lib.dec_set_active(handle, active, masked)
                lib.dec_launch_head(handle)
            lib.dec_tokens(handle, self.tokens.ctypes.data)
            for b, s in enumerate(slots):
                if s is None or b in finished:
                    continue
                token = int(self.tokens[b])
                if token == bos:
                    finished.append(b)
                    continue
                s.out.append(token)
                if s.request.constraint is not None:
                    s.request.constraint.push(token)
                if len(s.out) == s.request.max_tokens:
                    finished.append(b)
                elif s.request.stop is not None:
                    stopping.append(b)
            if finished:
                for b in finished:
                    yield finish(b)
                lib.dec_set_active(handle, active, 0)  # each choosing step sets its masks anew
