"""
Batched greedy decoding of a GPT (nanochat/model/gpt.py) by csrc/decode.cu: the model prefills each prompt but its last
token, then one captured CUDA graph advances every active slot a token per step until BOS, the token budget or the
caller's stop rule; a freed slot takes the next request (continuous batching). The decoder snapshots the model's
weights, context, windows and rotary tables at construction (a stage change needs a new decoder). fim_prompt builds FIM
prompts.
"""

import ctypes
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch

from nanochat.data.tokenizer import FIM_MIDDLE, FIM_PREFIX, FIM_SUFFIX
from nanochat.native import cuda_library


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


class KVCache:
    """The model's prefill cache (gpt.py's kv_cache) for one row of seq_len tokens from position 0: keys and values per
    layer in FlashAttention's (B, T, H, D) layout, filled in place by the attention layers."""

    def __init__(self, *, num_layers, seq_len, num_heads, head_dim, device):
        self.k_cache = torch.empty(num_layers, 1, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
        self.v_cache = torch.empty_like(self.k_cache)


@dataclass
class Slot:
    """A sequence in a decoder slot: its request's index, its budget and stop rule, the generated tokens."""
    index: int
    budget: int
    stop: Callable | None
    out: list = field(default_factory=list)


class NativeDecoder:
    """Greedy decoding by decode.cu of up to `batch` sequences at once (slots); a sequence's tokens do not depend on the
    batch (decode.cu computes each slot alone in a fixed order; each prompt is prefilled at batch 1).
    batch: dec_create's slot count, at most the active mask's bits, bounded by free device memory."""

    @torch.no_grad()
    def __init__(self, model, *, bos_token_id):
        lib = cuda_library("decode")
        lib.dec_create.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        lib.dec_create.restype = ctypes.c_void_p
        lib.dec_batch.argtypes = [ctypes.c_void_p]
        lib.dec_set_prefilled.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p,
                                          ctypes.c_void_p, ctypes.c_longlong]
        lib.dec_set_active.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong]
        lib.dec_launch.argtypes = [ctypes.c_void_p]
        lib.dec_destroy.argtypes = [ctypes.c_void_p]
        lib.dec_tokens.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib, self.model = lib, model
        cfg = model.config
        L, C, T = cfg.n_layer, cfg.n_embd, cfg.sequence_len
        self.device = model.wte.weight.device
        # bf16 copies of the matrices and tables (the forward's rounding at use); fp32 q/k gains, gate weights and
        # rotary table, as gpt.py uses them. Absent resid/x0 lambdas enter as their identity (1, 0).
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
        resid = torch.ones(L) if model.resid_lambdas is None else model.resid_lambdas.cpu()
        x0 = torch.zeros(L) if model.x0_lambdas is None else model.x0_lambdas.cpu()
        scal = torch.stack([resid, x0], 1).to(torch.bfloat16).float().numpy()
        # F: c_fc is (F, C)
        config = np.array([L, C, model.blocks[0].mlp.c_fc.weight.size(0), cfg.vocab_size, T], dtype=np.int32)
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

    @torch.no_grad()
    def prefill(self, b, tokens):
        """Start tokens in slot b: the model's prefill of all but the last token fills the slot's cache."""
        cfg, n = self.model.config, len(tokens) - 1
        cache = KVCache(num_layers=cfg.n_layer, seq_len=n, num_heads=cfg.n_head, head_dim=cfg.n_embd // cfg.n_head,
                        device=self.device)
        self.model.prefill(torch.tensor([tokens[:n]], dtype=torch.int32, device=self.device), cache)
        torch.cuda.synchronize()  # the decoder copies the cache on its own stream
        self.lib.dec_set_prefilled(self.handle, b, int(tokens[n]), n, cache.k_cache.data_ptr(), cache.v_cache.data_ptr(),
                                   cache.k_cache.stride(0))
        torch.cuda.synchronize()  # the copies read the cache tensors, freed on return

    @torch.no_grad()
    def run(self, requests):
        """Greedy continuations of requests, a lazily consumed iterable of (prompt: BOS and at least one more token;
        max_tokens; stop: None or a predicate of the generated ids), with len(prompt) + max_tokens - 1 <= sequence_len.
        Yields (request index, generated ids) in completion order: at BOS (excluded), after max_tokens, or once stop
        holds (that token included). A step's stop rule is evaluated while the next step runs; a stopped slot's extra
        token is dropped."""
        lib, handle, B, bos = self.lib, self.handle, self.batch, self.bos
        queue, slots, mask = enumerate(requests), [None] * B, 0
        stopping = []  # slots whose last token awaits its stop rule

        def start(b):
            nonlocal mask
            for i, (prompt, max_tokens, stop) in queue:
                self.prefill(b, prompt)
                slots[b] = Slot(i, max_tokens, stop)
                mask |= 1 << b
                return
            slots[b] = None
            mask &= ~(1 << b)

        def finish(b):
            s = slots[b]
            start(b)
            return s.index, s.out

        for b in range(B):
            start(b)
        lib.dec_set_active(handle, mask)
        while mask:
            lib.dec_launch(handle)
            finished = sorted(b for b in stopping if slots[b].stop(slots[b].out))  # meanwhile the step runs
            stopping = []
            lib.dec_tokens(handle, self.tokens.ctypes.data)
            for b, s in enumerate(slots):
                if s is None or b in finished:
                    continue
                token = int(self.tokens[b])
                if token == bos:
                    finished.append(b)
                    continue
                s.out.append(token)
                if len(s.out) == s.budget:
                    finished.append(b)
                elif s.stop is not None:
                    stopping.append(b)
            if finished:
                for b in finished:
                    yield finish(b)
                lib.dec_set_active(handle, mask)
