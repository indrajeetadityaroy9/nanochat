"""
Greedy decoding of a GPT (nanochat/model/gpt.py) on CUDA for a batch of independent sequences: the model's forward
pass prefills a sequence's prompt but its last token, then csrc/decode.cu advances every sequence of the batch by one
token per step, one captured CUDA graph (see its header), each until the model writes BOS, its token budget ends or the
caller's stop rule fires; a finished sequence's slot takes the next request (continuous batching). fim_prompt
assembles fill-in-the-middle prompts. The decoder reads the model's weights as they are when it is made (bf16 copies of
the matrices and tables) and the model's context, windows and rotary tables (a stage change needs a new decoder). Its
library is compiled from csrc/decode.cu once per source revision and GPU architecture (native.cuda_library).
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


def fim_prompt(tokenizer, prefix, suffix, mode: Literal["psm", "spm"]):
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

    def __init__(self, num_layers, seq_len, num_heads, head_dim, device):
        self.k_cache = torch.empty(num_layers, 1, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
        self.v_cache = torch.empty_like(self.k_cache)


@dataclass
class _Slot:
    """A sequence in a decoder slot: its request's index, its budget and stop rule, the generated tokens."""
    index: int
    budget: int
    stop: Callable | None
    out: list = field(default_factory=list)


class NativeDecoder:
    """Greedy decoding by decode.cu of up to `batch` sequences at once (slots). A sequence's tokens do not depend on the
    batch: decode.cu computes each slot alone in a fixed order, and each sequence is prefilled by the model's batch-1
    forward of its own prompt.
    batch: the most decode.cu runs (dec_max_batch) whose KV caches and buffers fit the free device memory. A step reads
    the weights once for all slots plus every slot's cached keys and values, so tokens per second grow with the batch
    until the KV caches' bytes dominate the step."""

    @torch.no_grad()
    def __init__(self, model, *, bos_token_id):
        lib = cuda_library("decode")
        lib.dec_create.argtypes = [ctypes.c_void_p] * 5
        lib.dec_create.restype = ctypes.c_void_p
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
        # decode.cu reads bf16 copies of the matrices and tables (the rounding the forward applies to the fp32
        # parameters at use) and fp32 copies of the q/k gains and value-embedding gate weights, which gpt.py uses in
        # fp32, as it does the rotary table. The weights are what the model has under its config's switches: value
        # embeddings and gates on the layers of model.value_embeds; q/k gains (with qk_gain off, init_buffers's ones);
        # resid lambdas 1 and x0 lambdas 0 when the model has none (its forward's identity: 1 x + 0 x0 = x, exact).
        def bf16(t):
            return t.to(torch.bfloat16)

        weights = [bf16(model.wte.weight), model.cos, model.sin, bf16(model.lm_head.weight)]
        for i, block in enumerate(model.blocks):
            attn, mlp = block.attn, block.mlp
            ve = str(i) in model.value_embeds
            weights += [bf16(torch.cat([attn.c_q.weight, attn.c_k.weight, attn.c_v.weight])), bf16(attn.c_o.weight),
                        bf16(mlp.c_fc.weight), bf16(mlp.c_proj.weight),
                        bf16(model.value_embeds[str(i)].weight) if ve else None,
                        attn.ve_gate.weight.clone() if ve else None, attn.q_gain.clone(), attn.k_gain.clone()]
        self._weights = weights  # referenced while the decoder reads them through pointers
        # the resid/x0 lambdas meet bf16 tensors as bf16 values (a 0-dim fp32 tensor times a bf16 tensor computes in
        # bf16), per layer (resid, x0)
        resid = torch.ones(L) if model.resid_lambdas is None else model.resid_lambdas.cpu()
        x0 = torch.zeros(L) if model.x0_lambdas is None else model.x0_lambdas.cpu()
        scal = torch.stack([resid, x0], 1).to(torch.bfloat16).float().numpy()
        # MLP hidden width F: c_fc is (F, C), c_proj (C, F)
        hidden = model.blocks[0].mlp.c_fc.weight.size(0)
        # a slot's device memory in bytes, 2 per bf16 value: KV caches 2 L T C, the per-vector buffers (x, x0, xm, h,
        # att, xo 6 C, qkv 3 C, MLP F); 4 per fp32 value: attention partials H splits (D sums + m, l) with splits at
        # most T (one key each)
        slot_bytes = 2 * (2 * L * T * C + 9 * C + hidden) + 4 * cfg.n_head * T * (C // cfg.n_head + 2)
        self.batch = min(lib.dec_max_batch(), torch.cuda.mem_get_info()[0] // slot_bytes)
        config = np.array([L, C, hidden, cfg.vocab_size, T, self.batch], dtype=np.int32)
        self.bos = bos_token_id
        # Every buffer decode.cu reads or writes is a tensor of this object or its own, and the stream is torch's (its
        # pool streams live as long as the process), so dec_destroy (the finalizer, before these tensors are freed) is
        # the decoder's only release of native resources.
        self._stream = torch.cuda.Stream(self.device)
        self._tokens = np.zeros(self.batch, dtype=np.int32)
        ptrs = np.array([0 if t is None else t.data_ptr() for t in weights], dtype=np.uint64)
        window = np.array(model.window_sizes, dtype=np.int32)
        torch.cuda.synchronize()  # the weight copies above are on the current stream; the decoder uses its own
        self.handle = lib.dec_create(config.ctypes.data, ptrs.ctypes.data, scal.ctypes.data, window.ctypes.data,
                                     self._stream.cuda_stream)
        weakref.finalize(self, lib.dec_destroy, self.handle)

    @torch.no_grad()
    def _prefill(self, b, tokens):
        """Start a sequence of tokens in slot b: the model's forward pass over all but its last token fills the slot's
        cache; the steps continue from the last token."""
        cfg, n = self.model.config, len(tokens) - 1
        cache = KVCache(cfg.n_layer, n, cfg.n_head, cfg.n_embd // cfg.n_head, self.device)
        self.model.prefill(torch.tensor([tokens[:n]], dtype=torch.int32, device=self.device), cache)
        torch.cuda.synchronize()  # the decoder copies the cache on its own stream
        self.lib.dec_set_prefilled(self.handle, b, int(tokens[n]), n, cache.k_cache.data_ptr(), cache.v_cache.data_ptr(),
                                   cache.k_cache.stride(0))
        torch.cuda.synchronize()  # the copies read the cache tensors, freed on return

    @torch.no_grad()
    def run(self, requests):
        """Greedy continuations of requests, an iterable (consumed lazily) of (prompt tokens: BOS and at least one more;
        max_tokens; stop: None or a callable of the generated token ids so far, true when the sequence is done), each
        with len(prompt) + max_tokens - 1 at most the model's sequence_len (the last step's position indexes the slot's
        cache and the rotary table). Yields (the request's index, its generated token ids) as each finishes, in
        completion order: at BOS (not included), after max_tokens, or once the stop rule held (the token that made it
        hold included). A finished sequence's slot takes the next request. The stop rule of a step's tokens is evaluated
        while the next step runs: a slot it ends computes one step more, whose token is dropped."""
        lib, handle, B, bos = self.lib, self.handle, self.batch, self.bos
        queue, slots, mask = enumerate(requests), [None] * B, 0
        stopping = []  # slots whose last token awaits its stop rule

        def start(b):
            nonlocal mask
            for i, (prompt, max_tokens, stop) in queue:
                self._prefill(b, prompt)
                slots[b] = _Slot(i, max_tokens, stop)
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
            lib.dec_tokens(handle, self._tokens.ctypes.data)
            for b, s in enumerate(slots):
                if s is None or b in finished:
                    continue
                token = int(self._tokens[b])
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
