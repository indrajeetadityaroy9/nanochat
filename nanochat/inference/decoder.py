"""
Greedy decoding of a GPT (nanochat/model/gpt.py) on CUDA for a batch of independent sequences: the model's forward
pass prefills a sequence's prompt but its last token, then decode.cu advances every sequence of the batch by one token
per step, two captured CUDA graphs (see its header), each until the model writes BOS, its token budget ends or the
caller's stop rule fires; a finished sequence's slot takes the next request (continuous batching). fim_prompt
assembles fill-in-the-middle prompts. The decoder snapshots the model's weights when it is made (the Engram tables
excepted: it gathers their rows from the model's parameters each step). Its library is compiled from decode.cu once per
source revision and GPU architecture (native.cuda_library).
"""

import ctypes
import os
import weakref
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

from nanochat.data.tokenizer import FIM_MIDDLE, FIM_PREFIX, FIM_SUFFIX
from nanochat.model.engram import engram_ids
from nanochat.model.mudd import MUDD_SCALE
from nanochat.native import cuda_library
from nanochat.ops.engram import ENGRAM_CONV_KERNEL

SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decode.cu")


def fim_prompt(tokenizer, prefix, suffix, mode):
    """The token ids of a fill-in-the-middle prompt for the text between prefix and suffix (strings, each encoded on its
    own), in the layout the packer trains (nanochat/data/pack.py, loader.py; Bavarian et al. arXiv 2207.14255 §3, App. D):
        psm: BOS <fim_prefix> prefix <fim_suffix> suffix <fim_middle>
        spm: BOS <fim_prefix> <fim_suffix> suffix <fim_middle> prefix
    The model continues with the middle and ends it with BOS (NativeDecoder.run stops there)."""
    special = tokenizer.special_tokens
    pre, suf, mid = special[FIM_PREFIX], special[FIM_SUFFIX], special[FIM_MIDDLE]
    p, s = tokenizer.enc.encode_ordinary(prefix), tokenizer.enc.encode_ordinary(suffix)
    if mode == "psm":
        return [tokenizer.bos_token_id, pre, *p, suf, *s, mid]
    return [tokenizer.bos_token_id, pre, suf, *s, mid, *p]


class KVCache:
    """The model's prefill cache (gpt.py's kv_cache) for one row of seq_len tokens from position 0: keys and values per
    layer in FlashAttention's (B, T, H, D) layout, filled in place by the attention layers, the last token's normalized
    embedding (smear), per Engram layer the (B, T, C) rows its short convolution reads (engram_taps) and, per key-offset
    layer, the last token's own stationary key dims before the offset, (B, H, D - rotary_dim) (key_tail; the cache holds
    the shifted keys)."""

    def __init__(self, num_layers, seq_len, num_heads, head_dim, engram_layers, device):
        self.k_cache = torch.empty(num_layers, 1, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
        self.v_cache = torch.empty_like(self.k_cache)
        self.prev_embedding = None
        self.engram_taps = [None] * engram_layers
        self.key_tail = [None] * num_layers


@dataclass
class _Slot:
    """A sequence in a decoder slot: its request's index, its budget and stop rule, the generated tokens."""
    index: int
    budget: int
    stop: Optional[Callable]
    out: list = field(default_factory=list)


class NativeDecoder:
    """Greedy decoding by decode.cu of up to `batch` sequences at once (slots). A sequence's tokens do not depend on the
    batch: decode.cu computes each slot alone in a fixed order, and each sequence is prefilled by the model's batch-1
    forward of its own prompt.
    Each step runs as decode.cu's front (the blocks before the first Engram layer) and back (the rest). Between them, on
    the decoder's stream, each slot's Engram rows of its input token are gathered on the GPU by the model's own
    functions (engram_ids over the slot's last max(orders) tokens, each Engram layer's Engram.rows) into the rows buffer
    the back reads; the host enqueues that lookup while the front runs (DeepSeek Engram §6.4). Each Engram layer's short
    convolution reads the u rows (nanochat/model/engram.py) of earlier positions: the step keeps them in a ring per
    layer and slot, which the prefill fills from the model's forward (kv_cache.engram_taps).
    batch: the most decode.cu runs (dec_max_batch) whose KV caches and buffers fit the free device memory. A step reads
    the weights once for all slots plus every slot's cached keys and values, so tokens per second grow with the batch
    until the KV caches' bytes dominate the step."""

    def __init__(self, model):
        lib = cuda_library("decode", SOURCE)
        lib.dec_create.argtypes = [ctypes.c_void_p] * 5
        lib.dec_create.restype = ctypes.c_void_p
        lib.dec_set_prefilled.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                          ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p,
                                          ctypes.c_void_p, ctypes.c_void_p]
        lib.dec_set_active.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong]
        lib.dec_launch.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.dec_destroy.argtypes = [ctypes.c_void_p]
        lib.dec_tokens.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib, self.model = lib, model
        cfg, blocks = model.config, model.transformer.h
        attn = blocks[0].attn
        L, C = cfg.n_layer, cfg.n_embd
        weights = [model.transformer.wte.weight, model.smear_gate.weight, model.cos, model.sin, model.lm_head.weight]
        engram = model.config.engram
        # read as fp32 by decode.cu: the rotary tables (as qk_prologue.cu), the Engram norm and conv weights (as
        # nanochat/ops/engram.cu) and the smear and value-embedding gate weights (gpt.py runs those gates in fp32 on the
        # fp32 parameters); every other weight (the attention output gate's included: gpt.py rounds it for its
        # fp32-accumulated logits) is snapshot as bf16, the rounding the training forward applies to the fp32
        # parameters. smear_lambda enters gpt.py's fp32 smear as is; the resid/x0 lambdas meet bf16 tensors as bf16
        # values (a 0-dim fp32 tensor times a bf16 tensor computes in bf16)
        rounded = lambda t: float(t.detach().to(torch.bfloat16))
        fp32 = [model.cos, model.sin, model.smear_gate.weight]
        scal = [float(model.smear_lambda.detach()), attn.ve_gate_scale, MUDD_SCALE]
        for i, block in enumerate(blocks):
            a, mlp = block.attn, block.mlp
            weights += [torch.cat([a.c_q.weight, a.c_k.weight, a.c_v.weight]), a.c_proj.weight, mlp.c_fc.weight,
                        mlp.c_proj.weight, model.value_embeds[str(i)].weight if str(i) in model.value_embeds else None,
                        a.ve_gate.weight if a.ve_gate is not None else None]
            fuse = model.engram[str(i)] if str(i) in model.engram else None
            weights += [fuse.w_kv.weight, fuse.norm_h, fuse.norm_k, fuse.norm_v, fuse.conv_w] if fuse else [None] * 5
            weights += [a.attn_gate.weight]
            fp32 += [fuse.norm_h, fuse.norm_k, fuse.norm_v, fuse.conv_w] if fuse else []
            fp32 += [a.ve_gate.weight] if a.ve_gate is not None else []
            scal += [rounded(model.resid_lambdas[i]), rounded(model.x0_lambdas[i]), a.qk_scale]
        # MUDD's two sites (gpt.py mudd_in, mudd_out): W1 rounded to bf16 as the model's fp32-accumulated first layer
        # reads it, W2 and b fp32 (an fp32 linear); the sources (model.mudd_sources) coded for decode.cu: -1 x0, l the
        # output of block l, L + l the value embedding of layer l
        sources = []
        code = lambda key: -1 if key == "x0" else key[1] if key[0] == "h" else L + key[1]
        for site, keys in ((model.mudd_in, model.mudd_sources[0]), (model.mudd_out, model.mudd_sources[1])):
            weights += [site.w1, site.w2, site.b]
            fp32 += [site.w2, site.b]
            sources += [len(keys), site.w1.size(0)] + [code(key) for key in keys]
        # snapshots, which the decoder reads through pointers
        dtype = lambda t: torch.float32 if any(t is p for p in fp32) else torch.bfloat16
        self._weights = [None if t is None else t.detach().to(dtype(t)).contiguous() for t in weights]
        # MLP hidden width h: c_proj is (C, h); c_fc is (2h, C), gate rows then up rows. Engram conv: (C, kernel) weights
        # with taps dilation apart; its ring holds the positions p - dilation (kernel - 1) .. p
        dilation = next(iter(model.engram.values())).dilation
        self._ring = dilation * (ENGRAM_CONV_KERNEL - 1) + 1
        hidden, cols, T, E = blocks[0].mlp.c_proj.weight.size(1), engram.slots * engram.dim, cfg.sequence_len, len(engram.layers)
        # a slot's device memory (bf16 unless noted): KV caches 2 L T C, key-offset slots 2 L C, Engram conv rings
        # E ring C and rows E cols, block outputs L C, the per-vector buffers (x0, xm, xf, h, att, xo, prev 7 C, qkv
        # 3 C, MLP h, Engram kv 2 C), fp32 value stream C, fp32 attention partials H splits (D + 2) with splits at most
        # T (one key each)
        slot_bytes = (2 * (2 * L * T * C + 2 * L * C + E * self._ring * C + E * cols + L * C + 12 * C + hidden)
                      + 4 * C + 4 * cfg.n_head * T * (attn.head_dim + 2))
        self.batch = min(lib.dec_max_batch(), torch.cuda.mem_get_info()[0] // slot_bytes)
        config = np.array([L, cfg.n_head, C, hidden, cfg.vocab_size, T, self.batch, model.smear_gate.weight.size(1),
                           attn.ve_gate_channels, cols, dilation, max(engram.orders)] + sources, dtype=np.int32)
        # the step's Engram rows, bf16 [Engram layers][slots][sub-tables x dim], gathered on the decoder's stream from
        # the slots' windows (decode.cu's k_window: per slot its last max(orders) tokens) by one compiled graph of the
        # fixed (batch, max(orders)) shape. Every buffer decode.cu reads or writes is a tensor of this object and the
        # stream is torch's (its pool streams live as long as the process), so dec_destroy (the finalizer, before these
        # tensors are freed) is the decoder's only release of native resources.
        self.bos = engram.bos
        device = model.get_device()
        self._rows = torch.empty(E, self.batch, cols, dtype=torch.bfloat16, device=device)
        self._window = torch.full((self.batch, max(engram.orders)), self.bos, dtype=torch.int64, device=device)
        layers = [model.engram[str(l)] for l in engram.layers]
        self._lookup = torch.compile(
            lambda window: torch.stack([m.rows(engram_ids(window, m.config, m.layer)[:, -1]) for m in layers]),
            dynamic=False, fullgraph=True)
        self._stream = torch.cuda.Stream(device)
        self._tokens = np.zeros(self.batch, dtype=np.int32)
        ptrs = [0 if t is None else t.data_ptr() for t in self._weights]
        ptrs = np.array(ptrs[:5] + [self._rows.data_ptr(), self._window.data_ptr()] + ptrs[5:], dtype=np.uint64)
        scal = np.array(scal, dtype=np.float32)
        self.key_offsets = list(model.key_offsets)
        layer = np.array([[w, k] for w, k in zip(model.window_sizes, self.key_offsets, strict=False)], dtype=np.int32)
        torch.cuda.synchronize()  # the weight copies above are on the current stream; the decoder uses its own
        self.handle = lib.dec_create(config.ctypes.data, ptrs.ctypes.data, scal.ctypes.data, layer.ctypes.data,
                                     self._stream.cuda_stream)
        weakref.finalize(self, lib.dec_destroy, self.handle)

    @torch.no_grad()
    def _prefill(self, b, tokens):
        """Start a sequence of tokens in slot b: the model's forward pass over all but its last token fills the slot's
        cache; the steps continue from the last token. The Engram conv rings take the u rows of the cached positions the
        steps' taps reach (position q in slot q % ring); the key-offset slots the last cached position's own stationary
        key dims (cache.key_tail), which the first step shifts into its key."""
        m, n_cached = self.model, len(tokens) - 1
        H, D = m.config.n_kv_head, m.transformer.h[0].attn.head_dim
        cache = KVCache(m.config.n_layer, n_cached, H, D, len(m.config.engram.layers), m.get_device())
        idx = torch.tensor([tokens[:n_cached]])
        m(idx.to(m.get_device()), kv_cache=cache)
        prev = cache.prev_embedding  # (1, 1, C): the last cached position's normalized embedding, which the first step smears
        taps = torch.stack(cache.engram_taps)[:, 0]  # (Engram layers, n_cached, C)
        ring = taps.new_zeros(taps.size(0), self._ring, taps.size(2))
        q = torch.arange(max(0, n_cached - self._ring + 1), n_cached, device=taps.device)
        ring[:, q % self._ring] = taps[:, q]
        # decode.cu reads a layer's slot at the stationary dims only (rotary_dim and on of each head)
        tail = torch.zeros(m.config.n_layer, H, D, dtype=torch.bfloat16, device=m.get_device())
        for l, on in enumerate(self.key_offsets):
            if on:
                tail[l, :, D - cache.key_tail[l].size(-1):] = cache.key_tail[l][0]
        prompt = np.asarray(tokens, dtype=np.int32)  # referenced until the copy: ctypes passes only its address
        torch.cuda.synchronize()  # the decoder copies the cache on its own stream
        self.lib.dec_set_prefilled(self.handle, b, prompt.ctypes.data, len(tokens), n_cached, cache.k_cache.data_ptr(),
                                   cache.v_cache.data_ptr(), cache.k_cache.stride(0), prev.data_ptr(), ring.data_ptr(),
                                   tail.data_ptr())
        torch.cuda.synchronize()  # the copies read the cache tensors, freed on return

    @torch.no_grad()
    def run(self, requests):
        """Greedy continuations of requests, an iterable (consumed lazily) of (prompt tokens: BOS and at least one more;
        max_tokens; stop: None or a callable of the generated token ids so far, true when the sequence is done). Yields
        (the request's index, its generated token ids) as each finishes, in completion order: at BOS (not included),
        after max_tokens, or once the stop rule held (the token that made it hold included). A finished sequence's slot
        takes the next request. The stop rule of a step's tokens is evaluated while the next step runs: a slot it ends
        computes one step more, whose token is dropped."""
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
            lib.dec_launch(handle, 0)
            with torch.cuda.stream(self._stream):  # after the front's k_window, before the back
                self._rows.copy_(self._lookup(self._window))
            lib.dec_launch(handle, 1)
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
