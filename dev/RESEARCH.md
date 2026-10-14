# Training techniques on Hopper and Blackwell (September 2026)

Scope: nanochat targets one DGX Spark (GB10, sm_121). Findings that only matter for H100 or multi-node clusters are kept as context but produce no recommendations.

## 1. State of the art

| Area | State of the art | Relevance to nanochat |
|---|---|---|
| Precision | Default everywhere is BF16 compute with FP32 master weights and FP32 gradient reduction. Tensorwise FP8 is the fastest FP8 recipe (+25% at 8B on H100 and on B200, per torchao). MXFP8 is Blackwell-only and slower than tensorwise there. NVFP4 is production only at NVIDIA (Nemotron 3), needs about 15% of layers in BF16 plus extra tricks, and isn't available on H100. | Keep the custom tensorwise FP8 (nanochat/fp8.py), opt-in via `--fp8`. GB10 (sm_12x) lags: in Transformer Engine, MXFP8 has no backward pass there. |
| Attention | FA3 is the best published on H100 (740 TFLOP/s). FA4 is the only kernel family covering sm_90, sm_100 and sm_12x: up to 1,613 TFLOP/s on B200, but on sm_12x it uses older (sm_80-style) instructions. | FA4 is the single attention path. |
| Compiler and kernels | PyTorch 2.14's compiler overlaps compute and communication by default. PyTorch 2.13 adds `nn.LinearCrossEntropyLoss` (up to ~4× less peak memory). | FA4 is registered as `torch.library` custom ops (nanochat/gpt.py), so the model compiles with no graph breaks. `nn.LinearCrossEntropyLoss` is not used: the logit softcap sits between `lm_head` and the loss. |
| Optimizer | Muon is replacing AdamW for hidden matrices (Kimi K2, GLM, DeepSeek-V4, Qwen3.8, Trinity); `torch.optim.Muon` ships in 2.14. Its advantage shrinks with size: 1.4× at 0.1B, 1.1× at 1.2B. | Keep Muon. nanochat/optim.py already uses Polar Express and NorMuon. |
| Parallelism | FSDP2 and Megatron TP/PP only pay off across nodes or for models that don't fit. SmolLM3 found plain data parallelism fastest. | Out of scope on one GPU. For multi-rank runs `MuonAdamW` already synchronizes gradients itself with ZeRO-2-style sharding (no DDP wrapper). |
| Throughput reference | Llama3-8B FP8: 744 TFLOP/s/GPU on H100, 1,576 on B200 (NVIDIA). Dense BF16 clusters reach about 38–43% MFU (Llama 3, OLMo 3). | Treat these as ceilings. |

## 2. What labs and papers report

- Hardware: 2025 runs were on Hopper; 2026 runs are on B200, B300 and GB200. Frameworks split between in-house stacks (DeepSeek, Moonshot, Meta), Megatron plus TE (NVIDIA, Apertus, Qwen3.8), and PyTorch-native FSDP2/torchtitan (OLMo, Arcee).
- FP8 is optional, not standard. Kimi K2, OLMo 3, Apertus and Arcee Trinity Large kept BF16 compute or backed out of FP8. Where FP8 is used, the LM head and the first and last layers stay in BF16.
- Stability: QK-norm is nearly universal and has replaced logit soft-capping (nanochat uses both). The usual schedule is warmup-stable-decay (WSD), which nanochat's warmup/warmdown schedule follows. Most mix sliding-window and full-attention layers (3:1 or 5:1), with the last layer full, the same shape as nanochat's SSSL pattern.
- Software versions are rarely disclosed. One exception: Apertus hit a libnvrtc bug in NGC 25.01, which is why pinning the NGC month and smoke-testing it matters.

## 3. Recommendations for nanochat

Change:

1. Smoke test: the image build only checks that `torch` and `flash_attn.cute` import. Add `uv pip check` to the build, and run one FA4 forward and backward pass in a new image before training on it.

Don't:

- Adopt MXFP8 or NVFP4.
- Switch to FSDP or tensor/pipeline parallelism.
- Install numpy from uv.lock: the image's numba (0.64 in NGC 26.08) breaks silently on numpy 2.5 or later, so docker/Dockerfile prunes numpy and uses the image's.
- Move the image build to `uv sync` or the PyTorch wheel index; both reinstall PyPI torch over NVIDIA's.
- Uninstall NVIDIA's FA2; FA4 imports through its `flash_attn` package.
