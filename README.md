# nanochat

![scaling laws](dev/scaling_laws_jan26.png)

nanochat is a minimal experimental harness for LLM training research on a single GPU node. It covers tokenization, pretraining, supervised finetuning, reinforcement learning, evaluation, and inference in one small, hackable codebase. Models are configured by a single complexity dial: `--depth`, the number of transformer layers. Width, head count, batch size, learning rates, weight decay, and training horizon are all derived from it, so sweeping depth yields a miniseries of compute-optimal models (GPT-2 capability is around d24–d26).

## Time-to-GPT-2 Leaderboard

The primary benchmark is "time to GPT-2": the wall-clock training time needed on an 8XH100 node to exceed the GPT-2 (1.6B) DCLM CORE score of 0.256525. [runs/speedrun.sh](runs/speedrun.sh) always reflects the reference recipe.

| # | time | val_bpb | CORE | Description | Date | Commit | Contributors |
|---|-------------|---------|------|-------------|------|--------|--------------|
| 0 | 168 hours | - | 0.2565 | Original OpenAI GPT-2 checkpoint | 2019 | - | OpenAI |
| 1 | 3.04 | 0.74833 | 0.2585 | d24 baseline, slightly overtrained | Jan 29 2026 | 348fbb3 | @karpathy |
| 2 | 2.91 | 0.74504 | 0.2578 | d26 slightly undertrained **+fp8** | Feb 2 2026 | a67eba3 | @karpathy |
| 3 | 2.76 | 0.74645 | 0.2602 | bump total batch size to 1M tokens | Feb 5 2026 | 2c062aa | @karpathy |
| 4 | 2.02 | 0.71854 | 0.2571 | change dataset to NVIDIA ClimbMix | Mar 4 2026 | 324e69c | @ddudek @karpathy |
| 5 | 1.80 | 0.71808 | 0.2690 | autoresearch [round 1](https://x.com/karpathy/status/2031135152349524125) | Mar 9 2026 | 6ed7d1d | @karpathy |
| 6 | 1.65 | 0.71800 | 0.2626 | autoresearch round 2 | Mar 14 2026 | a825e63 | @karpathy |

See [dev/LEADERBOARD.md](dev/LEADERBOARD.md) for how time is measured and how to interpret results.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra gpu    # CUDA (A100/H100/etc.)
uv sync --extra cpu    # (or) CPU-only / MPS
source .venv/bin/activate
```

`uv sync --extra gpu --group dev` adds pytest, matplotlib, ipykernel, and python-dotenv.

Artifacts (data shards, tokenizer, checkpoints, eval bundle, task data) live under `$NANOCHAT_BASE_DIR`, default `~/.cache/nanochat`.

## Reference pipeline

[runs/speedrun.sh](runs/speedrun.sh) runs the full pipeline on an 8XH100 node (~1.5 hours):

```bash
python -m nanochat.dataset -n 170                    # ClimbMix shards (+ the val shard)
python -m scripts.tok_train                          # BPE tokenizer, vocab 32768
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16   # CORE, bpb, samples
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft                  # ChatCORE
```

Optional stage: `scripts.chat_rl` (GRPO-style RL on GSM8K, evaluate with `chat_eval -i rl`).

- A100 nodes work, just slower.
- On a single GPU, omit `torchrun`; the scripts switch to gradient accumulation automatically and produce ~identical results.
- With less than 80GB of VRAM, reduce `--device-batch-size` (32 → 16, 8, 4, ...).

## Experiments

For quick iteration (~5 min pretraining runs), train a 12-layer model:

```
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run="d12" \
    --model-tag="d12" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1 \
```

This logs to wandb (run name "d12"), runs the CORE metric only on the last step, and skips sampling and intermediate checkpoints. Metrics to compare runs on:

1. `val_bpb` (validation loss in vocab-size-invariant bits per byte) against `step`, `total_training_time`, and `total_training_flops`.
2. `core_metric` (the DCLM CORE score).
3. VRAM utilization, `train/mfu` (model FLOPS utilization), `train/tok_per_sec` (throughput).

Changes must be principled enough to hold across all depths, not just the one tested. Two sweep drivers check this:

- [runs/miniseries.sh](runs/miniseries.sh): trains d12–d26 at the default data:param ratio and writes a results CSV.
- [runs/scaling_laws.sh](runs/scaling_laws.sh): trains depths 10–20 at fixed FLOP budgets (1e18–1e19) and writes a resumable CSV, analyzed in [dev/scaling_analysis.ipynb](dev/scaling_analysis.ipynb).

Experiment history, including negative results, is in [dev/LOG.md](dev/LOG.md).

## Precision / dtype

nanochat does not use `torch.amp.autocast`. Precision is managed explicitly through a single global `COMPUTE_DTYPE` (defined in `nanochat/common.py`), auto-detected from the hardware:

| Hardware | Default dtype | Why |
|----------|--------------|-----|
| CUDA SM 80+ (A100, H100, ...) | `bfloat16` | Native bf16 tensor cores |
| CUDA SM < 80 (V100, T4, ...) | `float32` | No bf16; fp16 available via `NANOCHAT_DTYPE=float16` (uses GradScaler) |
| CPU / MPS | `float32` | Safe default. On recent macOS, MPS also runs `NANOCHAT_DTYPE=bfloat16` fine (~25% less memory, similar speed) |

Override the default with the `NANOCHAT_DTYPE` environment variable:

```bash
NANOCHAT_DTYPE=float32 python -m scripts.base_eval --eval=sample           # force fp32
NANOCHAT_DTYPE=bfloat16 torchrun --nproc_per_node=8 -m scripts.base_train  # force bf16
```

Model weights are stored in fp32 (for optimizer precision), and the custom `Linear` layer casts them to `COMPUTE_DTYPE` during the forward pass. Embeddings are stored directly in `COMPUTE_DTYPE` to save memory. This gives the mixed-precision benefit of autocast with explicit control over what runs in which precision.

`float16` training automatically enables a `GradScaler` in `base_train.py` and `chat_sft.py`; RL does not support it yet. Inference in fp16 works everywhere. `--fp8` (CUDA only) converts eligible linear layers to FP8 matmuls with tensorwise scaling (`nanochat/fp8.py`).

## File structure

```
.
├── LICENSE
├── README.md
├── dev
│   ├── LEADERBOARD.md              # Time-to-GPT-2 benchmark rules and runs
│   ├── LOG.md                      # Experiment log
│   ├── estimate_gpt3_core.ipynb    # GPT-3 CORE estimate
│   ├── repackage_data_reference.py # Pretraining data shard generation
│   ├── scaling_analysis.ipynb      # Scaling-law fits from runs/scaling_laws.sh
│   └── scaling_laws_jan26.png
├── nanochat
│   ├── __init__.py                 # empty
│   ├── checkpoint_manager.py       # Save/Load model checkpoints
│   ├── common.py                   # Dtype, device, distributed, logging utilities
│   ├── core_eval.py                # Evaluates base model CORE score (DCLM paper)
│   ├── dataloader.py               # Tokenizing distributed data loader (BOS-aligned best-fit packing)
│   ├── dataset.py                  # Download/read utils for pretraining data
│   ├── engine.py                   # Efficient model inference with KV cache and calculator tool
│   ├── execution.py                # Sandboxed Python execution (HumanEval)
│   ├── flash_attention.py          # FA3 with PyTorch SDPA fallback
│   ├── fp8.py                      # Minimal FP8 (tensorwise) training
│   ├── gpt.py                      # The GPT nn.Module Transformer
│   ├── loss_eval.py                # Evaluate bits per byte (instead of loss)
│   ├── optim.py                    # AdamW + Muon optimizer, 1GPU and distributed
│   └── tokenizer.py                # BPE tokenizer (rustbpe train, tiktoken inference)
├── pyproject.toml
├── runs
│   ├── miniseries.sh               # Depth sweep
│   ├── scaling_laws.sh             # Scaling laws experiments
│   └── speedrun.sh                 # Reference GPT-2 speedrun (d24, fp8) + SFT
├── scripts
│   ├── base_eval.py                # Base model: CORE score, bits per byte, samples
│   ├── base_train.py               # Base model: train
│   ├── chat_eval.py                # Chat model: eval tasks, ChatCORE
│   ├── chat_rl.py                  # Chat model: reinforcement learning
│   ├── chat_sft.py                 # Chat model: train SFT
│   └── tok_train.py                # Tokenizer: train it
├── tasks
│   ├── arc.py                      # Multiple choice science questions
│   ├── common.py                   # Task base, TaskMixture, HubDataset
│   ├── gsm8k.py                    # 8K Grade School Math questions
│   ├── humaneval.py                # Misnomer; Simple Python coding task
│   ├── mmlu.py                     # Multiple choice questions, broad topics
│   └── smoltalk.py                 # Conglomerate dataset of SmolTalk from HF
├── tests
│   ├── test_attention_fallback.py  # FA3/SDPA attention fallback
│   ├── test_engine.py              # Inference engine, KV cache
│   ├── test_execution.py           # Sandboxed code execution
│   ├── test_optim.py               # MuonAdamW optimizer (needs GPU)
│   ├── test_tasks.py               # Task slicing, mixtures, HubDataset
│   └── test_tokenizer.py           # BPE round-trips, chat rendering
└── uv.lock
```

## Contributing

The goal of nanochat is to improve the state of the art in micro models that are accessible to work with end to end on budgets of < $1000 dollars. Accessibility is about overall cost but also about cognitive complexity - nanochat is not an exhaustively configurable LLM "framework"; there are no giant configuration objects, model factories, or if-then-else monsters in the code base. It is a single, cohesive, minimal, readable, hackable, maximally-forkable "strong baseline" codebase designed to run start to end and produce a ChatGPT model you can talk to. Currently, the most interesting part personally is speeding up the latency to GPT-2 (i.e. getting a CORE score above 0.256525). Currently this takes ~1.5 hours (down from 3h), but by improving the pretraining stage we can improve this further.

Current AI policy: disclosure. When submitting a PR, please declare any parts that had substantial LLM contribution and that you have not written or that you do not fully understand.

## Acknowledgements

- The name (nanochat) derives from my earlier project [nanoGPT](https://github.com/karpathy/nanoGPT), which only covered pretraining.
- nanochat is also inspired by [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt), which gamified the nanoGPT repo with clear metrics and a leaderboard, and borrows a lot of its ideas and some implementation for pretraining.
- Thank you to [HuggingFace](https://huggingface.co/) for fineweb and smoltalk.
- Thank you [Lambda](https://lambda.ai/service/gpu-cloud) for the compute used in developing this project.
- Thank you to chief LLM whisperer 🧙‍♂️ Alec Radford for advice/guidance.
- Thank you to the repo czar Sofie [@svlandeg](https://github.com/svlandeg) for help with managing issues, pull requests and discussions of nanochat.

## Cite

If you find nanochat helpful in your research cite simply as:

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## License

MIT
