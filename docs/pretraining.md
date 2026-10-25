# Pre-training a code base model on one GH200

A Python and C++ code-generation base model pre-trained on one NVIDIA Hopper GPU (sm_90: the GH200's, or an H100 80 GB
SXM5) at the most tokens its data holds without any stage reading a document twice. The code is the specification;
this document states each decision, its basis (a cited result, a measurement, or the law it is derived from) and the
file that defines it. Two companion documents own their parts, and this one links to them:

- `docs/corpus.md`: the corpus (sources, curation, mixture, FIM, packing, the tokenizer's data).
- `docs/ablations.md`: every ablation (method, shared tests, records, each experiment's arms, rule and results).

Numbers come from the code, from `results/`, or are derived from them [arithmetic]. Measurements whose records were not
kept (the pre-launch measurements of 2026-10-19 to 2026-10-22, mostly on an H100 80GB HBM3 SXM5 at its 700 W cap) are
marked [unverified]. Terminology follows DeepSeek-Coder: the base model, its pre-training, an ablation's baseline.

## Model

`model/modeling.py`: karpathy/nanochat 92d63d4 plus the wins of its experiment log ("LOG", `dev/LOG.md`), with
Qwen3's QK norm and learnable gains. Each layer mixes x0 into the stream and looks up its value embeddings
(`CausalLM.hidden`); its `DecoderLayer` then applies `Attention` and `MLP` to the RMS-normed stream, each added back.

| | d24 (the base model) |
|---|---|
| Shape | 24 layers × 768 (`model_config`: width 30 × num_hidden_layers rounded up to whole 128-dim heads; MobileLLM-350M's 32 × 960, arXiv 2402.14905 Table 9), 6 heads × 128, MHA |
| Attention | per-head RMS norm → learned per-dim gain → RoPE over the full head, base 10^6 (`model/config.py` `ROPE_THETA`); SSSL windows (S = 512), last layer full; FlashAttention-4 varlen with document masking |
| MLP | ReLU², hidden 4C |
| Residual | `x = resid_lambda·x + x0_lambda·x0` per layer; value embeddings on alternating layers (12 of 24) behind a per-head `2·sigmoid` gate of the normed stream, zero init (gate 1): modded-nanogpt record #14's per-layer tables, MoVE's LaVE (arXiv 2601.22887 §4.1 Eq. 6) |
| Head | untied `lm_head`, zero init (modded-nanogpt record #8), plain cross entropy, no z-loss; an output-logit softcap only as the D5 ablation's arm (`ModelConfig.final_logit_softcapping`) |
| Vocabulary | 2^15 ids (`model/config.py` `VOCAB_SIZE`; BOS and 3 FIM sentinels; `docs/ablations.md`, tokenizer_split_merge) |
| Parameters | 195,096,576 non-embedding parameters (`layers` + `lm_head`, `num_non_embedding_params`), what the token budget counts; 522,252,336 in all with the token table and 12 value-embedding tables (25,165,824 each) |
| FLOPs per token | 2.831 GFLOP non-embedding: DeepSeek LLM's M (arXiv 2401.02954 §3.2 Eq. 2), 72·n_layer·d² + 12·n_layer·d·l_seq at the full 8K context (`non_embedding_flops_per_token`) |
| Init | stream-reading matrices and tables uniform with std C^-½; `o_proj`, `down_proj` (modded-nanogpt record #5), `lm_head`, value-embedding gates zero; resid lambdas and gains 1, x0 lambdas 0 |
| Precision | fp32 parameters, gradients and optimizer state; bf16 activations and GEMM inputs |
| Lineage | From nanochat 92d63d4: the block layout, value-embedding layers, MLP and window pattern. Not taken: its smear, backout and logit softcap, QK order (RoPE → norm → ×1.2), short window (a quarter of the context), RoPE base (10^5) and init. The gate is nanochat e85db6b's (2·sigmoid, zero init) read over the full width rather than its first 32 channels; the full-width read is not ablated |

### Basis

- **Value embeddings, kept** (decided from the literature): MoVE's LaVE gains 0.016, 0.010 and 0.010 at D12, D20
  and D32 (arXiv 2601.22887 Table 1), ≈ −0.014 extrapolated to d24 in MoVE's coordinates [inference], against 4.7% of
  the d24 step [unverified]. No reference code model has them.
- **Learnable QK gains, kept** (D4): fixing the normalized q/k scale costs 0.11% of loss (nGPT, arXiv 2410.01131
  App. A.9 Table 5); per-head QK norm with learnable parameters in arXiv 2309.14322 §3.2.5.
- **Published, not tested here** (no reference code model ablates such details):
  - *ReLU² rather than SwiGLU:* nanochat measured SwiGLU worse at d12 and d24 under Muon (LOG 2026-02-05); a gated
    MLP would also need a gated decode MLP (`model/kernels/decode.cu`).
  - *SSSL windows:* Gemma 3 (arXiv 2503.19786 §5.2) measures local:global up to 7:1 and windows down to 512 nearly
    lossless in perplexity; StarCoder2 trains code under a 4,096-token window.
  - *Shape:* MobileLLM's deep-and-thin shape at d24's scaling parameters (35 × 640) takes 9.2% more FLOPs per token
    [unverified], so an equal-token comparison would give it more compute.
  - *head_dim 128:* compiled into the decoder's kernels (`model/config.py` `HEAD_DIM`); nanochat measured 128 better
    than 64 (LOG 2026-01-17).
  - *MHA, not grouped-query attention:* GQA's benefit is KV-cache memory; d24's cache for 256 of the decoder's
    slots (`model/kernels/decode.cu` `MAX_BATCH` 1,024, as many as fit memory) of the longest benchmark request
    (391 prompt + 768 new tokens) is 21.9 GB [arithmetic: 2 · 24 · 768 · 2 bytes · 1,158 · 256], which the GPU holds.
  - *RoPE base 10^6 from the first step* (D9): above arXiv 2405.14591's lower bound for a 32K context (Table 2:
    6.4e5), large from the start as Llama 3 (arXiv 2407.21783 §3.2); Barbero et al. (arXiv 2410.06205 Table 2) find
    10k against 500k at 8K within 0.3%, of mixed sign.
- **Not adopted:** Block Attention Residuals (a published 1.1% loss gain at 194M parameters, arXiv 2603.15031; +12% d24
  step time [unverified]), the partial key offset (modded-nanogpt record #49), either of which would need a native
  decode path; multi-token prediction, reported to hurt models of this size on code benchmarks (arXiv 2404.19737).

### Against the reference code models

| Choice | The base model | DeepSeek-Coder / Qwen2.5-Coder / StarCoder2 / Code Llama |
|---|---|---|
| Attention | SSSL: 3 of 4 layers a 512-token window | full attention (DeepSeek-Coder, Qwen2.5-Coder, Code Llama); StarCoder2 a 4,096-token window in a 16,384 context |
| MLP | ReLU², hidden 4C | SwiGLU (DeepSeek-Coder, Qwen2.5-Coder, Code Llama); StarCoder2 GELU |
| Value embeddings | alternating layers, gated | none |
| Optimizer | Muon on matrices, AdamW on tables and vectors | AdamW, β = (0.9, 0.95) (DeepSeek-Coder §3.4, Code Llama) |
| Peak lr | DeepSeek LLM's law × `ablations/peak_lr.py`'s multiplier, the peak of linear decay to zero | the law's lr held for 80% of tokens, then 31.6% and 10% of it (DeepSeek LLM §2.3, the schedule DeepSeek-Coder §3.4 follows) |
| Weight decay | τ law (Regime) | 0.1 (DeepSeek LLM §2.3) |
| Gradient clipping | 1.0 | 1.0 (DeepSeek LLM §2.3) |
| Context stages | 8K file level, then the last 5.45% of steps at 32K on repositories | Qwen2.5-Coder: the same 8K → 32K repository stage; DeepSeek-Coder: 16K repository-ordered files; Code Llama: a 16K long-context stage |
| Data | code:text:math 87:10:3 | DeepSeek-Coder 87% code, 10% code-related English, 3% Chinese; Qwen2.5-Coder code:text:math 70:20:10 |
| FIM | rate 0.5 per piece of the repository corpora's train documents, PSM/SPM 50/50, AST-FIM middles (`pack.FIM_RATE`, `SPM_RATE`; `docs/corpus.md` §8) | DeepSeek-Coder 0.5, PSM only (arXiv 2401.14196 §3.1.2); StarCoder2 ≈25% of chunks; Code Llama 90% |

What those models ablate: Code Llama the RoPE base of its long-context stage (Appendix G.4, Table 18) and FIM's cost
to left-to-right completion; DeepSeek-Coder the FIM rate (Fig. 3); StarCoder2 RoPE against learned positions; OpenCoder
(arXiv 2411.04905) its data. None ablates architecture details such as gains, scalars or attention windows.

## Data and stages

The corpus, its mixture and its two stages are specified in `docs/corpus.md`; the code is `corpus/`
(`mixture.py` the stages and shares, `pack.py` the rows and FIM, `loader.py` the micro-batches). In short:
code:text:math 87:10:3 in both stages; an 8,192-token file stage, then a 32,768-token repository stage over the last
`mixture.REPO_LEVEL_SHARE` (5.45%) of steps (Qwen2.5-Coder, arXiv 2409.12186 §3.2); 102.6B tokens in all
(`mixture.MAX_STEPS` × `BATCH_SIZE`), what the curated corpus holds without any stage reading a document twice. A run's
`tokens_per_param` defaults to its data's `train/stages.py` `max_tokens_per_param`. The tokenizer is decided by
`docs/ablations.md`, tokenizer_split_merge.

## Regime

| Setting | Value | Source |
|---|---|---|
| Batch | 1,474,560 tokens (15 micro-batches of 98,304) at 102.6B tokens | DeepSeek LLM's law B = 0.2920·C^0.3271, C = M·D (`train/schedule.py` `plan`), rounded to whole micro-batches |
| Steps | 69,580 (525.89 tokens per non-embedding parameter); repository stage from step 65,785 | `schedule.plan` (D11: `round(tokens_per_param · num_non_embedding_params)`) |
| Peak lr | the law's η = 0.3118·C^-0.125 = 8.6295e-4 times `ablations/peak_lr.py`'s multiplier: × 2.7972 = 2.4139e-3 at 102.6B tokens | DeepSeek LLM §3.1 Eq. 1; `conf/experiment/pretrain.yaml`'s `learning_rate_scale` resolves `peak_lr.py`'s `analysis` at the corpus's `max_tokens_per_param` (`docs/ablations.md`, peak_lr) |
| Warmup | 40 steps, linear | 2/(1 − β₂) (arXiv 1910.04209, for Adam); nanochat 92d63d4's 40 steps at d24 |
| Decay | linear to zero; the last step at 1/(69,580 − 40) = 1.44e-5 of peak (D10: exactly 0 is not required) | arXiv 2502.15938 §4 |
| Optimizers | AdamW (fused) on tables, `lm_head` and vectors; Muon (`torch.optim.Muon`'s update, Newton–Schulz batched by shape) with `match_rms_adamw` on the other matrices, one lr for every group | Moonlight, arXiv 2502.16982 §2.2 (`train/optim.py`) |
| AdamW | β = (0.9, 0.95) | DeepSeek LLM §2.3 |
| Weight decay | λ = 1/(η τ T), the EMA timescale τ = 1.084·TPP^−0.527 of the T steps (λ = 0.1492 at the base model's peak lr), on tables, `lm_head` and Muon matrices, 0 on vectors | Power Lines, arXiv 2505.13738 App. E.3.1 (`optim.tau_weight_decay`) |
| Gradient clipping | global norm 1.0 on the accumulated gradient, before both optimizers; the norm is logged every step | DeepSeek LLM §2.3 (D3; `optim.MAX_GRAD_NORM`, `update`) |
| Repository stage | continues the decay: 5.46% of peak at its first step, 0 at the last, over 3,795 steps | D2: decay to zero (arXiv 2502.15938); DeepSeek-Coder §3.6 and DeepSeek-V3 instead hold their final lr |

Batch, steps, lr and weight decay at 102.6B tokens are computed from the code (`schedule.plan` at d24); the run's own
values come from `max_tokens_per_param` on the adopted tokenizer's packs.

**Gradient clipping** at 1.0 (D3). Adam's update is invariant to the gradient's scale and Muon's Newton–Schulz divides
its input by its norm, so under a slowly varying scale the clip damps only spikes. The failure Megatron-LM PR 5395
describes for a pre-Muon clip needs gradient norms far above any recorded here; DeepSpeed issue 8439 clips after
orthogonalization, which this code does not.

**Warmup.** 40 steps, the value nanochat 92d63d4 uses at d24 and 2/(1 − β₂) for AdamW's β₂ (arXiv 1910.04209); the
lr ablation's 11 runs (`results/peak_lr.jsonl`, up to × 8 at d8) show no lr-dependent warmup instability. No paper fixes a
warmup for Muon.

## Runs

Every run, an ablation's or the base model's, is a `run.py` `RunConfig` (Hydra, hydra-core 1.3.7: `RunConfig`'s
defaults are the base model's recipe, `conf/experiment/<name>.yaml` sweeps fields against them), trained by
`train/loop.py` and evaluated by `eval/suite.py evaluate`: `python run.py experiment=<name>`. `pretrain.yaml` (d24 at
`max_tokens_per_param`, seed 0, every benchmark) waits for every ablation; the ablations, their order and records:
`docs/ablations.md`.

A run's name (`run.run_name`) is its `RunConfig` fields that differ from the defaults, sorted, `k=v` joined by `,`
(e.g. `num_hidden_layers=16,seed=1,tokenizer=sa-bpe`; `defaults` if none). Per run, in `artifacts/runs/<run_name>/`:
`.hydra/` (the composed config), `metrics.jsonl` (`metrics.py`: the run record holds the model config, the
`RunConfig` as `config`, the tokenizer's `data_key`, `num_parameters`, DeepSeek's M and PaLM's nominal model FLOPs
per token, `max_grad_norm`, the plan, `learning_rate` and weight decay per group; a train record per step its loss,
learning rate, gradient norm, step time, tokens per second, executed model FLOPs and MFU; the stage switch's and the
end record each stage's peak allocated and reserved memory; a failure record the step a run diverged at),
`checkpoint-<N>.pt`, `eval-<N>.json` and `run.log`. A rerun resumes: a run with a final evaluation or a divergence
is skipped, a stopped one resumes from its newest checkpoint, and a run whose recorded plan or data key differs stops
the sweep ("recipe changed") until its directory is deleted. Training and each checkpoint's evaluation run in fresh
spawned processes.

**Model FLOPs** (`metrics.py`): per PaLM (arXiv 2204.02311 App. B), 6 × the matmul parameters + 12 · hidden size per
attended key per layer. *Nominal* (`model_flops_per_token`): every key a query can attend, min(window, context), as
torchtitan counts. *Executed* (`executed_model_flops`, the MFU numerator): the keys each query attends within its
segment and window, causally (Megatron-LM's causal and per-document counting). DeepSeek LLM's M stays the scaling
plan's compute: it counts full attention in every layer, so its utilization exceeds 100% at 32K.

## GPU implementation

Ops in `model/ops/`, CUDA sources in `model/kernels/` (built on first use by `native.cuda_library`). Each is a
`torch.library` op, so `torch.compile(fullgraph=True)` traces one graph per stage.

| Op | Implementation |
|---|---|
| Projection, MLP, LM-head GEMMs | forward and dX: QuACK SM90 GEMMs autotuned by QuACK (`quack.gemm_interface`, `tuned=True`); fp32 weight gradients: cuBLAS `torch.mm` |
| ReLU² MLP | QuACK `gemm_act` / `gemm_dact` |
| Cross entropy | QuACK, bf16 logits read once, the logit gradient written over them; the softcap arm's capped loss fused in the same op (`model/ops/cross_entropy.py`) |
| Attention | FlashAttention-4 varlen over the micro-batch's real document segments only (`model/ops/flash_attn.py`), its default backward |
| q/k norm + gain + RoPE | `model/kernels/qk_prologue.cu`: one pass per direction; gain gradient without atomics (row math shared with the decoder, `model/kernels/qk_row.cuh`) |
| Value-embedding gate and mix | `model/kernels/ve_gate.cu` (`model/ops/ve_gate.py`): one launch forward, one pass backward, the gate's logits in fp32 |
| Muon | Newton–Schulz batched over same-shape matrices (`optim.Muon`) |
| Embeddings | `F.embedding`, bf16 cast fused by Inductor; backward adds rows with atomics |
| Decoding | `model/kernels/decode.cu` (`inference/decoder.py`): batched native decoder for the code benchmarks |

Measured choices [unverified: `torch.profiler` and CUDA-graph timings, records not kept]:
- **cuBLAS for the fp32 weight gradients:** dYᵀX has a small output (768 × 768 or 768 × 3,072) and K = 98,304; QuACK's
  split-K-1 tiles left most SMs idle (c_q/c_k/c_v/c_o dW 0.544 against 0.191 ms, relative error vs fp64 2.3e-4 against
  1.8e-5).
- **`model/kernels/qk_prologue.cu`** rather than Inductor's fusion of the PyTorch ops: the d24 step 3,792 against
  3,890 ms; and micro-batches of 98,304 tokens rather than 65,536 (1% faster per token).
- **FlashAttention-4** rather than cuDNN 9.26: forward + backward 1.47 against 4.96 ms at window 512, 1.83 against 5.85
  ms at 8,192 (d8 shapes); its full-context layers use the causal kernel (`window = None`), 1–2% faster than the local
  kernel at window T, outputs bitwise equal. FA4 on the real segments rather than the loader's padded `cu_seqlens`:
  losses bitwise, gradients within run-to-run spread.
- **Split q/k/v projections** rather than one GEMM on concatenated weights: fusion saved no measurable GEMM time and
  its layout needs copies.
- **Ours over QuACK's chunked linear cross entropy** for the LM-head loss (29.2 against ≥ 31.4 ms at d8); per-position
  evaluation losses by QuACK's `lse_target_epi` (no logits).
- **The value-embedding gate kernel** is not bitwise to the PyTorch expressions it replaced (its fp32 reduction order
  moves about 15% of elements by one bf16 ulp).
- **Rejected:** FP8 MLP GEMMs (e4m3 rowwise, d16 on real rows: mean loss over steps 250–299 +0.0048 over bf16 against
  a bf16 two-run spread of 0.0014, growing); two concurrent runs (−8% aggregate; one run saturates the GPU); Liger and
  Cut Cross-Entropy losses, fused RMSNorm/RoPE libraries, DCP checkpointing.

**The d24 step** [unverified]: on the H100 SXM5, 12 micro-batches, 3,741 ms (GEMM 53%, FA4 25%, Inductor 14%; peak
69.6 GiB); on the GH200 with the SM clock locked at 1,980 MHz and items above applied, about 3,490 ms for the same 12
micro-batches. The GEMMs run at the 700 W power cap's ceiling (703 of 989 dense bf16 TFLOP/s).

## Reproducibility

- **Fastest kernels.** FA4's and the embedding backward reduce with atomics, so repeated steps differ in the last bits;
  fixed-order reductions would cost about 1% of a step [unverified]. Python's garbage collector stays off after setup
  and collects only after a checkpoint save (`train/loop.py`). The forward is deterministic: Inductor's
  coordinate-descent tuning is off (`scripts/container.sh`), since it picks kernel configurations by timing, so two
  processes would compile different kernels.
- **Resume.** A relaunched run continues from its newest checkpoint (weights, optimizer state, data position), written
  to a temporary file and renamed, so a kill mid-save leaves the previous one.
- **Environment.** `docker/Dockerfile` pins base images by digest and git dependencies by commit; `uv.lock` the Python
  packages; fp32 matmuls stay fp32 (`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`).
- **Run identity.** The run's key and its recorded configuration (Runs); a run recorded under a different plan or
  data key stops its experiment.
- **Data order.** Each corpus's packed rows are permuted once by the run's seed (NumPy `RandomState`; `train/stages.py`
  `stage_batches`), which also seeds its initialization; rows go to corpora by the loader's Weyl sequence
  (`mixture.mixture_rows`).

## Reproducing the run

The pipeline splits at the tokenizer. Everything through curation is text and runs on the GB10; the curated shards go
to the private Hub dataset `iaroy/nanochat-data` (`corpus/publish.py` `DATASET`). The tokenizer ablation and decision,
tokenize, pack and training run on the GH200 from a pull of that dataset. Every command runs in the `nanochat` image
(`docker build -f docker/Dockerfile -t nanochat .`) from the repository's root: through `scripts/container.sh` on the
GH200, or on the GB10 as

```
docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -e PYTHONPATH=/workspace/nanochat -w /workspace/nanochat \
  -v $PWD:/workspace/nanochat nanochat python -m corpus.<step> ...
```

(steps that read the Hub add `-e HF_HOME=/tmp/hf -e HF_HUB_DISABLE_XET=1 -e HF_TOKEN=$(hf auth token | tail -1)`).

On the GB10, once (the dataset is constant afterwards):

1. **Prepare**: `python -m corpus.prepare` fetches and standardizes every source (`docs/corpus.md` §1–2);
   fetched files are kept, so a rerun downloads only what it lacks; a source read in part fetches its extent
   (`prepare.extent`).
2. **Curate**: `python -m corpus.curate`: the Python 3 check with repair, exact deduplication, the split, the
   curated shards (`CORPUS/curated/<source>/<split>/<k>.parquet`; `docs/corpus.md` §3).
3. **Publish**: `python -m corpus.publish push` writes the card, replaces the dataset's tree and squashes its
   history into one commit (the card turns the Hub's viewer off).

On a fresh Lambda `gpu_1x_gh200` (Ubuntu with Docker and the NVIDIA driver; the H100 SXM5 runs the same image):

```bash
git clone https://github.com/indrajeetadityaroy9/nanochat nanochat && cd nanochat
HF_TOKEN=<read token> scripts/setup.sh   # CDI spec, clock lock, docker group, image, artifacts/, the curated corpus
```

then, after logging in again (docker group), each through `scripts/container.sh`:

1. **Tokenizer ablation, intrinsic part and data** (`docs/ablations.md`, tokenizer_split_merge): `python -m
   ablations.tokenizer_split_merge vocabularies`, `intrinsic`, `data`. An arm's extent margin
   (`results/tokenizer_split_merge-intrinsic.json`) below 1 means a group
   read in part is too small for it: raise `mixture.SPREAD`, then prepare, curate and publish again on the GB10.
2. **Recipe ablations and tokenizer decision** (the user's approval): `python run.py experiment=<name>` for
   `muon_wd_timescale_20tpp`, `logit_softcap`, `muon_wd_timescale_200tpp`, `tokenizer_split_merge`, `adamw_group_lr` (records in `results/<name>.*`; runs in
   `artifacts/runs/<run_name>/`).
3. **Adopt the tokenizer**: `python -m ablations.tokenizer_split_merge adopt` writes the decided vocabulary to `CORPUS/tokenizer`
   (over any provisional one); every later step reads it.
4. **Tokenize**: `python -m corpus.tokenize`, ending with the 32K stage's repository partitions
   (`tokenize.tokenize_repo_level_stage`).
5. **Pack**: `python -m corpus.pack`; it stops naming any source that holds fewer rows than its draw.
6. **Smoke**: the first micro-batches of both stages (`loader.PretrainingBatches` at the stages' weights), decoded;
   each step's group composition and FIM share.
7. **Train** (the user's approval): `tmux new -d -s run 'scripts/container.sh python run.py experiment=pretrain'`.

Everything a run reads or writes lives in the checkout (`paths.py`): `artifacts/` (gitignored, created by
`scripts/setup.sh`) holds `corpus/`, `runs/<run_name>/` and `cache/`; `results/` (tracked) holds the data the paper's
tables and figures are drawn from.

`python run.py experiment=pretrain` (`conf/experiment/pretrain.yaml`) trains both stages with a checkpoint (the run's
full state) at each tenth of the steps and at the repository stage's start, evaluates every checkpoint with every
benchmark, and writes `results/pretrain.jsonl` and `results/pretrain-curves.jsonl`. **Expected time** [inference from the
unverified GH200 step]: 69,580 steps of 15 micro-batches at about 1.25 × 3.49 s, about 84 h, plus evaluation.

`scripts/release.sh <run_name> <repo>` uploads the run's checkpoints, records and `results/` to a Hugging Face model
repository; `scripts/container.sh hf download <repo> --local-dir /workspace/nanochat` restores them where `load_model`
and a resumed run read them.

### Hardware

The GH200 (one H100 GPU with 96 GB HBM3, 64 Neoverse-V2 cores, 525 GB of memory) is the host of the ablations and
the base model's pre-training; the H100 SXM5 (80 GB) is the measured fallback. What the code depends on:

- *Power.* The GPU's enforced limit is 700 W, set outside the VM (`nvidia-smi -pl 900` is accepted but not enforced),
  and the GH200 module (Grace + Hopper + memory) shares a 1,000 W budget with the CPU [unverified], so no heavy CPU job
  runs beside a measured run. `scripts/setup.sh` locks the SM clock at 1,980 MHz (speed only, not numerics).
- *Memory.* `corpus/mixture.py` `MICRO_BATCH_TOKENS` (98,304: 12 rows at 8K, 3 at 32K) is the most whole rows of both
  contexts whose d24 step fits 80 GB (peak 69.6 GiB [unverified]); on 96 GB it stays, as a larger micro-batch would
  change every run's batch rounding (`schedule.plan`).
- *Kernels.* Both GPUs are sm_90; QuACK's autotuner picks GEMM tiles by timing, per host, so steps need not be bitwise
  across hosts [inference].
- *ARM64.* `docker/Dockerfile` pins multi-arch index digests; the data order does not depend on the CPU [inference].
  `char` is unsigned on ARM64 and signed on x86, so a MultiPL-E C++ program that depends on it can score differently
  [inference].
- *Evaluation is where the CPU counts:* its pools are sized by `os.cpu_count()`.

## Pre-launch verification

**The model against its specification** [unverified: 2026-10-20 at `7faf11f`, before D1 and the kernels above; records
not kept]. Each op and the whole model against fp64 restatements written from the Model table, forward and backward,
on the GB10 (sm_121) and an H100 (SM90); relative error:

| Check | GB10 | H100 |
|---|---|---|
| GEMMs and ReLU² (fp32 weight gradients) | 3.7e-3 (4.4e-6) | 3.7e-3 (4.4e-6) |
| q/k prologue (gain gradients) | 1.7e-3 (1.5e-7) | 1.7e-3 (2.0e-7, bitwise repeatable) |
| FA4 output and gradients | 3.6e-3 | 3.6e-3 |
| FA4 window boundary: worst row, against either off-by-one | 2.7e-3 against 1.0–2.3 | 2.7e-3 against 1.06–2.20 |
| FA4 without the document mask | 4.1 | 4.2 |
| Cross entropy | 1.7e-3 | 1.7e-3 |
| Whole model (4 × 256, every component perturbed to matter): per-position loss | 5.0e-3 | 5.2e-3 mean, 2.5e-2 max |
| Restatements each breaking one rule (no x0, σ gate, no value embeddings, RoPE +θ, no document mask) | 5.1e-2–1.5e-1 | 4.4e-2–1.5e-1 |
| Gradients | 1.7e-2 | 2.1e-2 |
| Native decoder: greedy tokens the forward's argmax, the rest within | 499 of 512, 0.014 of its maximum logit | 502 of 512, 0.015 |

## Evaluation

Every metric is a published one, computed as its source computes it. `eval/suite.py evaluate` scores a checkpoint:
the base model's checkpoints take everything (`generate="benchmarks"`), the ablation runs the likelihood metrics and
the parse measure (`generate="parse"`). The GPU work runs back to back (the unconstrained decodes, the likelihood
metrics, then the filtered searches) while each benchmark's scoring, CPU work in its official evaluator, runs on a
scorer thread as soon as its completions exist; the generation code takes an engine (requests in, token ids out;
`NativeDecoder` here), so another decoder can replace it.

**Likelihood metrics** (`eval/bpb.py`, by `metrics.bpb`), in bits per byte (The Pile, arXiv 2101.00027: summed bits
over summed UTF-8 bytes, tokenizer-independent):

- *Held-out corpora:* each training corpus's val split at 8K and at 32K, per context bucket (≤ 8K, > 8K: Code Llama's
  perplexity by context length, arXiv 2308.12950 §3.3, Fig. 4a) and per byte-offset bucket (doublings from 1 KiB to
  128 KiB of the piece before a target, so tokenizers compare at the same bytes of context, arXiv 2503.13423 §3.1,
  App. B.1), pooled by the stage's mixture weights: the AR loss of arXiv 2207.14255 §2. Targets of no bytes (BOS,
  padding, sentinels) are not scored, as karpathy/nanochat's `evaluate_bpb`. A corpus's sample is its rows up to
  80 × 2^19 tokens' worth (`bpb.VAL_TOKENS`), systematic.
- *Reference solutions:* the bits of HumanEval+'s, MBPP+'s and HumanEval-X C++'s reference solutions given their
  prompts: the task loss of arXiv 2412.04403 §2.2; a continuous correct-answer likelihood makes HumanEval and MBPP
  decisions predictable where accuracy is trivial (DataDecide, arXiv 2504.11393 §3.4, Fig. 6); BPB raises the
  signal-to-noise ratio of most benchmarks (arXiv 2508.13144 §5.3); the loss on a benchmark's problems and reference
  solutions tracks pass@1 where held-out loss does not (arXiv 2604.07769 §3.3.1). EvalPlus's solutions begin with blank
  lines, which a tokenizer joins to the prompt's last line break, so those blank lines count as prompt.

**Parse measure** (`code_eval.parse_generation`): one greedy, unconstrained completion of each HumanEval+ and MBPP+
prompt with BOS and the prompt healed at its boundary, as arXiv 2608.18062 generates (§4.3; App. C: the trailing
prompt token dropped and generation constrained until its bytes are re-emitted); the share whose prompt + completion
compiles, fails with an IndentationError (TabError included) or with another SyntaxError (the companion post's
classification; a parser overflow counts as a SyntaxError), and the completions' EvalPlus pass@1. It departs from
TokEval's protocol in its prompts (EvalPlus's 0-shot base prompts; TokEval: lm-eval-harness, MBPP 3-shot), its budget
and stops (EvalPlus's 768 tokens and stop strings; the post: 256 tokens), and compiles the raw completion.

**Benchmarks** (`eval/code_eval.py`): EvalPlus (arXiv 2305.01210) HumanEval+ and MBPP+ (base and plus pass@1),
MultiPL-E (arXiv 2208.08227) HumanEval and MBPP in C++, and HumanEval single-line, multi-line and random-span
infilling (arXiv 2207.14255; PSM and SPM), each scored by its official evaluator, by up to three generation modes:

| Mode | Decoding | Score |
|---|---|---|
| `greedy` | left-to-right argmax, one completion per problem | pass@1 as DeepSeek-Coder (arXiv 2401.14196 §4.1), StarCoder2 (arXiv 2402.19173 §7.1.1) and EvalPlus's leaderboard report it |
| `sampled` (EvalPlus, MultiPL-E) | 20 samples per problem at temperature 0.2 from the nucleus (arXiv 1904.09751) top-p 0.95 (`code_eval.N_SAMPLE`, `TEMPERATURE`, `TOP_P`) | pass@1 by Codex's unbiased estimator (arXiv 2107.03374 §2.1, §3.2–3.3), as MultiPL-E reports it |
| `filtered` | left-to-right prompts healed (arXiv 2608.18062 App. C; token healing, Guidance 2023), Python and FIM middles under the syntax constraint; the greedy candidate, then seeded samples until one has no post-generation diagnostic and passes the prompt's own examples, at most 10 (`verify.ATTEMPTS`); the accepted candidate, else the greedy one | filtered pass@1 (arXiv 2107.03374 §3.5; AlphaCode, arXiv 2203.07814 §4.5): a system measurement, not comparable to pass@1 tables |

- **Comparability.** The sampled mode draws 20 samples per problem where Codex and MultiPL-E drew 200 and the BigCode
  leaderboard 50: the same estimate, with more variance per problem. HumanEval infilling is greedy, not arXiv
  2207.14255's sampling. The prompts' examples overlap the scored base tests, so the filtered mode favours base over
  plus pass@1. EvalPlus's verdicts are timed, so heavy CPU load can fail a slow correct solution.
- **Infilling** (human-eval-infilling): the middle is scored as generated, up to BOS or the budget (the package's
  protocol); `greedy_reference_lines` also scores the greedy single- and multi-line middles cut to the reference
  middle's line count (arXiv 2505.18789 Table 1, Fig. 4), a separate metric since it reads the reference. Both orders
  are scored greedy; the filtered mode runs SPM, its middles ending at BOS.
- **MultiPL-E data:** the dataset revision MultiPL-E's own generator pins (`multipl_e/completions.py`
  `DATASET_REVISION`).
- **Contamination.** The training corpora were not decontaminated against the benchmarks: a benchmark problem in the
  training data inflates its score.

**Decoder** (`inference/decoder.py`, `model/kernels/decode.cu`): batched continuous decoding of up to 1,024 slots,
each step one captured CUDA graph (a narrow copy over the first 256 slots while no higher slot is active). A new
request's prompt is prefilled: its positions run as rows of one batched pass through the step's body (up to 1,024
rows of any prompts per launch, each row computed alone as a step would), so its slot starts at the prompt's last
token; a prompt's K and V are kept in a paged prompt store (pages of 64 positions) and a later request with the same
prompt copies them instead. Every sum's order depends on the shapes and the slot's position alone, so a request's
tokens do not depend on the batch, the store or timing. `memory_fraction` caps the decoder's device memory (slots,
prefill rows, graphs, store) at a fraction of the device's total (the evaluation reads
`NANOCHAT_DECODER_MEMORY_FRACTION`; unset: all free memory), so a job can share the GPU. A `Request` carries the
prompt's `input_ids`, `max_new_tokens`, `stopping_criteria`,
`temperature` (0 = argmax, ties to the lowest id), `top_p` (the nucleus, arXiv 1904.09751 §3.1, as Hugging Face's
`TopPLogitsWarper` keeps it), `seed` (the sample at position *i* depends only on seed and *i*) and an optional
constraint (`heal_tokens()`'s `Reemit`, or the Python syntax constraint, which owes the healed bytes itself), whose
masks are written while the step's body runs: the syntax constraints' in one parallel call straight into the
decoder's pinned mask rows, each constraint's last token applied in the same call.

**vLLM and transformers** (`inference/export_hf.py`, `inference/vllm_model.py`, `inference/modeling_nanochat.py`,
`docker/Dockerfile.vllm`): `python -m inference.export_hf <checkpoint.pt> <dir> [<tokenizer dir>]` (nanochat image)
writes a Hugging Face model directory: `model.safetensors` (matrices and tables bf16, the per-dim and per-layer vectors
and value-embedding gates fp32), `config.json` (the per-layer windows as Gemma 3's `layer_types` + `sliding_window`,
FA4's `window_left` + 1), `generation_config.json` (BOS ends a generation) and `tokenizer.json` (transformers'
`TikTokenConverter`: the same ids as tiktoken, BOS prepended). The `nanochat-vllm` image (vLLM 0.32.0 on its own
torch 2.13, which the NGC-based nanochat image cannot hold) registers `NanochatForCausalLM` through vLLM's
`vllm.general_plugins` entry point, so `vllm serve <dir>` and `LLM(<dir>)` run the model with vLLM's layers (paged
attention with per-layer sliding windows and the hybrid KV cache, CUDA graphs, torch.compile); `model_impl=
"transformers"` (vLLM's Transformers backend) or transformers' `AutoModelForCausalLM.from_pretrained(<dir>,
trust_remote_code=True)` run `modeling_nanochat.py`, copied into the directory. No architectural change was needed;
the model's own pieces are PyTorch expressions at the training kernels' precision: the rotation by −θ with fp32
tables (vLLM's and transformers' rotary embeddings rotate by +θ from bf16 tables), the value embeddings (they read
`input_ids` in every layer) and the residual mix. Checked on a d16 checkpoint: `tokenizer.json` encodes 108,026
benchmark and corpus val texts exactly as tiktoken; vLLM's logits against the training forward on 207K real positions
(benchmarks, FIM, val documents to 32K): top-1 agreement 97.9–99.3%, mean KL 2.5–5.8·10⁻⁴ nats (the training forward
against itself at another row offset: 99.4%, 1.8·10⁻⁴), next-token loss within 0.002 nats; greedy HumanEval+
0.067/0.067, MBPP+ 0.056/0.045, FIM single-line line EM PSM 0.022, SPM 0.368, against NativeDecoder's
0.061/0.061, 0.045/0.037, 0.022, 0.367 on the same checkpoint. On the code benchmarks' request sets the native
decoder is faster at d16 for codegen: greedy + sampled (23,100 requests) 100 s against vLLM's 267 s, whose engine is
bound by its per-step host work at this model size (GPU 17–36% busy, `max_num_seqs` 1,024). FIM greedy (16,976
requests) is faster through vLLM, 45 s against 75 s. The native class and the Transformers backend run at the same rate.

**Standard-protocol cross-check** (`scripts/standard_eval.sh <exported dir> <out dir> [generate|score]`): the
harnesses' own CLIs, unmodified, on an export served by vLLM: EvalPlus's `evalplus.codegen --backend vllm --greedy` on
HumanEval+ and MBPP+ (direct completion; the evalplus commit `pyproject.toml` pins) scored by `evalplus.evaluate`, and
MultiPL-E's `automodel_vllm.py` on humaneval-cpp and mbpp-cpp (T = 0.2, top-p 0.95, 20 completions per problem; the
commit `docker/Dockerfile` pins) judged by its `evaluation/src/main.py` and `pass_k.py`. To run it on a checkpoint:
`scripts/container.sh python -m inference.export_hf <checkpoint.pt> <dir>`, then `scripts/standard_eval.sh <dir> <out>`
(the `generate` phase needs the GPU, in the `nanochat-vllm` image; the `score` phase is CPU work in the nanochat
image). Results: `<out>/evalplus/<dataset>/*_eval_results.json`, `<out>/multipl-e/`, `<out>/pass_k.csv`. It sets this
suite's numbers beside the protocols as published, which differ from ours in their prompts' tokenization (text through
`tokenizer.json`), MultiPL-E's 2,048-token context and their samplers. On the d16 checkpoint above, EvalPlus's CLI gives
HumanEval+ 0.067/0.067 and MBPP+ 0.053/0.042 (base/plus), this suite's greedy mode 0.061/0.061 and 0.045/0.037.

**Constrained decoding: syntax, not types.** Type-constrained decoding (arXiv 2504.09246) implements TypeScript only;
arXiv 2606.21619 found an incomplete constrainer lowers functional correctness, while a complete syntax constrainer
gained (its §5.3). Python 3 syntax can be constrained completely, and every Python file of the code categories parses
as Python 3 (`curate.repaired`, `docs/corpus.md` §3.1; UltraData-Code L3's exercises and the Kaggle notebooks are not
checked), so the model's distribution of program files lies inside the constrainer's language.
Types are checked after generation instead.

**`SyntaxConstraint`** (`inference/syntax_constraint.py`, the Rust crate `inference/syntax/`):
`SyntaxConstraint(tokenizer, prefix, *, suffix, owed=b"", lines=None)` (bytes); `masks(constraints, out, rows)` writes a
batch of constraints' next masks in parallel (rayon) without the GIL. Token *t* is allowed iff `prefix + generated + t`
can still become a valid Python 3.12 module, the text ending anywhere; while `owed` bytes (a healed prompt's dropped
token) are not re-emitted, only tokens that re-emit them; FIM sentinels never; a prefix that cannot become valid
Python 3.12 raises `ValueError`. Ends are decided once chosen: a chosen BOS stands iff the grammar ends a module after
the text and suffix and `ast.parse` accepts them (`ends`), else the slot takes the step's best other allowed token,
which the head computes beside its choice (the same choice as masking BOS beforehand, greedy or Gumbel-max; a
nucleus-sampled slot's BOS is decided before the step). A generation cut to `lines` lines (`code_eval.fim_lines`; the
evaluation's middles are uncut) ends at its `lines`-th line end: a token holding it is allowed iff the grammar ends a
module after the text up to that line end and the suffix, and stands iff `ast.parse` accepts them (else the slot's
head runs again without it); a BOS-ended cut middle gets its final newline before the check, as `fim_lines` scores
it; a line that no line end can end validly and only a line end continues (after a backslash) ends unchecked. Layers:
CPython 3.12.3's tokenizer
(`Parser/tokenizer.c`) as a character-level state machine (`lexer.rs`); INDENT/DEDENT and the tab-consistency check;
an LALR(1) grammar (`python.lark`) built by Lark at load and read by the crate (`parser.rs`), the soft keywords read
both ways until one fails; per lexer state, the vocabulary's events over its byte trie (`vocab.rs`). Where it is looser
than CPython it accepts rather than rejects. Checks: 0 false rejections over 11.4M token steps of 3,000 Python files
(the standard library, site-packages, this repository; BPE) and 9.2M randomly segmented, and no prefix that
`ast.parse` accepts without the grammar's end; the d16 model's unconstrained completions (EvalPlus greedy and
sampled, infilling SPM greedy and sampled) replayed through their constraints: every masked token invalid, the
constrained outputs token-identical wherever nothing was masked; all 8,488 infilling reference middles (BPE and two
random segmentations) accepted. Rejected alternatives: SynCode (one-regex f-strings
reject valid 3.12 code), llguidance (no INDENT/DEDENT, issue #107), arXiv 2402.17988 (no released library). C++ is not
constrained; it is checked after generation by `g++ -std=c++17 -fsyntax-only`, the MultiPL-E judge's compiler.

**Post-generation checks** (`eval/verify.py`): Python — the prompt plus completion parses, the EvalPlus entry function
exists and returns a value, and a static type checker (basedpyright, batched) limited to definite failures; then the
prompt's own examples (docstring `>>>` lines, MBPP+ `assert` lines) run by value in a subprocess with a timeout, under
EvalPlus's `reliability_guard` (as its own test runs).
C++ — `g++ -fsyntax-only`.

**Not used:** speculative decoding (lossless by construction; its gains target large-model serving); self-correction
prompting (a base model without instruction tuning).

## Post-training

Not implemented: the repository pre-trains and evaluates a base model. For HumanEval-style Python and C++ problem
solving, supervised fine-tuning on function-completion exercises after pre-training is the step with the largest
published effect at this scale:

- *Exercises:* fine-tuning on under 180M tokens of function-completion exercises raised a 1.3B model's HumanEval
  pass@1 from 29% to 50.6% (arXiv 2306.11644 §2.1, §2.2, Fig. 2.1).
- *Instruction tuning:* OpenCoder-1.5B from base to two-stage SFT, HumanEval+ 49.4 → 67.7 and MBPP+ 58.7 → 61.9
  (arXiv 2411.04905 Tables 6 and 7; not a controlled comparison).

**Format: completion, not chat.** An example is a prompt in the benchmarks' form, then its solution and BOS, the loss on
the solution's tokens and BOS alone; the tokenizer has no chat tokens (`corpus/tokenizer.py` `SPECIAL_TOKENS`).
Candidate data: `OpenCoder-LLM/opc-sft-stage2` (`educational_instruct`, `package_instruct`) and
`OpenCoder-LLM/opc-annealing-corpus` (`synthetic_code_snippet`, `algorithmic_corpus`). Before any trains the model:
decontamination against every scored benchmark by entry point and 10-gram overlap (as arXiv 2411.04905 §4.4); FIM
examples in `pack.py`'s format in the mix; the fine-tuned model reported beside the final base checkpoint.

## Layout

- `run.py`, `conf/`: the run app (Runs).
- `metrics.py`: the run log (`metrics.jsonl`) and every metric: bits per byte, parse outcomes, a run's compared
  measures; torch-free, imported by training, evaluation and the ablations.
- `paths.py`: the checkout's layout (`artifacts/`, `results/`). `native.py`: builds and loads the CUDA kernels and the
  Rust crates.
- `model/`: `modeling.py` (the model, non-embedding parameters and FLOPs per token), `config.py` (vocabulary, head
  width, RoPE theta), `ops/` and `kernels/` (the ops above and the decoder).
- `train/`: `loop.py` (setup, resume, the steps, checkpoints and records), `optim.py` (optimizers, weight
  decay, gradient clip), `schedule.py` (plan, each step's stage, `lr_lambda`), `stages.py` (`max_tokens_per_param`,
  each stage's micro-batches in the run's order), `checkpoint.py`.
- `eval/`: the checkpoint's evaluation (`suite.py`), the likelihood sums (`bpb.py`), the code benchmarks
  (`code_eval.py`) and the post-generation checks (`verify.py`).
- `inference/`: the native decoder (`decoder.py`), the Python 3.12 syntax constraint
  (`syntax_constraint.py`, its Rust crate and grammar in `syntax/`), the Hugging Face export (`export_hf.py`), the
  transformers model (`modeling_nanochat.py`) and the vLLM model (`vllm_model.py`).
- `corpus/`: the corpus pipeline, one module per step (`__init__.py` lists them; `mixture.py` also holds the
  micro-batch and repository-stage share, `publish.py` the dataset, `train_tokenizer.py` the tokenizer's sample and
  trainer); `crate/`, its Rust crate (`native.py`): the SA-BPE trainer (and frequency BPE), repository samples and
  the packer.
- `ablations/`: one module per experiment, named as it, each with its pre-registered `analysis` (`peak_lr.py`,
  `tokenizer_split_merge.py` (the tokenizer ablation), `logit_softcap.py`,
  `adamw_group_lr.py`, `muon_wd_timescale.py`); `contrast.py` (the pre-registered tests and adoption rules they decide
  by), `records.py` (an experiment's records in `results/`).
- `docs/`: `pretraining.md` (this document), `corpus.md` (the corpus's specification), `ablations.md`.
- `scripts/`: `setup.sh`, `container.sh`, `release.sh`. `docker/Dockerfile`: the image; `docker/Dockerfile.vllm`: the
  vLLM image. `results/`: the records.

## References

Papers this document cites, by arXiv identifier (`arxiv.org/abs/<id>`); `docs/corpus.md` cites its own inline by
identifier and section, `docs/ablations.md` in its References.

| arXiv | Title |
|---|---|
| 1904.09751 | The Curious Case of Neural Text Degeneration |
| 1910.04209 | On the adequacy of untuned warmup for adaptive optimization |
| 2101.00027 | The Pile: An 800GB Dataset of Diverse Text for Language Modeling |
| 2107.03374 | Evaluating Large Language Models Trained on Code |
| 2203.07814 | Competition-Level Code Generation with AlphaCode |
| 2204.02311 | PaLM: Scaling Language Modeling with Pathways |
| 2207.14255 | Efficient Training of Language Models to Fill in the Middle |
| 2208.08227 | MultiPL-E: A Scalable and Extensible Approach to Benchmarking Neural Code Generation |
| 2305.01210 | Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation |
| 2306.11644 | Textbooks Are All You Need |
| 2308.12950 | Code Llama: Open Foundation Models for Code |
| 2309.14322 | Small-scale proxies for large-scale Transformer training instabilities |
| 2401.02954 | DeepSeek LLM: Scaling Open-Source Language Models with Longtermism |
| 2401.14196 | DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence |
| 2402.14905 | MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases |
| 2402.17988 | Constrained Decoding for Fill-in-the-Middle Code Language Models via Efficient Left and Right Quotienting of Context-Sensitive Grammars |
| 2402.19173 | StarCoder 2 and The Stack v2: The Next Generation |
| 2404.19737 | Better & Faster Large Language Models via Multi-token Prediction |
| 2405.14591 | Base of RoPE Bounds Context Length |
| 2407.21783 | The Llama 3 Herd of Models |
| 2409.12186 | Qwen2.5-Coder Technical Report |
| 2410.01131 | nGPT: Normalized Transformer with Representation Learning on the Hypersphere |
| 2410.06205 | Round and Round We Go! What makes Rotary Positional Encodings useful? |
| 2411.04905 | OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models |
| 2412.04403 | Establishing Task Scaling Laws via Compute-Efficient Model Ladders |
| 2502.15938 | Straight to Zero: Why Linearly Decaying the Learning Rate to Zero Works Best for LLMs |
| 2502.16982 | Muon is Scalable for LLM Training |
| 2503.13423 | SuperBPE: Space Travel for Language Models |
| 2503.19786 | Gemma 3 Technical Report |
| 2504.09246 | Type-Constrained Code Generation with Language Models |
| 2504.11393 | DataDecide: How to Predict Best Pretraining Data with Small Experiments |
| 2505.13738 | Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training |
| 2505.18789 | From Output to Evaluation: Does Raw Instruction-Tuned Code LLMs Output Suffice for Fill-in-the-Middle Code Generation? |
| 2508.13144 | Signal and Noise: A Framework for Reducing Uncertainty in Language Model Evaluation |
| 2601.22887 | MoVE: Mixture of Value Embeddings -- A New Axis for Scaling Parametric Memory in Autoregressive Models |
| 2603.15031 | Attention Residuals |
| 2604.07769 | An Empirical Study on Influence-Based Pretraining Data Selection for Code Large Language Models |
| 2606.21619 | The Alignment Problem in Constrained Code Generation |
| 2608.18062 | TokEval: A Tokenizer Evaluation Suite |

Repositories, at the commit, file or record cited: `karpathy/nanochat` (92d63d4, e85db6b, `dev/LOG.md`);
`KellerJordan/modded-nanogpt` (records #5, #8, #14, #49); `Dao-AILab/flash-attention` (FlashAttention-4,
`flash_attn.cute`), `Dao-AILab/quack`; `evalplus/evalplus`, `nuprl/MultiPL-E`, `openai/human-eval-infilling`;
`huggingface/transformers` (`modeling_qwen3.py`, `TopPLogitsWarper`), `python/cpython` 3.12.3 (`Parser/tokenizer.c`);
`guidance-ai/llguidance` (issue #107); `NVIDIA/Megatron-LM` (PR 5395), `deepspeedai/DeepSpeed` (issue 8439);
`OpenCoder-LLM/opc-sft-stage2`, `OpenCoder-LLM/opc-annealing-corpus` (Hugging Face datasets). Token healing: "The Art
of Prompt Design: Prompt Boundaries and Token Healing" (Guidance, 2023).
