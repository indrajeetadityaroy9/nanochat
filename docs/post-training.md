# Post-training

The method that turns the base model into a code model that reasons before it answers: its starting point (§1), the
published evidence each choice rests on (§2), the format (§3), the supervised fine-tuning recipe the repository
implements (§4), its evaluation (§5), the code RL stage after it (§6), what the pipeline leaves out (§7) and the
attempts to remove the SFT model's reasoning loops directly, none adopted (§8). The SFT corpus is
`docs/corpus.md` §10; the shared training loop `docs/training.md`; the natural-termination probe
`docs/evaluation.md` §10; measured results `docs/results.md`.

Markers: [measured] from a record or a run; [arithmetic] derived from code numbers; [inference] reasoning, not
measured; [estimate]; [unverified] a figure without a record, or a paper value its extracted text lacks. Below 1B the
cited evidence is MobileLLM-R1 (140M–950M), Qwen3-0.6B, Qwen2.5-Coder-0.5B and two 0.5B math studies; every transfer
to the 522M base model is [inference].

## Code

| File | Section |
|---|---|
| `corpus/posttrain/sft.py` (`THINK`, `END_THINK`), `corpus/loader.py` (`PackedSplit.contexts`, `PretrainingBatches.assemble`) | §3 markers and loss (the corpus: `docs/corpus.md` §10) |
| `run.py` (`RunConfig.base`, `base_step`, `sft_epochs`, `generate` "none"; `base_plan`, `plan_of`, `validation_steps`, `job`), `conf/experiment/sft.yaml`, `train/stages.py` (`stage_batches`: epochs past the first; `val_batches`) | §4 |
| `train/schedule.py` (`sft_plan`, `RunPlan.weight_decay`, `last_stage`), `train/loop.py` (base weights, `validate`), `eval/sft_val.py` | §4 |
| `conf/experiment/sft_lr.yaml`, `ablations/sft_lr.py` (`sft_lr_fit`, `sft_lr_scale`), `ablations/records.py` (`record`, `results_dir`), `results/posttrain/sft_lr.jsonl` and its `-curves.jsonl` | §4 |
| `scripts/official_reasoning.sh`, `eval/official_reasoning_parser.py`, `eval/official_lcb.py`, `eval/official_lcb_shards.py`, `eval/official_tokenize.py`, `eval/official_evalplus.py`, `eval/official_reasoning.py`, `eval/official_loop_stop.py`, `inference/export_hf.py` (`--chat`, `CHAT_TEMPLATE`) | §5 |
| `eval/official_termination.py`, `scripts/official_termination.sh`, `scripts/official_termination_watch.sh`, `eval/serving_check.py`, `scripts/serving_check.sh` | §5.1 (the method: `docs/evaluation.md` §10) |
| `onpolicy/prompts.py`, `onpolicy/rollouts.py`, `onpolicy/verify.py`, `onpolicy/outcomes.py`, `scripts/onpolicy.sh`, `docker/Dockerfile.onpolicy` | §5 (the KodCode val evaluation), §6 (the RL stage's prompts and tests) |
| `onpolicy/rl.py`, `onpolicy/sampler.py`, `scripts/rl.sh`, `inference/export_hf.py` (`hf_weights`), `scripts/post_sft_queue.sh` | §6 |

## 1. Starting point

- **The model**: d24, 522,252,336 parameters (`docs/architecture.md` §1), 103.03B tokens (`docs/training.md` §2),
  a completion model: the tokenizer's special tokens are BOS and the three FIM sentinels (`corpus/tokenizer.py`
  `SPECIAL_TOKENS`), so there is no chat template. An example is a prompt and a response, the reasoning between
  plain-text `<think>` markers and then the answer (§3); the loss falls on the response and its closing BOS.
- **The vocabulary**: 32,768 ids of the repository's own tokenizer (`docs/corpus.md` §5); no published model shares
  it, which decides how teachers can be used (§2.5).
- **The measures**: the base model's official-tool scores and likelihoods (`docs/results.md` §1, §4; protocol
  `docs/evaluation.md` §4); a post-trained model is measured in its reasoning format (§5).
- **What to keep**: the 32K context. Infilling stays the base model's: no SFT source holds FIM examples
  (`docs/corpus.md` §10), and the post-trained model answers in one format (§3).
- **Contamination**: every SFT source passes the pretraining rule before use, a 10-gram of words shared with a scored
  benchmark's prompt or reference drops the example (`docs/corpus.md` §3, §10); OpenCoder removes HumanEval/MBPP
  entry points and 10-gram overlaps (arXiv 2411.04905 §4.4).

## 2. Design basis

### 2.1 Current pipelines

| Model (size) | Stages after pre-training | Source |
|---|---|---|
| DeepSeek-V3.2 | one RL specialist per domain (math, programming, reasoning, agents, agentic coding, search) from the same base; their outputs as SFT data; one mixed GRPO stage with rule-based outcome rewards and a length penalty, rubric-based generative rewards for general tasks | arXiv 2512.02556 §3 |
| Kimi K3 (2.8T MoE, 104B active) | SFT; RL experts for three domains at three reasoning-effort levels; multi-teacher on-policy distillation into one model | arXiv 2607.24653 §4.1 |
| GLM-5 (744B, 40B active) | SFT; reasoning RL; asynchronous agentic RL; general RL; on-policy distillation from the preceding stages' final checkpoints | arXiv 2602.15763 §3 |
| MiMo-V2-Flash (309B, 15B active) | SFT; domain teachers trained by RL or SFT; multi-teacher on-policy distillation | arXiv 2601.02780 §4.1 |
| Qwen3-Coder-Next (80B, 3B active) | mid-training; SFT; experts (web development, user experience, single-turn RL on unit tests, multi-turn software-engineering RL) distilled into the SFT model | arXiv 2603.00729 §3–§4 |
| Olmo 3 Think (7B, 32B) | SFT; DPO on delta pairs (§2.4); RL with verifiable rewards | arXiv 2512.13961 §4 |
| Nemotron-Cascade (8B, 14B) | two SFT stages; RL by domain in turn: RLHF, instruction following, math, code, software engineering | arXiv 2512.13607 §3–§4 |
| Nemotron-Cascade 2 (30B MoE, 3B active) | one packed SFT stage; RL in turn: instruction following, multi-domain, multi-domain on-policy distillation, RLHF, long context, code, software engineering | arXiv 2603.19220 §3–§4 |
| Seed-Coder-8B reasoning | long-CoT warmup on thousands of examples, from the base model; GRPO | arXiv 2506.03524 §3.2 |
| Qwen3 0.6B–14B | off-policy distillation, then on-policy distillation from Qwen3-32B or Qwen3-235B-A22B; no RL | arXiv 2505.09388 §4.5 |
| Phi-4-mini-reasoning (3.8B) | distillation as mid-training; distillation SFT; DPO on rejected teacher rollouts; RL | arXiv 2504.21233 §3 |
| MobileLLM-R1 (140M, 360M, 950M) | logit distillation from Llama-3.1-8B-Instruct in mid-training; general SFT; reasoning SFT; no RL | arXiv 2509.24945 §3.2, §7.4 |

What these replaced:

| Earlier form | Current form | Section |
|---|---|---|
| LoRA or QLoRA instruction tuning | full-parameter updates in every report above | §2.2 |
| unverified instructions written by an LLM (Evol-Instruct, OSS-Instruct) | execution-checked, difficulty-filtered chain-of-thought traces from a reasoning teacher | §2.3 |
| offline DPO on judged pairs | DPO on strong-versus-weak model pairs, or on-policy pairs labeled in a sandbox | §2.4 |
| RL for small models | on-policy distillation from a stronger teacher | §2.5 |
| PPO's value model; GRPO's KL, std normalization and several PPO epochs per batch | group baseline, no KL, mean-only or batch-level normalization, one update per batch or truncated importance weights | §2.6 |
| a reward model trained on compiler outcomes | executing verified tests | §2.7 |
| sequential RL stages | one mixed stage, or stages consolidated by on-policy distillation | §2.5, §2.6 |

### 2.2 Full-parameter updates

- **DeepSeek-Coder uses no QLoRA**: its instruction tuning ("a cosine schedule with 100 warm-up steps and an initial
  learning rate 1e-5 … a batch size of 4M tokens and 2B tokens in total", arXiv 2401.14196 §3.7) names no LoRA,
  QLoRA, low-rank or parameter-efficient method [inference: every weight trains]. DeepSeek-Coder-V2: SFT on 300M
  tokens at lr 5e-6, batch 1M tokens, 1B tokens; then GRPO on about 40K prompts with test cases (arXiv 2406.11931
  §3.5); no PEFT either. None of the models in §2.1, nor Qwen2.5-Coder, OpenCoder, AceReason, rStar-Coder or
  OpenThoughts, describes an adapter for its own training; GrandCode uses LoRA only for test-time RL on one problem
  (arXiv 2604.02721 §10.3).
- **Measured on code**: "FFT generally leads to the best downstream performance across all scales" (StarCoder
  1B–16B, PEFT on 8-bit models; arXiv 2401.00788 §1, Fig. 1). Students below 14B train with full parameters, LoRA
  above (arXiv 2502.12143 §3.1). Distillation, SFT and LoRA together on Qwen2.5-0.5B-Instruct scored 0.7% on
  OlympiadBench and 1.2% on MATH500 (arXiv 2506.13404 §4.2), and SFT with LoRA plus RL collapsed in training (§4.3).
- **RL with adapters** (1.5B, already distilled): LoRA rank 32 on every linear layer 42.5 against full parameters'
  44.9 and DoRA's 46.6, PiSSA 0.2 (arXiv 2512.23165 Table 3); Tina's rank-32 GRPO reached its best checkpoint for $9
  (arXiv 2504.15777 Table 1), and 4 of its 5 models scored above their full-parameter baselines (Table 3).
- **Capacity** (code): in continued pre-training of Llama-2-7B on StarCoder-Python, LoRA at rank 256 peaked at
  HumanEval 0.224 against full fine-tuning's 0.263; in code instruction tuning ranks 16 and 64 scored 0.358 and 0.417
  at epoch 4 and rank 256 0.498, against full fine-tuning's 0.497 at epoch 8 (arXiv 2405.09673 §4.1); full
  fine-tuning on code "does not learn low-rank perturbations" (§4.6). LoRA matches full fine-tuning on
  "small-to-medium-sized" SFT sets and falls behind on sets that "exceed LoRA capacity"; in RL it matches "even with
  ranks as low as 1" at about 10× full fine-tuning's learning rate (Schulman and Thinking Machines Lab, "LoRA Without
  Regret", 2025).
- **Compute and memory**: for a d_in × d_out matrix full fine-tuning costs 3·d_in·d_out multiply-adds per token and
  LoRA 2·d_in·d_out + 3R(d_in + d_out) ("LoRA Without Regret"), so LoRA saves compute only below the rank
  d_in·d_out / (3(d_in + d_out)): 128 for each of d24's separate 768 × 768 attention projections (`model/modeling.py`
  `Attention`) and 205 for the MLP's matrices [arithmetic]. QLoRA's 4-bit base exists to fit "a 65B parameter model on
  a single 48GB GPU" (arXiv 2305.14314 Abstract); pre-training already holds the fp32 weights, gradients and both
  optimizers' state of all 522,252,336 parameters in 7.47 GiB (`docs/training.md` §7).
- **Here** [inference]: full parameters for the SFT (§4): its billions of tokens are the volume where LoRA falls
  behind, and the rank that kept up on code (256) costs more than full fine-tuning at hidden size 768. LoRA fits RL
  (§6).

### 2.3 Supervised fine-tuning and off-policy distillation

Off-policy distillation is SFT on a teacher's text, so it needs no shared vocabulary (§2.5). Published effects at
1.5B and below:

| Model | Data | Result | Source |
|---|---|---|---|
| Qwen2.5-Coder-0.5B, base → instruct | tens of millions of synthetic instructions, then millions chosen by rejection sampling; FIM instructions; offline DPO | HumanEval 28.0 → 61.6, HumanEval+ 57.3, LiveCodeBench 2.0 | arXiv 2409.12186 §4, §6.1, Table 16 |
| DeepSeek-Coder-1.3B, base → instruct | 2B tokens (§2.2) | HumanEval 34.8 → 65.2 | arXiv 2401.14196 Table 3 |
| OpenCoder-1.5B, base → instruct | two SFT stages | HumanEval+ 49.4 → 67.7, MBPP+ 58.7 → 61.9 | arXiv 2411.04905 Tables 6, 7 (not controlled) |
| Qwen2.5-Coder-1.5B: OpenCodeInstruct against its Instruct | 5M instruction–solution pairs, sequences ≤ 2,048 tokens | HumanEval+ 73.8 against 66.5, MBPP+ 68.3 against 59.4 | arXiv 2504.04030 Table 3 |
| Qwen2.5-Coder-1.5B-Instruct + rStar-Coder | 580K verified long-CoT solutions, 6 epochs | LiveCodeBench 6.5 → 40.1, HumanEval+ 81.7, MBPP+ 60.8 | arXiv 2505.21297 Table 2, §4.2 |
| MobileLLM-R1-950M, after a distillation mid-training stage | general SFT, then 6.2M reasoning samples (2.2M from OpenCodeReasoning-2) at 32K | LiveCodeBench v6 19.9 (Qwen3-0.6B 14.9) | arXiv 2509.24945 Tables 2, 5, 7 |
| Qwen3-0.6B, distilled (§2.5) | — | LiveCodeBench v5 12.3 thinking, 3.6 not thinking | arXiv 2505.09388 Tables 19, 20 |

- **Prompts before answers**: in a fit z = a·log₂x + b·log₂y + c of the score against unique prompts x and answers
  per prompt y (seven runs, 7B, math), a = 4.831 and b = 2.635 (R² 0.989; arXiv 2506.13284 §4.4.2); at a fixed budget
  64K prompts × 1 beat 16K × 4 and 8K × 8 (arXiv 2601.06953 §3.2; the ordering only, its scores are in Fig. 3);
  37.7K problems × 16 solutions scored LiveCodeBench 54.7 against the full set's 57.3 (7B; arXiv 2505.21297 App.
  A.1). OpenThoughts samples 16 answers per question to grow a source (arXiv 2506.04178 §4.4).
- **Epochs**: gains plateau at 5–6, and more answers per prompt "also greatly benefits from multi-epoch training"
  (arXiv 2506.13284 §4.4.3); long traces need 8–10 epochs, short ones 2–3; at 8 epochs long traces scored
  LiveCodeBench v5 60.3 against short ones' 43.1 (7B; arXiv 2601.06953 Table 4).
- **Long or short traces at small sizes**: with 7.5K MATH prompts, short chains beat long ones at 3B and below
  (Qwen2.5-0.5B 19.5 against 14.8, Qwen2.5-3B 43.4 against 40.3) and a small teacher beats a large one (arXiv
  2502.12143 Tables 1–2); one long example in five is best at 3B (Table 3); base models show "a more significant
  learnability gap" than the Instruct students (§3.4). The 950M and 1.5B models above trained on 580K–6.2M long traces
  [inference: the gap was measured at 7.5K prompts and is untested at that data scale below 1B].
- **The valley** (7B and 8B students, LiveCodeBench): 1K distilled examples more than halve the baseline, 10K lift it
  about 50%, 30K a further 50–100%; easy and medium problems help more than hard ones (+41% against +7%); correct and
  incorrect R1 traces distill alike (0.185 against 0.182) (arXiv 2510.06101 §4, Fig. 1, Table 2).
- **Sources and filters** (OpenThoughts, 7B student; 31.6K examples per arm): two sources beat sixteen (41.3 against
  36.4, Table 4); for code the best question filter is difficulty judged by an LLM (Table 5); QwQ-32B teaches better
  than DeepSeek-R1 by 1.9 average points on code data though it scores lower itself (Table 8). No answer filter beat
  none: verification lowered the 7B (41.9 against 45.0) and raised the 32B (64.5 against 62.1) (Table 15), and
  unit-test filtering scored LiveCodeBench 36.0 against 38.5 on 16K examples (Table 18); removing the examples whose
  traces exceed 8,192 tokens cost 17.6% relative on a 12K-instance set (Table 22) (arXiv 2506.04178).
- **Staging**: general instructions before reasoning data beat one joint stage at 950M on MATH and GSM8K (57.8 and
  68.5 against 56.2 and 53.1), not on LiveCodeBench (13.7 against 14.9); math-only reasoning data scored LiveCodeBench
  0.0, code-only 12.0 (arXiv 2509.24945 Table 8). Phi-4-mini-reasoning packs examples in distillation mid-training and
  not in the last SFT, "where we teach the model to decide where to stop generating" (arXiv 2504.21233 §3.2).
- **Template**: in Olmo 3's RL-Zero (RLVR from the base, no SFT), "'simple' prompt templates greatly outperform
  standard post-trained templates (e.g., <think></think>)" (arXiv 2512.13961 §6); it does not test an SFT template.
  Seed-Coder's warmup starts from the base, not the instruct model, which "frequently collapses into typical SFT
  training data patterns during RL" (arXiv 2506.03524 §3.2).
- **Learning rate and forgetting** (a 1B student, KL distillation of 150 steps): out-of-domain scores move by at
  most 1.3 points at lr 1e-5 and by −11.2 to −14.0 at 5e-5 (arXiv 2609.35259 §4.2).
- **Published settings**: rStar-Coder 6 epochs, lr 4e-5, batch 96, 16K (§4.1); MobileLLM-R1's reasoning SFT 4
  epochs, lr 8e-5, 32K (Table 2); OpenThoughts 5 epochs, lr 8e-5, batch 512, packed (App. Table 9); Olmo 3 Think 2
  epochs, lr 5e-5, 1M-token batches of packed 32K rows (App. A.6.1); Nemotron-Cascade lr 5e-5 then 2e-5, batch 256
  (Table 14).
- **Here** [inference]: thinking-mode traces only, whole rows up to 32,768 tokens, one packed stage (§4) over a corpus
  past the valley, at most a few traces per prompt (`docs/corpus.md` §10).

### 2.4 Preference tuning

- **Delta learning** (Olmo 3 Think 7B): the chosen response from Qwen3-32B thinking, the rejected one from Qwen3-0.6B
  thinking, unfiltered; "the quality of preference data depends primarily on the quality of the delta between chosen
  and rejected responses rather than the quality of either response individually" (arXiv 2512.13961 §4.3). Against
  the SFT checkpoint's average 70.3, SFT continued on the chosen responses scored 64.5 and delta DPO 72.9 (Table 21).
  RL from the DPO checkpoint beat RL from SFT, average 74.1 against 71.9 and LiveCodeBench 73.4 against 70.7 (§4.5,
  Table 22). Length-normalized DPO, β = 5, lr 8e-8, 150K pairs, one epoch (Table 48).
- **Rollout pairs**: DPO on Phi-4-mini-reasoning's rejected R1 rollouts took AIME 43.3 → 50.0 (arXiv 2504.21233
  Table 3); Seed-Coder samples about 20K pairs on-policy and labels them by sandbox tests (arXiv 2506.03524 §3.1).
- **Here** [inference]: no DPO on teacher pairs; preference at the model's own loops (FTPO, then DPO) was tried and
  not adopted (§8).

### 2.5 On-policy distillation

The student samples, the teacher scores every sampled token, and the loss is the per-token reverse KL; in its
sampled-token form the advantage is sg[log π_T(y_t) − log π_θ(y_t)] (arXiv 2602.15763 Eq. 2; arXiv 2604.13010
Eq. 2). Kimi K3 clips it to ±R_max; its top-k objectives showed "no clear advantage in either convergence speed or
final performance" (arXiv 2607.24653 §4.1.3, Eq. 15).

- **Against RL** (Qwen3-8B from one off-policy distilled checkpoint, math and code queries; arXiv 2505.09388 §4.7,
  Table 21): "reinforcement learning does not lead to any improvement in pass@64 scores".

  | Arm | AIME'24 (pass@64) | LiveCodeBench v5 | GPU hours |
  |---|---|---|---|
  | off-policy distillation | 55.0 (90.0) | 42.0 | — |
  | + RL | 67.6 (90.0) | 52.9 | 17,920 |
  | + on-policy distillation | 74.4 (93.3) | 60.3 | 1,800 |

  Llama-Nemotron: RL "yields suboptimal results for smaller models compared to distillation" (arXiv 2505.00949 §5).
  At 32B, distillation from R1 scored LiveCodeBench 57.2 against RL's 40.2 (arXiv 2501.12948 (version 1) §4.1,
  Table 6).
- **Against SFT** (4B): AIME24 56.7 → 65.4 and LiveCodeBench v6 31.5 → 39.3; precomputing the teacher's
  log-probabilities on the SFT model's rollouts (Lightning OPD) matches it at 3.6–4.0× less compute (arXiv 2604.13010
  Tables 1–2). At 1.7B, OPD lowered thinking-mode math 59.2 → 57.1, and "prolonged training can even lead to
  performance degradation"; rewarding the teacher's gain over its own pre-reasoning base (OPD²) raised code
  10.5 → 29.4 against OPD's 21.0 (arXiv 2607.15161 Tables 2–5, §3.5).
- **Controlled at 1B–1.5B** (arXiv 2609.35259): on-policy and off-policy rollouts reach 72% and 73% in-domain; the KL
  direction matters more, forward KL robust to the rollout policy and reverse KL needing the student's rollouts (§4.2,
  §5.2); off-policy checkpoints gave the strongest RL afterwards (§5.3).
- **Constraints**: one vocabulary: "Teachers and students are taken from the same model family to ensure a shared
  tokenizer and vocabulary" (arXiv 2609.35259 §4.1); "Most OPD objectives assume that teacher and student share the
  same vocabulary" (arXiv 2604.00626 §8). At a fixed OPD teacher a different SFT teacher costs 0.4–4.2 AIME24 points
  in standard OPD and 0.6–8.8 in Lightning OPD (arXiv 2604.13010 Tables 3–4). Teachers may be the model's own earlier
  stages (GLM-5 §3.5) or "even the student model itself" (arXiv 2601.02780 §4.1).
- **Here** [inference]: the vocabulary (§1) rules out every published teacher for OPD; external teachers enter as text
  (§2.3); with one RL domain there are no specialists to consolidate (§7).

### 2.6 Reinforcement learning with verifiable rewards

GRPO (arXiv 2402.03300 §4.1) samples G outputs per prompt and uses the group's mean reward as the baseline in place of
PPO's value model, "another model of comparable size". Current recipes keep the group baseline and change the rest:

| Component | Current choice | Evidence |
|---|---|---|
| KL to a reference | none | "removing it does not lead to over-optimization or destabilized training" (arXiv 2512.13961 §4.4); β = 0 in AceReason, Nemotron-Cascade, Seed-Coder, GLM-5 and ScaleRL; a reference KL "leads to a degradation in performance" (arXiv 2505.22617 §4.1). Kept by DeepSeek-V3.2 with an importance-corrected estimator, weak or none for math (§3.1), and by ProRL from a distilled 1.5B with periodic reference resets (arXiv 2505.24864 §2.3.1) |
| Advantage | r − mean(r), no std division | the std term's difficulty bias (§2.8, Dr. GRPO); OlmoRL (arXiv 2512.13961 §4.4) and DeepSeek-V3.2 (Eq. 5); ScaleRL divides by the batch's std, "marginally better" (arXiv 2510.13786 §3.2) |
| Aggregation | over the batch's tokens | DAPO (§2.8), OlmoRL, AceReason; prompt-level in ScaleRL; token-level "best suited for base models" (arXiv 2508.08221 §4.3) |
| Update | one gradient step per batch, so the ratio is 1 (REINFORCE with group-normalized rewards); truncated importance weights once rollouts lag (CISPO, §2.8) | 2 or 4 updates per batch "led to rapid entropy collapse around 100 steps" (arXiv 2505.16400 §3.2); Nemotron-Cascade likewise (§4.1) |
| Clipping, where a PPO-form loss stays | ε_low 0.2, ε_high 0.28 or 0.272 | GLM-5 §3.2; Seed-Coder §3.2; OlmoRL App. Table 49 (§2.8, DAPO) |
| Sampler–trainer mismatch | FP16 in sampler and trainer; truncated importance weights; fp32 logits | FP16's sequence-level mismatch is "approximately 24× smaller" than BF16's (arXiv 2510.26788 §3.5); in BF16 GRPO collapsed after peaking and only sequence-level MIS stayed stable, which FP16 beat (§4.2); OlmoRL caps the weight at 2; GLM-5 masks tokens outside [1/2, 2] (§3.2); fp32 logits raised ScaleRL's asymptote 0.52 → 0.61 (§3.2); in a dense model the displacement is "concentrated near zero and bounded, consistent with bf16 rounding error" (arXiv 2609.32444 §3.1) |
| Prompts | drop groups whose rewards are all equal and refill the batch; drop prompts solved ≥ 90% | OlmoRL active sampling; ScaleRL's No-Positive-Resampling (§3.2); Nemotron-Cascade drops problems solved 8/8, or 0/8 by a strong model (§4.6); 2.2K hard prompts beat the full 49K, AIME24 65.9 against 63.3 (arXiv 2505.16400 Table 3) |
| Rollouts | 8–16 per prompt | OlmoRL 8, AceReason 8–16, ScaleRL 16; more as compute grows, fewer on hard sets (arXiv 2603.12151 §4) |
| Temperature | entropy about 0.3 | "setting the training temperature such that the temperature-adjusted entropy remains around 0.3 typically leads to effective RL training" (arXiv 2506.13284 §4.5.2); code RL is "sensitive to the temperature" (arXiv 2512.13607 §4.6) |
| Length | short first, then longer | 8K → 16K → 24K → 32K beat starting at 16K or 24K (arXiv 2505.16400 §4.3); an 8K stage first, AIME25 56.7 against 51.8 (arXiv 2506.13284 §4.5.4); masking truncated outputs helps at 8K–16K and hurts at 32K (§4.5.3, Fig. 9); 10–15% truncated outputs destabilize (arXiv 2510.13786 App. A.15) |
| Entropy | a candidate: covariance-targeted KL-Cov or Clip-Cov | 32B average 52.2 against clip-higher's 47.2 and GRPO's 45.8 (arXiv 2505.22617 Table 2) |

- **Small models**: MobileLLM-R1-950M after RL from the base scored GSM8K 57.0 against SFT's 74.0; "For
  high-performing small models fully fine-tuned on SFT data, additional RL does not observe a significant performance
  improvement" (arXiv 2509.24945 §7.4). Where RL helps at 1.7B and below it starts from distilled or instruct models:
  ProRL at 1.5B (code +14.4% over more than 2K steps, §3), Code-A1 at 1.5B and Murphy at 1.7B (§2.7). From a base
  model at 0.5B on math: average 12.1 → 20.9 (arXiv 2503.18892 Table 1). Score Centering trains Qwen3-0.6B under a
  deliberately mismatched sampler (arXiv 2609.20807 §5).
- **Order**: math-only RL raised LiveCodeBench v5 by 6.8 at 7B and 5.8 at 14B, and code RL from a math-RL checkpoint
  ends higher (arXiv 2505.16400 Table 1, §4.3); Nemotron-Cascade orders RL from general to specialized domains and
  finds it "resistant to catastrophic forgetting" (§4.1); DeepSeek-V3.2 merges the domains into one stage (§2.1).
- **Coverage**: base models catch up with RL-trained ones at large k (arXiv 2504.13837 §3.2), and RL left Qwen3-8B's
  pass@64 unchanged (§2.5); AceReason's RL raised LiveCodeBench pass@k up to k = 64 by about 10 points (§4.3), and
  ProRL solves tasks where the base never succeeds (§4).
- **Here**: bf16 in the sampler and the trainer on identically rounded weights (§6), no importance weights
  [inference]. On the base model's final checkpoint (CPU, three code files of 1,024 tokens) the largest activations
  are 145 (residual stream), 241 (ReLU²), 46.8 (attention score) and 29.5 (logit before the cap), and the weights
  bound the last three by 11,195, 68.2 and 75, all below fp16's 65,504; with activations rounded to fp16 the mean KL to
  fp32 is 1.5e-5, to bf16 1.0e-3; 77–99.9% of the residual stream's gradient values lie below fp16's smallest normal
  (6.1e-5), so FP16 training would need a scaled loss [measured].

### 2.7 Code rewards

- **Binary on verified tests**: reward 1 "if and only if the generated code successfully passes all test cases"
  (arXiv 2505.16400 §3.3); Nemotron-Cascade (§4.6), KodCode (arXiv 2503.02951 §4.1) and Seed-Coder alike. The tests
  decide the result: false negatives leave RL at "a sub-optimal point"; false positives (weak tests, DeepCoder's set
  among them) bring "a sudden drop on testing set accuracy midway" (arXiv 2505.16400 §4.3); AceReason drops
  special-judge and interactive problems, those with incorrect tests or no golden solution, and those with weak tests
  (App. A.4).
- **Dense rewards**: at 32B, a reward of 1.1 when every test passes, −0.1 when none does and 0.1 times the pass
  fraction otherwise beat the binary reward by 0.55–1.19 LiveCodeBench points, the most with unvalidated tests (arXiv
  2608.24135 §3.3, Table 3); X-Coder's fraction reward drew late reward hacking that exploits "edge cases for partial
  rewards" (arXiv 2601.06953 §5); Nanbeige4.1-3B rewards the pass rate and adds an efficiency term only once every
  test passes (arXiv 2602.13367 §2.3).
- **Making tests**: majority agreement among 16 sampled solutions labels outputs with 96.8% accuracy against GPT-4o's
  12.7% (arXiv 2505.21297 Table 5); Qwen3-Coder-Next keeps the tests "that achieve the highest consensus under
  majority voting" (§4.2); at a fixed budget of about 40 tests, tests synthesized against near-correct faulty programs
  and then selected for diversity add 2.4 LiveCodeBench points, the synthesis alone 1.0 (arXiv 2608.24135 Table 2).
- **Learned rewards**: DeepSeek-Coder-V2 trained a reward model on compiler outcomes, the raw 0–1 signal "noisy and
  sub-optimal" with sparse tests (arXiv 2406.11931 §3.5); later code recipes execute tests directly. A reasoning reward
  model gated on all tests passing raised P-GRPO's average 54.9 → 57.4; gated on the pass rate, the policy
  "excessively exploits these noisy signals" (arXiv 2508.05170 Table 1, Fig. 4).
- **Tests by the model**: a separate test writer trained adversarially beat human tests at 1.5B, 56.95 against 56.23,
  where one model writing both scored 55.88 (arXiv 2603.15611 §4.2); CURE at 4B–14B (arXiv 2506.03136 §4.2).
- **Multi-turn**: RL over execution feedback matches single-turn GRPO when evaluated in one turn and adds up to 8
  points over compute-matched GRPO in three iterations (Qwen3-1.7B, Qwen3-4B and OLMo-2-7B; arXiv 2511.07833 §5.1,
  Table 1).
- **Here** [inference]: binary, on the last code block after `</think>` and every test of the problem (§6); no
  dense or pass-fraction arm, no neural reward.

### 2.8 GRPO variants

| Variant | Change to GRPO | Problem addressed | Evidence (setting) | Here [inference] |
|---|---|---|---|---|
| Dr. GRPO (arXiv 2503.20783 §3) | drops the 1/\|o_i\| length normalization and the std division; the advantage becomes RLOO's up to G/(G−1) (App. A) | length bias and a question-difficulty bias | math; its whole 1.5B recipe 33.1 → 42.1 over the base (Table 4); either removal alone beats GRPO (App. C) | the token-level loss has no 1/\|o_i\|; the advantage divides by the batch's std, one number per batch, so no group's difficulty reweights it (arXiv 2508.08221 §4.1; §6) |
| DAPO (arXiv 2503.14476 §3–§4) | clip [1 − 0.2, 1 + 0.28] (Clip-Higher); dynamic sampling keeps groups with 0 < correct < G; token-level loss; overlong filtering and a soft length penalty; no KL | entropy collapse, zero-advantage groups, length effects | math, Qwen2.5-32B: AIME24 30 → 50 (Table 1); clip-higher "gradually becomes unstable" (arXiv 2505.22617 §4.3); under DAPO Phi-4-mini-reasoning's AIME consensus "consistently degrades" (arXiv 2504.21233 §5.5) | the token-level loss; zero-advantage groups left out of each batch, without resampling; overlong filtering; no Clip-Higher (one update per batch); no length penalty (§6) |
| CISPO (arXiv 2506.13585 §3.1) | clips the stop-gradient importance weight instead of the update; token-level normalization; no KL | PPO clipping drops rare reflective tokens after the first off-policy update | Qwen2.5-32B zero-RL: DAPO's AIME in half the steps (Fig. 2); ScaleRL: "the best balance of stability and robustness to hyperparameters" (arXiv 2510.13786 App. A.17) | not used: one update per batch (§6) |
| GSPO (arXiv 2507.18071 §4–§5) | sequence-level ratio (π/π_old)^{1/\|y\|}, clipped per sequence at [1 − 3e-4, 1 + 4e-4] | token-ratio noise and collapse; MoE routing drift; tolerates precision mismatch (§5.4) | Qwen3-30B-A3B MoE (Fig. 1); in ScaleRL its runs "diverged mid-training" (App. A.17) | not first |
| P-GRPO (arXiv 2508.05170 §2.4) | reward R^f + R^o + R^o·R^t: a reward model's reasoning score counts only when the code passes all tests | outcome rewards ignore the reasoning; all-correct groups get no advantage | Qwen2.5-Coder-7B-Instruct: MBPP+ 76.2 against GRPO's 75.1, HumanEval+ 81.1 both (Table 1) | not first: needs a reward model |
| DGPO (arXiv 2601.20614 §3–§4) | advantage over the mean absolute deviation; softmax difficulty weights (T = 2) | GRPO's total update per question, 2G·√(p(1 − p)), peaks at pass rate 0.5 (Thm. 1) | math only (1.5B–7B): average 39.79 against 37.61 | no code evidence |

### 2.9 The pipeline

SFT on a reasoning-code mixture: the code traces of Nemotron-Cascade's SFT (arXiv 2512.13607 §3) and of Nemotron's
competitive-programming set, with increased verified function-level coverage (KodCode's R1 traces, each solution
passing its tests, beside Exercism's function stubs; the composition: `docs/corpus.md` §10), in one format
(reasoning, then the answer), with full parameters (§2.2); then RL with verifiable rewards on KodCode's problems
(§6). Below 1B the published pipelines end at SFT or distillation: MobileLLM-R1 (140M–950M) has no RL, "additional RL
does not observe a significant performance improvement" (arXiv 2509.24945 §7.4), and RL "yields suboptimal results for
smaller models compared to distillation" (arXiv 2505.00949 §5); Nemotron-Cascade's code RL added 2.4–4.1
LiveCodeBench v6 points at 8B–14B (Table 7). Here RL runs from the SFT model for 2 hours on the one GPU (§6), and its
evaluation tests that evidence at 522M. Each choice is [inference] from the cited evidence.

| Stage | From | Data | Objective | Lengths | Gate |
|---|---|---|---|---|---|
| 0. Base | — | — | — | — | the base run's last checkpoint (`base_step` 69,874; `docs/results.md` §1, §4) |
| 1. SFT (§4) | the base | the SFT corpus's reasoning traces (`docs/corpus.md` §10) | next-token loss on the response | rows ≤ 32,768 tokens, packed | reasoning pass@1 on HumanEval+ and MBPP+ above the base's greedy pass@1 (§5) |
| 2. RL (§6) | the SFT model | KodCode train problems with tests, 180,911 usable | the token-level policy gradient, one on-policy update per batch, DAPO's soft length penalty | responses ≤ 16,384 | KodCode val pass@1 of the last policy against the SFT model's (§6); the reasoning protocol after it (§5) |

## 3. Format

- **Thinking-mode only**: no non-thinking mode. An example renders as BOS, the prompt, `\n\n<think>\n`, the
  reasoning, `\n</think>\n\n`, the answer and BOS (`corpus/posttrain/sft.py` `THINK`, `END_THINK`). The markers are
  ordinary text (6 and 6 tokens here), so the vocabulary stays (§1). Nemotron-Cascade 2 opens a thinking turn with
  `<think>` and a newline (arXiv 2603.19220 §3.1.2); its predecessor's code traces are "formatted in the thinking
  mode" (arXiv 2512.13607 §3.2.3), the reasoning inline between the markers (Cascade-1) or in a separate field placed
  between them (CP-v2; `docs/corpus.md` §10).
- **Loss**: on the response, from the reasoning through the closing BOS; the prompt and its `\n\n<think>\n` are
  context, their labels ignored by the loader (`corpus/loader.py` `PretrainingBatches.assemble`, the context lengths
  in `<split>.context`, `docs/corpus.md` §10). Every prompt ends with `\n\n<think>\n` in SFT, RL and evaluation
  alike; the context is encoded alone, as a prompt is at inference.
- **Positions**: a packed row's documents take their positions in the row (`model/kernels/qk_prologue.cu`: the rotary
  table at the row position), while inference starts each prompt at 0. With RoPE the q·k score depends only on the
  offset n − m (arXiv 2104.09864 §3.2.2 Eq. 16), and the per-document attention segments (`docs/architecture.md` §1)
  and the windows are relative, so global and reset positions give the same attention, up to the rounding of the
  fp32 tables at different positions [inference].

## 4. SFT recipe

- **Plan: one stage, 2 epochs** (`run.py` `RunConfig.base`, `sft_epochs`; `conf/experiment/sft.yaml`;
  `train/schedule.py` `sft_plan`; `train/stages.py` `stage_batches`): the base run (`conf/experiment/pretrain.yaml`'s,
  named at the measured multiplier `${lr_multiplier:}`), its weights after `base_step` steps (default its last,
  69,874, read with its batch and peak from its `run` record, `run.py` `base_plan`) with fresh optimizer state; the
  base's batch, 1,474,560 tokens: 15 micro-batches of 3 rows of 32,768 tokens (`SFT_STAGE`, as the 32K stage);
  ⌊2 × 172,726 / 45⌋ = 7,676 steps, 11.32B tokens (58.02 per parameter) [arithmetic]. The first epoch reads every
  train row in the seeded permutation; the second is a fresh permutation (seed + 1) read for its first 172,694 rows,
  so no row repeats within an epoch, and the last 32 packed rows of the second permutation stay unread (7,676 × 45 =
  345,420 of 2 × 172,726 = 345,452 [arithmetic]); a resume continues at its row across the boundary. ≈10.9 h at the
  base's exclusive 287,476 tok/s (`docs/training.md` §7) [estimate], without the 9 validation passes (≈0.16 h
  [estimate]).
- **Basis of the epochs** [inference]: the published recipes run 1.5–8 epochs (Nemotron-Cascade 2's single packed
  stage "reaches optimal performance after approximately 1.5 epochs", arXiv 2603.19220 §3.1.1; Olmo 3 Think 2;
  others §2.3), and short traces need 2–3 (arXiv 2601.06953 Table 4). Nemotron-Cascade runs one epoch per stage over
  8.3 and 17.6 traces per prompt (arXiv 2512.13607 §3.1.1, §3.2.3); here 1.48 traces per prompt over 2 epochs give
  2.96 trace passes per prompt [arithmetic]. The epoch count and the corpus's cap are set together by a token budget
  (`docs/corpus.md` §10), not by a measured optimum.
- **Optimizer**: the base's NorMuon and AdamW with one learning rate, warmup over 40 steps and linear decay to zero
  (`docs/training.md` §3–§4); the peak is `learning_rate_scale` times the base run's recorded peak, 2.3454e-3.
- **No weight decay**, in every group (`sft_plan` `weight_decay` 0.0, which `train/optim.py` `optimizers` gives the
  NorMuon matrices and the AdamW tables; the 1-D parameters never decay): an independent choice, not tied to the
  peak's calibration, so the total decay does not depend on which learning rate wins. The closest-size recipe,
  MobileLLM-R1-950M, fine-tunes with none (arXiv 2509.24945 §3.2), as does OpenThoughts (arXiv 2506.04178 App. D.2);
  Nemotron-Cascade uses 1e-4 (arXiv 2512.13607 App. D Table 14) and Nemotron-Cascade 2 0.1 (Σ_t η_t λ ≈ 0.106 over
  its run, arXiv 2603.19220 App. B Table 7 [arithmetic]). The τ law (`docs/training.md` §3) sets a from-scratch
  timescale: at the SFT's 58.02 tokens per parameter it gives λ = 4.4 at 1/10 of the base peak, which would leave
  about 2% of the base weights' scale [arithmetic].
- **Peak calibration** (`conf/experiment/sft_lr.yaml`, `ablations/sft_lr.py` `sft_lr_fit`): calibrated, not
  borrowed, since Nemotron-Cascade's AdamW 5e-5 has no NorMuon counterpart. Three complete schedules at 1/10, 1/3
  and 1 times the base peak over the same first 2% of the first epoch's order (`sft_epochs` 0.02), trained only
  (`generate` "none": no evaluation, no step-0 validation), each scored by the response loss on the whole val split
  after its last step; an arm without a val record (a diverged one) stops the fit with its name. The rule: the vertex
  of the least-squares parabola of those losses against u = ln(scale) (`metrics.parabola_vertex`, as
  `docs/training.md` §5 calibrates the base's), which must have positive curvature and lie inside the grid (else the
  fit raises, and an arm 3× beyond the edge with the lowest loss is added); the full run's scale is the vertex times
  (run steps / arm steps)^−0.30, the steps from the plans (`conf/experiment/sft.yaml` passes its `sft_epochs` to the
  resolver).
- **Calibration result** (`results/posttrain/sft_lr.jsonl`): the arms ran 76 steps over the first 3,420 train rows;
  final losses 1.00505, 0.98195 and 1.03143 at 1/10, 1/3 and 1 [measured], vertex ×0.2575 (curvature 0.0279); times
  (7,676 / 76)^−0.30 = 0.2504 the scale is 0.064496, a peak of 1.5127e-4 [arithmetic].
- **The peak is an empirically transferred estimate** (≈1.5e-4), not a measured optimum of the full run. Three arms
  determine the parabola exactly: the fit has no residual, so neither the vertex nor the curvature has an error bar,
  and each arm is one seed. 40 of each arm's 76 steps are warmup (53%; 0.5% of the full run), outside the regime of
  the horizon fits below. The horizon exponent's basis [inference]: at a fixed batch the optimum falls with the run's
  length with exponent 0.32 (arXiv 2409.19913 §6 Eq. 5; 0.30 in its §4.1, Llama-1); the repository's d8 pair, net of
  its doubled batch by a square-root rule, gives 0.24–0.30 (`docs/training.md` §5) and DeepSeek LLM's law 0.29 at a
  fixed batch [arithmetic]; all of them pre-training measurements, their transfer to SFT from a trained base
  [inference]. The one counter-claim, that the optimum hardly moves under linear decay to zero (arXiv 2502.15938
  §4.2), is contradicted by that d8 pair. The band the exponent implies: 0.24 gives scale 0.085 (peak 2.0e-4), 0.5
  gives 0.026 (6.0e-5) [arithmetic]. Erring low is the cheaper error, forgetting growing with the learning rate
  (§2.3). Cost: ≈25 min of GPU [estimate].
- **Validation and checkpoints**: the response loss, mean nats per label (each response and its closing BOS), over
  the whole val split (1,720 rows, 53.5M labels) before the full run's first step, every 1,000 steps and after the
  last (`run.py` `validation_steps`, `train/loop.py` `validate`; `val` records); full states every 1,000 steps (resume
  points) and at the end; the full run trains with `generate` "none" (its run name carries it), and its last
  checkpoint is evaluated by the reasoning protocol (§5). **Per source**: `python -m eval.sft_val <run_name> <step>`
  scores a saved model on the same rows and labels and pools each val document's scored labels
  (`docs/corpus.md` §10, val labels) per group, per KodCode style and in all (the `val` record's loss, up to
  summation order), into `RUNS/<run>/sft-val-<step>.json` (`loss` in nats per label, `labels`, `documents`; and
  `document_losses`, each document's summed loss and label count, for paired comparisons across checkpoints). The
  intermediate checkpoints are points on the 2-epoch schedule's linear decay, not annealed: at the end of epoch 1
  (step 3,838) the lr is still 0.50 of the peak and at 1.5 epochs (step 5,757) 0.25 (`train/schedule.py` `lr_lambda`
  [arithmetic]), so a comparison of 1.0 or 1.5 epochs against 2 needs runs of those lengths with their own decay to
  zero [inference].
- **Runs and records**: an SFT run is a `run.py` run whose `RunConfig.base` names the base run; its directory, name
  and log follow the base's (`docs/training.md` §6). A sweep's records go to `results/posttrain/` when its runs start
  from a base checkpoint, else to `results/pretrain/` (`ablations/records.py` `results_dir`); the calibration's are
  `results/posttrain/sft_lr.jsonl` and `-curves.jsonl`; a single run's curves, `python -m ablations.records
  <run_name> sft` (`results/posttrain/sft-curves.jsonl`, `docs/training.md` §6).

## 5. Evaluation

- **Reasoning protocol** (`scripts/official_reasoning.sh`; the tools and pins: `docs/evaluation.md` §4): the
  checkpoint exported with the chat template (`inference/export_hf.py --chat`: BOS, the messages' contents joined by
  "\n\n" and stripped, then §3's `\n\n<think>\n`, the SFT context as `corpus/posttrain/build.py` `render` encodes it)
  and served by `vllm serve` with the `nanochat` reasoning parser (`eval/official_reasoning_parser.py`: the answer is
  the text after the last `</think>`; a response without it is all reasoning, its answer empty). Before generation,
  `eval/official_tokenize.py` checks that no message either tool will send holds `"\r"` (`docs/evaluation.md` §4)
  and, through the server's `/tokenize`, that the first problem's messages render as BOS +
  `encode_ordinary(prompt.strip() + "\n\n<think>\n")` (a calibration arm's checkpoint: equal for EvalPlus's 153 and
  LiveCodeBench's 431 ids [measured]).
- **Benchmarks**: HumanEval+ and MBPP+ by EvalPlus's OpenAI backend (`evalplus.codegen --backend openai` through
  `eval/official_evalplus.py`: its chat instruction, "Please provide a self-contained Python script that solves the
  following problem in a markdown code block:", and the prompt in a python fence), run greedy (`--greedy`: n 1, T 0)
  and sampled (n 20, T 0.6, top-p 0.95 set by its `openai_request.make_request`), scored by `evalplus.evaluate`;
  `--max_new_tokens` 32,768 reaches the requests only through `eval/official_evalplus.py` (EvalPlus's OpenAI backend
  drops it and asks 768, `docs/evaluation.md` §8).
  LiveCodeBench v6 (arXiv 2403.07974) by lcb_runner (`eval/official_lcb.py`: the problems of release_v6 with a
  contest date in [2024-08-01, 2025-05-01), Nemotron-Cascade's "08/24–05/25", arXiv 2512.13607 Table 1; its system
  message and question as two messages, line-ending-normalized, n 16, T 0.6, top-p 0.95), scored by its
  `codegen_metrics`. T 0.6 and top-p 0.95 are DeepSeek-R1's (arXiv 2501.12948 v1 §3) and Nemotron-Cascade's (arXiv
  2512.13607 §2); every request asks 32,768 new tokens, which vLLM caps at the context less the prompt, in place of
  Nemotron-Cascade's 64K. Reported: each tool's pass@1 (avg@n), in `docs/results.md`.
- **Endings apart from scores**: a response that never closes `</think>` scores 0 whatever its reasoning, so
  termination is reported beside pass@1, not folded into it. `eval/official_evalplus.py` asks the server for each
  sample's generated ids (`return_token_ids`) and records its finish reason (`stop` at BOS, `length` at the budget)
  and whether `</think>` closed (`<samples>.endings.jsonl`); `eval/official_reasoning.py` writes `summary.json`: per
  benchmark and mode, the closure rate, the finish reasons, pass@1 over every sample and pass@1 over the closed
  samples (pooled), on the base and the plus tests. The server stops loops as the termination probe does
  (`eval/official_loop_stop.py`, `docs/evaluation.md` §10): such a response has no `</think>`, so it scores 0 as it
  would after running on to the budget; each sample's `loop_stopped` and both tools' stop counts are in `summary.json`.
- **Contamination**: the SFT prompts' measured overlap with LiveCodeBench v6 is `docs/corpus.md` §10's.
- **The base's records** (likelihood, the official benchmarks; `docs/evaluation.md` §1, §4) are taken at every stage
  to measure what it loses, not as a gate.
- **No selection on the benchmarks**: RL runs for a fixed time budget and is evaluated on held-out KodCode val
  problems (§6); HumanEval+, MBPP+ and LiveCodeBench are read once per stage, at its end.
- **KodCode val** (`scripts/onpolicy.sh`; `onpolicy/rollouts.py`, `onpolicy/verify.py`, `onpolicy/outcomes.py`): 512
  problems of the SFT val split held out from the KodCode train pool by question and by test file
  (`onpolicy/prompts.py` `held_out`: SFT splits by problem, a normalized statement, which differs between the styles
  in which KodCode words one question, so another style of a train question can sit in val),
  4 samples each at T 0.6 and top-p 0.95 within 16,384 tokens and under the probe's loop stop, each answer run against
  its problem's tests (`onpolicy/verify.py`); pass@1 over the 2,048, a pass on a problem whose wrong program passes not
  counted. The SFT model: 0.325, pass@4 0.551, 43.5% of the responses loops [measured:
  `onpolicy-7676/eval-model/val/selection.json`]. RL's evaluations reuse these prompts and seeds (§6).

### 5.1 Natural termination and evaluation-time settings

- **Natural termination**: every full-state checkpoint is probed for whether its responses close and end, apart from
  their scores (method: `docs/evaluation.md` §10; results: `docs/results.md` §6). In Olmo3-7B's SFT termination rose
  from 24% at one epoch to ≈ 89% at 32 (arXiv 2602.11149 §4.2); whether it moves likewise over this run's epochs is
  what the probe measures [inference].
- **Secondary settings, disclosed**: forced closure (s1's budget forcing, arXiv 2501.19393 §3.1: `</think>` appended
  at B reasoning tokens) and a sampling temperature above the protocol's 0.6 change pass@1. s1 reports budget forcing
  in its headline numbers (arXiv 2501.19393 §3.1, §4); a forced stop recovered a correct answer in 0.364 of the
  unfinished samples of the officially post-trained Qwen3-0.6B after 20 thinking-mode on-policy distillation steps
  on problems it had solved 3–4 times in 4, and 0.049 on unsolved ones (arXiv 2609.37326 §5, App. E Table 18). Here
  B and the temperature are chosen on the probe's held-out SFT prompts only, fixed before a benchmark is read, and
  reported beside the protocol's natural number with the forced fraction; the step-2000 measurements they are chosen
  from are `docs/results.md` §6.4 (T 1.0 closes 59 of 64 HumanEval+ responses against 42 at T 0.6; forcing at 4K
  recovers 7 plus passes of 22 unclosed) [measured].

## 6. Code RL

RL with verifiable rewards from the SFT model (`scripts/rl.sh`) on the one GPU for a fixed time budget: a trainer
(`onpolicy/rl.py`) in the training image, on the native model and its fused kernels (`model/modeling.py`), beside a
sampler (`onpolicy/sampler.py`) in the vLLM image (`docker/Dockerfile.onpolicy`). The images hold different torch
builds (the training image's NGC torch with FlashAttention-4, vLLM's torch 2.13), so the two processes exchange the
policy and its batches through files under `/dev/shm`, each written then renamed: the trainer publishes policy k in
the export's format, the sampler loads it into its engine in place (vLLM's `reload_weights`), samples and judges
batch k, and the trainer updates on it. Sampling and the update alternate on the GPU, as on-policy RL requires.

- **Objective** (`onpolicy/rl.py`): one update per batch, so the sampler–trainer ratio is 1 (2 or 4 updates per batch
  "led to rapid entropy collapse around 100 steps", arXiv 2505.16400 §3.2; Nemotron-Cascade's code RL updates once,
  arXiv 2512.13607 §4.1.2); the loss summed over the batch's tokens and divided by their count (DAPO, arXiv 2503.14476
  §3.3, Eq. 12); A = (r − the group's mean) / the batch's std (arXiv 2508.08221 §4.1); a group whose rewards are all
  equal left out (zero advantage, DAPO §3.2); no KL and no entropy term (DAPO §2.3).
- **Reward** (`onpolicy/verify.py` `judge`): 1 when the answer, the text after the last `</think>`, holds code that
  passes every test of the problem (`check`), else 0 (no `</think>`, no code, a failure or a time-out); a problem
  whose wrong program (`stub`) passes is left out. A loop the sampler aborts scores 0, a failure rather than an answer
  cut short: an exact cycle repeats its tokens to the limit (§8) [inference]. To the reward is added DAPO's soft
  overlong punishment (arXiv 2503.14476 §3.4, Eq. 13; `onpolicy/rl.py` `overlong`): 0 up to 12,288 tokens, then
  falling linearly to −1 at the 16,384-token limit, so a cut response scores −1 (`L_CACHE` 4,096, DAPO's "soft punish
  cache", §4.1). Masking cut responses instead (overlong filtering, DAPO §3.4; "clear benefits when the token limit is
  short (e.g., 8K or 16K)", arXiv 2506.13284 §4.5.3) was tried first and left nothing against length: from the SFT
  model the cut share of the six batches was 3.7%, 3.9%, 7.9%, 9.2%, 11.4% and 19.5%, the mean response grew from
  7,676 to 9,735 tokens and the gradient norm from 0.147 to 0.179, while the pass rate over all responses fell from
  0.420 to 0.391 (the logged reward, over the uncut responses only, 0.436 to 0.479) [measured:
  `onpolicy-7676/rl/attempt-16k-masked-train.jsonl`], past the 10–15% truncation that "typically destabilized
  training" (arXiv 2510.13786 App. A.15). DAPO added the soft punishment on top of filtering and Clip-Higher, AIME24
  38 → 41 (Table 1); ScaleRL found a length penalty no better than interruptions at 16K GPU-hours (App. A.10), but
  an interruption trains answers the evaluation, which cuts at 16,384, never asks for.
- **Problems** (`onpolicy/prompts.py`): KodCode's train pool, its usable rows (named tests, no Docs subset): 180,911,
  of them 107,244 complete, 52,740 instruct and 20,927 online judge [measured]; a seeded permutation, batch k its k-th
  slice of 128 prompts × 8 samples (Lite PPO's 1,024, arXiv 2508.08221 §3; Nemotron-Cascade's 128 × 8, arXiv
  2512.13607 App. D Table 19). Every batch's prompts are new, so its reward is a held-out estimate of the policy.
  Python only: C++ comes from the SFT stage.
- **Sampling**: T 1.0 and top-p 1.0, the trainer's distribution exactly (DAPO §4.1; rollouts at T 0.6 destabilized
  ProRL's training against 1.2, arXiv 2507.12507 §5.1); at most 16,384 tokens: the SFT model's first T 1.0 batch
  averaged 7,582 tokens a response and 3.6% (37 of 1,024) reached the limit, where an 8,192 limit cut 42% (430 of
  1,024) [measured], above the 10–15% that destabilizes training (arXiv 2510.13786 App. A.15).
- **Policy match**: the trainer's weights are fp32 and its forward multiplies by bf16 copies rounded to nearest; the
  published policy (`inference/export_hf.py` `hf_weights`, the export's rounding) holds the same bf16 values, and the
  trainer's head runs in fp32 as vLLM's does, so the sampler draws from the policy the trainer differentiates up to
  the two engines' kernels. FP16 with sequence-level importance weights (§2.6) is not used: the native kernels are
  bf16 [inference: a short run on matched weights].
- **Optimizer**: AdamW at a constant 2e-6 [estimate: between Precision-RL's 1e-6 for its 14B dense and 30B MoE runs
  (arXiv 2510.26788 App. A) and Nemotron-Cascade's 3e-6–4e-6 at 8B–30B], β (0.9, 0.95), no weight decay, the gradient
  norm clipped at 1.0, ε 1e-15 (an ε near the gradients' scale clips them, arXiv 2510.13786 App. A.3).
- **Throughput** (GH200, batch 0) [measured]: one vLLM engine with the batch's 1,024 requests in flight holds the GPU
  at 100% utilization and 660–690 W. Driven by `LLMEngine.step` with token deltas, a response is aborted as a loop at
  the first check, every 128 tokens past 1,024, where its tail is an exact cycle or the protocol's rule holds
  (`onpolicy/rollouts.py` `cycle`, `stopped`): loops hold 68.9% of the SFT val rollouts' tokens, and stopping each
  exact cycle at its first detectable point removes 23.0% of the tokens and 38.6% of the KV reads (`eval-model/val`).
  Each finished response is judged at once by a pool of spawned workers, so the tests run while generation continues
  (18 s of judging after the last response). The trainer packs the trained responses into rows of 32,768 tokens (first
  fit decreasing, each its own attention segment) and computes the head and its loss only at the trained positions
  (`Head`: gradients in its forward, 8,192 rows at a time, the head in fp32). The masked attempt's batch 0: 421 s of
  generation (7.86M tokens, 18.7K tokens/s) and 31 s for the update (606 responses, 4.14M tokens), about 7.5 minutes
  an update (`onpolicy-7676/rl/attempt-16k-masked-train.jsonl`).
- **Budget and evaluation**: the trainer stops at the batch whose end would let the next overrun `BUDGET`, 2 hours:
  about 15 updates at 7.5 minutes [arithmetic]; Nemotron-Cascade's code RL ran 22 to 90 steps (arXiv 2603.19220
  App. B; arXiv 2512.13607 App. D Table 19). The sampler evaluates the first and the last policy on `eval-model/val`'s
  512 KodCode val prompts × 4 at the protocol's T 0.6 and top-p 0.95, with their seeds and its own loop stop: the SFT
  model's pass@1 is 0.334 there against 0.325 under `onpolicy/rollouts.py`'s round-based stop [measured]. The last
  policy is then evaluated by the reasoning protocol (§5) beside the SFT model (`scripts/post_sft_queue.sh`); results
  in `docs/results.md` §6.

## 7. Not in the pipeline

| Stage | Why not |
|---|---|
| A non-thinking mode | the format is thinking-mode only (§3) |
| LoRA, QLoRA | at hidden size 768 LoRA saves compute only below rank 128–205, where it lags full fine-tuning on code; QLoRA saves memory the run does not need (§2.2) |
| DPO on teacher pairs (§2.4) | Nemotron-Cascade goes from SFT to RL without it; preference at the model's own loops (FTPO, DPO) was tried and not adopted (§8) |
| Math RL | the SFT data holds no math; Nemotron-Cascade's math RL moved LiveCodeBench v6 by −0.3 to +2.2 (arXiv 2512.13607 §4.5) |
| RLHF, instruction-following, multi-domain and long-context RL | chat, judges or tools the model is not trained for; reward models of 72B–235B (arXiv 2512.13607 §4.2–§4.4; arXiv 2603.19220 §4.2–§4.6) |
| Software-engineering RL | a 72B or 120B judge, or agentic execution over 256K tokens (arXiv 2512.13607 §4.7; arXiv 2603.19220 §4.8) |
| On-policy distillation (§2.5) | one RL domain leaves no specialists to merge; Nemotron-Cascade 2's teachers are its own per-domain checkpoints (arXiv 2603.19220 §4.4) |

## 8. Loop removal after SFT: tried, not adopted

**Why.** The SFT model (step 7676) loops: on 512 KodCode val prompts × 4 samples at T 0.6 (top-p 0.95), pass@1 is
0.325 and 891 of 2,048 responses (43.5%) end in a loop, on 72.1% of the prompts; 888 of them are stopped by the
protocol's loop stop and 802 end in an exact cycle (`onpolicy-7676/eval-model/val/selection.json` `all`) [measured].
The termination probe (128 EvalPlus + 128 val prompts) stops 103 of 256 as loops, 40.2%
(`onpolicy-7676/eval-model/termination.json` `unclosed_loop_stopped`) [measured]. Every looped response scores as a
failure, so the loops bound pass@1 [inference]. Repetition is self-reinforcing: "the more times a sentence is repeated
in the context, the higher the probability of continuing to generate that sentence" (arXiv 2206.02369, abstract).
The loops are near-deterministic: the loop token leads its runner-up by a median 5.5 logits and p(loop token) at T 0.6
has median 1.00; continuations resampled from loop histories escaped 0 of 88; 545 of the 802 exact-cycle loops are
on prompts the model never solved in 4 samples (`onpolicy/train.py` docstring, branch `onpolicy-experiments`)
[unverified: no record in the run directory; 545 recomputed, see below].

| Attempt | What | Result | Record |
|---|---|---|---|
| FTPO, every parameter, lr 1.5e-6 | final-token preference (arXiv 2510.15061 §5; Antidoom [blog, not peer-reviewed]): reject the loop's first repeated token, choose the model's own alternatives; stop at held-out chosen_win 0.35 | stopped at step 100, chosen_win 0.366 [measured]; KodCode val pass@1 0.360 → 0.000 [unverified] | `onpolicy-7676/ftpo-attempts/full-parameter-lr1.5e-6.json` `monitor` |
| FTPO, last 5 layers + head, lr 1e-5 | same rows (4,887 train, 256 held out), guard: NLL of the SFT model's passing val answers ≤ +10% | guard tripped at step 20: NLL 0.1902 → 0.2364, +24.3% [measured]; chosen_win 0.027; pass@1 0.226 [unverified] | `onpolicy-7676/ftpo-attempts/last5-and-head-lr1e-5.json` `monitor`, `written.reason` |
| FTPO, retention-gated | + KL to the reference at 12 normal anchors per row (×200); gate at step 20: NLL rise ≤ 2%, specificity ≤ 0.2 | aborted at step 20: NLL +4.2%, specificity 0.357, chosen_win 0.029; nothing written [measured] | `onpolicy-7676/ftpo-attempts/retention-gated-aborted.json` |
| Forced commit (measurement) | `</think>` forced at the loop's cut; the model writes its answer | 24.7–25.6% of loops pass; +0.089 to +0.100 pass@1 if every loop committed [measured/arithmetic] | `onpolicy-7676/forced-commit/*/verdicts.jsonl` |
| Commit training | NLL of `</think>` at loop cuts + KL retention at anchors + one-sided tether on "</" | commit learned only as fast as at normal positions; gate failed, kept step 30, no effect [measured] | `onpolicy-7676/commit/train-log.json` |
| Forced commit at inference | the served loop stop forces `</think>` instead of ending the response | built, not adopted: it changes decoding, not the model [inference] | `eval/official_loop_stop.py` `LoopCommitLogitsProcessor` (branch) |

**FTPO diagnosis.** Moving the loop token's logit moved normal text with it: 66–98% of the change in the loop lead
appeared at normal positions, and the rejected tokens are the SFT model's top-1 at 57.7% of normal positions, so the
update acts as a global "trust the top-1 less" [unverified: 66–98% is stated in the `onpolicy/train.py` docstring;
57.7% has no record]. In the full-parameter run the held-out non-target tether term rose 0 → 1.86 by step 100 while
chosen_win reached 0.366 [measured, `full-parameter-lr1.5e-6.json` `monitor`]. FTPO's own sources report the risk:
"Training longer tended to degrade the model, often creating new doom-looping issues" [blog]; no result below 2.6B
parameters is published [blog].

**Forced commit.** The cut (`onpolicy/selection.py` `cut`) is placed at the cycle's first repeat (onset + period),
after one verbatim repeat (onset + 2 period), or at the first line start in the copy after one repeat; `</think>` is
appended there, as s1's budget forcing appends the end-of-thinking delimiter "to early exit the thinking stage and make
the model provide its current best answer" (arXiv 2501.19393 §3.1). Uplift = passes / 2,048 [arithmetic].

| Cut | Loops cut | Pass | Pass@1 uplift | Pass by prompt difficulty (SFT solved 0 / 1–2 / 3–4 of 4) |
|---|---|---|---|---|
| first repeat | 802 | 205 (25.6%) | +0.100 | 14.5% / 43.5% / 85.3% (545 / 223 / 34 loops) |
| mid-line, after one repeat | 802 | 199 (24.8%) | +0.097 | 14.1% / 42.2% / 82.4% |
| line start, after one repeat | 737 | 182 (24.7%) | +0.089 | 14.2% / 40.5% / 87.5% (500 / 205 / 32 loops) |

Records: `onpolicy-7676/forced-commit/{cut-first-repeat,cut-after-one-repeat,cut-line-start}/verdicts.jsonl`, the
difficulty split from `onpolicy-7676/eval-model/val/verdicts.jsonl` [measured; shares arithmetic]. At line starts
737 of the 802 exact-cycle loops have a cut (the copy holds a line start), and 730 of the 737 forced answers (99.1%)
ended cleanly (`cut-line-start/rollouts.parquet` `ended`) [measured]. Most loops sit on prompts the model cannot solve
(545 of 802 on 0-of-4 prompts), so committing recovers about a quarter of them [measured].

**Commit training** (`onpolicy/train.py` `commit_loss`; 5,414 training rows at line-start cuts, 256 held out; lr
1.5e-6; RETAIN 200; gate at step 20: guard NLL rise ≤ 2%, hazard ≤ 0.05; stop at commit_win 0.4). Held-out
`monitor` (`onpolicy-7676/commit/train-log.json`) [measured]:

| Step | log p(commit) at cuts | p(commit) at T 0.6 | log p(commit) at normal line starts | Specificity | Hazard | Guard NLL rise |
|---|---|---|---|---|---|---|
| 0 | −22.16 | 0.00 | −29.03 | – | 0 | 0 |
| 20 | −17.83 | 0.00 | −25.00 | 0.93 | 0 | −0.96% |
| 30 | −11.07 | 0.00 | −18.54 | 0.95 | 0.0009 | −0.98% |
| 40 | −5.19 | 0.07 | −15.00 | 0.83 | 0.148 | +8.34% |

Specificity is the gain at normal positions over the gain at cuts (0.83–0.95 from step 20): the model learned to commit
almost everywhere at once, adding 0.15 premature commits per response by step 40 (hazard) [measured]. Step 40 failed
the guard and the hazard bound; the last passing point, step 30, still commits with p ≈ 0 at T 0.6, so the written
model is the SFT model in effect (`written.step` 30) [measured; "in effect" inference].

**Verdict.** No variant removed loops without damaging normal text: FTPO's update is not specific to loop positions,
and commit training raises `</think>` at normal positions nearly as fast as at cuts. The loops are absorbed by the RL
stage instead (§6), whose reward already counts a loop as a failure [inference].

Provenance: code on branch `onpolicy-experiments` @ `7257448` (`onpolicy/train.py`, `onpolicy/selection.py`,
`onpolicy/rollouts.py`, `eval/official_loop_stop.py`); records under the SFT run's `onpolicy-7676/`; the planned FTPO →
DPO design as it stood: `docs/post-training.md` §8 at commit `9b106ca`.

## References

| arXiv | Title |
|---|---|
| 2104.09864 | RoFormer: Enhanced Transformer with Rotary Position Embedding |
| 2206.02369 | Learning to Break the Loop: Analyzing and Mitigating Repetitions for Neural Text Generation |
| 2305.14314 | QLoRA: Efficient Finetuning of Quantized LLMs |
| 2401.00788 | Astraios: Parameter-Efficient Instruction Tuning Code Large Language Models |
| 2401.14196 | DeepSeek-Coder: When the Large Language Model Meets Programming - The Rise of Code Intelligence |
| 2402.03300 | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models |
| 2403.07974 | LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code |
| 2405.09673 | LoRA Learns Less and Forgets Less |
| 2406.11931 | DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence |
| 2409.12186 | Qwen2.5-Coder Technical Report |
| 2409.19913 | Scaling Optimal LR Across Token Horizons |
| 2411.04905 | OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models |
| 2501.12948 | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning |
| 2501.19393 | s1: Simple test-time scaling |
| 2502.12143 | Small Models Struggle to Learn from Strong Reasoners |
| 2502.15938 | Straight to Zero: Why Linearly Decaying the Learning Rate to Zero Works Best for LLMs |
| 2503.02951 | KodCode: A Diverse, Challenging, and Verifiable Synthetic Dataset for Coding |
| 2503.14476 | DAPO: An Open-Source LLM Reinforcement Learning System at Scale |
| 2503.18892 | SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild |
| 2503.20783 | Understanding R1-Zero-Like Training: A Critical Perspective |
| 2504.04030 | OpenCodeInstruct: A Large-scale Instruction Tuning Dataset for Code LLMs |
| 2504.13837 | Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? |
| 2504.15777 | Tina: Tiny Reasoning Models via LoRA |
| 2504.21233 | Phi-4-Mini-Reasoning: Exploring the Limits of Small Reasoning Language Models in Math |
| 2505.00949 | Llama-Nemotron: Efficient Reasoning Models |
| 2505.09388 | Qwen3 Technical Report |
| 2505.16400 | AceReason-Nemotron: Advancing Math and Code Reasoning through Reinforcement Learning |
| 2505.21297 | rStar-Coder: Scaling Competitive Code Reasoning with a Large-Scale Verified Dataset |
| 2505.22617 | The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models |
| 2505.24864 | ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models |
| 2506.03136 | Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning |
| 2506.03524 | Seed-Coder: Let the Code Model Curate Data for Itself |
| 2506.04178 | OpenThoughts: Data Recipes for Reasoning Models |
| 2506.13284 | AceReason-Nemotron 1.1: Advancing Math and Code Reasoning through SFT and RL Synergy |
| 2506.13404 | Effective Learning for Small Reasoning Models: An Empirical Study on 0.5B Reasoning LLMs |
| 2506.13585 | MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention |
| 2507.12507 | Scaling Up RL: Unlocking Diverse Reasoning in LLMs via Prolonged Training |
| 2507.18071 | Group Sequence Policy Optimization |
| 2508.05170 | Posterior-GRPO: Rewarding Reasoning Processes in Code Generation |
| 2508.08221 | Tricks or Traps? A Deep Dive into RL for LLM Reasoning |
| 2509.24945 | MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes |
| 2510.06101 | The Valley of Code Reasoning: Scaling Knowledge Distillation of Large Language Models |
| 2510.13786 | The Art of Scaling Reinforcement Learning Compute for LLMs |
| 2510.15061 | Antislop: A Comprehensive Framework for Identifying and Eliminating Repetitive Patterns in Language Models |
| 2510.26788 | Defeating the Training-Inference Mismatch via FP16 |
| 2511.07833 | Murphy: Multi-Turn GRPO for Self-Correcting Code Generation |
| 2512.02556 | DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models |
| 2512.13607 | Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for General-Purpose Reasoning Models |
| 2512.13961 | Olmo 3 |
| 2512.23165 | Evaluating Parameter Efficient Methods for RLVR |
| 2601.02780 | MiMo-V2-Flash Technical Report |
| 2601.06953 | X-Coder: Advancing Competitive Programming with Fully Synthetic Tasks, Solutions, and Tests |
| 2601.20614 | Harder Is Better: Boosting Mathematical Reasoning via Difficulty-Aware GRPO and Multi-Aspect Question Reformulation |
| 2602.11149 | Data Repetition Beats Data Scaling in Long-CoT Supervised Fine-Tuning |
| 2602.13367 | Nanbeige4.1-3B: A Small General Model that Reasons, Aligns, and Acts |
| 2602.15763 | GLM-5: from Vibe Coding to Agentic Engineering |
| 2603.00729 | Qwen3-Coder-Next Technical Report |
| 2603.12151 | IsoCompute Playbook: Optimally Scaling Sampling Compute for LLM RL |
| 2603.15611 | Code-A1: Adversarial Evolving of Code LLM and Test LLM via Reinforcement Learning |
| 2603.19220 | Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation |
| 2604.00626 | A Survey of On-Policy Distillation for Large Language Models |
| 2604.02721 | GrandCode: Achieving Grandmaster Level in Competitive Programming via Agentic Reinforcement Learning |
| 2604.13010 | Lightning OPD: Efficient Post-Training for Large Reasoning Models with Offline On-Policy Distillation |
| 2607.15161 | On-Policy Delta Distillation |
| 2607.24653 | Kimi K3: Open Frontier Intelligence |
| 2608.24135 | Robust Code RL via Faulty-Code-Driven Test case Synthesis and Dense Reward Shaping |
| 2609.20807 | Score Centering Stabilizes Off-policy Reinforcement Learning |
| 2609.32444 | Rethinking Training-Inference Mismatch in LLM Reinforcement Learning: Where It Arises and How to Correct It |
| 2609.35259 | On-Policy or Off-Policy Learning? A Systematic Study of Distillation Dynamics |
| 2609.37326 | Solving Without Stopping: On-Policy Distillation at Small Scale |
