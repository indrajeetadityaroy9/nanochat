# Post-training

What would turn the base model into a code model that answers a prompt and reasons before it answers: the pipelines
published in 2025–2026 (§2), how they update the weights (§3), supervised fine-tuning and off-policy distillation
(§4), preference tuning (§5), on-policy distillation (§6), reinforcement learning with verifiable rewards (§7) and its
code rewards (§8), the GRPO variants (§9), the data (§10) and the pipeline chosen from them (§11). §11.2's SFT is
implemented (its corpus, its runs through `run.py`); nothing else here is: the repository pre-trains
(`docs/training.md`) and evaluates (`docs/evaluation.md`) the base model it starts from.

Markers: [arithmetic] derived from code numbers; [inference] reasoning, not measured; [estimate]; [unverified] a
figure without a record, or a paper value its extracted text lacks. Below 1B the cited evidence is MobileLLM-R1
(140M–950M), Qwen3-0.6B, Qwen2.5-Coder-0.5B and two 0.5B math studies; every transfer to the 522M base model is
[inference].

## Code

| File | Section |
|---|---|
| `posttrain/sft.py` (`NAME`, `SFT_STAGE`, `THINK`, `END_THINK`) | §11.1 markers; §11.2 the corpus's stage |
| `posttrain/sft_data.py` | §11.1 format; §11.2 data, filters, split by problem, packing |
| `corpus/loader.py` (`PackedSplit.contexts`, `PretrainingBatches.assemble`) | §11.1 loss: the context's labels ignored |
| `run.py` (`RunConfig.base`, `base_step`, `sft_rows`, `generate` "none"; `plan_of`, `validation_steps`, `job`), `conf/experiment/sft.yaml` | §11.2 run |
| `train/schedule.py` (`sft_plan`, `RunPlan.weight_decay`, `last_stage`), `train/stages.py` (`val_batches`), `train/loop.py` (base weights, `validate`) | §11.2 run |
| `conf/experiment/sft_lr.yaml`, `ablations/sft_lr.py` (`sft_lr_fit`, `sft_lr_scale`), `ablations/records.py` (`record`) | §11.2 peak calibration |

The data of §10 is in `artifacts/posttrain/<id>`; the SFT corpus in `artifacts/corpus/tokenized/sft-<data key>/`, its
counts in `stats.json` there.

## 1. Starting point

- **The model**: d24, 522,252,336 parameters (`docs/architecture.md` §1), 103.03B tokens (`docs/training.md` §2),
  a completion model: the tokenizer's special tokens are BOS and the three FIM sentinels (`corpus/tokenizer.py`
  `SPECIAL_TOKENS`), so there is no chat template. An example is a prompt and a response, the reasoning between
  plain-text `<think>` markers and then the answer (§11.1); the loss falls on the response and its closing BOS.
- **The vocabulary**: 32,768 ids of the repository's own tokenizer (`docs/corpus.md` §5); no published model shares
  it, which decides how teachers can be used (§6).
- **The measures** (`docs/evaluation.md` §3–§4): greedy pass@1, pass@k from the sampled mode's 20 completions per
  problem, and the parse failures. At 5 tokens per parameter the d24 runs scored HumanEval+ 0.067–0.110, MBPP+
  0.114–0.132 and parsed 0.775–0.894 of MBPP+ (`docs/training.md` §5); the base model's own values come from its
  evaluation after training. A post-trained model is measured in its reasoning format (§11.4).
- **What to keep**: the 32K context. Infilling stays the base model's: no source of §10 holds FIM examples, and the
  post-trained model answers in one format (§11.1).
- **Contamination**: every source passes the pretraining rule before use, a 10-gram of words shared with a scored
  benchmark's prompt or reference drops the example (`docs/corpus.md` §3); OpenCoder removes HumanEval/MBPP entry
  points and 10-gram overlaps (arXiv 2411.04905 §4.4).

## 2. Current pipelines

| Model (size) | Stages after pre-training | Source |
|---|---|---|
| DeepSeek-V3.2 | one RL specialist per domain (math, programming, reasoning, agents, agentic coding, search) from the same base; their outputs as SFT data; one mixed GRPO stage with rule-based outcome rewards and a length penalty, rubric-based generative rewards for general tasks | arXiv 2512.02556 §3 |
| Kimi K3 (2.8T MoE, 104B active) | SFT; RL experts for three domains at three reasoning-effort levels; multi-teacher on-policy distillation into one model | arXiv 2607.24653 §4.1 |
| GLM-5 (744B, 40B active) | SFT; reasoning RL; asynchronous agentic RL; general RL; on-policy distillation from the preceding stages' final checkpoints | arXiv 2602.15763 §3 |
| MiMo-V2-Flash (309B, 15B active) | SFT; domain teachers trained by RL or SFT; multi-teacher on-policy distillation | arXiv 2601.02780 §4.1 |
| Qwen3-Coder-Next (80B, 3B active) | mid-training; SFT; experts (web development, user experience, single-turn RL on unit tests, multi-turn software-engineering RL) distilled into the SFT model | arXiv 2603.00729 §3–§4 |
| Olmo 3 Think (7B, 32B) | SFT; DPO on delta pairs (§5); RL with verifiable rewards | arXiv 2512.13961 §4 |
| Nemotron-Cascade (8B, 14B) | two SFT stages; RL by domain in turn: RLHF, instruction following, math, code, software engineering | arXiv 2512.13607 §3–§4 |
| Nemotron-Cascade 2 (30B MoE, 3B active) | one packed SFT stage; RL in turn: instruction following, multi-domain, multi-domain on-policy distillation, RLHF, long context, code, software engineering | arXiv 2603.19220 §3–§4 |
| Seed-Coder-8B reasoning | long-CoT warmup on thousands of examples, from the base model; GRPO | arXiv 2506.03524 §3.2 |
| Qwen3 0.6B–14B | off-policy distillation, then on-policy distillation from Qwen3-32B or Qwen3-235B-A22B; no RL | arXiv 2505.09388 §4.5 |
| Phi-4-mini-reasoning (3.8B) | distillation as mid-training; distillation SFT; DPO on rejected teacher rollouts; RL | arXiv 2504.21233 §3 |
| MobileLLM-R1 (140M, 360M, 950M) | logit distillation from Llama-3.1-8B-Instruct in mid-training; general SFT; reasoning SFT; no RL | arXiv 2509.24945 §3.2, §7.4 |

What these replaced:

| Earlier form | Current form | Section |
|---|---|---|
| LoRA or QLoRA instruction tuning | full-parameter updates in every report above | §3 |
| unverified instructions written by an LLM (Evol-Instruct, OSS-Instruct, 2023) | execution-checked, difficulty-filtered chain-of-thought traces from a reasoning teacher | §4 |
| offline DPO on judged pairs | DPO on strong-versus-weak model pairs, or on-policy pairs labeled in a sandbox | §5 |
| RL for small models | on-policy distillation from a stronger teacher | §6 |
| PPO's value model; GRPO's KL, std normalization and several PPO epochs per batch | group baseline, no KL, mean-only or batch-level normalization, one update per batch or truncated importance weights | §7 |
| a reward model trained on compiler outcomes | executing verified tests | §8 |
| sequential RL stages | one mixed stage, or stages consolidated by on-policy distillation | §6, §7 |

## 3. Full-parameter updates

- **DeepSeek-Coder uses no QLoRA.** Its instruction tuning, in full: "We develop DeepSeek-Coder-Instruct by
  enhancing the DeepSeek-Coder-Base through instruction-based fine-tuning using high-quality data … structured by the
  Alpaca Instruction format … For training, we use a cosine schedule with 100 warm-up steps and an initial learning
  rate 1e-5. We also use a batch size of 4M tokens and 2B tokens in total" (arXiv 2401.14196 §3.7). The text names
  no LoRA, QLoRA, low-rank or parameter-efficient method, nor says that every weight trains [inference: it does].
  DeepSeek-Coder-V2: SFT on 300M tokens of instructions at lr 5e-6, batch 1M tokens, 1B tokens; then GRPO on about
  40K prompts with test cases (arXiv 2406.11931 §3.5); no PEFT either.
- **The other reports**: none of the models in §2, nor Qwen2.5-Coder, OpenCoder, AceReason, rStar-Coder or
  OpenThoughts, describes an adapter for its own training. GrandCode uses LoRA only for test-time RL on one problem
  (arXiv 2604.02721 §10.3).
- **Measured on code**: "FFT generally leads to the best downstream performance across all scales" (StarCoder 1B–16B,
  PEFT on 8-bit models; arXiv 2401.00788 §1, Fig. 1); at 1B LoRA trails P-Tuning and (IA)³ (§1). Students below 14B
  train with full parameters, LoRA above (arXiv 2502.12143 §3.1). Distillation plus SFT with LoRA on
  Qwen2.5-0.5B-Instruct scored 0.7% on OlympiadBench and 1.2% on MATH500, and adding RL collapsed (arXiv 2506.13404
  §4.2–§4.3; its tables are empty in the extracted text [unverified]).
- **RL with adapters** (1.5B, already distilled): LoRA rank 32 on every linear layer 42.5 against full parameters'
  44.9 and DoRA's 46.6, PiSSA 0.2 (arXiv 2512.23165 Table 3); Tina's rank-32 GRPO reached its best checkpoint for $9
  (arXiv 2504.15777 Table 1), and 4 of its 5 models scored above their full-parameter baselines on the same data
  (Table 3).
- **Capacity** (code): in continued pre-training of Llama-2-7B on StarCoder-Python, LoRA at rank 256 peaked at 20B
  tokens with HumanEval 0.224, about full fine-tuning's 0.218 at 4B, and full fine-tuning reached 0.263; in code
  instruction tuning ranks 16 and 64 scored 0.358 and 0.417 at epoch 4 and rank 256 0.498, against full
  fine-tuning's 0.497 at epoch 8 (arXiv 2405.09673 §4.1); full fine-tuning on code "does not learn low-rank
  perturbations" (§4.6). LoRA matches full fine-tuning on "small-to-medium-sized" SFT sets and falls behind on sets
  that "exceed LoRA capacity"; in RL it matches "even with ranks as low as 1", at about 10× full fine-tuning's
  learning rate (Schulman and Thinking Machines Lab, "LoRA Without Regret", 2025).
- **Compute and memory**: per N×N weight matrix, full fine-tuning costs 3N² multiply-adds per token and LoRA
  2N² + 6NR ("LoRA Without Regret"); for a d_in × d_out matrix, 3·d_in·d_out against 2·d_in·d_out + 3R(d_in + d_out)
  [arithmetic: the same count], so LoRA saves compute only below the rank d_in·d_out / (3(d_in + d_out)): 128 for
  each of d24's `q_proj`, `k_proj`, `v_proj` and `o_proj` (separate 768 × 768 matrices, `model/modeling.py`
  `Attention`; 192 only for one adapter on a fused qkv) and 205 for the MLP's matrices [arithmetic]; it saves nothing
  on attention, 34.7% of the 32K step's GPU time [unverified] (`docs/training.md` §7). QLoRA's 4-bit base exists to
  fit "a 65B parameter model on a single 48GB GPU" (arXiv 2305.14314 Abstract).
- **Here** [inference]: full parameters for the SFT (§11.2): its 2.3–27.5B tokens are the pre-training-like volume
  where LoRA falls behind, and the rank that kept up on code (256) costs more than full fine-tuning at hidden size
  768. Memory gives no reason for adapters: pre-training holds the fp32 weights, gradients and both optimizers' state
  of all 522,252,336 parameters in 7.47 GiB (`docs/training.md` §7). LoRA fits RL (§11.3).

## 4. Supervised fine-tuning and off-policy distillation

Off-policy distillation is SFT on a teacher's text, so it needs no shared vocabulary (§6). Published effects at 1.5B
and below:

| Model | Data | Result | Source |
|---|---|---|---|
| Qwen2.5-Coder-0.5B, base → instruct | tens of millions of synthetic instructions, then millions chosen by rejection sampling; FIM instructions; offline DPO | HumanEval 28.0 → 61.6, HumanEval+ 57.3, LiveCodeBench 2.0 | arXiv 2409.12186 §4, §6.1, Table 16 |
| DeepSeek-Coder-1.3B, base → instruct | 2B tokens (§3) | HumanEval 34.8 → 65.2 | arXiv 2401.14196 Table 3 |
| OpenCoder-1.5B, base → instruct | two SFT stages (§10) | HumanEval+ 49.4 → 67.7, MBPP+ 58.7 → 61.9 | arXiv 2411.04905 Tables 6, 7 (not controlled) |
| Qwen2.5-Coder-1.5B: OpenCodeInstruct against its Instruct | 5M instruction–solution pairs, sequences ≤ 2,048 tokens | HumanEval+ 73.8 against 66.5, MBPP+ 68.3 against 59.4 | arXiv 2504.04030 Table 3 |
| Qwen2.5-Coder-1.5B-Instruct + rStar-Coder | 580K verified long-CoT solutions, 6 epochs | LiveCodeBench 6.5 → 40.1, HumanEval+ 81.7, MBPP+ 60.8 | arXiv 2505.21297 Table 2, §4.2 |
| MobileLLM-R1-950M, after a distillation mid-training stage | general SFT, then 6.2M reasoning samples (2.2M from OpenCodeReasoning-2) at 32K | LiveCodeBench v6 19.9 (Qwen3-0.6B 14.9) | arXiv 2509.24945 Tables 2, 5, 7 |
| Qwen3-0.6B, distilled (§6) | — | LiveCodeBench v5 12.3 thinking, 3.6 not thinking | arXiv 2505.09388 Tables 19, 20 |

- **Prompts before answers**: in a fit z = a·log₂x + b·log₂y + c of the score against unique prompts x and answers
  per prompt y (both standardized; seven runs, each trained until its accuracy plateaus; 7B, math), a = 4.831 and
  b = 2.635 (R² 0.989; arXiv 2506.13284 §4.4.2); at a fixed budget 64K prompts × 1 beat 16K × 4 and 8K × 8 (arXiv
  2601.06953 §3.2; the ordering only, its scores are in Fig. 3); 37.7K problems × 16 solutions scored LiveCodeBench
  54.7 against the full set's 57.3 (7B; arXiv 2505.21297 App. A.1). OpenThoughts samples 16 answers per question to
  grow a source (arXiv 2506.04178 §4.4).
- **Epochs**: gains plateau at 5–6, and more answers per prompt "also greatly benefits from multi-epoch training"
  (arXiv 2506.13284 §4.4.3); long traces need 8–10 epochs, short ones 2–3; at 8 epochs long traces scored
  LiveCodeBench v5 60.3 against short ones' 43.1 (7B; arXiv 2601.06953 Table 4).
- **Long or short traces at small sizes**: with 7.5K MATH prompts, short chains beat long ones at 3B and below
  (Qwen2.5-0.5B 19.5 against 14.8, Qwen2.5-3B 43.4 against 40.3) and a small teacher beats a large one (arXiv
  2502.12143 Tables 1–2); one long example in five is best at 3B (Table 3); base models show "a more significant
  learnability gap" than the Instruct students (§3.4). The 950M and 1.5B models above trained on 580K–6.2M long
  traces [inference: the gap was measured at 7.5K prompts and is untested at that data scale below 1B].
- **The valley** (7B and 8B students, LiveCodeBench): 1K distilled examples more than halve the baseline, 10K lift it
  about 50%, 30K a further 50–100%; easy and medium problems help more than hard ones (+41% against +7%); correct and
  incorrect R1 traces distill alike (0.185 against 0.182) (arXiv 2510.06101 §4, Fig. 1, Table 2).
- **Sources and filters** (OpenThoughts, 7B student; 31.6K examples per arm in Tables 4, 5 and 8): two sources beat
  sixteen (41.3 against 36.4, Table 4); for code the best question filter is difficulty judged by an LLM (Table 5);
  QwQ-32B teaches better than DeepSeek-R1 by 1.9 average points on code data (code average 2.3) though it scores
  lower itself (Table 8). No answer filter beat none: verification lowered the 7B (41.9 against 45.0) and raised the
  32B (64.5 against 62.1) on OpenThoughts-114K (Table 15), and unit-test filtering scored LiveCodeBench 36.0 against
  38.5 on 16K examples (Table 18); removing the examples whose traces exceed 8,192 tokens cost 17.6% relative on a
  12K-instance set (Table 22).
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
- **Here** [inference]: reasoning traces only, whole rows up to 32,768 tokens, one packed stage of one epoch
  (§11.2): Cascade-1's Python code traces, at most 4 per prompt in each group (677,871 of 2,694,422 after the
  filters), past the valley; CP-v2's Python and C++ traces that fit (183,768) and its Exercism function-stub traces
  (68,500).

## 5. Preference tuning

- **Delta learning** (Olmo 3 Think 7B): the chosen response from Qwen3-32B thinking, the rejected one from
  Qwen3-0.6B thinking, unfiltered; "the quality of preference data depends primarily on the quality of the delta
  between chosen and rejected responses rather than the quality of either response individually" (arXiv 2512.13961
  §4.3). Against the SFT checkpoint's average 70.3, SFT continued on the chosen responses scored 64.5 and delta DPO
  72.9 (Table 21). RL from the DPO checkpoint beat RL from SFT, average 74.1 against 71.9 and LiveCodeBench 73.4
  against 70.7: "In no situation does the SFT model improve over the DPO model after RL" (§4.5, Table 22).
  Length-normalized DPO, β = 5, lr 8e-8, 150K pairs, one epoch (Table 48).
- **Rollout pairs**: DPO on Phi-4-mini-reasoning's rejected R1 rollouts took AIME 43.3 → 50.0 (arXiv 2504.21233
  Table 3); Seed-Coder samples about 20K pairs on-policy and labels them by sandbox tests (arXiv 2506.03524 §3.1).
- **Here** [inference]: not used; Nemotron-Cascade goes from SFT to RL (§11.5).

## 6. On-policy distillation

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
  Tables 1–2); AIME24 "rises steeply in the first 50 OPD steps and saturates" (Fig. 3b). At 1.7B, OPD lowered
  thinking-mode math 59.2 → 57.1, and "prolonged training can even lead to performance degradation"; rewarding the
  teacher's gain over its own pre-reasoning base (OPD²) raised code 10.5 → 29.4 against OPD's 21.0 (arXiv 2607.15161
  Tables 2–5, §3.5).
- **Controlled at 1B–1.5B** (arXiv 2609.35259): on-policy and off-policy rollouts reach 72% and 73% in-domain; the KL
  direction matters more, forward KL robust to the rollout policy and reverse KL needing the student's rollouts (§4.2,
  §5.2); on-policy rollouts add 10–15% pass@k on a harder task, and off-policy checkpoints gave the strongest RL
  afterwards (§5.3).
- **Constraints**: one vocabulary: "Teachers and students are taken from the same model family to ensure a shared
  tokenizer and vocabulary" (arXiv 2609.35259 §4.1); "Most OPD objectives assume that teacher and student share the
  same vocabulary" (arXiv 2604.00626 §8). One teacher for SFT and OPD: a different SFT teacher costs 2–6.8 AIME24
  points (arXiv 2604.13010 Tables 3–4).
- **Teachers from the model itself**: GLM-5's are its own earlier stages, recovering what later stages lost (§3.5);
  MiMo-V2-Flash's teacher "can be a specialized RL-derived model with strong capabilities, a different SFT model, or
  even the student model itself" (arXiv 2601.02780 §4.1).
- **Here** [inference]: the vocabulary (§1) rules out every published teacher for OPD; external teachers enter as text
  (§4–§5); with one RL domain there are no specialists to consolidate (§11.5).

## 7. Reinforcement learning with verifiable rewards

GRPO (arXiv 2402.03300 §4.1) samples G outputs per prompt and uses the group's mean reward as the baseline in place of
PPO's value model, "another model of comparable size". The 2025–2026 recipes keep the group baseline and change the
rest:

| Component | Current choice | Evidence |
|---|---|---|
| KL to a reference | none | "removing it does not lead to over-optimization or destabilized training" (arXiv 2512.13961 §4.4); β = 0 in AceReason, Nemotron-Cascade, Seed-Coder, GLM-5 and ScaleRL; a reference KL "leads to a degradation in performance" (arXiv 2505.22617 §4.1). Kept by DeepSeek-V3.2 with an importance-corrected estimator, weak or none for math (§3.1), and by ProRL from a distilled 1.5B with periodic reference resets (arXiv 2505.24864 §2.3.1) |
| Advantage | r − mean(r), no std division | the std term's difficulty bias (§9, Dr. GRPO); OlmoRL (arXiv 2512.13961 §4.4) and DeepSeek-V3.2 (Eq. 5); ScaleRL divides by the batch's std, "marginally better" (arXiv 2510.13786 §3.2) |
| Aggregation | over the batch's tokens | DAPO (§9), OlmoRL, AceReason; prompt-level in ScaleRL; token-level "best suited for base models" (arXiv 2508.08221 §4.3) |
| Update | one gradient step per batch, so the ratio is 1 (REINFORCE with group-normalized rewards); truncated importance weights once rollouts lag (CISPO, §9) | 2 or 4 updates per batch "led to rapid entropy collapse around 100 steps" (arXiv 2505.16400 §3.2); Nemotron-Cascade likewise (§4.1) |
| Clipping, where a PPO-form loss stays | ε_low 0.2, ε_high 0.28 or 0.272 | GLM-5 §3.2; Seed-Coder §3.2; OlmoRL App. Table 49 (§9, DAPO) |
| Sampler–trainer mismatch | FP16 in sampler and trainer; truncated importance weights; fp32 logits | FP16's sequence-level mismatch is "approximately 24× smaller" than BF16's (arXiv 2510.26788 §3.5); in BF16 GRPO collapsed after peaking (73% VeRL, 84% Oat training accuracy) and only sequence-level MIS stayed stable, which FP16 beat (§4.2); OlmoRL caps the weight at 2; GLM-5 masks tokens outside [1/2, 2] (§3.2); fp32 logits raised ScaleRL's asymptote 0.52 → 0.61 (§3.2); in a dense model the displacement is "concentrated near zero and bounded, consistent with bf16 rounding error" (arXiv 2609.32444 §3.1) |
| Prompts | drop groups whose rewards are all equal and refill the batch; drop prompts solved ≥ 90% | OlmoRL active sampling; ScaleRL's No-Positive-Resampling (§3.2); Nemotron-Cascade drops problems solved 8/8, or 0/8 by a strong model (§4.6); 2.2K hard prompts beat the full 49K, AIME24 65.9 against 63.3 (arXiv 2505.16400 Table 3) |
| Rollouts | 8–16 per prompt | OlmoRL 8, AceReason 8–16, ScaleRL 16; more as compute grows, fewer on hard sets (arXiv 2603.12151 §4) |
| Temperature | entropy about 0.3 | "setting the training temperature such that the temperature-adjusted entropy remains around 0.3 typically leads to effective RL training" (arXiv 2506.13284 §4.5.2); code RL is "sensitive to the temperature" (arXiv 2512.13607 §4.6) |
| Length | short first, then longer | 8K → 16K → 24K → 32K beat starting at 16K or 24K (arXiv 2505.16400 §4.3); an 8K stage first, AIME25 56.7 against 51.8 (arXiv 2506.13284 §4.5.4); masking truncated outputs helps at 8K–16K and hurts at 32K (Table 2); 10–15% truncated outputs destabilize (arXiv 2510.13786 App. A.15) |
| Entropy | a candidate: covariance-targeted KL-Cov or Clip-Cov | 32B average 52.2 against clip-higher's 47.2 and GRPO's 45.8 (arXiv 2505.22617 Table 2) |

- **Small models**: MobileLLM-R1-950M after RL from the base scored GSM8K 57.0 against SFT's 74.0; "For
  high-performing small models fully fine-tuned on SFT data, additional RL does not observe a significant performance
  improvement" (arXiv 2509.24945 §7.4). Where RL helps at 1.7B and below it starts from distilled or instruct models:
  ProRL at 1.5B (code +14.4% over more than 2K steps, §3), Code-A1 at 1.5B and Murphy at 1.7B (§8). From a base model
  at 0.5B on math: average 12.1 → 20.9 (arXiv 2503.18892 Table 1). Score Centering trains Qwen3-0.6B under a
  deliberately mismatched sampler (arXiv 2609.20807 §5).
- **Order**: math-only RL raised LiveCodeBench v5 by 6.8 at 7B and 5.8 at 14B, and code RL from a math-RL checkpoint
  ends higher (arXiv 2505.16400 Table 1, §4.3); Nemotron-Cascade orders RL from general to specialized domains and
  finds it "resistant to catastrophic forgetting" (§4.1); DeepSeek-V3.2 merges the domains into one stage (§2).
- **Coverage**: base models catch up with RL-trained ones at large k (arXiv 2504.13837 §3.2), and RL left Qwen3-8B's
  pass@64 unchanged (§6); AceReason's RL raised LiveCodeBench pass@k up to k = 64 by about 10 points (§4.3), and ProRL
  solves tasks where the base never succeeds (§4); delta DPO raised Olmo 3's pass@k (§4.5).
- **Here** [inference]: sampler and trainer both in FP16 (§11.3), the remaining mismatch under sequence-level truncated
  weights. On the final checkpoint (CPU, three code files of 1,024 tokens [measured]) the largest activations are 145
  (residual stream), 241 (ReLU²), 46.8 (attention score) and 29.5 (logit before the cap), and the weights bound the
  last three by 11,195, 68.2 and 75, all below fp16's 65,504; with activations rounded to fp16 the mean KL to fp32 is
  1.5e-5, to bf16 1.0e-3. 77–99.9% of the residual stream's gradient values lie below fp16's smallest normal
  (6.1e-5), so the loss is scaled.

## 8. Code rewards

- **Binary on verified tests**: reward 1 "if and only if the generated code successfully passes all test cases"
  (arXiv 2505.16400 §3.3); Nemotron-Cascade (§4.6), KodCode (arXiv 2503.02951 §4.1) and Seed-Coder alike. The tests
  decide the result: false negatives (unverifiable tests, a reference that fails them) leave RL at "a sub-optimal
  point"; false positives (weak tests, DeepCoder's set among them) bring "a sudden drop on testing set accuracy
  midway" (arXiv 2505.16400 §4.3); AceReason drops special-judge, interactive, untested and weak-test problems (App.
  A.4).
- **Pass fraction**: at 32B, rewarding the fraction of tests passed beat the binary reward by 0.55–1.19 LiveCodeBench
  points, most with unvalidated tests (arXiv 2608.24135 Table 3); X-Coder's fraction reward drew late reward hacking
  that exploits "edge cases for partial rewards" (arXiv 2601.06953 §3.3); Nanbeige4.1-3B rewards the pass rate and adds
  an efficiency term only once every test passes (arXiv 2602.13367 §2.3); OlmoRL tried both, its final choice unstated
  (arXiv 2512.13961 §4.4).
- **Making tests**: majority agreement among 16 sampled solutions labels outputs with 96.8% accuracy against GPT-4o's
  12.7% (arXiv 2505.21297 Table 5); Qwen3-Coder-Next keeps the tests "that achieve the highest consensus under
  majority voting" (§4.2); tests aimed at near-correct faulty programs add 2.4 points at a fixed budget of about 40
  tests (arXiv 2608.24135 Table 2).
- **Learned rewards**: DeepSeek-Coder-V2 (2024) trained a reward model on compiler outcomes, the raw 0–1 signal "noisy
  and sub-optimal" with sparse tests (§3.5); the code recipes of 2025–2026 above execute tests directly. A reasoning
  reward model gated on all tests passing raised P-GRPO's average 54.9 → 57.4; gated on the pass rate, the policy
  "excessively exploits these noisy signals" (arXiv 2508.05170 Table 1, Fig. 4).
- **Tests by the model**: a separate test writer trained adversarially beat human tests at 1.5B, 56.95 against 56.23,
  where one model writing both scored 55.88 (arXiv 2603.15611 §4.2); CURE at 4B–14B (arXiv 2506.03136 §4.2).
- **Multi-turn**: RL over execution feedback matches single-turn GRPO when evaluated in one turn and adds up to 8
  points over three (Qwen3-1.7B and 4B; arXiv 2511.07833 App. D).
- **Here** [inference]: binary, on the last code block after `</think>` and every test of the problem (§11.3); no
  pass-fraction arm, no neural reward.

## 9. GRPO variants

| Variant | Change to GRPO | Problem addressed | Evidence (setting) | Here [inference] |
|---|---|---|---|---|
| Dr. GRPO (arXiv 2503.20783 §3) | drops the 1/\|o_i\| length normalization and the std division (optionally a constant normalizer); the advantage becomes RLOO's up to G/(G−1) (App. A) | length bias (favors short correct and long wrong answers) and a question-difficulty bias | math; its whole 1.5B recipe 33.1 → 42.1 over the base (Oat-Zero-1.5B, Table 4); either removal alone beats GRPO (App. C); OlmoRL and DeepSeek-V3.2 drop the std (§7) | not adopted: Nemotron-Cascade's std stays (§11.3); after the 0/8–8/8 filter, 1/std of a group of 8 binary rewards varies by at most 1.51× [arithmetic] |
| DAPO (arXiv 2503.14476 §3–§4) | clip [1 − 0.2, 1 + 0.28] (Clip-Higher); dynamic sampling keeps groups with 0 < correct < G; token-level loss over the group; overlong filtering and a soft length penalty; no KL | entropy collapse, zero-advantage groups, length effects | math, Qwen2.5-32B: AIME24 30 → 50 over the additions (Table 1); its three core pieces are in OlmoRL, GLM-5 and Seed-Coder (§7); clip-higher "gradually becomes unstable" (arXiv 2505.22617 §4.3); under DAPO Phi-4-mini-reasoning's AIME consensus "consistently degrades" (arXiv 2504.21233 §5.5) | the token-level loss; dynamic sampling as §11.3's static filter; no Clip-Higher (no PPO-form loss); §7's length curriculum in place of length shaping |
| CISPO (arXiv 2506.13585 §3.1) | clips the stop-gradient importance weight instead of the update, so every token keeps its gradient; token-level normalization; no KL | PPO clipping drops rare reflective tokens after the first off-policy update | Qwen2.5-32B zero-RL: DAPO's AIME in half the steps (Fig. 2); ScaleRL, 8B dense and a MoE: "the best balance of stability and robustness to hyperparameters" (arXiv 2510.13786 App. A.17) | not used: one update per batch (§11.3) |
| GSPO (arXiv 2507.18071 §4–§5) | sequence-level ratio (π/π_old)^{1/\|y\|}, clipped per sequence at [1 − 3e-4, 1 + 4e-4] | token-ratio noise and collapse; MoE routing drift (no Routing Replay); tolerates training/inference precision mismatch (§5.4) | Qwen3-30B-A3B MoE; math and code curves (Fig. 1); in ScaleRL its runs "diverged mid-training" (App. A.17) | not first |
| P-GRPO (arXiv 2508.05170 §2.4) | reward R^f + R^o + R^o·R^t: a reward model's reasoning score R^t counts only when the code passes all tests (R^o = 1) | outcome rewards ignore the reasoning; all-correct groups get no advantage | code, Qwen2.5-Coder-7B-Instruct: MBPP+ 76.2 against GRPO's 75.1, HumanEval+ 81.1 both (Table 1); math 7B 51.5 against 48.0 (Table 3) | not first: needs a reward model and a reasoning format |
| DGPO (arXiv 2601.20614 §3–§4) | advantage over the mean absolute deviation, so each question's total update is G; softmax difficulty weights (T = 2); loss over valid questions | GRPO's total update per question, 2G·√(p(1 − p)), peaks at pass rate p = 0.5 and shrinks toward 0 and 1 (Thm. 1) | math only (1.5B–7B): average 39.79 against 37.61 | no code evidence |

## 10. Data

§11's sources: rows from the dataset cards and the Hub's file listings; tokens in this tokenizer from ranged reads of
30–300 rows per file [estimate]:

| Dataset | Stage | Rows | Tokens (ours) | Content |
|---|---|---|---|---|
| `nvidia/Nemotron-Cascade-SFT-Stage-1` `code` | SFT | 1,301,591 of 165,393 prompts | ≈9.9B; per row p50 6,658, max 18,273 | Python; DeepSeek-R1 traces; OpenCodeReasoning 763,495, opc-sft-stage2 323,163, leetcode 126,878, TACO 56,694, Magicoder 27,625, APPS 3,736 |
| `nvidia/Nemotron-Cascade-SFT-Stage-2` `code` | SFT | 1,392,831 of 78,852 prompts | ≈17.6B; per row p50 11,775, 0.2% above 32,768 | Python; DeepSeek-R1-0528 traces; OpenCodeReasoning 1,143,266, opc-sft-stage2 188,063, TACO 58,273, synthetic 1,654, APPS 1,575 |
| `nvidia/Nemotron-SFT-Competitive-Programming-v2` `competitive_programming_{python,cpp}` | SFT, rows ≤ 32,768 tokens | 333,561 and 332,186 (counted from the files) of 34,799 prompts; 157,326 and 156,672 fit (§11.2) | per row p50 56,573 and 59,871 (all rows) | stdin problems, Codeforces ratings up to 3,500; the only C++ traces |
| `nvidia/Nemotron-SFT-Competitive-Programming-v2` `exercism` | SFT | 79,244 (counted from the file), 79,103 with one Python file listing, 78,156 prompts | reasoning (the answer's prose before its file) p50 302, code p50 180 | function stubs from Aider's Exercism protocol, rendered as in §11.1 (§11.2) |
| `allenai/Dolci-Think-RL-7B` `code`, `code_stdio` | RL | 12,121 and 9,264 (counted from the files) | — | assert tests; input/output tests |
| `KodCode/KodCode-Light-RL-10K` | RL | 10,000 (counted from the files) | — | function form; KodCode's tests (below) |
| `nvidia/Nemotron-RL-coding-competitive_coding` | RL | 16,083 | prompt p50 640 | Python stdin problems, tests inline: a median 55.5 per problem in its 25K file |

- **Not used**: `nvidia/Nemotron-Cascade-2-SFT-Data` (24,824,269 rows in 8 configurations, ≈171B tokens here
  [estimate]) has no code-reasoning configuration, only math, science, chat, instruction following, safety, agents
  and software engineering; Nemotron-Cascade 2's code traces (1.9M Python, 1.0M C++, 1.3M with tool calls; arXiv
  2603.19220 §3.2.2) and the code-RL problems of both Cascade models are not released.
- **Fetched, not in §11** (rows counted from the files):

| Dataset | Role | Rows | Tests |
|---|---|---|---|
| `OpenCoder-LLM/opc-sft-stage1` | short-solution SFT, stage 1 | 4,216,321 | — |
| `OpenCoder-LLM/opc-sft-stage2` | short-solution SFT, stage 2 | 436,347: educational 118,278, evol 111,183, McEval 35,943, package 170,943 | educational subset only |
| `nvidia/OpenCodeInstruct` | short-solution SFT | 5,000,000 | 10 per sample with execution status, an LLM judge's score (arXiv 2504.04030 §2.3–§2.5) |
| `KodCode/KodCode-V1` | short-solution SFT, RL problems | 487,432 | GPT-4o solutions and tests kept only when they pass with 100% branch coverage, 7.52 tests per question (arXiv 2503.02951 §2.2, §3.2) |
| `microsoft/rStar-Coder` (`seed_sft`, `synthetic_sft`, `synthetic_rl`) | long-trace SFT; RL problems | seed_sft 591,660 (281,361 with tests, 125,521 of those passing); synthetic_sft 398,183; synthetic_rl 398,107 problems | not present (separate test-case subsets) |
| `nvidia/OpenCodeReasoning-2` (`train/python`) | long-trace SFT | 1,398,166: R1 solutions with QwQ-32B critiques, judged right 1,027,261 and wrong 370,905 | a pass rate per row |
| `open-thoughts/OpenThoughts3-1.2M` | long-trace SFT | 1,200,000: math 850,000, code 250,000 (OpenCodeReasoning questions 125,840, StackExchange code golf 124,160), science 100,000 | none (QwQ-32B answers, unfiltered, §4) |
| `nvidia/OpenCodeReasoning` | long-trace SFT | 735,255 (two splits, 567,850 and 167,405) | none |
| `KodCode/KodCode-V1-SFT-R1` | long-trace SFT | 483,437 | R1 solutions with their test results |
| `agentica-org/DeepCoder-Preview-Dataset` | RL | 24,287 train (PrimeIntellect 16,252, TACO 7,436, LiveCodeBench v5 599); 687 test (LiveCodeBench v5 279, Codeforces 408) | yes; weak enough for false-positive rewards (§8) |

- **Decontamination as published**: Nemotron-Cascade drops SFT samples sharing a 9-gram with coding benchmarks
  (arXiv 2512.13607 §3.2.3); KodCode flags 94 of 447K questions above cosine 0.95 to HumanEval, MBPP, BigCodeBench or
  LiveCodeBench v5 (§3.2), and its files carry `benchmark_similarity` and `benchmark_task_id` per row; rStar-Coder
  removes 16-gram overlaps with HumanEval(+), MBPP(+), LiveCodeBench and USACO 2025 (§3.3); OpenCodeReasoning's 90
  flagged pairs were judged not duplicates, none removed (§2.1); OpenCodeInstruct's procedure is [unverified]
  (§2.2.1 empty in the extracted text). None of these is the pretraining rule (§1), which every source still needs.

## 11. Pipeline [inference]

SFT only: Nemotron-Cascade's SFT (arXiv 2512.13607 §3) cut to the one domain the model is for (code) and one format
(reasoning, then the answer), with full parameters (§3). Below 1B the published pipelines end at SFT or
distillation: MobileLLM-R1 (140M–950M) has no RL, "additional RL does not observe a significant performance
improvement" (arXiv 2509.24945 §7.4), and RL "yields suboptimal results for smaller models compared to distillation"
(arXiv 2505.00949 §5). Nemotron-Cascade's code RL added 2.4–4.1 LiveCodeBench v6 points at 8B–14B (Table 7); here
it is deferred (§11.3): it needs an image, an export and a reward the repository lacks, and its step time on this
model is unmeasured. Each choice is [inference] from the cited evidence; none is measured on this model.

| Stage | From | Data | Objective | Lengths | Gate |
|---|---|---|---|---|---|
| 0. Base | — | — | — | — | the evaluations of steps 66,063 and 69,874 (`docs/evaluation.md` §1) pick the start |
| 1. SFT | the base | reasoning traces, at most 4 per prompt in each group: Cascade-1's Python, CP-v2's Python, C++ and Exercism; 920,350 traces of 279,152 prompts (220,456 problems), 8.03B tokens, 5,453 steps, ≈7.8 h (§11.2) | next-token loss on the response | rows ≤ 32,768 tokens, packed | reasoning pass@1 on HumanEval+ and MBPP+ above the base's greedy pass@1 (§11.4) |
| Code RL, deferred | the SFT model | problems with tests, 47,468 before filtering (§11.3) | Nemotron-Cascade's GRPO, one update per batch, sequence-level truncated importance weights; FP16 | responses ≤ 16,384, then ≤ 28,672 | pass@1 up, pass@20 not down beyond its error bar (§11.4) |

### 11.1 Format

- **Reasoning is the only format**: no non-thinking mode. An example renders as BOS, the prompt, `\n\n<think>\n`, the
  reasoning, `\n</think>\n\n`, the answer and BOS. The markers are ordinary text (6 and 6 tokens here), so the
  vocabulary stays (§1). Nemotron-Cascade 2 opens a thinking turn with `<think>` and a newline (arXiv 2603.19220
  §3.1.2); §10's code traces are "formatted in the thinking mode" (arXiv 2512.13607 §3.2.3), the reasoning inline
  between the markers (Cascade-1) or in a separate field placed between them (CP-v2).
- **Loss**: on the response, from the reasoning through the closing BOS; the prompt and its `\n\n<think>\n` are
  context, their labels ignored by the loader (§11.2, Packing). Every prompt ends with `\n\n<think>\n` in SFT, RL and
  evaluation alike; the context is encoded alone, as a prompt is at inference.

### 11.2 SFT

- **Data** (§10; `posttrain/sft_data.py` `GROUPS`): five groups, Cascade-1's Stage-1 and Stage-2 code and CP-v2's
  Python, C++ and Exercism traces. A Cascade-1 row's reasoning is its reply between the leading `<think>` and
  `</think>`, its answer the rest; a CP-v2 row's are its `reasoning_content` and `content` (no prompt is a `-`
  placeholder). An Exercism row is Aider's 9-message protocol: the prompt is the stub's file listing and the problem
  statement without Aider's editing instructions, the reasoning the reply's prose before its file, the answer that
  file's code in a Python block; these are the only function-stub prompts (HumanEval's shape) with short reasoning in
  these releases. CP-v2's Python and C++ are Codeforces-level problems harder than Cascade-1's, and the only C++.
- **Filters**, in order, each row counted at the first it fails: no reasoning or no answer; reasoning under 20
  tokens; above 32,768 tokens rendered, dropped, not cut (a cut trace loses its answer; Nemotron-Cascade drops
  over-length samples from its math data, arXiv 2512.13607 §3.2.2, and states nothing for code); §1's 10-gram rule on
  prompt, reasoning and answer (`corpus/decontaminate.py`); exact duplicates of prompt, reasoning and answer (the
  first kept, groups in the table's order); at most 4 traces per exact prompt in each group, drawn at random (seed
  0), so a prompt in several groups keeps up to 4 in each (282,138 distinct prompts, 341,740 prompt-group pairs in
  the corpus); last, train problems near a val problem (below). No correctness filter: Nemotron-Cascade (arXiv
  2512.13607) states none; Nemotron-Cascade 2 keeps only correct traces where tests exist (arXiv 2603.19220 §3.2.2);
  at 7–8B unfiltered traces matched or beat filtered ones (arXiv 2506.04178 Tables 15, 18; arXiv 2510.06101 Table 2).
- **Split by problem** (`posttrain/sft_data.py` `problem`, `OVERLAP_N`): one problem recurs across groups and
  sources under different wrappers (CP-v2's language header, Nemotron-Cascade's format request or function header)
  and whitespace, so the unit is the problem: the prompt without those wrappers (`WRAPPERS`), lowercased, its words
  (runs of letters and digits) joined by single spaces. 1% of the distinct problems (seed 0) is the val split; train
  then loses every problem sharing a run of 46 words with a val problem (`corpus_crate().Grams`, the matcher of §1's
  rule). The 46 is measured on this corpus: near-duplicate problems (an exercise's variants, a problem restated, its
  easy and hard versions) share runs from 40 words up, and the sources' stock phrases (Codeforces' and CodeChef's
  input conventions, a standard definition) reach 45, one 53-word Codeforces input block excepted. Split by exact
  prompt, 928 of 2,921 val problems had their statement in train (52.4% of val labels) and 1,199 their first 300
  normalized characters (65.9%); split by problem without the filter, 0 and 453 (43.8%); with it, 0 and 1 (below
  0.01%), the first 300 characters after whitespace-only normalization 2 (below 0.01%). Measured (`stats.json`;
  prompts distinct within a group, "All" across groups; tokens include both BOS, the loss tokens are each response
  and its closing BOS):

| Group | Rows | No reasoning or answer | Over 32,768 | 10-gram | Duplicate | Over 4 per prompt | Near a val problem | Train rows (prompts) | Val rows (prompts) | Train tokens (loss) |
|---|---|---|---|---|---|---|---|---|---|---|
| Cascade-1 Stage-1 | 1,301,591 | 0 | 0 | 60,732 | 22,785 | 829,881 | 3,776 | 380,561 (149,153) | 3,856 (1,517) | 2.59B (2.48B) |
| Cascade-1 Stage-2 | 1,392,831 | 0 | 7,251 | 51,943 | 0 | 1,036,331 | 3,852 | 290,356 (73,536) | 3,098 (783) | 3.37B (3.27B) |
| CP-v2 Python | 333,561 | 0 | 176,235 | 3,142 | 0 | 60,727 | 1,707 | 90,725 (24,099) | 1,025 (270) | 0.99B (0.94B) |
| CP-v2 C++ | 332,186 | 0 | 175,514 | 287 | 0 | 62,670 | 1,697 | 90,983 (24,019) | 1,035 (275) | 1.01B (0.96B) |
| CP-v2 Exercism | 79,244 | 336: 170 without prose, 141 without a file, 25 under 20 tokens | 0 | 1,395 | 0 | 339 | 8,674 | 67,725 (67,317) | 775 (771) | 0.073B (0.039B) |
| All | 3,439,413 | 336 | 359,000 | 117,499 | 22,785 | 1,989,948 | 19,706 | 920,350 (279,152; 220,456 problems) | 9,789 (2,986; 2,324 problems) | 8.03B (7.70B) |

- **Tokens against time**: the cap trades answers per prompt, worth less than prompts (coefficients 2.635 against
  4.831; at a fixed budget 64K prompts × 1 beat 16K × 4, §4), for time. Before the filters, Cascade-1 alone, one
  epoch at the 32K stage's 287K tok/s (`docs/training.md` §7) [estimate]:

| Traces per prompt | Cascade-1 rows | Tokens | Steps | Hours |
|---|---|---|---|---|
| 1 | 244,245 | 2.3B | 1,553 | 2.2 |
| 2 | 460,865 | 4.4B | 2,957 | 4.2 |
| 4 (chosen) | 883,895 | 8.4B | 5,717 | 8.1 |
| all | 2,694,422 | 27.5B | 18,650 | 26.6 |

- **Packing**: the documents, in a seeded random order, are packed by the base's packer (`corpus/pack.py`,
  Best-Fit-Decreasing over shards of 4,096 documents, `docs/corpus.md` §8), none cut, one attention segment each:
  245,413 train and 2,621 val rows of 32,768 tokens, 99.90% and 99.88% of input positions in use. Each document's
  context length (the prompt and `\n\n<think>\n`) is in `<split>.context`; the loader sets those labels to
  `IGNORE_INDEX` (`corpus/loader.py` `PretrainingBatches.assemble`), so 7.70B of the corpus's 8.03B train tokens carry
  loss [arithmetic: 95.8%].
- **One stage, one epoch** (`run.py` `RunConfig.base`, `conf/experiment/sft.yaml`, `train/schedule.py` `sft_plan`):
  the base run's weights after `base_step` steps (default 69,874, its last) with fresh optimizer state; 3 rows per
  micro-batch and 15 micro-batches, 1,474,560 tokens, per step as in the base's 32K stage; 5,453 steps over a seeded
  permutation of the train rows (28 rows unread), ≈7.8 h [estimate], without the 7 validation passes (≈0.2 h
  [estimate]). Nemotron-Cascade trains one epoch per stage over 8.3 and 17.6 traces per prompt (arXiv 2512.13607
  §3.1.1, §3.2.3); Nemotron-Cascade 2's single packed stage peaks after about 1.5 epochs (arXiv 2603.19220 §3.1.1).
  The other recipes run 2–8 epochs (MobileLLM-R1 4 at 950M, rStar-Coder 6 at 1.5B, AceReason's plateau at 5–6,
  X-Coder 8–10 for long traces; §4): one epoch over ≈3.3 traces per prompt here is a time choice.
- **Optimizer**: the base's NorMuon and AdamW with one learning rate, warmup over 40 steps and linear decay to zero
  (`docs/training.md` §3–§4); the peak is `learning_rate_scale` times the base run's recorded peak, 2.3454e-3. No
  weight decay (`sft_plan`): the closest-size recipe, MobileLLM-R1-950M, fine-tunes with none (arXiv 2509.24945 §3.2),
  as does OpenThoughts (arXiv 2506.04178 App. D.2); Nemotron-Cascade uses 1e-4 (arXiv 2512.13607 App. D Table 14),
  and Nemotron-Cascade 2's 0.1 (Σ_t η_t λ ≈ 0.106 over its run, arXiv 2603.19220 App. B Table 7 [arithmetic]) is the
  exception; with λ = 0 the total decay does not depend on which learning rate wins. The τ law (`docs/training.md`
  §3) sets a from-scratch timescale: at the SFT's 41.21 tokens per parameter it gives λ = 5.1 at 1/10 of the base
  peak, which would leave about 4% of the base weights' scale [arithmetic].
- **Peak calibration** (`conf/experiment/sft_lr.yaml`, `ablations/sft_lr.py` `sft_lr_fit`): calibrated, not
  borrowed, since Nemotron-Cascade's AdamW 5e-5 has no NorMuon counterpart. Three complete schedules at 1/10, 1/3
  and 1 times the base peak over the same first 2% of the train permutation (109 steps each), trained only
  (`generate` "none": no evaluation, no step-0 validation), each scored by the response loss on the whole val split
  after its last step. The rule: the vertex of the least-squares parabola of those losses against u = ln(scale)
  (`metrics.parabola_vertex`, as `docs/training.md` §5 calibrates the base's), which must have positive curvature and
  lie inside the grid (else the fit raises, and an arm 3× beyond the edge with the lowest loss is added); the full
  run's scale is the vertex times (5,453 / 109)^−0.30 = 0.309, the step counts from the plans [arithmetic]. Basis
  [inference]: at a fixed batch the optimum falls with the run's length with exponent 0.30–0.32 (arXiv 2409.19913 §6
  Eq. 5); the repository's d8 pair, net of its doubled batch by a square-root rule, gives 0.24–0.30 (`docs/training.md`
  §5) and DeepSeek LLM's law 0.29 at a fixed batch [arithmetic]; the one counter-claim, that the optimum hardly moves
  under linear decay to zero (arXiv 2502.15938 §4.2), is contradicted by that d8 pair; erring low is the cheaper error,
  forgetting growing with the learning rate (§4). At 1B, in KL distillation of 150 steps, AdamW at 5e-5 against 1e-5
  cost 11–14 points out of domain (arXiv 2609.35259 §4.2). Cost: ≈36 min of GPU [estimate].
- **Validation and checkpoints**: the response loss, mean nats per label (each response and its closing BOS), over
  the whole val split (2,621 rows, 82.0M labels) before the full run's first step, every 1,000 steps and after the
  last (`run.py` `validation_steps`, `train/loop.py` `validate`; `val` records); full states every 1,000 steps (resume
  points) and at the end, the full run's last evaluated by the reasoning protocol (`generate: reasoning`, §11.4).

### 11.3 Code RL, deferred

- **When**: after §11.2's model passes its gate, if time allows; the recipe below is fixed so that it starts from
  settled choices. LoRA fits this stage: RL matched full fine-tuning at rank 1 ("LoRA Without Regret", §3), and FP16
  kept LoRA RL stable where "BF16-based LoRA training collapses after roughly 600 steps" (arXiv 2510.26788 §5.2).
- **Framework**: upstream verl (main of 2026-10-30) in its own image, built from its `docker/Dockerfile.uv.cu130` for
  arm64 (torch 2.13, vLLM 0.32.0, transformers 5.16.1); FP16 training and sampling are part of it (verl #4036). The
  Precision-RL fork (verl 0.4.1.dev of 2025-07-16) does not run here: it pins vLLM ≤ 0.8.5, whose
  `vllm.worker.worker_base` its sampler imports and vLLM 0.12.0 removed, and transformers 4, against the model
  file's 5.
- **Model**: the SFT checkpoint through `inference/export_hf.py` with fp32 weights (bf16 today; fp16 from the fp32
  master rounds 8× less, relative error 2.1e-4 against 1.7e-3 [measured]) and a chat template that renders BOS, the
  prompt and `\n\n<think>\n` (verl applies a template to every prompt). The LM head with its cap and the gate of the
  value embeddings run with autocast off: autocast casts their fp32 `F.linear` inputs to fp16
  (`inference/modeling_nanochat.py`).
- **Precision** (arXiv 2510.26788): the trainer computes in fp16 on fp32 master weights and fp32 AdamW state,
  reducing gradients in fp32, under a dynamic loss scale (a step with a non-finite gradient is skipped and the scale
  halved; it doubles after 400 clean steps); the sampler, vLLM, runs in fp16. The model's ranges in fp16: §7.
- **Objective**: Nemotron-Cascade's GRPO (arXiv 2512.13607 §4.1.2, Eq. 1): 8 rollouts per prompt, the advantage
  (r − mean)/std over the group, one gradient update per batch, the loss averaged over the batch's tokens, no KL and
  no entropy term. One update per batch leaves only the sampler–trainer ratio, which Nemotron-Cascade takes as 1;
  here each sequence's gradient is weighted by min(ρ, 3), ρ the product over its tokens of π_train/π_sampler
  (PG-Seq-TIS, arXiv 2510.26788 Eq. 6, C = 3 as in §4.1).
- **verl settings**: `algorithm.adv_estimator` grpo, `norm_adv_by_std_in_grpo` true, `algorithm.rollout_correction`
  {`rollout_is` sequence, `rollout_is_threshold` 3.0, `bypass_mode` true, `loss_type` reinforce} (bypass mode skips
  the old-policy forward), `ppo_epochs` 1 with the mini-batch the whole batch, `loss_agg_mode` token-mean,
  `use_kl_loss` false, `entropy_coeff` 0; the actor engine's `dtype` float16 with `model_dtype` fp32, the rollout's
  `dtype` float16 and `calculate_log_probs` true; SDPA attention, no padding removal, no gradient checkpointing (the
  model file has none), FSDP wrapping by `NanochatDecoderLayer`.
- **Reward**: 1 if the last fenced code block after `</think>` passes every test of the problem, else 0 (no
  `</think>`, no block, a cut response, an error or a time-out included), as Nemotron-Cascade (§4.6.1); run by the
  repository's checks (`eval/verify.py`) as verl's custom reward function.
- **Problems** (§10): Dolci-Think-RL's code (12,121) and code_stdio (9,264), KodCode-Light-RL-10K (10,000) and
  Nemotron-RL-coding-competitive_coding (16,083): 47,468 [arithmetic]. Dropped: a problem whose reference solution
  fails its own tests; §1's 10-gram rule; a problem the SFT model solves 8 or 0 times of 8 at the rollout settings
  (Nemotron-Cascade drops those a 7B solves 8/8 and those R1-0528 solves 0/8, §4.6.1; here the policy decides both).
  DeepCoder-Preview is left out for its weak tests (§8). All Python: C++ comes from the SFT stage only.
- **Sampling**: temperature 1.0 and top-p 1.0, so that the sampler's distribution is the policy's, as the FP16
  analysis samples to compare them (arXiv 2510.26788 §3.5); Nemotron-Cascade 2 samples so in its other RL stages and
  at top-p 0.95 in code RL (arXiv 2603.19220 App. B).
- **Batch and lengths**: 128 prompts × 8 rollouts per update (arXiv 2512.13607 App. D Table 19); prompts ≤ 4,096
  tokens; responses ≤ 16,384, raised to 28,672 (32,768 − 4,096) once more than 10% are cut (§7, Length); a cut
  response scores 0 and stays in the batch (Nemotron-Cascade's code RL filters no overlong response, §4.6.2).
- **Optimizer**: AdamW at a constant 2e-6 [estimate: between Precision-RL's 1e-6 at 1.5B–30B (arXiv 2510.26788
  App. A) and Nemotron-Cascade's 3e-6–4e-6 at 8B–30B], β (0.9, 0.95), no weight decay, gradient norm clipped at 1.0.
- **Stopping and watches**: every 10 steps, pass@1 on 5% of the filtered problems held out; stop when it stops
  rising. Nemotron-Cascade's code RL ran 22 to 90 steps (arXiv 2603.19220 App. B; arXiv 2512.13607 App. D Table 19).
  Watched: the entropy (temperature 1.0 → 0.8 if it climbs, as Nemotron-Cascade's unified 8B run, Table 19), steps
  skipped by the loss scale, the sampler–trainer log-probability gap, the share of cut responses.

### 11.4 Evaluation of each stage

- **Reasoning protocol** (`eval/code_eval.py` `reasoning_generation`, the evaluation's `generate="reasoning"`;
  `docs/evaluation.md` §4): the prompt is EvalPlus's instruction for chat models, "Please provide a self-contained
  Python script that solves the following problem in a markdown code block:" (`evalplus/codegen.py`), then the
  benchmark's prompt in a fenced block, then §11.1's `\n\n<think>\n`; MultiPL-E's C++ prompts the same with "C++" for
  "Python". 20 samples at temperature 0.6 and top-p 0.95, Nemotron-Cascade's evaluation setting (arXiv 2512.13607
  §2), each until BOS or the context (32,768 tokens with the prompt) in place of its 64K. The answer is the last
  fenced block after `</think>`, scored by EvalPlus as a chat model's solution and by MultiPL-E's judge with the
  prompt's `#include` lines and the tests. Reported: pass@1 (the mean over the 20) and pass@20, the share of responses
  cut at the budget, with no `</think>` or with no block after it, the response lengths.
- **The base's records** (likelihood, greedy completion, infilling; `docs/evaluation.md` §1) are taken at every stage
  to measure what it loses, not as a gate.
- **No selection on the benchmarks**: RL, when run, stops on its held-out problems (§11.3); HumanEval+, MBPP+ and
  MultiPL-E are read once per stage, at its end.

### 11.5 Not in the pipeline

| Stage | Why not |
|---|---|
| A non-thinking mode | reasoning is the default and only format (§11.1) |
| LoRA, QLoRA | at hidden size 768 LoRA saves compute only below rank 128–205, where it lags full fine-tuning on code; QLoRA saves memory the run does not need (§3) |
| DPO (§5) | Nemotron-Cascade goes from SFT to RL without it |
| Math RL | the SFT data holds no math; Nemotron-Cascade's math RL moved LiveCodeBench v6 by −0.3 to +2.2 (arXiv 2512.13607 §4.5) |
| RLHF, instruction-following, multi-domain and long-context RL | chat, judges or tools the model is not trained for; reward models of 72B–235B (arXiv 2512.13607 §4.2–§4.4; arXiv 2603.19220 §4.2–§4.6) |
| Software-engineering RL | a 72B or 120B judge, or agentic execution over 256K tokens (arXiv 2512.13607 §4.7; arXiv 2603.19220 §4.8) |
| On-policy distillation (§6) | one RL domain leaves no specialists to merge; Nemotron-Cascade 2's teachers are its own per-domain checkpoints (arXiv 2603.19220 §4.4) |

### 11.6 Code

1. `posttrain/sft_data.py` reads §10's SFT files from `artifacts/posttrain/`, renders §11.1's format, applies §11.2's
   filters, cap and split, tokenizes and packs, the prompts' labels ignored by the loader.
2. SFT as a `run.py` run from a base checkpoint (`conf/experiment/sft.yaml`) and its peak calibration
   (`conf/experiment/sft_lr.yaml`).
3. `eval/code_eval.py` `reasoning_generation`: the reasoning protocol (§11.4).
4. Not written, for §11.3 only: the verl image; `inference/export_hf.py` and `inference/modeling_nanochat.py` (fp32
   weights, the chat template, the head and gate outside autocast); `posttrain/rl_data.py` (problems and tests in
   verl's format, the 8-sample filter sampled by the decoder); `posttrain/reward.py` (§11.3's reward); the run's
   settings.

## References

| arXiv | Title |
|---|---|
| 2305.14314 | QLoRA: Efficient Finetuning of Quantized LLMs |
| 2401.00788 | Astraios: Parameter-Efficient Instruction Tuning Code Large Language Models |
| 2401.14196 | DeepSeek-Coder: When the Large Language Model Meets Programming - The Rise of Code Intelligence |
| 2402.03300 | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models |
| 2405.09673 | LoRA Learns Less and Forgets Less |
| 2406.11931 | DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence |
| 2409.12186 | Qwen2.5-Coder Technical Report |
| 2409.19913 | Scaling Optimal LR Across Token Horizons |
| 2411.04905 | OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models |
| 2501.12948 | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning |
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
| 2507.18071 | Group Sequence Policy Optimization |
| 2508.05170 | Posterior-GRPO: Rewarding Reasoning Processes in Code Generation |
| 2508.08221 | Tricks or Traps? A Deep Dive into RL for LLM Reasoning |
| 2509.24945 | MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes |
| 2510.06101 | The Valley of Code Reasoning: Scaling Knowledge Distillation of Large Language Models |
| 2510.13786 | The Art of Scaling Reinforcement Learning Compute for LLMs |
| 2510.26788 | Defeating the Training-Inference Mismatch via FP16 |
| 2511.07833 | Murphy: Multi-Turn GRPO for Self-Correcting Code Generation |
| 2512.02556 | DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models |
| 2512.13607 | Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for General-Purpose Reasoning Models |
| 2512.13961 | Olmo 3 |
| 2512.23165 | Evaluating Parameter Efficient Methods for RLVR |
| 2601.02780 | MiMo-V2-Flash Technical Report |
| 2601.06953 | X-Coder: Advancing Competitive Programming with Fully Synthetic Tasks, Solutions, and Tests |
| 2601.20614 | Harder Is Better: Boosting Mathematical Reasoning via Difficulty-Aware GRPO and Multi-Aspect Question Reformulation |
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
