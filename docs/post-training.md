# Post-training

The method that turns the base model into a code model that reasons before it answers: its starting point (§1), the
published evidence each choice rests on (§2), the format (§3), the supervised fine-tuning recipe the repository
implements (§4), its evaluation (§5), the code RL stage it defers (§6) and what the pipeline leaves out (§7). The SFT
corpus is `docs/corpus.md` §10; the shared training loop `docs/training.md`; measured results `docs/results.md`.

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
| `scripts/official_reasoning.sh`, `eval/official_reasoning_parser.py`, `eval/official_lcb.py`, `eval/official_tokenize.py`, `eval/official_evalplus.py`, `eval/official_reasoning.py`, `inference/export_hf.py` (`--chat`, `CHAT_TEMPLATE`) | §5 |
| `eval/official_termination.py`, `scripts/official_termination.sh`, `scripts/official_termination_watch.sh` | §5.1 |
| `eval/serving_check.py`, `scripts/serving_check.sh` | §5.1 serving faithfulness |

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
  above (arXiv 2502.12143 §3.1). Distillation plus SFT with LoRA on Qwen2.5-0.5B-Instruct scored 0.7% on
  OlympiadBench and 1.2% on MATH500, and adding RL collapsed (arXiv 2506.13404 §4.2–§4.3; its tables are empty in the
  extracted text [unverified]).
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
- **Here** [inference]: not used; Nemotron-Cascade goes from SFT to RL (§7).

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
  same vocabulary" (arXiv 2604.00626 §8). A different SFT teacher costs 2–6.8 AIME24 points (arXiv 2604.13010
  Tables 3–4). Teachers may be the model's own earlier stages (GLM-5 §3.5) or "even the student model itself" (arXiv
  2601.02780 §4.1).
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
| Length | short first, then longer | 8K → 16K → 24K → 32K beat starting at 16K or 24K (arXiv 2505.16400 §4.3); an 8K stage first, AIME25 56.7 against 51.8 (arXiv 2506.13284 §4.5.4); masking truncated outputs helps at 8K–16K and hurts at 32K (Table 2); 10–15% truncated outputs destabilize (arXiv 2510.13786 App. A.15) |
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
- **Here**: sampler and trainer both in FP16 (§6), the remaining mismatch under sequence-level truncated weights
  [inference]. On the base model's final checkpoint (CPU, three code files of 1,024 tokens) the largest activations
  are 145 (residual stream), 241 (ReLU²), 46.8 (attention score) and 29.5 (logit before the cap), and the weights
  bound the last three by 11,195, 68.2 and 75, all below fp16's 65,504; with activations rounded to fp16 the mean KL to
  fp32 is 1.5e-5, to bf16 1.0e-3; 77–99.9% of the residual stream's gradient values lie below fp16's smallest normal
  (6.1e-5), so the loss is scaled [measured].

### 2.7 Code rewards

- **Binary on verified tests**: reward 1 "if and only if the generated code successfully passes all test cases"
  (arXiv 2505.16400 §3.3); Nemotron-Cascade (§4.6), KodCode (arXiv 2503.02951 §4.1) and Seed-Coder alike. The tests
  decide the result: false negatives leave RL at "a sub-optimal point"; false positives (weak tests, DeepCoder's set
  among them) bring "a sudden drop on testing set accuracy midway" (arXiv 2505.16400 §4.3); AceReason drops
  special-judge, interactive, untested and weak-test problems (App. A.4).
- **Pass fraction**: at 32B, rewarding the fraction of tests passed beat the binary reward by 0.55–1.19 LiveCodeBench
  points, most with unvalidated tests (arXiv 2608.24135 Table 3); X-Coder's fraction reward drew late reward hacking
  that exploits "edge cases for partial rewards" (arXiv 2601.06953 §3.3); Nanbeige4.1-3B rewards the pass rate and adds
  an efficiency term only once every test passes (arXiv 2602.13367 §2.3).
- **Making tests**: majority agreement among 16 sampled solutions labels outputs with 96.8% accuracy against GPT-4o's
  12.7% (arXiv 2505.21297 Table 5); Qwen3-Coder-Next keeps the tests "that achieve the highest consensus under
  majority voting" (§4.2); tests aimed at near-correct faulty programs add 2.4 points at a fixed budget of about 40
  tests (arXiv 2608.24135 Table 2).
- **Learned rewards**: DeepSeek-Coder-V2 trained a reward model on compiler outcomes, the raw 0–1 signal "noisy and
  sub-optimal" with sparse tests (arXiv 2406.11931 §3.5); later code recipes execute tests directly. A reasoning reward
  model gated on all tests passing raised P-GRPO's average 54.9 → 57.4; gated on the pass rate, the policy
  "excessively exploits these noisy signals" (arXiv 2508.05170 Table 1, Fig. 4).
- **Tests by the model**: a separate test writer trained adversarially beat human tests at 1.5B, 56.95 against 56.23,
  where one model writing both scored 55.88 (arXiv 2603.15611 §4.2); CURE at 4B–14B (arXiv 2506.03136 §4.2).
- **Multi-turn**: RL over execution feedback matches single-turn GRPO when evaluated in one turn and adds up to 8
  points over three (Qwen3-1.7B and 4B; arXiv 2511.07833 App. D).
- **Here** [inference]: binary, on the last code block after `</think>` and every test of the problem (§6); no
  pass-fraction arm, no neural reward.

### 2.8 GRPO variants

| Variant | Change to GRPO | Problem addressed | Evidence (setting) | Here [inference] |
|---|---|---|---|---|
| Dr. GRPO (arXiv 2503.20783 §3) | drops the 1/\|o_i\| length normalization and the std division; the advantage becomes RLOO's up to G/(G−1) (App. A) | length bias and a question-difficulty bias | math; its whole 1.5B recipe 33.1 → 42.1 over the base (Table 4); either removal alone beats GRPO (App. C) | not adopted: Nemotron-Cascade's std stays (§6); after the 0/8–8/8 filter, 1/std of a group of 8 binary rewards varies by at most 1.51× [arithmetic] |
| DAPO (arXiv 2503.14476 §3–§4) | clip [1 − 0.2, 1 + 0.28] (Clip-Higher); dynamic sampling keeps groups with 0 < correct < G; token-level loss; overlong filtering and a soft length penalty; no KL | entropy collapse, zero-advantage groups, length effects | math, Qwen2.5-32B: AIME24 30 → 50 (Table 1); clip-higher "gradually becomes unstable" (arXiv 2505.22617 §4.3); under DAPO Phi-4-mini-reasoning's AIME consensus "consistently degrades" (arXiv 2504.21233 §5.5) | the token-level loss; dynamic sampling as §6's static filter; no Clip-Higher (no PPO-form loss); §2.6's length curriculum in place of length shaping |
| CISPO (arXiv 2506.13585 §3.1) | clips the stop-gradient importance weight instead of the update; token-level normalization; no KL | PPO clipping drops rare reflective tokens after the first off-policy update | Qwen2.5-32B zero-RL: DAPO's AIME in half the steps (Fig. 2); ScaleRL: "the best balance of stability and robustness to hyperparameters" (arXiv 2510.13786 App. A.17) | not used: one update per batch (§6) |
| GSPO (arXiv 2507.18071 §4–§5) | sequence-level ratio (π/π_old)^{1/\|y\|}, clipped per sequence at [1 − 3e-4, 1 + 4e-4] | token-ratio noise and collapse; MoE routing drift; tolerates precision mismatch (§5.4) | Qwen3-30B-A3B MoE (Fig. 1); in ScaleRL its runs "diverged mid-training" (App. A.17) | not first |
| P-GRPO (arXiv 2508.05170 §2.4) | reward R^f + R^o + R^o·R^t: a reward model's reasoning score counts only when the code passes all tests | outcome rewards ignore the reasoning; all-correct groups get no advantage | Qwen2.5-Coder-7B-Instruct: MBPP+ 76.2 against GRPO's 75.1, HumanEval+ 81.1 both (Table 1) | not first: needs a reward model |
| DGPO (arXiv 2601.20614 §3–§4) | advantage over the mean absolute deviation; softmax difficulty weights (T = 2) | GRPO's total update per question, 2G·√(p(1 − p)), peaks at pass rate 0.5 (Thm. 1) | math only (1.5B–7B): average 39.79 against 37.61 | no code evidence |

### 2.9 The pipeline

SFT only, on a reasoning-code mixture: the code traces of Nemotron-Cascade's SFT (arXiv 2512.13607 §3) and of
Nemotron's competitive-programming set, with increased verified function-level coverage (KodCode's R1 traces, each
solution passing its tests, beside Exercism's function stubs; the composition: `docs/corpus.md` §10), in one format
(reasoning, then the answer), with full parameters (§2.2). Below 1B the published pipelines end at SFT or
distillation: MobileLLM-R1 (140M–950M) has no RL, "additional RL does not observe a significant performance
improvement" (arXiv 2509.24945 §7.4), and RL "yields suboptimal results for smaller models compared to distillation"
(arXiv 2505.00949 §5). Nemotron-Cascade's code RL added 2.4–4.1 LiveCodeBench v6 points at 8B–14B (Table 7); here it
is deferred (§6): it needs an image, an export and a reward the repository lacks, and its step time on this model is
unmeasured. Each choice is [inference] from the cited evidence.

| Stage | From | Data | Objective | Lengths | Gate |
|---|---|---|---|---|---|
| 0. Base | — | — | — | — | the base run's last checkpoint (`base_step` 69,874; `docs/results.md` §1, §4) |
| 1. SFT (§4) | the base | the SFT corpus's reasoning traces (`docs/corpus.md` §10) | next-token loss on the response | rows ≤ 32,768 tokens, packed | reasoning pass@1 on HumanEval+ and MBPP+ above the base's greedy pass@1 (§5) |
| Code RL, deferred (§6) | the SFT model | problems with tests, 47,468 before filtering | Nemotron-Cascade's GRPO, one update per batch, sequence-level truncated importance weights; FP16 | responses ≤ 16,384, then ≤ 28,672 | pass@1 up, pass@20 not down beyond its error bar (§5) |

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
  samples (pooled), on the base and the plus tests.
- **Contamination**: the SFT prompts' measured overlap with LiveCodeBench v6 is `docs/corpus.md` §10's.
- **The base's records** (likelihood, the official benchmarks; `docs/evaluation.md` §1, §4) are taken at every stage
  to measure what it loses, not as a gate.
- **No selection on the benchmarks**: RL, when run, stops on its held-out problems (§6); HumanEval+, MBPP+ and
  LiveCodeBench are read once per stage, at its end.

### 5.1 Natural-termination evaluation on held-out prompts

- **Purpose**: the teacher-forced response loss on the val split (§4) does not measure whether generation ends. The
  model must itself emit `</think>`, a fenced answer and the closing BOS; a response that never closes its reasoning
  has no answer and scores 0 in every benchmark (§5). Natural closure is the primary measure: a forced thinking budget
  (s1's budget forcing, which appends the end-of-thinking delimiter at a maximum token count, arXiv 2501.19393 §3.1)
  would mix learned and imposed termination. In Olmo3-7B's SFT, "Single-epoch models terminate only 24% of
  generations, while 32-epoch models approach the rate of 89%", termination tracking accuracy (arXiv 2602.11149
  §4.2); whether termination moves likewise over the epochs of this run is what the evaluation measures [inference].
- **Prompt sets** (`eval/official_termination.py`): the first 64 HumanEval+ tasks as EvalPlus's chat path renders
  them (§5; each checked to render as BOS + `encode_ordinary(prompt.strip() + "\n\n<think>\n")`) and 64 SFT val
  documents drawn by `default_rng(0)`, their contexts as the SFT trained on them; the val problems are disjoint from
  the train split's by the problem-level split (`docs/corpus.md` §10). Each val prompt carries its source group and
  style (`val-labels.jsonl`) and its teacher's reasoning length.
- **Protocol**: the `--chat` export (`inference/export_hf.py`), vLLM offline in bf16 with the head in fp32, eos = BOS,
  T 0.6, top-p 0.95, one sample per prompt seeded by 0 + its index, max new tokens the context less the prompt; every
  full-state checkpoint of the run, with the same prompts and seeds at every step (`scripts/official_termination.sh`
  per checkpoint, `scripts/official_termination_watch.sh` over a run as its checkpoints appear;
  `scripts/official_reasoning.sh` runs the probe on its export). Beside the training job the probe takes 0.12 of the
  GPU's memory and is skipped if less than 15 GB is free at its start. Generation runs in rounds of 2,048 new tokens
  with prefix caching, and a response with no `</think>` whose last 4,096 tokens have repetition ≥ 0.99 after a round
  is stopped there (outcome "unclosed, loop-stopped"; with "unclosed at the limit" it compares with a one-shot probe's
  "unclosed at the limit"): loops take most of a one-shot probe's tokens, and none has been seen to close. Each round
  restarts a request's seeded generator (seed + round × 1,000,003), so only the first 2,048 tokens equal a one-shot
  generation's.
- **Metrics** (per set and pooled; `RUNS/<run>/termination-<step>.json`): five exclusive outcomes, closed and ended
  (BOS), closed with the answer at the limit, never closed at the limit, never closed and loop-stopped, never closed
  and ended; the closing-time curve
  P(closed with reasoning ≤ B) for B = 1K, 2K, 4K, 8K, 16K, 32K tokens; reasoning tokens p50/p90/p95 (an unclosed
  response all reasoning) and answer tokens p50 given closure; the shares of closed answers with a fenced code block
  and with the closing BOS; the HumanEval+ responses' pass on the base and the plus tests (EvalPlus's own sanitizer
  and `check_correctness`, single-threaded BLAS), overall and given closure; outcomes per source group and style and
  per teacher-length bucket (< 2K, 2–4K, 4–8K, 8–16K, ≥ 16K) with the rank correlation of teacher length and
  closure; for never-closed responses the share of their last 4,096 tokens inside a repeated 32-token window, and every
  response's last 2,000 characters, to test whether they run away in loops. Each response keeps its full answer (an
  evalplus one also its full reasoning); scoring runs after generation in a CPU container of its own with
  single-threaded BLAS (`--score`), never in a process forked from vLLM's: there EvalPlus's 4 GiB address-space guard
  fails tests that allocate much (HumanEval/15's plus tests failed so at steps 1000 and 2000, whose
  `termination-<step>.json` plus counts are therefore low by at least one [measured: the stored tails re-scored]).
- **Budget forcing and sampling sensitivity** (`--budget`, `RUNS/<run>/termination-budget-<step>.json`): both sets
  under four samplings (the protocol's; T 0.8 and T 1.0 at top-p 0.95; Qwen3's top-k 20 with presence penalty 1.0),
  reasoning stopped at `END_THINK` or 16,384 tokens; closure, the closing-time curve and loops (repetition ≥ 0.99) per
  sampling; the HumanEval+ answers after the reasoning cut at B = 4K, 8K, 16K tokens (s1's budget forcing, arXiv
  2501.19393 §3.1: `END_THINK` appended to the first B tokens of a response not closed by B) under the protocol's
  sampling, and at 16K under the others, scored as above, split into naturally closed and forced. `--dry`: the same
  under the protocol's sampling with and without DRY on the reasoning (`eval/dry.py`, a vLLM logits processor;
  arXiv 2608.22761 App. A Eq. 1 with its Table 8 setting: multiplier 0.8, base 1.75, allowed length 2, range 1,024;
  breakers the tokens holding a newline, colon, quote or asterisk; off once `</think>` is emitted).
- **Serving faithfulness** (`scripts/serving_check.sh`, `RUNS/<run>/serving-check-<step>.json`): the export in vLLM
  against the training model on SFT val documents. Prefill: per-token NLL of 8 documents of 1K–32.7K tokens by
  `model.losses` on the val rows and by vLLM's prompt logprobs. Decode: vLLM's greedy generations (≤ 8,192 tokens)
  from the 4 val prompts with the longest teacher responses, teacher-forced through the training model: argmax
  agreement, the margin at disagreements, and both paths' log-probability of the chosen tokens. At step 2000
  [measured]: prefill mean NLL differs by ≤ 0.0009 nats in every position bucket (0–512, 512–2K, 2–8K, 8–16K, 16–32K;
  98,470 tokens); decode agrees on 98.7% of tokens 0–512, 99.85% at 512–2K and 100% at 2–8K, disagreement margins
  ≤ 0.42 nats and not growing, mean chosen-token log-probability equal to 0.001 nats: the export's 513-token windows
  and RoPE are honoured in both paths.
- **Cost** [measured: `termination-1000-v1.json`, the step-1000 probe beside the training job]: 929 s per probe
  (854 s generating); the run's median tokens/s fell 34% while it generated (312,840 → 205,817), about 290 s of run
  time per probe.

## 6. Code RL, deferred

- **When**: after §4's model passes its gate, if time allows; the recipe below is fixed so that it starts from
  settled choices. LoRA fits this stage: RL matched full fine-tuning at rank 1 ("LoRA Without Regret", §2.2), and FP16
  kept LoRA RL stable where "BF16-based LoRA training collapses after roughly 600 steps" (arXiv 2510.26788 §5.2).
- **Framework**: upstream verl in its own image, built from its `docker/Dockerfile.uv.cu130` for arm64 (torch 2.13,
  vLLM 0.32.0, transformers 5.16.1); FP16 training and sampling are part of it (verl #4036). The Precision-RL fork
  (verl 0.4.1.dev) does not run here: it pins vLLM ≤ 0.8.5, whose `vllm.worker.worker_base` its sampler imports and
  vLLM 0.12.0 removed, and transformers 4, against the model file's 5.
- **Model**: the SFT checkpoint through `inference/export_hf.py` with fp32 weights (bf16 today; fp16 from the fp32
  master rounds 8× less, relative error 2.1e-4 against 1.7e-3 [measured]) and a chat template that renders BOS, the
  prompt and `\n\n<think>\n`. The LM head with its cap and the gate of the value embeddings run with autocast off:
  autocast casts their fp32 `F.linear` inputs to fp16 (`inference/modeling_nanochat.py`).
- **Precision** (arXiv 2510.26788): the trainer computes in fp16 on fp32 master weights and fp32 AdamW state,
  reducing gradients in fp32, under a dynamic loss scale (a step with a non-finite gradient is skipped and the scale
  halved; it doubles after 400 clean steps); the sampler, vLLM, runs in fp16. The model's ranges in fp16: §2.6.
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
- **Problems** (rows counted from the files): `allenai/Dolci-Think-RL-7B` `code` (12,121, assert tests) and
  `code_stdio` (9,264, input/output tests), `KodCode/KodCode-Light-RL-10K` (10,000, function form with KodCode's
  tests) and `nvidia/Nemotron-RL-coding-competitive_coding` (16,083 per its card; Python stdin problems with tests
  inline, a median 55.5 per problem in its 25K file, prompt p50 640 tokens): 47,468 [arithmetic]. Dropped: a problem
  whose reference solution fails its own tests; the pretraining 10-gram rule (`docs/corpus.md` §3); a problem the SFT
  model solves 8 or 0 times of 8 at the rollout settings (Nemotron-Cascade drops those a 7B solves 8/8 and those
  R1-0528 solves 0/8, §4.6.1; here the policy decides both). DeepCoder-Preview is left out for its weak tests (§2.7).
  All Python: C++ comes from the SFT stage only.
- **Sampling**: temperature 1.0 and top-p 1.0, so that the sampler's distribution is the policy's, as the FP16
  analysis samples to compare them (arXiv 2510.26788 §3.5); Nemotron-Cascade 2 samples so in its other RL stages and
  at top-p 0.95 in code RL (arXiv 2603.19220 App. B).
- **Batch and lengths**: 128 prompts × 8 rollouts per update (arXiv 2512.13607 App. D Table 19); prompts ≤ 4,096
  tokens; responses ≤ 16,384, raised to 28,672 (32,768 − 4,096) once more than 10% are cut (§2.6, Length); a cut
  response scores 0 and stays in the batch (Nemotron-Cascade's code RL filters no overlong response, §4.6.2).
- **Optimizer**: AdamW at a constant 2e-6 [estimate: between Precision-RL's 1e-6 at 1.5B–30B (arXiv 2510.26788
  App. A) and Nemotron-Cascade's 3e-6–4e-6 at 8B–30B], β (0.9, 0.95), no weight decay, gradient norm clipped at 1.0.
- **Stopping and watches**: every 10 steps, pass@1 on 5% of the filtered problems held out; stop when it stops
  rising. Nemotron-Cascade's code RL ran 22 to 90 steps (arXiv 2603.19220 App. B; arXiv 2512.13607 App. D Table 19).
  Watched: the entropy (temperature 1.0 → 0.8 if it climbs, as Nemotron-Cascade's unified 8B run, Table 19), steps
  skipped by the loss scale, the sampler–trainer log-probability gap, the share of cut responses.
- **Not written**: the verl image; the fp32 export and the head and gate outside autocast in `inference/export_hf.py`
  and `inference/modeling_nanochat.py`; the problems and tests in verl's format with the 8-sample filter; the reward
  function; the run's settings.

## 7. Not in the pipeline

| Stage | Why not |
|---|---|
| A non-thinking mode | the format is thinking-mode only (§3) |
| LoRA, QLoRA | at hidden size 768 LoRA saves compute only below rank 128–205, where it lags full fine-tuning on code; QLoRA saves memory the run does not need (§2.2) |
| DPO (§2.4) | Nemotron-Cascade goes from SFT to RL without it |
| Math RL | the SFT data holds no math; Nemotron-Cascade's math RL moved LiveCodeBench v6 by −0.3 to +2.2 (arXiv 2512.13607 §4.5) |
| RLHF, instruction-following, multi-domain and long-context RL | chat, judges or tools the model is not trained for; reward models of 72B–235B (arXiv 2512.13607 §4.2–§4.4; arXiv 2603.19220 §4.2–§4.6) |
| Software-engineering RL | a 72B or 120B judge, or agentic execution over 256K tokens (arXiv 2512.13607 §4.7; arXiv 2603.19220 §4.8) |
| On-policy distillation (§2.5) | one RL domain leaves no specialists to merge; Nemotron-Cascade 2's teachers are its own per-domain checkpoints (arXiv 2603.19220 §4.4) |

## References

| arXiv | Title |
|---|---|
| 2104.09864 | RoFormer: Enhanced Transformer with Rotary Position Embedding |
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
