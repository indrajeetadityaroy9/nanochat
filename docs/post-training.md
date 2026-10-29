# Post-training

What would turn the base model into a code model that answers a prompt and reasons before it answers: the pipelines
published in 2025–2026 (§2), how they update the weights (§3), supervised fine-tuning and off-policy distillation
(§4), preference tuning (§5), on-policy distillation (§6), reinforcement learning with verifiable rewards (§7) and its
code rewards (§8), the GRPO variants (§9), the data (§10) and a plan (§11). Nothing here is implemented: the
repository pre-trains (`docs/training.md`) and evaluates (`docs/evaluation.md`) a base model.

Markers: [arithmetic] derived from code numbers; [inference] reasoning, not measured; [estimate]; [unverified] a
figure without a record, or a paper value its extracted text lacks. Below 1B the cited evidence is MobileLLM-R1
(140M–950M), Qwen3-0.6B, Qwen2.5-Coder-0.5B and two 0.5B math studies; every transfer to the 522M base model is
[inference].

## Code

No code. The data of §10 is in `artifacts/posttrain/<id>`.

## 1. Starting point

- **The model**: d24, 522,252,336 parameters (`docs/architecture.md` §1), 103.03B tokens (`docs/training.md` §2),
  a completion model: the tokenizer's special tokens are BOS and the three FIM sentinels (`corpus/tokenizer.py`
  `SPECIAL_TOKENS`), so there is no chat template. An example is a prompt (a signature and docstring, as EvalPlus
  prompts a base model, `docs/evaluation.md` §4) and a completion; the loss falls on the completion and its closing
  BOS. The cited recipes train chat or `<think>` formats; their prompts become docstrings here [inference].
- **The vocabulary**: 32,768 ids of the repository's own tokenizer (`docs/corpus.md` §5); no published model shares
  it, which decides how teachers can be used (§6).
- **The measures** (`docs/evaluation.md` §3–§4): greedy pass@1, pass@k from the sampled mode's 20 completions per
  problem, and the parse failures. At 5 tokens per parameter the d24 runs scored HumanEval+ 0.067–0.110, MBPP+
  0.114–0.132 and parsed 0.775–0.894 of MBPP+ (`docs/training.md` §5); the base model's own values come from its
  evaluation after training.
- **What to keep**: infilling (FIM examples in the packer's format, `docs/corpus.md` §8, in every SFT mix) and the
  32K context [inference].
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
  PEFT on 8-bit models; arXiv 2401.00788 Fig. 1, §2.2); at 1B LoRA trails P-Tuning and (IA)³. Students below 14B
  train with full parameters, LoRA above (arXiv 2502.12143 §3.1). At 0.5B, SFT with LoRA scored 0.7% on OlympiadBench
  and 1.2% on MATH500, and SFT plus LoRA plus RL collapsed (arXiv 2506.13404 §4.2–§4.3; its tables are empty in the
  extracted text [unverified]).
- **RL with adapters** (1.5B, already distilled): LoRA rank 32 on every linear layer 42.5 against full parameters'
  44.9 and DoRA's 46.6, PiSSA 0.2 (arXiv 2512.23165 Table 3); Tina's rank-32 GRPO reached its best checkpoint for $9,
  at or above published full-parameter models on the same data (arXiv 2504.15777 Table 3).
- **Here**: full parameters in every stage. Pre-training already holds the fp32 weights, gradients and both
  optimizers' state of all 522,252,336 parameters in 7.47 GiB (`docs/training.md` §7), so memory gives no reason for
  adapters [inference].

## 4. Supervised fine-tuning and off-policy distillation

Off-policy distillation is SFT on a teacher's text, so it needs no shared vocabulary (§6). Published effects at 1.5B
and below:

| Model | Data | Result | Source |
|---|---|---|---|
| Qwen2.5-Coder-0.5B, base → instruct | tens of millions of synthetic instructions, then millions chosen by rejection sampling; FIM instructions; offline DPO | HumanEval 28.0 → 61.6, HumanEval+ 57.3, LiveCodeBench 2.0 | arXiv 2409.12186 §4, §6.1, Table 16 |
| DeepSeek-Coder-1.3B, base → instruct | 2B tokens (§3) | HumanEval 34.8 → 65.2 | arXiv 2401.14196 Table 3 |
| OpenCoder-1.5B, base → instruct | two SFT stages (§10) | HumanEval+ 49.4 → 67.7, MBPP+ 58.7 → 61.9 | arXiv 2411.04905 Tables 6, 7 (not controlled) |
| Qwen2.5-Coder-1.5B: OpenCodeInstruct against its Instruct | 5M short solutions | HumanEval+ 73.8 against 66.5, MBPP+ 68.3 against 59.4 | arXiv 2504.04030 Table 3 |
| Qwen2.5-Coder-1.5B-Instruct + rStar-Coder | 580K verified long-CoT solutions, 6 epochs | LiveCodeBench 6.5 → 40.1, HumanEval+ 81.7, MBPP+ 60.8 | arXiv 2505.21297 Table 2, §4.2 |
| MobileLLM-R1-950M, from pre-training | general SFT, then 6.2M reasoning samples (2.2M from OpenCodeReasoning-2) at 32K | LiveCodeBench v6 19.9 (Qwen3-0.6B 14.9) | arXiv 2509.24945 Tables 2, 5, 7 |
| Qwen3-0.6B, distilled (§6) | — | LiveCodeBench v5 12.3 thinking, 3.6 not thinking | arXiv 2505.09388 Tables 19, 20 |

- **Prompts before answers**: in a fit of score against unique prompts and answers per prompt, the prompts'
  coefficient is 4.831 and the answers' 2.635 (R² 0.989; arXiv 2506.13284 §4.4.2); at a fixed budget 64K prompts × 1
  beat 16K × 4 and 8K × 8 (arXiv 2601.06953 §3.2); 37.7K problems × 16 solutions scored LiveCodeBench 54.7 against the
  full set's 57.3 (arXiv 2505.21297 Table 6). OpenThoughts samples 16 answers per question to grow a source (arXiv
  2506.04178 §4.4).
- **Epochs**: gains plateau at 5–6 (arXiv 2506.13284 §4.4.3); long traces need 8–10 epochs, short ones 2–3; at 8
  epochs long traces scored LiveCodeBench v5 60.3 against short ones' 43.1 (7B; arXiv 2601.06953 Table 4).
- **Long or short traces at small sizes**: with 7.5K MATH prompts, short chains beat long ones below 3B
  (Qwen2.5-0.5B 19.5 against 14.8) and a small teacher beats a large one (arXiv 2502.12143 Tables 1–2); one long
  example in five is best at 3B (Table 3). The 950M and 1.5B models above trained on 580K–6.2M long traces
  [inference: the gap was measured at 7.5K prompts and is untested at that data scale below 1B].
- **The valley** (7B and 8B students, LiveCodeBench): 1K distilled examples more than halve the baseline, 10K lift it
  about 50%, 30K a further 50–100%; easy and medium problems help more than hard ones (+41% against +7%); correct and
  incorrect R1 traces distill alike (0.185 against 0.182) (arXiv 2510.06101 §4, Fig. 1, Table 2).
- **Sources and filters** (OpenThoughts, 7B student, 31.6K examples per arm): two sources beat sixteen (41.3 against
  36.4, Table 4); for code the best question filter is difficulty judged by an LLM (Table 5); QwQ-32B teaches better
  than DeepSeek-R1 by 1.9 code points though it scores lower itself (Table 8). No answer filter beat none:
  verification lowered the 7B (41.9 against 45.0) and raised the 32B (64.5 against 62.1) (Table 15), and unit-test
  filtering scored LiveCodeBench 36.0 against 38.5 (Table 18); cutting traces at 8,192 tokens cost 17.6% (Table 22).
- **Staging**: general instructions before reasoning data beat one joint stage at 950M on MATH and GSM8K (57.8 and
  68.5 against 56.2 and 53.1), not on LiveCodeBench (13.7 against 14.9); math-only reasoning data scored LiveCodeBench
  0.0, code-only 12.0 (arXiv 2509.24945 Table 8). Phi-4-mini-reasoning packs examples in distillation mid-training and
  not in the last SFT, "where we teach the model to decide where to stop generating" (arXiv 2504.21233 §3.2).
- **Template**: from a base model, "'simple' prompt templates greatly outperform standard post-trained templates
  (e.g., <think></think>)" (arXiv 2512.13961 §6). Seed-Coder's warmup starts from the base, not the instruct model,
  which "frequently collapses into typical SFT training data patterns during RL" (arXiv 2506.03524 §3.2).
- **Learning rate and forgetting** (1B and 1.5B students): out-of-domain scores move by at most 1.3 points at lr
  1e-5 and by −11.2 to −14.0 at 5e-5 (arXiv 2609.35259 §4.2).
- **Published settings**: rStar-Coder 6 epochs, lr 4e-5, batch 96, 16K (§4.1); MobileLLM-R1's reasoning SFT 4
  epochs, lr 8e-5, 32K (Table 2); OpenThoughts 5 epochs, lr 8e-5, batch 512, packed (App. Table 9); Olmo 3 Think 2
  epochs, lr 5e-5, 1M-token batches of packed 32K rows (App. A.6.1); Nemotron-Cascade lr 5e-5 then 2e-5, batch 256
  (Table 14).
- **Here** [inference]: short solutions with FIM first, then long traces behind a plain-text delimiter (no new
  tokens); at least 100K long traces, past the valley; arms long, short and one long example in five.

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
- **Here** [inference]: delta pairs are two teachers' text, so any vocabulary works; one arm between SFT and RL.

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
  (§4–§5); OPD consolidates the model's own RL-trained specialists (§11).

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
| Sampler–trainer mismatch | truncated importance weights; fp32 logits | OlmoRL caps the weight at 2; GLM-5 masks tokens outside [1/2, 2] (§3.2); fp32 logits raised ScaleRL's asymptote 0.52 → 0.61 (§3.2); in a dense model the displacement is "concentrated near zero and bounded, consistent with bf16 rounding error" (arXiv 2609.32444 §3.1) |
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
- **Here** [inference]: rollouts come from the decoder, whose KV cache is e4m3 while the trainer attends in bf16
  (`docs/evaluation.md` §6); with an FP8 sampler plain policy gradient stayed stable at 30B, an FP4 KV cache
  collapsed it (arXiv 2609.20807 §5.4). The per-token log-probability gap between decoder and trainer is measured
  before choosing one step per batch alone or with truncated weights.

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
- **Here** [inference]: binary on tests verified against a reference solution; the pass fraction the one alternative
  arm; no neural reward.

## 9. GRPO variants

| Variant | Change to GRPO | Problem addressed | Evidence (setting) | Here [inference] |
|---|---|---|---|---|
| Dr. GRPO (arXiv 2503.20783 §3) | drops the 1/\|o_i\| length normalization and the std division (optionally a constant normalizer); the advantage becomes RLOO's up to G/(G−1) (App. A) | length bias (favors short correct and long wrong answers) and a question-difficulty bias | math; its whole 1.5B recipe 33.1 → 42.1 over the base (Oat-Zero-1.5B, Table 4); either removal alone beats GRPO (App. C); OlmoRL and DeepSeek-V3.2 drop the std (§7) | adopt |
| DAPO (arXiv 2503.14476 §3–§4) | clip [1 − 0.2, 1 + 0.28] (Clip-Higher); dynamic sampling keeps groups with 0 < correct < G; token-level loss over the group; overlong filtering and a soft length penalty; no KL | entropy collapse, zero-advantage groups, length effects | math, Qwen2.5-32B: AIME24 30 → 50 over the additions (Table 1); its three core pieces are in OlmoRL, GLM-5 and Seed-Coder (§7); clip-higher "gradually becomes unstable" (arXiv 2505.22617 §4.3); under DAPO Phi-4-mini-reasoning's AIME consensus "consistently degrades" (arXiv 2504.21233 §5.5) | adopt dynamic sampling and the token-level loss; Clip-Higher with a PPO-form loss; §7's length curriculum in place of length shaping |
| CISPO (arXiv 2506.13585 §3.1) | clips the stop-gradient importance weight instead of the update, so every token keeps its gradient; token-level normalization; no KL | PPO clipping drops rare reflective tokens after the first off-policy update | Qwen2.5-32B zero-RL: DAPO's AIME in half the steps (Fig. 2); ScaleRL, 8B dense and a MoE: "the best balance of stability and robustness to hyperparameters" (arXiv 2510.13786 App. A.17) | the loss once rollouts lag the policy |
| GSPO (arXiv 2507.18071 §4–§5) | sequence-level ratio (π/π_old)^{1/\|y\|}, clipped per sequence at [1 − 3e-4, 1 + 4e-4] | token-ratio noise and collapse; MoE routing drift (no Routing Replay); tolerates training/inference precision mismatch (§5.4) | Qwen3-30B-A3B MoE; math and code curves (Fig. 1); in ScaleRL its runs "diverged mid-training" (App. A.17) | not first |
| P-GRPO (arXiv 2508.05170 §2.4) | reward R^f + R^o + R^o·R^t: a reward model's reasoning score R^t counts only when the code passes all tests (R^o = 1) | outcome rewards ignore the reasoning; all-correct groups get no advantage | code, Qwen2.5-Coder-7B-Instruct: MBPP+ 76.2 against GRPO's 75.1, HumanEval+ 81.1 both (Table 1); math 7B 51.5 against 48.0 (Table 3) | not first: needs a reward model and a reasoning format |
| DGPO (arXiv 2601.20614 §3–§4) | advantage over the mean absolute deviation, so each question's total update is G; softmax difficulty weights (T = 2); loss over valid questions | GRPO's total update per question, 2G·√(p(1 − p)), peaks at pass rate p = 0.5 and shrinks toward 0 and 1 (Thm. 1) | math only (1.5B–7B): average 39.79 against 37.61 | no code evidence |

## 10. Data

Rows are counted from the files.

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
| `allenai/Dolci-Think-RL-7B` | RL | 102,014: code 12,121 (assert tests), code_stdio 9,264 (input/output tests), math 30,180, instruction following 29,813, general 20,636; each with the pass rate of Olmo 3's starting checkpoint | yes |
| `KodCode/KodCode-Light-RL-10K` | RL | 10,000 | yes |
| `agentica-org/DeepCoder-Preview-Dataset` | RL | 24,287 train (PrimeIntellect 16,252, TACO 7,436, LiveCodeBench v5 599); 687 test (LiveCodeBench v5 279, Codeforces 408) | yes; weak enough for false-positive rewards (§8) |

- **Decontamination as published**: KodCode flags 94 of 447K questions above cosine 0.95 to HumanEval, MBPP,
  BigCodeBench or LiveCodeBench v5 (§3.2), and its files carry `benchmark_similarity` and `benchmark_task_id` per row;
  rStar-Coder removes 16-gram overlaps with HumanEval(+), MBPP(+), LiveCodeBench and USACO 2025 (§3.3);
  OpenCodeReasoning's 90 flagged pairs were judged not duplicates, none removed (§2.1); OpenCodeInstruct's procedure
  is [unverified] (§2.2.1 empty in the extracted text). None of these is the pretraining rule (§1), which every source
  still needs.

## 11. Plan [inference]

1. **Base evaluation**: the final checkpoint's benchmarks-mode record (`docs/evaluation.md` §1): greedy pass@1,
   pass@k from 20 samples, parse failures. A wide pass@1–pass@k gap is the room RL can use (§7).
2. **SFT**, full parameters (§3), two stages (§4): short solutions with FIM (OpenCodeInstruct's score-5 samples,
   OpenCoder stage 2, KodCode-V1); then long traces behind a plain-text delimiter (rStar-Coder, OpenCodeReasoning-2's
   Python, OpenThoughts3's code), at least 100K examples, several epochs at 32K; arms long, short and one long example
   in five. All decontaminated (§1).
3. **DPO** on delta pairs from a strong and a weak teacher (§5), against none.
4. **RL** with §7's components and a binary reward on verified tests (§8): Dolci-Think-RL's code, KodCode-Light-RL-10K,
   DeepCoder-Preview's train problems the policy solves sometimes; 8K responses first. Arms: the pass fraction; math RL
   before code RL.
5. **Consolidation**: if separate RL runs (code, math) each win where the other loses, on-policy distillation from
   them into one model (§6).
6. **Each stage's gate**: pass@1 up on HumanEval+ and MBPP+, pass@k at k = 20 not down beyond its error bar, no parse
   regression, infilling not worse (`docs/evaluation.md` §4).

## References

| arXiv | Title |
|---|---|
| 2401.00788 | Astraios: Parameter-Efficient Instruction Tuning Code Large Language Models |
| 2401.14196 | DeepSeek-Coder: When the Large Language Model Meets Programming - The Rise of Code Intelligence |
| 2402.03300 | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models |
| 2406.11931 | DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence |
| 2409.12186 | Qwen2.5-Coder Technical Report |
| 2411.04905 | OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models |
| 2501.12948 | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning |
| 2502.12143 | Small Models Struggle to Learn from Strong Reasoners |
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
| 2604.00626 | A Survey of On-Policy Distillation for Large Language Models |
| 2604.02721 | GrandCode: Achieving Grandmaster Level in Competitive Programming via Agentic Reinforcement Learning |
| 2604.13010 | Lightning OPD: Efficient Post-Training for Large Reasoning Models with Offline On-Policy Distillation |
| 2607.15161 | On-Policy Delta Distillation |
| 2607.24653 | Kimi K3: Open Frontier Intelligence |
| 2608.24135 | Robust Code RL via Faulty-Code-Driven Test case Synthesis and Dense Reward Shaping |
| 2609.20807 | Score Centering Stabilizes Off-policy Reinforcement Learning |
| 2609.32444 | Rethinking Training-Inference Mismatch in LLM Reinforcement Learning: Where It Arises and How to Correct It |
| 2609.35259 | On-Policy or Off-Policy Learning? A Systematic Study of Distillation Dynamics |
