# Post-training

What would turn the base model into a code model that answers a prompt: supervised fine-tuning (SFT), distillation
from stronger models, and reinforcement learning with unit-test rewards by GRPO and its variants; the published
evidence each choice rests on, and the fetched data. Nothing here is implemented: the repository pre-trains
(`docs/training.md`) and evaluates (`docs/evaluation.md`) a base model.

Markers: [arithmetic] derived from code numbers; [inference] reasoning, not measured; [estimate]; [unverified] a
figure without a record, or a paper value its extracted text lacks. No cited paper post-trains a model below 1B on
code; every transfer to the 522M base model is [inference].

## Code

No code. §6 fetches the data into `artifacts/posttrain/` (not versioned).

## 1. Starting point

- **The model**: d24, 522,252,336 parameters (`docs/architecture.md` §1), 103.03B tokens (`docs/training.md` §2),
  a completion model: the tokenizer's special tokens are BOS and the three FIM sentinels (`corpus/tokenizer.py`
  `SPECIAL_TOKENS`), so there is no chat template. An example is a prompt (a signature and docstring, as EvalPlus
  prompts a base model, `docs/evaluation.md` §4) and a completion; the loss falls on the completion and its closing
  BOS. The cited recipes train chat or `<think>` formats; their prompts become docstrings here [inference].
- **The measures** (`docs/evaluation.md` §3–§4): greedy pass@1, pass@k from the sampled mode's 20 completions per
  problem, and the parse failures. At 5 tokens per parameter the d24 runs scored HumanEval+ 0.067–0.110, MBPP+
  0.114–0.132 and parsed 0.775–0.894 of MBPP+ (`docs/training.md` §5); the base model's own values come from its
  evaluation after training.
- **What to keep**: infilling (FIM examples in the packer's format, `docs/corpus.md` §8, in every SFT mix) and the
  32K context [inference].
- **Contamination**: every source passes the pretraining rule before use, a 10-gram of words shared with a scored
  benchmark's prompt or reference drops the example (`docs/corpus.md` §3); OpenCoder removes HumanEval/MBPP entry
  points and 10-gram overlaps (arXiv 2411.04905 §4.4).

## 2. Supervised fine-tuning

Published effects on code at the smallest sizes reported:

| Model | Data | HumanEval (+) | MBPP (+) | Source |
|---|---|---|---|---|
| phi-1, 1.3B | 180M tokens of exercises | 29% → 50.6% | — | arXiv 2306.11644 §2, Fig. 2.1, Table 1 |
| OpenCoder-1.5B, base → instruct | two SFT stages (§6) | 54.3 / 49.4 → 72.5 / 67.7 | 70.6 / 58.7 → 72.7 / 61.9 | arXiv 2411.04905 Tables 6, 7 (not controlled) |
| Qwen2.5-Coder-1.5B: OpenCodeInstruct vs its Instruct | 5M samples | 78.7 / 73.8 vs 70.7 / 66.5 | 80.2 / 68.3 vs 69.2 / 59.4 | arXiv 2504.04030 Table 3 |
| Llama-3.2-1B: OpenCodeInstruct vs its Instruct | 5M samples | HumanEval+ 50.0 vs 26.8 | — | arXiv 2504.04030 Table 3 |

- **Selection** (OpenCodeInstruct, 7B, 500K samples per arm, Table 4): an LLM judge's score 5 does best (HumanEval+
  80.5, MBPP+ 71.4) against a random 500K (77.8, 70.1), test-passing samples (78.4, 70.4) and failing ones (75.3,
  69.8); natural-language-to-code beats code-to-code (HumanEval+ 79.9 against 74.4, Table 5); gains continue to the
  full 5M (Fig. 3).
- **Stages**: OpenCoder's stage 1 (4.0M: theoretical computer-science question-answer pairs and broad instructions)
  then stage 2 (436K practical tasks from GitHub code) improves on stage 1 alone and on one mixed stage (§6.4; Table
  10's numbers [unverified]: absent from the extracted text).
- **Settings published**: OpenCodeInstruct 3 epochs, lr 5e-6, batch 2,048, length 2,048 (§3); OpenCoder stage 1
  one epoch at 2e-5, batch 4,096, stage 2 three epochs at 5e-5, batch 512 (§4.3); KodCode 2–3 epochs at 1e-5, batch
  128 (arXiv 2503.02951 App. C.1, 32B only).

## 3. Distillation

SFT on a stronger model's outputs.

- **Against RL at one size**: Qwen-32B-Base after "over 10K steps" of RL scores LiveCodeBench 40.2, distilled from
  DeepSeek-R1 57.2: "distilling more powerful models into smaller ones yields excellent results, whereas smaller models
  relying on the large-scale RL … may not even achieve the performance of distillation" (arXiv 2501.12948 (version 1)
  §4.1, Table 6).
- **Code traces**: OpenCodeReasoning, 736,712 Python samples over 28,904 competitive-programming problems from R1
  (16K-token outputs), syntax-checked but not execution-filtered; 7B–32B students, LiveCodeBench 51.3 at 7B, the data
  curve "does not plateau" at 736K (arXiv 2504.01943 §2.2–§2.4, Table 2); training on failing samples beat passing
  ones, attributed to harder problems (§4.1).
- **Small students**: "small models (≤3B parameters) do not consistently benefit from long chain-of-thought (CoT)
  reasoning or distillation from larger models" (arXiv 2502.12143 Abstract; math). Long against short CoT: Qwen2.5-0.5B
  14.8 against 19.5, 1.5B −7.1, 32B +13.7 (Table 1); large against small teacher, 0.5B −3.5 (Table 2); base models
  show a larger gap than instruct ones (§3.4). Mixing one long-CoT example in five (α = 0.2) is best at 3B (Table 3),
  untested below.
- **The valley** (arXiv 2510.06101 §4, Fig. 1; 7B and 8B students, LiveCodeBench): 1K distilled examples more than
  halve the baseline, 10K lift it about 50%, 30K a further 50–100%; easy and medium problems help more than hard ones
  (Table 2: +41% against +7% at 4K); correct and incorrect traces do the same (0.185 against 0.182).
- **Here** [inference]: short solutions first (OpenCodeInstruct, OpenCoder, KodCode's GPT-4o solutions); long R1
  traces (KodCode-V1-SFT-R1, OpenCodeReasoning) only as a minority of an SFT mix, at 10K examples or more, kept only if
  they beat the mix without them.

## 4. Reinforcement learning: GRPO

**GRPO** (arXiv 2402.03300 §4.1.1 Eq. 3) samples G outputs per prompt and replaces PPO's value model, "another model
of comparable size", with the group's mean reward; with one reward per output, every token's advantage is
Â = (r − mean(r)) / std(r) (§4.1.2). The PPO-clipped ratio is averaged over each output's tokens and the outputs, and a
KL penalty to the reference policy joins the loss through the unbiased estimator π_ref/π − log(π_ref/π) − 1 (Eq. 4).
DeepSeekMath's reward is a trained reward model; its settings: G = 64, lr 1e-6, β = 0.04, length 1,024, batch 1,024
(§4.2); from its 7B instruct model, GSM8K 82.9 → 88.2 and MATH 46.8 → 51.7 (Table 5). "RL enhances Maj@K's
performance but not Pass@K" (§5.2.2).

- **Rewards**: DeepSeek-R1 trains with rule rewards only, accuracy and format; for code "a compiler can be utilized to
  evaluate the model's responses against a suite of predefined test cases"; a neural reward model is avoided as
  "susceptible to reward hacking" (arXiv 2501.12948 §2.2).
- **On code**: KodCode trains GRPO with reward 1 iff a solution "passes all unit tests" (256 steps, lr 5e-7, 16
  rollouts, batch 256, 4,096 tokens, KL 0.001): Qwen2.5-7B-Instruct-1M HumanEval+ 79.3 → 86.0 and MBPP+ 69.3 → 72.8;
  the stronger Qwen2.5-Coder-7B-Instruct 85.4 → 85.4 and 71.7 → 74.1 (arXiv 2503.02951 §4.1, Table 4). On the
  DeepCoder data GRPO took Qwen2.5-Coder-7B-Instruct's MBPP+ 71.7 → 75.1 and HumanEval+ 84.1 → 81.1; SFT on the same
  problems lowered HumanEval+ to 57.3 (arXiv 2508.05170 Table 1).
- **Small models** (math): from the base model with GRPO, Qwen2.5-0.5B's six-benchmark average 12.1 → 20.9 and MATH500
  15.8 → 34.4 (arXiv 2503.18892 Table 1; baselines greedy, trained models sampled at 1.0). The data must match the
  model ("data must align with the base model's inherent reasoning capabilities", §3.2: MATH levels 1–4 for 0.5B); a −1
  format penalty collapsed Llama-3.1-8B (§3.1); temperature 1.0 or 1.2 beat 0.6–0.8 at 0.5B, and 1 or 4 rollouts per
  prompt were unstable at 24B (App. G); 8 rollouts, KL 1e-4 below 14B (App. B.5). SFT before RL lowered the RL
  result at 24B (§4, App. F).
- **Limits**: RL-trained models win at small k, but "as k increases to the tens or hundreds, base models consistently
  catch up and surpass" them, code following math (arXiv 2504.13837 §3.2, Fig. 2–4); problems the RL model solves are
  solvable by the base (§4.1), with exceptions on code (App. C.7, Table 6); GRPO's pass@1 trails the base's pass@256 by
  43.9 points (§4.3). Against this, zero-RL lifted pass@k "by 10-30 absolute points" at 24B (arXiv 2503.18892 §1).
- **Here** [inference]: a binary all-tests-pass reward, no neural reward model and no format reward; problems the
  policy solves sometimes; pass@1 and pass@k tracked together, coverage loss being RL's known cost.

## 5. GRPO variants

| Variant | Change to GRPO | Problem addressed | Evidence (setting) | Here [inference] |
|---|---|---|---|---|
| Dr. GRPO (arXiv 2503.20783 §3) | drops the 1/\|o_i\| length normalization and the std division (optionally a constant normalizer); the advantage becomes RLOO's up to G/(G−1) (App. A) | length bias (favors short correct and long wrong answers) and a question-difficulty bias | math; its whole 1.5B recipe 33.1 → 42.1 over the base (Oat-Zero-1.5B, Table 4); either removal alone beats GRPO (App. C) | adopt |
| DAPO (arXiv 2503.14476 §3–§4) | clip [1 − 0.2, 1 + 0.28] (Clip-Higher); dynamic sampling keeps groups with 0 < correct < G; token-level loss over the group; overlong filtering and a soft length penalty; no KL | entropy collapse, zero-advantage groups, length effects | math, Qwen2.5-32B: AIME24 30 → 50 over the additions (Table 1) | adopt Clip-Higher, dynamic sampling, token-level loss; length shaping only if outputs reach the cap |
| GSPO (arXiv 2507.18071 §4–§5) | sequence-level ratio (π/π_old)^{1/\|y\|}, clipped per sequence at [1 − 3e-4, 1 + 4e-4] | token-ratio noise and collapse; MoE routing drift (no Routing Replay); tolerates training/inference precision mismatch (§5.4) | Qwen3-30B-A3B MoE; math and code curves (Fig. 1) | a candidate if rollouts from the bf16 decoder (`docs/evaluation.md` §6) disagree with the training forward enough to destabilize token ratios |
| P-GRPO (arXiv 2508.05170 §2.4) | reward R^f + R^o + R^o·R^t: a reward model's reasoning score R^t counts only when the code passes all tests (R^o = 1) | outcome rewards ignore the reasoning; all-correct groups get no advantage | code, Qwen2.5-Coder-7B-Instruct: MBPP+ 76.2 against GRPO's 75.1, HumanEval+ 81.1 both (Table 1); math 7B 51.5 against 48.0 (Table 3) | not first: needs a reward model and a reasoning format |
| DGPO (arXiv 2601.20614 §3–§4) | advantage over the mean absolute deviation, so each question's total update is G; softmax difficulty weights (T = 2); loss over valid questions | GRPO's total update per question, 2G·√(p(1 − p)), peaks at pass rate p = 0.5 and shrinks toward 0 and 1 (Thm. 1) | math only (1.5B–7B): average 39.79 against 37.61 | no code evidence |

## 6. Data

Fetched with `hf download <id> --type dataset --revision <revision> --local-dir artifacts/posttrain/<id>`, each at the
revision resolved at fetch time (`artifacts/posttrain/manifest.jsonl` holds id, revision and bytes), 36 GB in all.
Rows are counted from the files; licenses are the dataset cards'.

| Dataset | Role | Rows | Tests | License |
|---|---|---|---|---|
| `OpenCoder-LLM/opc-sft-stage1` | SFT, stage 1 | 4,216,321 | — | MIT |
| `OpenCoder-LLM/opc-sft-stage2` | SFT, stage 2 | 436,347: educational 118,278, evol 111,183, McEval 35,943, package 170,943 | educational subset only | MIT |
| `nvidia/OpenCodeInstruct` | SFT | 5,000,000 | 10 per sample with execution status, an LLM judge's score (arXiv 2504.04030 §2.3–§2.5) | CC-BY-4.0 |
| `KodCode/KodCode-V1` | SFT, RL problems | 487,432 | GPT-4o solutions and tests kept only when they pass with 100% branch coverage, 7.52 tests per question (arXiv 2503.02951 §2.2, §3.2) | CC-BY-NC-4.0 |
| `KodCode/KodCode-V1-SFT-R1` | distillation | 483,437 | R1 solutions with their test results | CC-BY-NC-4.0 |
| `nvidia/OpenCodeReasoning` | distillation | 735,255 (two splits, 567,850 and 167,405) | none | CC-BY-4.0 (and a per-row field) |
| `KodCode/KodCode-Light-RL-10K` | RL | 10,000 | yes | CC-BY-NC-4.0 |
| `agentica-org/DeepCoder-Preview-Dataset` | RL | 24,287 train (PrimeIntellect 16,252, TACO 7,436, LiveCodeBench v5 599); 687 test (LiveCodeBench v5 279, Codeforces 408) | yes | MIT |

- **Decontamination as published**: KodCode flags 94 of 447K questions above cosine 0.95 to HumanEval, MBPP,
  BigCodeBench or LiveCodeBench v5 (§3.2), and its files carry `benchmark_similarity` and `benchmark_task_id` per row;
  OpenCodeReasoning's 90 flagged pairs were judged not duplicates, none removed (§2.1); OpenCodeInstruct's procedure
  is [unverified] (§2.2.1 empty in the extracted text). None of these is the pretraining rule (§1), which every source
  still needs.
- **KodCode's RL subset**: its paper trains GRPO on "KodCode-RL-10K" (9.5K train, 0.5K validation, §4.1);
  `KodCode-Light-RL-10K` holds 10,000 rows [inference: the same set].
- **Licenses**: KodCode's CC-BY-NC-4.0 limits it to non-commercial use.
- **Not fetched**: `OpenCoder-LLM/opc-annealing-corpus` (75.7 GB, an annealing-phase pretraining corpus);
  `open-r1/codeforces-cots` (already in the 32K stage, `docs/corpus.md` §7); `PrimeIntellect/verifiable-coding-problems`
  and `likaixin/TACO-verified` (sources of DeepCoder-Preview).

## 7. Plan [inference]

1. **Base evaluation**: the final checkpoint's benchmarks-mode record (`docs/evaluation.md` §1): greedy pass@1,
   pass@k from 20 samples, parse failures. A wide pass@1–pass@k gap is the room RL can use (§4).
2. **SFT**: OpenCodeInstruct's score-5 samples, OpenCoder stage 2 and KodCode-V1's verified solutions, with FIM
   examples, all decontaminated (§1); lr and epochs from §2's published range.
3. **Distillation**: R1 traces as one example in five of the SFT mix (§3), kept only if they beat SFT alone.
4. **GRPO**: all-tests-pass reward; Dr. GRPO's advantage and DAPO's Clip-Higher, dynamic sampling and token-level
   loss (§5); 8–16 rollouts at temperature 1.0; KL 1e-4 or none; KodCode-Light-RL-10K, then DeepCoder-Preview's train
   problems the policy solves sometimes; a zero-RL arm from the base model alongside, given §4's SFT-before-RL result.
5. **Each stage's gate**: pass@1 up on HumanEval+ and MBPP+, pass@k at k = 20 not down beyond its error bar, no parse
   regression, infilling not worse (`docs/evaluation.md` §4).

## References

| arXiv | Title |
|---|---|
| 2306.11644 | Textbooks Are All You Need |
| 2402.03300 | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models |
| 2411.04905 | OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models |
| 2501.12948 | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning |
| 2502.12143 | Small Models Struggle to Learn from Strong Reasoners |
| 2503.02951 | KodCode: A Diverse, Challenging, and Verifiable Synthetic Dataset for Coding |
| 2503.14476 | DAPO: An Open-Source LLM Reinforcement Learning System at Scale |
| 2503.18892 | SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild |
| 2503.20783 | Understanding R1-Zero-Like Training: A Critical Perspective |
| 2504.01943 | OpenCodeReasoning: Advancing Data Distillation for Competitive Coding |
| 2504.04030 | OpenCodeInstruct: A Large-scale Instruction Tuning Dataset for Code LLMs |
| 2504.13837 | Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? |
| 2507.18071 | Group Sequence Policy Optimization |
| 2508.05170 | Posterior-GRPO: Rewarding Reasoning Processes in Code Generation |
| 2510.06101 | The Valley of Code Reasoning: Scaling Knowledge Distillation of Large Language Models |
| 2601.20614 | Harder Is Better: Boosting Mathematical Reasoning via Difficulty-Aware GRPO and Multi-Aspect Question Reformulation |

Datasets: the eight of §6 on the Hugging Face Hub.
