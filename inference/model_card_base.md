---
library_name: transformers
pipeline_tag: text-generation
tags:
- code
- python
- cpp
- custom_code
---

# nanochat d24 base

A 0.52B-parameter (195M non-embedding) decoder-only language model for Python and C++ code, pretrained from scratch on
103.03B tokens on one GH200. This is the base model: the pretraining run's final checkpoint (step 69,874), with no
instruction tuning. Every statement below is taken from the repository's documents
([github.com/indrajeetadityaroy9/nanochat](https://github.com/indrajeetadityaroy9/nanochat), `docs/`), which give the
code, records and papers behind each number.

## Usage

The model class is `modeling_nanochat.py` in this repository (torch and transformers only), so loading needs
`trust_remote_code=True`. It was exported and checked with transformers 5.21.0.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("<repo>")
model = AutoModelForCausalLM.from_pretrained("<repo>", trust_remote_code=True)
ids = tok("def fibonacci(n):\n", return_tensors="pt")  # the tokenizer prepends BOS
out = model.generate(**ids, max_new_tokens=128, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))
```

BOS (id 32,764) also ends a document: the model stops by emitting it. Fill-in-the-middle prompts use the three FIM
sentinels after BOS in prefix-suffix-middle order (`corpus/tokenizer.py` `fim_prompt`). The benchmark numbers below
were produced with vLLM 0.32.0 serving this export (Transformers backend); on the first 8 HumanEval problems the
transformers model file's greedy completions (128 new tokens) equal vLLM's.

## Model

| | |
|---|---|
| Layers, hidden size, heads | 24, 768, 6 heads of 128 (multi-head attention) |
| Parameters | 195,096,576 non-embedding; 522,252,336 in all (token table, 12 value-embedding tables, untied head) |
| Vocabulary | 32,768 (byte-level BPE: 256 bytes, 32,508 merges, BOS and 3 FIM sentinels) |
| Context | 32,768 tokens; 18 layers attend a 512-token sliding window, every fourth layer and the last the whole context |
| Blocks | pre-norm, gain-free RMS norm; QK-Norm with learned gains; RoPE (θ = 10^6); ReLU² MLP of 4× width |
| Additions | value embeddings in alternating layers (12 of 24) with a learned gate; learned residual and embedding mixing weights per layer |
| Output | logits capped as 15·tanh(z/15) |
| Weights | bf16 embeddings and matrices; fp32 gains, mixing weights and gates (`model.safetensors`, 1.04 GB) |

Details and the basis for each choice: `docs/architecture.md`.

## Training data

A Python and C++ code corpus with some text and mathematics (176.6M train documents from 18 sources: Stack-Edu,
UltraData-Stack-v3 and UltraData-Code, Nemotron rewriting and transpilation, CodeXGLUE, Kaggle notebooks,
Codeforces-CoTs, UltraData-Math and ClimbMix). The train splits are decontaminated against the EvalPlus, MultiPL-E,
HumanEval-X and HumanEval-infilling texts by 10-gram overlap. Training mixes, at 8K context, about 87% code (59%
repository files, 26% synthetic code), 10% text and 3% mathematics, then, at 32K context, whole repositories. Part of
the code is trained as fill-in-the-middle examples. Sources, curation, decontamination, tokenizer and mixture:
`docs/corpus.md`.

## Training

| | |
|---|---|
| Tokens | 103.03B (528 per non-embedding parameter): 97.41B as 8,192-token files, then 5.62B as 32,768-token repositories over the last 5.45% of steps |
| Steps, batch | 69,874 steps of 1,474,560 tokens |
| Optimizers | NorMuon for matrices, AdamW for embeddings, head and vectors; one learning rate and weight decay |
| Learning rate | peak 2.3454e-3 (DeepSeek LLM's compute law, times a calibrated multiplier), 40 warmup steps, linear decay to zero |
| Weight decay | 0.15322, from the Power Lines EMA-timescale law (arXiv 2505.13738) |
| Hardware | one NVIDIA GH200, about 80 hours of training |

Plan, optimizer, schedule, calibration and throughput: `docs/training.md`.

## Results

Greedy pass@1 by each benchmark's official tool, vLLM on this export, with the 95% normal interval of the problem
mean (`docs/results.md` §1; protocol `docs/evaluation.md`):

| Benchmark (tool) | n | pass@1 | 95% interval |
|---|---|---|---|
| HumanEval (EvalPlus) | 164 | 0.268 | [0.200, 0.336] |
| HumanEval+ (EvalPlus) | 164 | 0.220 | [0.156, 0.283] |
| MBPP (EvalPlus) | 378 | 0.444 | [0.394, 0.495] |
| MBPP+ (EvalPlus) | 378 | 0.365 | [0.316, 0.414] |
| HumanEval-C++ (MultiPL-E) | 161 | 0.217 | [0.153, 0.281] |
| MBPP-C++ (MultiPL-E) | 397 | 0.368 | [0.320, 0.415] |
| MBPP 3-shot (lm-evaluation-harness, `mbpp_lf`) | 500 | 0.322 | [0.281, 0.363] |
| HumanEval infilling, single-line / multi-line / random-span (PSM) | 1,033 / 5,815 / 1,640 | 0.553 / 0.261 / 0.113 | — |

`mbpp_lf` is lm-evaluation-harness's MBPP prompt with its few-shot solutions' CRLF line endings normalized as the
corpus normalizes them; on the unmodified CRLF prompt the model scores 0.184 (`docs/results.md` §1). Held-out bits per
byte at the end of training: 0.3584 at 8K and 0.3515 at 32K (code 0.2932 / 0.2827, text 0.8424 / 0.8430, mathematics
0.3266 / 0.3263; `docs/results.md` §4). The comparison with published base models of similar size is
`docs/results.md` §2.

## Limitations

- **A base model.** It continues code; it is not trained to follow instructions or chat.
- **It often does not stop.** 13–18% of the EvalPlus completions run to the 768-token budget, and infilling middles
  often overrun the hole (`docs/results.md` §3). Use stop sequences or a token limit.
- **Wrong logic is the main failure**: half of HumanEval+ fails on a wrong output; no EvalPlus failure is unparseable
  Python, and 7–15% of the C++ completions do not compile (`docs/results.md` §3).
- **Narrow domain.** Python and C++ with some English text and mathematics; prose-heavy tasks are its weakest
  (held-out text 0.84 bits per byte against 0.29 for code).
- **Sensitive to carriage returns.** Training text has none (CRLF and CR are mapped to LF); MBPP 3-shot falls from
  0.322 to 0.184 when its few-shot solutions keep CRLF line endings (`docs/results.md` §1).
- The original MBPP texts are outside the decontamination (`docs/evaluation.md` §4).

## Citation

The method and every number above are documented in the repository: `docs/architecture.md`, `docs/corpus.md`,
`docs/training.md`, `docs/evaluation.md` and `docs/results.md`
([github.com/indrajeetadityaroy9/nanochat](https://github.com/indrajeetadityaroy9/nanochat)).
