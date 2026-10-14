# Data mixture for code-focused pretraining: DeMix with X-MDE features and code validation loss, on one GB10

The mixture of the five registered corpora (`nanochat/data/pretrain/sources.py`) is searched with the methodology of
four papers, stored as Markdown in `dev/papers/`:

| Paper | File |
|---|---|
| DeMix: Li et al., *Decouple Searching from Training*, arXiv 2602.00747v3 (ICML 2026) | `papers/demix_2602.00747v3.md` |
| MDE: Belenki et al., *Optimizing Pre-Training Data Mixtures with Mixtures of Data Expert Models*, arXiv 2502.15950v1 | `papers/mde_2502.15950v1.md` |
| Xing et al., *Influence-Based Pretraining Data Selection for Code LLMs*, arXiv 2604.07769v1 (ICPC 2026) | `papers/discore_code_influence_2604.07769v1.md` |
| FastMix: Tan et al., arXiv 2606.14971v1 | `papers/fastmix_2606.14971v1.md` |

## 1. What each paper contributes

| Paper | Adopted | Evidence in the paper |
|---|---|---|
| DeMix | The whole pipeline (§2.2–2.4, §3): a base model; one component model per corpus, trained from the base on the corpus mixed half and half with general data (β = 0.5); a proxy for mixture α that is the linear merge $\sum_i\alpha_i\Theta_i$ (Eq. 6); three rounds of 64, 32 and 16 merged models, each round's next mixtures the best of a LightGBM predictor (learning rate 0.02, 300 iterations); the final mixture the average of the predictor's top 128 | Table 2: merged proxies reach code Spearman 0.81 at 211B total tokens, against 0.60 for trained proxies at 224B; the only one of the four built for a joint general/math/code mixture |
| MDE | The regressor's features (§3.2, "X-MDE"): the mixture weights plus each validation domain's MDE loss, $-\frac{1}{\vert V_j\vert}\sum_t\ln\sum_i\alpha_i p_i^{(t)}$, from each component's cached next-token probabilities (§3.1, Algorithm 1) | Fig. 2: adding MDE features raises the ET+SP ranking correlation of Linear from 0.646 to 0.953 and of GBM (RegMix's regressor) from 0.814 to 0.955 |
| Xing et al. | The validation signal (§3.2.2): each code benchmark's validation loss on its reference solutions, at most 200 problems per benchmark ("For datasets with over 200 problems, we randomly sampled 200") | Table 2: across checkpoints this loss has Spearman 0.958 with HumanEval-Py pass@1 and 0.923 with MBPP, against 0.398 and 0.502 for held-out pretraining loss |
| FastMix | Nothing | Its α gradient (Eq. 7) is derived "in the special case $n_2=1$ with SGD updates"; nanochat trains with Muon + AdamW |

Xing et al.'s DIScore selects individual documents; the corpora are used as published, so it is not applied.

The signal replaces DeMix's OpenCompass pass@1 because pass@1 is at its floor at this scale: Xing et al.'s 1.1B
code-only model scores 2.35% HumanEval-Py pass@1 after 5B tokens (Fig. 9). The objective is code only: the average,
over HumanEval and MBPP, of each merged model's rank (DeMix §2.4 ranks over general, math and code benchmarks).

## 2. Scale: DeMix's budgets per parameter on nanochat d8

DeMix's token budgets are kept as tokens per parameter of its Qwen3-1.7B backbone; everything else follows DeMix §3
or nanochat's training recipe.

| | DeMix (§3, App. A.1) | Here |
|---|---|---|
| Model | Qwen3-1.7B | nanochat d8: 41,943,232 scaling parameters (transformer matrices 25,166,016 + lm_head 16,777,216), vocab 32,768 |
| Candidate sets | 7: 1 general, 3 math, 3 code tiers | 5 corpora: `climbmix` (general), `opc_fineweb_math`, `opc_fineweb_code`, `stack_edu`, `stack_v3` |
| Base model | 50B tokens (29.41 per parameter) | 1,233,387,520 tokens, uniform over the 5 corpora (0.2 each) |
| Components | 30B tokens (17.65 per parameter), β = 0.5 | 17.65 per parameter, β = 0.5 with `climbmix`; the `climbmix` component is trained on `climbmix` alone |
| Final validation | 50B tokens on the final mixture | 29.41 per parameter, from the base |
| Optimizer, schedule | LR 3e-4, cosine to 20%, batch 512 × 8192 | nanochat: Muon + AdamW, warmup / constant / warmdown to 5%, batch from its scaling rule (262,144 tokens), sequence 2048 |
| Merge, search | linear merge; 64 + 32 + 16 merged models; LightGBM lr 0.02, 300 iterations; top 128 averaged | same (`scripts/demix.py`); 200,000 candidates per round, as in DeMix's released predictor |

A merge weight α stands for the data $0.5\cdot\text{general} + 0.5\sum_i\alpha_iD_i$ (App. A.1: "we can only search
within the subspace where non-general data accounts for less than 50%"); that mixture is what `scripts/demix.py`
writes for `base_train`. A general corpus receives its 0.5 share only if it is also a component, so `climbmix` must be
passed in both `--components` and `--general`; otherwise it is dropped from the output mixture.

DeMix measures proxy accuracy against 96 reference models trained on sampled mixtures (§3.3.1). At 2.95 h per d8
reference that is 283 GPU-hours, so it is not reproduced. The final mixture is compared with two trained baselines
from the same base and budget: uniform over the five corpora, and uniform merge weights (the centre of the searched
subspace).

## 3. Hardware (GB10)

- FlashAttention-4 is registered as a `torch.library` op, so `torch.compile(fullgraph=True)` compiles the whole step
  (`nanochat/gpt.py`). d12: 30,640 → 53,160 tok/s, peak 63.4 → 27.4 GiB.
- d8 at batch 64: 111,711–116,300 tok/s.
- Runs go one at a time: two concurrent runs time-slice to 26,230 + 26,291 = 52,521 tok/s in total.
- Memory: one d8 batch-64 run holds about 47 GiB of the 121.7 GiB unified pool:
  - 36.7 GiB of device memory (`nvidia-smi`);
  - an 8 GiB shared host mapping the size of the bf16 logits (B·T·V·2 bytes);
  - 2.5 GiB anonymous and 1.4 GiB file-mapped memory.

  Three concurrent runs exhausted the pool. The kernel swapped instead of killing for 30 minutes, which stalled
  `tailscaled` and dropped SSH (28 Sep, 01:15–01:46).

## 4. Implementation

| Piece | Where |
|---|---|
| Weighted mixture stream: corpus of each row from a Weyl sequence over the cumulative weights, every corpus continuing where the run it starts from stopped | `nanochat/data/pretrain/stream.py` (`parse_mixture`, `mixture_rows`, `PretrainingBatches`) |
| Mixture training, continued training from a checkpoint | `scripts/base_train.py` (`--dataset name:weight,...`, `--init-from`) |
| One tokenizer over several corpora, an equal share of characters each | `nanochat/tokenizer.py` (`--dataset a,b,...`) |
| Per-corpus bits per byte of a mixture checkpoint | `scripts/base_eval.py` |
| MBPP (full configuration, pinned) | `nanochat/data/eval/mbpp.py` |
| DeMix search with X-MDE features; code validation loss of trained models | `scripts/demix.py` (`--components`, `--evaluate`) |

Commands, from the repository root inside the container:

```
python -m nanochat.tokenizer --dataset=climbmix,opc_fineweb_code,opc_fineweb_math,stack_edu,stack_v3
python -m nanochat.data.pretrain.compile --dataset=<corpus> --max-files=<N>          # each corpus
python -m scripts.base_train --depth=8 --device-batch-size=64 --model-tag=demix-base \
    --dataset=climbmix:0.2,opc_fineweb_code:0.2,opc_fineweb_math:0.2,stack_edu:0.2,stack_v3:0.2 --target-param-data-ratio=29.41
python -m scripts.base_train --depth=8 --device-batch-size=64 --model-tag=demix-<corpus> --init-from=demix-base \
    --dataset=<corpus>:0.5,climbmix:0.5 --target-param-data-ratio=17.65   # each non-general corpus
python -m scripts.base_train --depth=8 --device-batch-size=64 --model-tag=demix-climbmix --init-from=demix-base \
    --dataset=climbmix --target-param-data-ratio=17.65
python -m scripts.demix --components=climbmix=demix-climbmix,opc_fineweb_code=demix-opc_fineweb_code,... --general=climbmix
python -m scripts.base_train ... --model-tag=demix-final --init-from=demix-base --dataset=<search.json mixture> --target-param-data-ratio=29.41
python -m scripts.demix --evaluate=demix-final,demix-uniform,demix-merge-uniform
```

## 5. Status

Inactive: the proof-of-concept data and runs were purged, and the current compiled data uses a ClimbMix-only tokenizer,
so the five-corpus tokenizer above must be trained (and every corpus recompiled) first.
