# nanochat

A fork of [karpathy/nanochat](https://github.com/karpathy/nanochat) for pretraining a Python and C++ code-generation
base model on one NVIDIA H100 80 GB SXM5 (sm_90): [DeepSeek-Coder](https://arxiv.org/abs/2401.14196)'s next-token +
fill-in-the-middle objective and [Qwen2.5-Coder](https://arxiv.org/abs/2409.12186)'s file-level (8K, RoPE base 1e4)
then repository-level (32K, RoPE base 1e6) stages.

The model is a Qwen3-style dense decoder with nanochat's measured wins: MHA with head dim 128, QK-norm with learnable
per-dim gains before RoPE, sliding windows SSSL (S = 512, last layer full), ReLU² MLP (4C), value embeddings on
alternating layers with a sigmoid gate, x0/residual lambdas, untied lm_head, zero-init output projections, no logit
softcap; every matrix and table initialized variance preserving (std C^-1/2). d24 is 24 x 768, vocabulary 2^15.

The regime is DeepSeek LLM's compute-optimal batch size and learning rate for the run's compute, linear warmup over
2/(1-beta2) steps, constant, then a (1-sqrt) cooldown over the last 20% of steps; AdamW (0.9, 0.95, wd 0.1) on the
embeddings and lm_head, Muon (Polar Express, Moonlight's update RMS matching, shared lr and wd) on every block matrix.

## Setup

On a fresh single-H100 host with Docker and the NVIDIA driver (Lambda's Ubuntu image), from this checkout: the CDI spec,
the docker group, the image (`docker/Dockerfile`) and the tokenizer and compiled + packed data from the host that
prepared them (an ssh alias or user@host); then, from a new login, the production run.

```bash
runs/setup.sh <source-host>
tmux new -d -s h100 'cd ~/projects/nanochat && runs/h100.sh'
```

Every command runs in the image through `runs/container.sh`, which mounts this checkout and the experiment directory
`~/nanochat-runs` (tokenizer, data store, checkpoints, metrics, eval results, kernel caches). Only the data
preparation reads the Hugging Face token, the Hub library's file under the image's `HF_HOME`:
`runs/container.sh hf auth login --token <token>` writes it once.

## Pipeline

```bash
# the mixture is config.MIXTURE: code:text:math = 87:10:3 (DeepSeek-Coder's code share), each corpus at its natural
# share of its group's packed rows (computed at run start)

# fetch, normalize (Parquet store) and curate every source: the code group whole, text and math up to 7:2:1 shares of
# the code group's horizon; exact and near duplicates dropped (StarCoder2's MinHash LSH, one copy across all corpora and
# splits) with GitHub Linguist's vendored / generated files; 1% val by group; repository files as repository samples
# (files in import-dependency order, DeepSeek-Coder)
runs/container.sh python -m nanochat.data.prepare
# SA-BPE tokenizer of config.VOCAB_SIZE (2^15 ids: merges + BOS + 3 FIM sentinels) on equal byte shares of every source,
# each the smallest source's train split
runs/container.sh python -m nanochat.data.tokenizer
# tokenize every curated document once (independent of the sequence length; a repository sample's files after their
# path comments, LF line ends, no BOM)
runs/container.sh python -m nanochat.data.compile
# cut long documents (repository samples at statement boundaries), make half of the train pieces of the repository
# corpora AST-FIM fill-in-the-middle examples, Best-Fit-Decreasing pack them into rows: an index and the FIM tokens;
# --stage file: 8K rows of every corpus, --stage repo: 32K rows of the repository corpora
runs/container.sh python -m nanochat.data.pack --stage file
runs/container.sh python -m nanochat.data.pack --stage repo
# the most tokens per scaling parameter a d24 run fits without reading a document twice
runs/container.sh python -m nanochat.train.stages 24
# train (tag d<depth>-r<ratio>-s<seed>); a restart resumes from the tag's newest checkpoint
runs/container.sh python -m scripts.base_train --depth=24 --target-param-data-ratio=<ratio> --seed=42 \
  --save-every=<steps>
# evaluate the final checkpoint: per-corpus val bpb, solution bpb, EvalPlus HumanEval(+) / MBPP(+), MultiPL-E C++,
# HumanEval infilling (official evaluators, each in its own process)
runs/container.sh python -m scripts.base_eval --model-tag=<tag>
```

The production run (`runs/h100.sh`: d24 at the data's most tokens per scaling parameter, both stages, a checkpoint
every 1% of steps, then base_eval; a relaunch resumes from the newest checkpoint; meant for tmux, see Setup).

## Layout

- `nanochat/config.py`: the settings every run shares (context, mixture, micro-batch, run tag).
- `nanochat/data/`: sources (pinned Hub revisions), prepare, compile, pack, the loader, the tokenizer (SA-BPE training,
  tiktoken encoding) and `bpe/`, the Rust crate (PyO3): SA-BPE trainer, repository samples
  (tree-sitter imports, petgraph ordering) and packer (tree-sitter statement cuts, AST-FIM examples encoded by
  tiktoken-rs).
- `nanochat/model/gpt.py`: the model (configuration, GPT, its scaling parameters and FLOPs per token).
- `nanochat/ops/`: custom ops and kernels (FlashAttention-4, QuACK GEMMs and cross-entropy, q/k norm + RoPE prologue;
  CUDA).
- `nanochat/train/`: optimizers (torch.optim AdamW + Muon), run plan and schedule, the stages' rows (`stages.py`), the
  checkpoint, metrics records.
- `nanochat/eval/`: bpb and the code benchmarks; `nanochat/inference/`: the batched native CUDA decoder.
- `scripts/`: `base_train.py`, `base_eval.py`.
- `runs/`: `setup.sh` (fresh host), `container.sh` (run in the image), `h100.sh` (the production run).
