# Actions from the subagents (living document)

Tracked in the working repo (`ACTIONS.md`), updated as each subagent reports. Every open item is answered from the
literature, citing exact sections, or, when only a measurement can settle it, designed as an ablation (§8).

| Subagent | State | Report |
|---|---|---|
| EvalInference-2 (refactor: eval, inference, syntax crate) | done | §2 |
| Ablations (refactor: harness, lr, tokenizer ablation, pretrain script) | done | §2 |
| TrainModel (refactor: model, ops, csrc, train, config, paths, native) | done | §2 |
| ArchitectureCheck (model and regime vs papers) | done | §1, §5 |
| CorpusCuration (refactor: schema, sources, parallel, prepare, curate, publish) | done | §2, §6 |
| CorpusTokens (refactor: tokenizer, mixture, compile, pack, loader, data crate) | running | pending |
| EfficiencyResearch (GH200 throughput) | running | §3 (partial) |
| RegimeAnswers (D1, D2, D10, D11, Muon warmup, d8@20 record) | running | §8 |
| OptimizerAnswers (D3, D6, D7) | running | §8 |
| ArchitectureAnswers (D4, D5, D9, value embeddings) | running | §8 |
| DataAnswers (D8, FIM rates and SPM share, M1) | running | §8 |
| EngineeringAnswers (A1, viable_end, I1, E1, lr.json keys) | running | §8 |

Order: decide §1 → implement and verify on the GH200 → integrate §2 and commit → tokenizer ablation `data` and runs
(§7). Anything in §1 that changes the model or the run plan must land before the ablation's runs; the intrinsic record
(tokenizer-only) is unaffected by all of it.

## 1. Decisions: model and training (ArchitectureCheck; affect every run, so the tokenizer ablation)

| # | Finding (evidence in `local://ArchitectureCheck.md` §B, §C) | Options | Effect if changed |
|---|---|---|---|
| D1 | **FLOP count M** (`gpt.py:201-205`, `schedule.py:41`): M uses each layer's attention window and counts the ve_gate matrices (nanochat 92's convention), not DeepSeek LLM Eq. 2's full-context M. C is 1.82× below Eq. 2's at d24/8K: B_opt 12.06 vs 14.67 micro-batches, law lr ×1/0.928 | keep 92's convention (state it) / adopt Eq. 2's M, the definition the law's constants were fit with | batch, steps and lr of every run change; the lr ablation's multipliers are relative to the old law lr and need re-basing |
| D2 | **Repository-stage lr**: the linear decay continues through the 32K stage; DeepSeek-Coder §3.6 holds the final lr, Qwen2.5-Coder gives no schedule (unsupported claim) | keep (our design, decay-to-zero per 2502.15938) and say so / hold the final lr | schedule of the last 5.45% of steps |
| D3 | **Gradient clip 1.0 before Muon** (`optim.py:11`): only DeepSeek LLM §2.3 states 1.0; Megatron PR 5395 (mis-described) skips global clipping for Muon and notes a pre-Muon clip can push gradients under Newton–Schulz's eps floor | keep / skip clipping for Muon matrices | update of Muon parameters on large-norm steps |
| D4 | **Learnable QK gains** (`gpt.py:3-4,145`): no measured basis (they come from transformers' Qwen3 code; the Qwen3 paper and nanochat's LOG do not measure them) | keep, cite transformers / remove | model parameters and forward |
| D5 | **No output-logit softcap**: the stated reason ("QK norm bounds the attention logits") does not cover 92's softcap, which acts on the lm_head output | keep without softcap, give the real reason / add 92's softcap | loss and logits |
| D6 | **One lr and weight decay for every optimizer group** (`optim.py:31-33`): no source; 92 uses per-group lrs (embeddings, head, scalars) | keep / per-group lrs | every update |
| D7 | **τ weight-decay law on Muon** (`optim.py`): fit on AdamW only (Power Lines §2.3); Moonlight §2.2 is partial support | keep with the stated assumption / separate Muon decay | weight decay of matrices |
| D8 | **Synthetic code at 30% of code** (`mixture.py:80-83`): Kang et al.'s ~30% is for rephrased web text | keep, mark [inference] / change the share | mixture (and the corpus extents: §6) |
| D9 | **RoPE base 10^6 from the start**: Qwen2.5-Coder uses 10^4 at 8K and 10^6 at 32K; Llama 3 §3.2 supports a large base from the start (stated in the code) | keep (supported) | none |
| D10 | **Last step's lr is 1.15e-5 of peak, not 0** (`lr_multiplier`) | doc fix / make the last step 0 | negligible |
| D11 | **`plan` truncates `int(ratio * params)`**: 86,974 steps at the exact ratio, 86,975 at max_ratio's bisected ratio | keep (mixture.STEPS matches the bisected run) | none |

Recommendation to confirm with the user: D1 is the one with a material, source-backed fix (the law's own M); D3 and
D5 follow EfficiencyResearch/ablation evidence; D2, D4, D6, D7, D8 are kept unless measured otherwise, with their
assumptions stated; D9-D11 need doc fixes only.

## 2. Refactor slices: integration

Done and verified on the GH200 (ref vs new): identical outputs, bitwise where deterministic, FA4 gradients within the
reference's own spread, CUDA SASS/PTX identical, no interface or record-format change.
- EvalInference-2: `warnings.catch_warnings(action=...)`, list comprehension in `multipl_e`, duplicate import, an
  unreachable `top_p` check, lexer constants and inlined helpers. Kept on purpose and worth a decision:
  - **A1** the syntax constraint builds its tables from `get_tokenizer()` (the corpus tokenizer), not the caller's
    tokenizer. Harmless today (ablation runs evaluate with "parse", unconstrained), wrong for generating with an arm's
    vocabulary: pass the tokenizer in (interface change).
  - `viable_end` treats a parse failure as "no partial token" while `allowed` treats it as false (kept).
- Ablations: one cached `shard_bytes` (duplicate footer reads), Latin-1 in place of the GPT-2 byte alphabet (one to one;
  identical ids on 23,000 pieces per arm), an unreachable `fit` guard removed, duplicate import.
- TrainModel: `layer_windows` as one expression, stale docstrings, `CACHE` dropped from `paths.__main__` (created by
  `NATIVE`), line reflows (CUDA host code differs only in `__LINE__` constants).
- CorpusCuration: one `undropped(paths)` SQL helper shared by `shuffle` and `shard_documents` (same SQL text), one
  `pa.array` pass with nulls in `check_row_group` (no trailing cast), the standardizer looked up once per file,
  `safe.directory={ROOT}`, stale references (`nanochat/data`, `packed.json`), `fissix` imported at module level.
  Verified identical: `texts`/`text_bytes`/`curated_shards` on all 1,812 shards and 2.1M rows; `curate.main()` end to
  end on synthetic standardized files for all 18 sources (1,822 output files byte-identical); `prepare` on one real Hub
  file per category (raw, standardized, manifest, extent identical; the extent matches the published file list);
  `publish` push/pull with the Hub faked. Kept on purpose: `Category.name`, `push`'s `if old:`, resumable raw/checked
  files, the repair `except` set, the sample-cut loop.
- Pending: CorpusTokens. Then a second independent audit pass over every slice, ruff and the 120-char check on the
  GH200, one commit pushed from the GH200.
- Verification hazard (fixed): ref and new shared `artifacts/cache/native`, and cargo's single fingerprint let a new
  build reuse the reference library; the poisoned data-crate `.so` was deleted, comparisons redone with private caches.

## 3. Throughput (EfficiencyResearch, partial)

| # | Finding | Action | State |
|---|---|---|---|
| E1 | GPU power: requested 900 W, enforced 700 W by the hypervisor (KVM guest); during the d24 8K step the SW power cap is active in every sample, SM clock avg 1,564 MHz of 1,980 | ask Lambda to raise the VM's GPU power limit to 900 W | open (provider) |
| E2 | Locking the SM clock at 1,980 MHz: d24 8K step −3.0% (3,585-3,590 vs 3,682-3,696 ms), d16 −1.7%; speed only | locked machine-wide; `sudo nvidia-smi -lgc 1980,1980` added to `runs/setup.sh` (not persistent across reboot) | done |
| E3 | Profile, kernel breakdown, ranked CUDA/C plan | pending report | running |

## 4. Infrastructure

- **I1** Declare the ablation's direct imports in `pyproject.toml` at the versions it ran with: `tokenizers==0.23.2`
  (now only in the NGC image, via torchtitan) and `regex==2026.9.29` (locked via tiktoken); `uv lock`; rebuild the image
  after the slices finish. `basedpyright` stays (eval/verify.py runs it).
- **I2** `runs/setup.sh`: clock lock (E2) added; long line reflowed.

## 5. Documentation (after the code is final)

Consolidate the four docs (`docs/h100-pretraining.md`, `nanochat/corpus/normalization_doc.md`, `ablations/tokenizer.md`,
`nanochat/corpus/gh200.md`) from the final code: cutover done, `data` step, DeepSeek-Coder terminology. Corrections from
ArchitectureCheck (no code effect):
- wrong locations: MobileLLM 32×960 is Table 9; nanochat Muon 0.02 at 92 `gpt.py:419/450`, √ factor at `optim.py:393`;
  zero-init lm_head is modded-nanogpt record #8; Power Lines constants in App. E.3.1; DataDecide's ~80% in the abstract
  and §3.1; `harness.production_plan` → `base_plan`, `production` fields → `baseline`;
- wrong values: MoVE d24 extrapolation ≈ −0.014 in MoVE's coordinates (the cited gains are LaVE rows); deep-and-thin
  d24 analogue +9.2% FLOPs; LONG threshold 190.5; Step-Law's fit uses 1,912 samples from 7 dense sizes; "within 7% of
  nanochat's lr" fails like-for-like (1.24× and 1.69×); StarCoder2's FIM ≈25% of chunks, DeepSeek-Coder PSM only;
  Power coefficient 0.0122; minor KV-cache and fp32-angle bounds;
- unsupported: ResFormer citation for per-layer value embeddings (they are modded-nanogpt's / MoVE's LaVE); 2410.23922
  read as covering Muon; 2405.18392 cited for crossing curves; d8@20 ×2 lr results without a record in `results/`;
  `fit` called a polynomial fit per 2409.19913 (it is a 3-point local parabola).
- record: `results/lr.json` carries `production_*` keys while `lr.py` writes `base_*` (rename the recorded keys or
  regenerate).

## 6. Dataset `iaroy/nanochat-data`

No change required for training or the ablation: 1,812 shards + manifest (identical to the pulled copy, code
`aa37215`, an ancestor) + a data-only card; nothing computational reads the manifest's `bytes` or `revision`
(`prepare` compares each source's file list, the card reads `repo` and `rows`).
- **M1 (decision)** the published manifest is not what the current `publish.py` writes: its split `bytes` are
  `text_bytes + 4 × rows` (the publishing commit's count, e.g. stack_edu_python train 66,812,305,784 vs 66,714,195,924,
  gap exactly 4 × 24,527,465), and its sources carry `revision` fields the current `prepare` no longer writes (revision
  pinning removed in `69c198e`). A future `publish push` rewrites both. Options: leave the published manifest as the
  publishing commit's record / upload a regenerated `manifest.json` alone (one small commit) so the Hub matches the
  code.
- Trigger for a data change: a §1 decision that raises the base run's token budget (`mixture.STEPS`) or a mixture share
  (D8) beyond the extents' margins → extend on the GB10, prepare, curate, `publish push`.

## 7. Tokenizer ablation (after §1 lands)

`python -m ablations.tokenizer data` (the d16 set compiled and packed under every arm, `results/tokenizer-data.json`),
then `runs` (8 d16 runs, decision), then `adopt`, compile, pack, pre-training.

## 8. Answers and ablation designs

One entry per open item of §1-§6: the question, then either the literature answer (paper id, exact section, table or
equation, the decisive sentence) or an ablation design (hypothesis, arms, fixed configuration, scale, seeds, measures,
pre-registered decision rule, GH200 cost). Filled in as the research agents report.
