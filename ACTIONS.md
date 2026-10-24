# Actions from the subagents (living document)

Tracked in the working repo (`ACTIONS.md`), updated as each subagent reports. Every open item is answered from the
literature, citing exact sections, or, when only a measurement can settle it, designed as an ablation (§8). Rule
(user, Oct 24): no complete ablation experiment, nor the base model's pre-training, runs without the user's explicit
approval; proof-of-concept runs for development and verification are allowed, on real corpus data only.

| Subagent | State | Report |
|---|---|---|
| EvalInference-2, Ablations, TrainModel, CorpusCuration, CorpusTokens (KISS refactor) | done | §2 |
| ArchitectureCheck (model and regime vs papers) | done | §1, §5 |
| RegimeAnswers, OptimizerAnswers, ArchitectureAnswers, DataAnswers, EngineeringAnswers | done | §8 |
| EfficiencyResearch (GH200 throughput: the training step) | finishing: item 4 accepted (−0.5% d24); 5-8 dropped (each ≤1% realizable; the step's GEMMs, ~58%, run at the 700 W cap's ceiling, FA4 ~19%); applies item 4's wiring after the Hydra commit | §3, §8 |
| HydraRuns (Hydra run configuration, the D5/D6/D7 knobs) | running: knob and sweep checks, waiting for the eval-exit fix | §0 |
| RunUtilization (GPU and CPU utilization of a run's and a sweep's lifecycle outside the step: compile, evaluation, start-up, gaps; eval/ and inference/ hot loops into Rust where Python binds) | running: the eval-exit deadlock fix | §0 |
| CorpusParallel (CPU parallelism of the corpus pipeline and the data crate: serial phases such as `stage_parts`, Python hot loops into rayon; outputs byte-identical) | running | §0 |
| AblationsAudit (user's audit and KISS refactor of `ablations/` and its pipeline/tokenizer integration, `local://ABLATIONS-AUDIT.md`; corpus changes via CorpusParallel) | plan approved; edits after the Hydra commit; a separate reviewer does the second audit | §0 |

## 0. Status: closed and open

### Closed (decided; applied where code changes)

| Item | Decision | Where |
|---|---|---|
| D1 FLOP count M | adopt DeepSeek LLM Eq. 2's M; lr ablation re-based (base ×2.7972, lr 2.4138e-3), its 11 runs reused | `ba01bb5` |
| D2 repository-stage lr | keep decay to zero (2502.15938); citation fixed | `ba01bb5` |
| D3 clip before Muon | keep 1.0 (DeepSeek LLM §2.3); citation fixed | `a81410c` |
| D4 learnable QK gains | keep (nGPT App. A.9 Table 5); citation fixed | `ba01bb5` |
| D9 RoPE base | keep 10^6 from the start; citation fixed | `ba01bb5` |
| D10 last-step lr | keep (1.15e-5 of peak, 87× below 2502.15938's smallest minimum); doc wording in §5 | none |
| D11 plan rounding | `round(ratio * params)` | `ba01bb5` |
| Value embeddings | keep; cited as modded-nanogpt #14 / MoVE's LaVE, not ResFormer | `ba01bb5` |
| FIM configuration | keep rate 0.5, PSM/SPM 50/50, AST-FIM; PSM/SPM cited to Bavarian §3.1, §4.3, §8.1 | working tree |
| Muon warmup | keep 40 steps (nanochat 92's at d24; no lr-dependent warmup instability in the 11 lr runs); doc in §5 | none |
| d8@20 ×2 lr records | unrecoverable (H100, no record); delete or mark unrecorded in the docs (§5) | §5 |
| M1, manifest | withdrawn; `manifest.json` removed, Hub holds 1,812 shards + card (`b091cb47`) | `4ade225` |
| A1 syntax constraint | takes the caller's tokenizer; tables cached per tokenizer | `ba01bb5` |
| viable_end | an unparseable prefix is not viable (as `allowed`) | `ba01bb5` |
| I1 dependencies | `tokenizers==0.23.3`, `regex==2026.9.29` declared and locked | `ba01bb5` |
| `results/lr.json` keys | `production_*` → `base_*` | `ba01bb5` |
| E2 clock lock | 1,980 MHz machine-wide; `runs/setup.sh` | `4ade225` |
| E3 throughput plan | reported; items 1-2 (FA4 real segments, batched Newton-Schulz) applied, d16 −10%, d24 −5.3% | `a81410c` |
| KISS refactor | five slices verified identical to the reference | `4ade225` |
| Data provenance | every run reads iaroy/nanochat-data; 1,812 shards SHA-256-equal to `b091cb47` | §8 |
| Model-generated sources | stay in the mixture as designed (user) | none |
| D8 synthetic-code share | keep 0.30 of code (user: LLM-generated code is valid data; no ablation, no literature study); Kang et al.'s ~30% marked [inference] for code | `679e7d9` |
| Tokenizer `data` step | the d16 set under every arm; max_ratio sa-bpe 31.42, sa-bpe-fused 29.48, hybrid 31.49, bpe 31.42, every arm 3 micro-batches (one train shard per repository source exceeds d16 at 20's draw) | `679e7d9` |
| Run configuration | Hydra (`hydra-core==1.3.7`) replaces the custom ablation driver (user); contract `local://HYDRA.md` | done (§8) |

### Open (remaining)

| Item | Next action | Waits on |
|---|---|---|
| Hydra migration | code done, image rebuilt, lint clean; HydraRuns runs the HEAD-equivalence check (d8, 200 steps, real sa-bpe packs), knob checks, sweep mechanics; then I commit | HydraRuns |
| Eval-exit deadlock | every `nanochat.run` job hung after its first evaluation: `code_eval.sanitizers()`'s cached forkserver executor is joined by a spawned child's exit before `concurrent.futures` shuts it down (RunUtilization's repro); fix: the executor scoped to `suite.evaluate`; lands with the Hydra commit | RunUtilization |
| Provisional corpus tokenizer | sa-bpe copied to `CORPUS/tokenizer` (SHA-256-equal), full corpus compile + pack running (`corpus-sabpe`, cores 8-63): D7's 200 half needs ~2.94B tokens, the ablation set holds ~1.3B; the base model's data if sa-bpe is adopted, else replaced at `adopt` | running (CPU) |
| D7 Muon τ | `muon_tau_20.yaml` (d8 at 20, sa-bpe set, 4 seeds) and `muon_tau_200.yaml` (d8 at 200, `tokenizer=corpus`, 2 seeds); ×½, ×1, ×2 | Hydra commit; the provisional corpus for the 200 half; user approval |
| D5 softcap | `softcap.yaml`: d16 at ratio 20.0 on the sa-bpe set, lr ×2.7972 (the D1-rebased base multiplier; `results/lr.json`'s stored analysis predates D1 and is regenerated at `adopt`), cap 15 vs none, 4 seeds | Hydra commit; user approval |
| Tokenizer ablation | 8 d16 runs at each arm's max_ratio (4 arms × 2 seeds, ~7,151 steps each, `experiment=tokenizer`), decision, `adopt`, compile, pack | D5 and D7 decided; user approval |
| D6 AdamW-group lr | `adamw_lr.yaml`: d16 on the sa-bpe set at its max_ratio, ×¼ and ×4, 2 seeds; its ×1 arm is the tokenizer ablation's sa-bpe baseline (same run key), so 4 runs train | the tokenizer runs; user approval |
| Throughput items 4-8 | fused kernels (value-embedding gate: written, not bitwise, accepted if its fp64 error ≤ Inductor's and a d16 loss curve stays within the reference's two-run spread; FA4 backward pre/post, q/k norm + RoPE, RMSNorm, weight-gradient accumulation), each verified and committed | EfficiencyResearch |
| Run-lifecycle utilization | RunUtilization measures a real proof-of-concept job's GPU timeline (compile, steady state, checkpoint, each evaluation part, gaps between jobs) and removes idle time with records unchanged; each fix verified, then committed | RunUtilization |
| E1 GPU power cap | Lambda raises the host's 700 W cap to 900 W (request text in `ANS-EngineeringAnswers.md`); not changed between an ablation's runs | the user (provider) |
| Base model pre-training | `experiment=base` (d24, the decided recipe and tokenizer) | every ablation above; user approval |
| Docs consolidation (§5) | the four docs from the final code, with §5's corrections, D10 and warmup wording, the d8@20 records, FIM citations, stale line-widening/manifest text, `mixture.py`'s "86,975 steps of 12 micro-batches" (the D1 plan: 69,580 of 15, same tokens) | final code |
| Second-pass audit | KISS audit of the final code | final code |

Order: Hydra migration; provisional corpus → D7, D5 → tokenizer runs (with D6's arms) → adopt, compile, pack → base
model pre-training; throughput items land as they verify; docs and audit last.

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

**Status: §0.** Every row above is decided (D8: keep 0.30) except D5, D6 and D7, whose ablations (§8) wait for the
user's approval.

## 2. Refactor slices: integration

Done and verified on the GH200 (ref vs new): identical outputs, bitwise where deterministic, FA4 gradients within the
reference's own spread, CUDA SASS/PTX identical, no interface or record-format change.
- EvalInference-2: `warnings.catch_warnings(action=...)`, list comprehension in `multipl_e`, duplicate import, an
  unreachable `top_p` check, lexer constants and inlined helpers. The two items kept for a decision are closed (§0):
  **A1** (the syntax constraint now takes the caller's tokenizer) and `viable_end` (an unparseable prefix is not
  viable, as `allowed`), both in `ba01bb5`.
- Ablations: one cached `shard_bytes` (duplicate footer reads), Latin-1 in place of the GPT-2 byte alphabet (one to one;
  identical ids on 23,000 pieces per arm), an unreachable `fit` guard removed, duplicate import.
- TrainModel: `layer_windows` as one expression, stale docstrings, `CACHE` dropped from `paths.__main__` (created by
  `NATIVE`), line reflows (CUDA host code differs only in `__LINE__` constants).
- CorpusCuration: one `undropped(paths)` SQL helper shared by `shuffle` and `shard_documents` (same SQL text), one
  `pa.array` pass with nulls in `check_row_group` (no trailing cast), the standardizer looked up once per file,
  `safe.directory={ROOT}`, stale references (`nanochat/data`, `packed.json`), `fissix` imported at module level.
  Verified identical: `texts`/`text_bytes`/`curated_shards` on all 1,812 shards and 2.1M rows; `curate.main()` end to
  end on synthetic standardized files for all 18 sources (1,822 output files byte-identical); `prepare` on one real Hub
  file per category (raw, standardized, extent identical; the extent matches the published file list);
  `publish` push/pull with the Hub faked. Kept on purpose: `Category.name`, `push`'s `if old:`, resumable raw/checked
  files, the repair `except` set, the sample-cut loop.
- CorpusTokens: the data crate's FIM line widening removed (the only caller passed `line=0.0`, the corpus decides "no
  line widening", §6; `random_bool(0.0)` was each piece's last draw, so no ChaCha8 draw moves), a dict literal,
  stale comments. Verified byte-identical: compile (val of 6 sources, FIM train shards with both grammars, the
  repositories corpora's val), packs at both contexts, `PretrainingBatches` (both readings, nonzero starts), `weights`,
  `mixture_rows`, the crate's `train` (both scores) and `repository_samples`: 141 files, 6.15 GB.
- Committed and pushed from the GH200: `4ade225`.
- Verification hazard (fixed): ref and new shared `artifacts/cache/native`, and cargo's single fingerprint let a new
  build reuse the reference library; the poisoned data-crate `.so` was deleted, comparisons redone with private caches.

## 3. Throughput (EfficiencyResearch)

| # | Finding | Action | State |
|---|---|---|---|
| E1 | GPU power: requested 900 W, enforced 700 W by the hypervisor (KVM guest); during the d24 8K step the SW power cap is active in every sample, SM clock avg 1,564 MHz of 1,980 | ask Lambda to raise the VM's GPU power limit to 900 W | open (provider) |
| E2 | Locking the SM clock at 1,980 MHz: d24 8K step −3.0% (3,585-3,590 vs 3,682-3,696 ms), d16 −1.7%; speed only | locked machine-wide; `sudo nvidia-smi -lgc 1980,1980` added to `runs/setup.sh` (not persistent across reboot) | done |
| E3 | Profile, kernel breakdown, ranked CUDA/C plan (`local://EfficiencyResearch.md`) | items 1-2 applied (`a81410c`); 4-8, the FP8 proof of concept and real-data rechecks running | partly done |

## 4. Infrastructure

- **I1** ✓ (`ba01bb5`): `tokenizers==0.23.3` (0.23.2 conflicts with huggingface-hub 2.2.0; fidelity identical) and
  `regex==2026.9.29` declared and locked; the image is rebuilt with the Hydra migration's `hydra-core`. `basedpyright`
  stays (eval/verify.py runs it).
- **I2** `runs/setup.sh`: clock lock (E2) added; long line reflowed.

## 5. Documentation (after the code is final)

Consolidate the four docs (`docs/h100-pretraining.md`, `nanochat/corpus/normalization_doc.md`, `ablations/tokenizer.md`,
`nanochat/corpus/gh200.md`) from the final code: cutover done, `data` step, DeepSeek-Coder terminology. Corrections from
ArchitectureCheck (no code effect):
- wrong locations: MobileLLM 32×960 is Table 9; nanochat Muon 0.02 at 92 `gpt.py:419/450`, √ factor at `optim.py:393`;
  zero-init lm_head is modded-nanogpt record #8; Power Lines constants in App. E.3.1; DataDecide's ~80% in the abstract
  and §3.1; `production_plan` → `base_plan` (now `nanochat/run.py`), `production` fields → `baseline`;
- wrong values: MoVE d24 extrapolation ≈ −0.014 in MoVE's coordinates (the cited gains are LaVE rows); deep-and-thin
  d24 analogue +9.2% FLOPs; LONG threshold 190.5; Step-Law's fit uses 1,912 samples from 7 dense sizes; "within 7% of
  nanochat's lr" fails like-for-like (1.24× and 1.69×); StarCoder2's FIM ≈25% of chunks, DeepSeek-Coder PSM only;
  Power coefficient 0.0122; minor KV-cache and fp32-angle bounds;
- unsupported: ResFormer citation for per-layer value embeddings (code fixed in `ba01bb5`; the docs' text remains);
  2410.23922 read as covering Muon; 2405.18392 cited for crossing curves; d8@20 ×2 lr results without a record in
  `results/` (unrecoverable: delete or mark unrecorded); `fit` called a polynomial fit per 2409.19913 (it is a 3-point
  local parabola).
- record: `results/lr.json`'s `production_*` keys renamed `base_*` ✓ (`ba01bb5`).
- also: D10's wording (1.15e-5 of peak is intended), the warmup text (40 steps per nanochat 92; 2410.23922 is not about
  Muon), FIM citations (§8), stale line-widening, manifest and ablation-driver text, `mixture.py:67`'s plan comment (D1).

## 6. Dataset `iaroy/nanochat-data`

**Resolved (user decision): no manifest.** `manifest.json` is not needed for the research and is gone: `prepare`
prepares every source (fetching keeps raw files), `publish` writes shards and card only, `pull` fetches shards. The Hub
dataset holds the 1,812 shards, the regenerated card and `.gitattributes` (commit `b091cb47`).
- Trigger for a data change: a §1 decision that raises the base run's token budget (`mixture.STEPS`) or a mixture share
  (D8) beyond the extents' margins → extend on the GB10, prepare, curate, `publish push`.

## 7. Tokenizer ablation (`data` running; runs wait for the user's approval)

`python -m ablations.tokenizer data` (the d16 set compiled and packed under every arm, `results/tokenizer-data.json`),
then `python -m nanochat.run experiment=tokenizer` (8 d16 runs: 4 arms × seeds 0 and 1; the decision in
`results/tokenizer.json`), then `adopt`, compile, pack, pre-training (`python -m nanochat.run experiment=base`).

## 8. Answers and ablation designs

One entry per open item of §1-§6: the question, then either the literature answer (paper id, exact section, table or
equation, the decisive sentence) or an ablation design (hypothesis, arms, fixed configuration, scale, seeds, measures,
pre-registered decision rule, GH200 cost). Filled in as the research agents report.


### D1 FLOP count M (RegimeAnswers): answered; adopted
- DeepSeek LLM's constants assume Eq. 2's M at full attention over l_seq (arXiv 2401.02954 §3.2: "M includes the
  computational overhead of attention operation … C=MD", Eq. 2 "M = 72 n_layer d_model² + 12 n_layer d_model
  l_seq"; §3.2 Table 3: every model at l_seq 4,096, e.g. "8 512 102400 4096 25.2M 77.6M 352M" = 72·8·512² +
  12·8·512·4096). The code's per-window M was nanochat 92's convention, ×1.82 below Eq. 2 at d24/8K.
- Applied (`gpt.non_embedding_flops_per_token`, l_seq the model's context). Plans: d16 at 20: 2 → 3 micro-batches,
  6,828 → 4,552 steps, law lr ×0.914; d24 base: 12 → 15 micro-batches, 86,975 → 69,580 steps on the same 102.6B
  tokens. The lr ablation's three settings keep their batch and steps, so its optima carry over: `lr.analysis`
  re-bases the vertices (×1.1186 at d8, ×1.0776 at d24), base multiplier ×2.7972, base lr 2.4138e-3 (verified on the
  GH200). No batch ablation: Muon's loss is flat across a 2× batch at 8× Chinchilla (arXiv 2509.02046 App. C.9
  Table 85).

### D2 repository-stage lr (RegimeAnswers): answered; keep decay to zero
Straight to Zero (arXiv 2502.15938) favours decaying to zero; Llama 3, Kimi K2 and Moonlight decay through their final
or long-context stage. DeepSeek-Coder §3.6 and DeepSeek-V3 §4.3 hold their final lr; Qwen2.5-Coder gives no schedule.
Citation fixed in `schedule.py`.

### D3 gradient clip before Muon (OptimizerAnswers): answered; keep 1.0
The Megatron PR 5395 stall needs gradient norms ~5e7-2e11 (below Newton-Schulz's 1e-7 floor); the largest recorded
here is 10.4. DeepSpeed #8439's bug clips after orthogonalization, which this code does not. Kimi K2's MuonClip clips
q/k weights, not gradients. The only source for 1.0 is DeepSeek LLM §2.3.

### D4 learnable QK gains (ArchitectureAnswers): answered; keep
nGPT (arXiv 2410.01131 App. A.9 Table 5): fixing the normalized q/k scale costs 0.11% of loss; transformers' Qwen3,
Gemma 3 and OLMo 2 learn it; per-head QK norm with learnable parameters in arXiv 2309.14322 §3.2.5. Resolving the
~0.0006 bpb effect would take ~247 seeds per arm: no ablation.

### D5 output-logit softcap (ArchitectureAnswers): ablation, before the tokenizer runs
- The old reason was wrong: QK norm "does not solve the problem of logit shift in the final softmax" (Chameleon,
  arXiv 2405.09818 §2.3). z-loss not needed: weight decay mitigates output-logit divergence (arXiv 2309.14322 §3.1.2),
  `lm_head` is decayed, and no lr-sweep run diverged up to ×8. Softcap evidence is small and single-run
  (modded-nanogpt record #9 −0.0053 nats at cap 30; nanochat LOG 2026-03-02: caps 10-30 within 1e-3).
- Design: d16 at 20, cap 15 vs none, 4 paired seeds; adopt iff code bpb better by > 1.96·0.0034·√(2/4) = 0.0047 and
  MBPP+ not worse (sign test); ~9 GPU-h. `RunConfig.softcap` (`GPTConfig.softcap`: c·tanh(z/c) in the training loss,
  eval losses, decoder sampling), `conf/experiment/softcap.yaml`, analysis in `ablations/recipe.py`. Runs on the
  sa-bpe arm's data (its d16 max_ratio is 31.42) at lr_scale 2.7972 (lr.py's base multiplier under D1), since the
  corpus tokenizer does not exist before the tokenizer ablation's adopt.

### D6 one lr/weight decay for all groups (OptimizerAnswers): source found; ablation after the tokenizer runs
Moonlight §2.2 shares lr and weight decay "among matrix and non-matrix parameters" (its matched-RMS argument covers
Muon's update size, not the AdamW groups' lr). Design: `RunConfig.adamw_lr_scale` (AdamW lr × s, weight decay / s;
`conf/experiment/adamw_lr.yaml`) ×¼ and ×4 at d16, 2 seeds, decay
timescale of the tables held; adopt iff better by > 0.0067 bpb and MBPP+ not worse; ~4 GPU-h.

### D7 τ weight-decay law on Muon (OptimizerAnswers): ablation, before the tokenizer runs
The law's form carries to Muon (Power Lines App. B; arXiv 2512.05620 App. H.5 reproduces it for Muon), its constants
do not (no Muon fit; doubling Muon's decay cost 0.010-0.019 loss in arXiv 2509.02046). Design:
`RunConfig.muon_tau_scale` (Muon weight decay / s; `conf/experiment/muon_tau_20.yaml` on the sa-bpe arm's data, then
`muon_tau_200.yaml` on the corpus's packs, as d8 at 200 reads 2.94B tokens, beyond the arm's set) ×½ and ×2 on Muon
matrices only, d8 at 20 (4 seeds) and at 200 (2 seeds); keep the law
unless an arm clears its margin, else fit Muon's own law through the two horizons (`ablations/recipe.py`
`muon_tau_law`); ~3.2 GPU-h.

### D9 RoPE base (ArchitectureAnswers): answered; keep 10^6 from the start
Barbero et al. (arXiv 2410.06205 Table 2) Gemma 2B at 8K, 10k vs 500k: differences under 0.3%, mixed sign; Llama 3,
OLMo 2 and Gemma 3's global layers pretrain at a large base; 10^6 is above 2405.14591's 32K bound (6.4e5).

### D10, D11 (RegimeAnswers): answered
D10: exactly 0 is not required (1.15e-5 of peak is 87× below 2502.15938's smallest nonzero minimum). D11: `round`
applied (`int` dropped a token in 119,167 of 2M micro-batch counts at d24).

### Value embeddings (ArchitectureAnswers): answered; keep, citation fixed
Per-layer token-indexed tables are modded-nanogpt record #14's, MoVE's LaVE (arXiv 2601.22887 §4.1 Eq. 6), not
ResFormer (which reuses the first layer's values); LaVE gains 0.016/0.010/0.010 at D12/D20/D32 (Table 1).

### A1, viable_end, I1, E1, lr.json (EngineeringAnswers): engineering; applied except E1
- A1 ✓: `SyntaxConstraint(tokenizer, ...)`, tables per tokenizer (a mismatch masked the true token in 1,475 of
  3,199 steps). viable_end ✓: `None => false` (no mask changes over 8,002,126 prefixes). I1 ✓: `tokenizers==0.23.3`
  (0.23.2 conflicts with huggingface-hub 2.2.0), `regex==2026.9.29`; fidelity identical under 0.23.3. lr.json ✓: keys
  renamed `production_*` → `base_*`.
- E1: ask Lambda to raise the host/BMC GPU power limit to 900 W (the guest's 900 W request is enforced at 700 W; the
  step's SW power cap is active in every sample, SM clock avg 1,492-1,564 MHz).

### Throughput (EfficiencyResearch): items 1-2 applied; 4-8 in progress
1. ✓ FA4 on the real `cu_seqlens` only (the loader padded to 2,796 segments for ~94 real); losses bitwise, gradients
   within the reference's run-to-run spread. 2. ✓ Muon Newton-Schulz batched by shape (momentum buffers bitwise,
   parameters at bf16 GEMM rounding). Together, clock locked: d16 8K 266 → 239.5 ms (−10%), d24 8K 3,709 → 3,511 ms
   (−5.3%); rechecked on real rows, real gradients and the real benchmark tasks (loss and bpb bitwise, Muon buffers
   bitwise). 3. FP8 MLP GEMMs (e4m3 rowwise, d16, 300 steps on real rows): mean loss over steps 250-299 +0.0048 over
   bf16 against a bf16 two-run spread of 0.0014, growing: rejected. Two concurrent d16 runs: −8% aggregate
   (+1.6% under MPS): not used, one run saturates the GPU. 4-8 (fused kernels: value-embedding gate, FA4 backward
   pre/post-processing, q/k norm + RoPE into the projection GEMM, RMSNorm into the next GEMM, weight-gradient
   accumulation in the GEMM) in progress; the value-embedding gate kernel is not bitwise (its fp32 reduction order
   moves ~15% of elements by one bf16 ulp), accepted on the §0 conditions. The GEMMs already run at the 700 W cap's
   ceiling (703 TFLOP/s of 989 peak).

### Data provenance (verified Oct 24): every run reads iaroy/nanochat-data
- `artifacts/corpus` holds only `curated/` (`publish pull`) and what compile/pack derive from it. All 1,812 curated
  shards (160,061,439,381 bytes) are SHA-256-identical to the Hub's revision `b091cb47`; none missing, none extra.
  The code reads no other data (loader: packed shards; bpb: compiled val splits; eval: the EvalPlus/HumanEval-X
  benchmarks); `register_fake` are custom-op shape functions, the one RNG is the seeded data-order permutation.
- Verification and measurement use real inputs only: packed corpus rows, activations and gradients from the model on
  them, the real benchmark tasks (agents' earlier random-tensor and synthetic-task checks are being redone).
- The dataset's model-generated sources (UltraData-Code L3, Nemotron Synthetic-Code, UltraData-Math L3,
  codeforces_cots) stay in the mixture as designed (user, Oct 24).

### D8 synthetic-code share (DataAnswers): closed, keep 0.30 (user)
- No paper fixes the share for this mixture. Kang et al. (arXiv 2510.01631) measure rephrased web text, and its
  Limitations say the findings "may not directly apply to … synthetic code". Code papers only bound it: To Code
  (arXiv 2408.10914 §3.4) 10% synthetic helps from scratch at 470M; UltraData-Code (arXiv 2610.23960 Table 6) 1:1 real
  and synthetic helps in continued pretraining; CodeAlchemy (arXiv 2606.10087 §3, RQ1/RQ3) rewritten-only code drops
  MBPP to 5.6 against 58.7 for raw, a 50-50 mix keeps 60.3. Positive results span about 4-50% of code.
- Extents: the curated corpus holds a share up to ≈0.329 (≈0.305 under sa-bpe-fused); lower shares need only the
  constant (repository code holds 1.44× the draw at 0%); above 0.329 the corpus must be extended on the GB10.
- Closed (user, Oct 24): LLM-generated code is valid, acceptable data; the share stays 0.30 of code, with no ablation
  and no further literature study; `mixture.py` marks Kang et al.'s ~30% (rephrased web text) as [inference] for code.

### FIM configuration (DataAnswers): answered by the literature; keep rate 0.5 per piece, PSM/SPM 50/50, AST-FIM
- SPM share: Bavarian et al. (arXiv 2207.14255 §4.3, Table 1) is the one controlled PSM/SPM/joint comparison: at the
  same 0.9 rate joint training matches or beats both single modes in every cell but one near-tie; a pure-SPM model
  cannot infill in PSM (0.023), pure PSM loses SPM random-span infilling (0.274); §8.1 recommends joint training; App. D
  "variant 2" is the layout `pack.rs` writes. AST-FIM (arXiv 2506.00204 §5.1) uses 50/50 without testing it.
- Rate: Bavarian §4.1-4.2, no left-to-right cost up to 0.9 from scratch at 50M-6.9B; DeepSeek-Coder (arXiv 2401.14196
  §3.1.2) chose 0.5, 100% gave the weakest completion; AST-FIM Table 1, at 0.7 AST middles kept HumanEval+/MBPP+ at the
  no-FIM baseline. Ours is 29.7% of file-stage tokens (StarCoder2 ≈25%, Code Llama 90%).
- Fixes: citations (`pack.py` 50/50 → Bavarian §3.1, §4.3, §8.1 ✓ in the working tree; normalization_doc §6: App. D supports the layout,
  not the probability; h100 doc: StarCoder2 ≈25%, DeepSeek-Coder PSM only, its line-widening text is stale). Not
  covered by any paper: FIM over whole multi-file repository texts in the 32K stage; the ablation runs' evaluation
  measures no infilling.

### Run configuration: Hydra (hydra-core 1.3.7)
- Replaces `ablations/harness.py` (`Run`, `python -m ablations.harness '<json>'`), `scripts/pretrain.py`,
  `config.run_tag`'s hashed tags, `suite.export`, the `variant` record field, lr.py's driver and the tokenizer
  ablation's `runs` verb. `conf/run.yaml` holds `nanochat/run.py`'s `RunConfig` defaults, the base model's recipe
  (depth 24, ratio null = `train/stages.py` max_ratio, seed 0, tokenizer corpus, scales 1.0, softcap null);
  `conf/experiment/<name>.yaml` sweeps fields against it (base, tokenizer, softcap, adamw_lr, muon_tau_20/200).
- A run's key is the fields differing from the defaults, sorted, `k=v` joined by `,` (e.g.
  `depth=16,seed=1,tokenizer=sa-bpe`; `defaults` if none); a run shared by two experiments is one run. Everything of
  it is in `artifacts/runs/<key>/`: `.hydra/`, `metrics.jsonl` (its run record holds `config` and `data_key`),
  `step<N>.pt`, `eval-step<N>.json`, `run.log`. A recorded run whose plan or data key differs raises "recipe changed";
  a finished or diverged run is skipped; training and each checkpoint's evaluation run in fresh spawned processes;
  any other failure stops the sweep. `ablations/records.py`'s callback writes `results/<experiment>.jsonl`,
  `-curves.jsonl` and, with an analysis, `results/<experiment>.json` = {"analysis": ...}.
- Commands: `runs/container.sh python -m nanochat.run experiment=<name>`; ad hoc
  `python -m nanochat.run depth=8 ratio=0.5 tokenizer=sa-bpe`.
