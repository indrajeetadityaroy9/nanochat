#!/bin/bash
# The GPU queue after an SFT run (docs/post-training.md §8.7). While the run trains it waits; once the run has written
# its end record (metrics.jsonl kind "end") and its container CONTAINER (default sft-run) is gone, it stops the run's
# termination watchers (scripts/official_termination_watch.sh, if any), lets a probe already started finish (the
# probes' lock), and runs, in order:
#   1. alone at PROBE_UTIL (default 0.85) of the GPU's memory: the final step's natural-termination probe on
#      PROBE_PROMPTS (default 128) prompts per set (scripts/official_termination.sh; termination-<step>.json, or
#      termination-<step>-p<PROBE_PROMPTS>.json beside an earlier probe of another size), then the export and its
#      KodCode val evaluation (scripts/onpolicy.sh, STAGES evaluate, EVAL_MODELS model; the probe copied);
#   2. the gate: FTPO and DPO only if that probe's responses loop (never closed, repetition >= 0.99) in at least
#      FTPO_ABOVE (default 0.05) of both sets together; else 3 runs the SFT model's evaluation alone;
#   3. at once: the reasoning protocol (scripts/official_reasoning.sh) of the final step in the background (port 8000,
#      CPUs OFFICIAL_CPUS, SERVE_UTIL, default 0.45), and the on-policy stage (scripts/onpolicy.sh, STAGES "rollouts
#      train dpo evaluate": mining waves, FTPO, DPO waves and DPO, the evaluation of model/, ftpo/ and dpo/) at
#      ONPOLICY_UTIL (default 0.45, which FTPO and DPO training fit in) on CPUs PICKED_CPUS; when the on-policy
#      stage ends, the reasoning protocol of the final model, dpo/ if written, else ftpo/ (MODEL/TAG dpo or ftpo:
#      official-<step>-<tag>/, port PICKED_PORT, default 8001, CPUs PICKED_CPUS, SERVE_UTIL); then both awaited;
#   4. alone at PROBE_UTIL: the final step's DRY comparison (scripts/official_termination.sh --dry) only if the final
#      model (else the SFT model) loops in at least DRY_BELOW (default 0.05) of its probe's responses.
# The shares: KV cache per token 2 x 24 layers x 6 heads x 128 x 2 bytes = 72 KiB, so of 94.5 GiB, 0.45 holds 20
# sequences (EvalPlus's n) of about 28K tokens after the weights; two 0.45 shares run side by side in 3. The CPUs:
# official scoring runs EvalPlus's evaluate with its default cpu_count // 2 workers and lcb_runner with
# --num_process_evaluate 32, so the SFT model's scoring gets its own max(nproc / 2, 32) cores (OFFICIAL_CPUS, the
# first) and the on-policy stage's verification, then the final model's scoring, the rest (PICKED_CPUS), so no timed
# test shares a core with another's. Every item whose output exists is skipped; a failed item is logged and the queue
# goes on. The body is one function, parsed before it runs, so a checkout updated while the queue waits does not
# change the running script.
# DRY_RUN=1 logs each command instead of running it (the waiting, watcher, skip and gate logic run as they would; with
# no probe written the gate is open, and without dpo/ or ftpo/ the final model is dpo/, so every item shows).
#
#   [DRY_RUN=1] scripts/post_sft_queue.sh <run name> [<final step, default 7676>]
#
# Output: one line per event in LOG (default RUNS/<run>/post-sft-queue.log); each item's own outputs and logs.
main() {
  set -uo pipefail
  local ROOT RUN DIR FINAL
  ROOT=$(cd "$(dirname "$0")/.." && pwd)
  RUN=$1 FINAL=${2:-7676}
  DIR=${RUNS:-$ROOT/artifacts/runs}/$RUN
  local LOG=${LOG:-$DIR/post-sft-queue.log} CONTAINER=${CONTAINER:-sft-run} POLL=${POLL:-60}
  local DRY_BELOW=${DRY_BELOW:-0.05} FTPO_ABOVE=${FTPO_ABOVE:-0.05} PROBE_PROMPTS=${PROBE_PROMPTS:-128}
  local PROBE_UTIL=${PROBE_UTIL:-0.85} SERVE_UTIL=${SERVE_UTIL:-0.45} ONPOLICY_UTIL=${ONPOLICY_UTIL:-0.45}
  local PICKED_PORT=${PICKED_PORT:-8001}
  local LOCK=${LOCK:-$ROOT/artifacts/cache/termination.lock}  # scripts/official_termination.sh's
  local ONPOLICY=$DIR/onpolicy-$FINAL cores official_cores
  cores=$(nproc)
  official_cores=$((cores / 2 > 32 ? cores / 2 : 32))
  local OFFICIAL_CPUS=${OFFICIAL_CPUS:-0-$((official_cores - 1))}
  local PICKED_CPUS=${PICKED_CPUS:-$official_cores-$((cores - 1))}

  log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; echo "$(date -u +%FT%TZ) $*"; }
  item() {  # <name> <log file> <command...>: run (or log, DRY_RUN) with its output in <log file>; log the outcome
    local name=$1 out=$2 start=$SECONDS code=0
    shift 2
    if [ -n "${DRY_RUN:-}" ]; then log "dry run: $name: $* > $out"; return 0; fi
    log "start $name"
    "$@" >> "$out" 2>&1 || code=$?
    log "end $name: exit $code, $((SECONDS - start)) s ($out)"
  }
  ended() {  # the run has written its end record and its container is gone
    python3 -c 'import json, sys
sys.exit(0 if any(json.loads(l).get("kind") == "end" for l in open(sys.argv[1]) if l.strip()) else 1)' \
      "$DIR/metrics.jsonl" 2> /dev/null || return 1
    [ -z "$(docker ps -q -f name="^$CONTAINER\$")" ]
  }
  saved() { [ -f "$DIR/checkpoint-$1.pt" ] || [ -f "$DIR/weights-$1.pt" ]; }  # as the scripts it runs find it
  stop_watcher() {  # this run's termination watchers
    local pid
    for pid in $(pgrep -f official_termination_watch.sh); do
      if tr '\0' ' ' < "/proc/$pid/cmdline" 2> /dev/null | grep -qF -- "$RUN"; then
        if [ -n "${DRY_RUN:-}" ]; then log "dry run: stop watcher $pid"
        else kill "$pid" && log "watcher $pid stopped"; fi
      fi
    done
  }
  probe_record() {  # the final step's probe of PROBE_PROMPTS prompts per set, if written
    local f
    for f in "$DIR/termination-$FINAL.json" "$DIR/termination-$FINAL-p$PROBE_PROMPTS.json"; do
      if [ -f "$f" ] && [ "$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["prompts_per_set"])' \
        "$f")" = "$PROBE_PROMPTS" ]; then echo "$f"; return 0; fi
    done
    return 1
  }
  probe() {  # the final step's termination probe of PROBE_PROMPTS prompts per set unless done
    local tag=""
    saved "$FINAL" || { log "probe $FINAL: no checkpoint"; return 0; }
    flock "$LOCK" true  # a probe of this step started elsewhere has finished
    probe_record > /dev/null && { log "probe $FINAL: done before"; return 0; }
    [ -f "$DIR/termination-$FINAL.json" ] && tag=p$PROBE_PROMPTS  # an earlier probe of another size
    [ -n "${DRY_RUN:-}" ] || rm -f "$DIR/termination-$FINAL${tag:+-$tag}.skipped"
    item "probe $FINAL" "$DIR/termination-$FINAL${tag:+-$tag}.log" env VLLM_GPU_MEMORY_UTILIZATION="$PROBE_UTIL" \
      ${tag:+TAG="$tag"} "$ROOT/scripts/official_termination.sh" "$RUN" "$FINAL" "$PROBE_PROMPTS"
  }
  onpolicy() {  # <stages> <eval models> <GPU memory share> [<cpu set>]
    item "onpolicy $1${2:+ ($2)}" "$DIR/onpolicy-$FINAL-${1// /-}${2:+-${2// /-}}.log" env STAGES="$1" \
      EVAL_MODELS="$2" VLLM_GPU_MEMORY_UTILIZATION="$3" PROBE_PROMPTS="$PROBE_PROMPTS" ${4:+CPU_SET="$4"} \
      "$ROOT/scripts/onpolicy.sh" "$RUN" "$FINAL"
  }
  official() {  # <port> <cpu set> <tag or empty> [<model dir>]
    local out=$DIR/official-$FINAL${3:+-$3}
    [ -f "$out/summary.json" ] && { log "official ${3:-sft} $FINAL: done before"; return 0; }
    item "official ${3:-sft} $FINAL" "$out.log" env VLLM_GPU_MEMORY_UTILIZATION="$SERVE_UTIL" PORT="$1" \
      CPU_SET="$2" ${4:+MODEL="$4"} ${3:+TAG="$3"} "$ROOT/scripts/official_reasoning.sh" "$RUN" "$FINAL"
  }
  loop_share() {  # <probe record>: the share of its responses (both sets) that loop (never closed, repetition >= 0.99)
    python3 -c 'import json, sys
rows = [r for rs in json.load(open(sys.argv[1]))["responses"].values() for r in rs]
print(sum(not r["closed"] and (r.get("repetition") or 0) >= 0.99 for r in rows) / len(rows))' "$1"
  }
  below() { python3 -c 'import sys; sys.exit(float(sys.argv[1]) >= float(sys.argv[2]))' "$1" "$2"; }

  [ -f "$DIR/metrics.jsonl" ] || { echo "no metrics.jsonl in $DIR" >&2; return 1; }
  log "queue started: run $RUN, final step $FINAL${DRY_RUN:+ (dry run)}"
  until ended; do sleep "$POLL"; done
  log "the run has ended"
  stop_watcher
  mkdir -p "$(dirname "$LOCK")"
  flock "$LOCK" true

  # 1. the SFT model's probe, export and KodCode val evaluation
  probe
  onpolicy evaluate model "$PROBE_UTIL"

  # 2. the gate
  local sft_probe loops ftpo=1 final="" final_dir=""
  if ! sft_probe=$(probe_record) || ! loops=$(loop_share "$sft_probe"); then
    log "gate: no probe of $PROBE_PROMPTS prompts per set; FTPO and DPO run"
  elif below "$loops" "$FTPO_ABOVE"; then
    log "gate: the SFT model loops in $loops of its probe's responses, below $FTPO_ABOVE; no FTPO or DPO"
    ftpo=""
  else
    log "gate: the SFT model loops in $loops of its probe's responses; FTPO and DPO run"
  fi

  # 3. the SFT model's reasoning protocol beside the on-policy stage, then the final model's
  official 8000 "$OFFICIAL_CPUS" "" &
  local sft_official=$!
  if [ -n "$ftpo" ]; then
    onpolicy "rollouts train dpo evaluate" "" "$ONPOLICY_UTIL" "$PICKED_CPUS"
    for final in dpo ftpo ""; do
      [ -n "$final" ] && [ -f "$ONPOLICY/$final/model.safetensors" ] && break
    done
    [ -z "$final" ] && [ -n "${DRY_RUN:-}" ] && final=dpo
    final_dir=${final:+$ONPOLICY/$final}
    log "final model: ${final:-none}"
    [ -z "$final" ] || official "$PICKED_PORT" "$PICKED_CPUS" "$final" "$final_dir"
  fi
  wait "$sft_official"
  log "official evaluations done"

  # 4. DRY, only if the final model (else the SFT model) still loops
  local final_probe
  if [ -n "$final" ]; then final_probe=$ONPOLICY/eval-$final/termination.json
  else final_probe=$(probe_record) || final_probe=""; fi
  if [ -f "$DIR/termination-dry-$FINAL.json" ]; then
    log "dry $FINAL: done before"
  elif [ -f "$final_probe" ] && loops=$(loop_share "$final_probe") && below "$loops" "$DRY_BELOW"; then
    log "dry $FINAL: skipped, ${final:-the SFT model} loops in $loops of its probe's responses"
  else
    item "dry $FINAL" "$DIR/termination-dry-$FINAL.log" \
      env VLLM_GPU_MEMORY_UTILIZATION="$PROBE_UTIL" "$ROOT/scripts/official_termination.sh" --dry "$RUN" "$FINAL"
  fi
  log "queue done"
}

main "$@"
