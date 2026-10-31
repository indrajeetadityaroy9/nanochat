#!/bin/bash
# The GPU queue after an SFT run, back to back with no idle gap (docs/post-training.md §8). While the run trains it
# waits, and stops the run's termination watcher (scripts/official_termination_watch.sh) once the watcher has settled
# step WATCH_LAST (default 4000; its termination-<step>.json or .skipped), so that the watcher, which probes every
# checkpoint left once the run ends, probes none of the queue's. Once the run has written its end record
# (metrics.jsonl kind "end") and its container CONTAINER (default sft-run) is gone, a probe already started finishes
# (the probes' lock) and the queue runs, on the free GPU, the critical path first:
#   1. the natural-termination probe of the final step (scripts/official_termination.sh, chunked) at
#      VLLM_GPU_MEMORY_UTILIZATION PROBE_UTIL, the DPO stage's before-measurement;
#   2. the on-policy stage (scripts/onpolicy.sh): rollouts, then RSFT and DPO, then the DPO model's evaluation (the
#      same probe on the same prompts and seeds, KodCode val pass@1);
#   3. the reasoning protocol (scripts/official_reasoning.sh, served at SERVE_UTIL) of the final step, then of the
#      DPO model (MODEL/TAG dpo: official-<step>-dpo/);
#   4. the export's and the RSFT model's evaluations (onpolicy.sh);
# and then the deferred items: the probes of each step in LATER_PROBES (default 3000 5000 6000 7000) without its
# termination-<step>.json (a .skipped marker is retried); the per-source held-out loss (eval/sft_val.py) of each step
# in VAL_STEPS (default the final, 4000, 6000) without its sft-val-<step>.json; the final step's DRY comparison
# (scripts/official_termination.sh --dry) unless the DPO model loops in fewer than DRY_BELOW (default 5%) of its probe's
# responses. Every item whose output exists is skipped; a failed item is logged and the queue goes on. The body is one
# function, parsed before it runs, so a checkout updated while the queue waits does not change the running script.
# DRY_RUN=1 logs each command instead of running it (the waiting, watcher and skip logic run as they would).
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
  local WATCH_LAST=${WATCH_LAST:-4000} LATER_PROBES=${LATER_PROBES:-"3000 5000 6000 7000"}
  local VAL_STEPS=${VAL_STEPS:-"$FINAL 4000 6000"} DRY_BELOW=${DRY_BELOW:-0.05}
  local PROBE_UTIL=${PROBE_UTIL:-0.85} SERVE_UTIL=${SERVE_UTIL:-0.85}
  local LOCK=${LOCK:-$ROOT/artifacts/cache/termination.lock}  # scripts/official_termination.sh's
  local ONPOLICY=$DIR/onpolicy-$FINAL

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
  probe() {  # <step>: the termination probe unless done
    saved "$1" || { log "probe $1: no checkpoint"; return 0; }
    flock "$LOCK" true  # a probe of this step started elsewhere has finished
    [ -f "$DIR/termination-$1.json" ] && { log "probe $1: done before"; return 0; }
    [ -n "${DRY_RUN:-}" ] || rm -f "$DIR/termination-$1.skipped"
    item "probe $1" "$DIR/termination-$1.log" \
      env VLLM_GPU_MEMORY_UTILIZATION="$PROBE_UTIL" "$ROOT/scripts/official_termination.sh" "$RUN" "$1"
  }
  onpolicy() {  # <stages> <eval models>
    item "onpolicy $1${2:+ ($2)}" "$DIR/onpolicy-$FINAL.log" env STAGES="$1" EVAL_MODELS="$2" \
      VLLM_GPU_MEMORY_UTILIZATION="$PROBE_UTIL" "$ROOT/scripts/onpolicy.sh" "$RUN" "$FINAL"
  }
  official() {  # <tag or empty> [<model dir>]
    local out=$DIR/official-$FINAL${1:+-$1}
    [ -f "$out/summary.json" ] && { log "official ${1:-sft} $FINAL: done before"; return 0; }
    item "official ${1:-sft} $FINAL" "$out.log" env VLLM_GPU_MEMORY_UTILIZATION="$SERVE_UTIL" \
      ${2:+MODEL="$2"} ${1:+TAG="$1"} "$ROOT/scripts/official_reasoning.sh" "$RUN" "$FINAL"
  }
  loop_share() {  # the share of a probe's responses that loop (never closed, repetition >= 0.99)
    python3 -c 'import json, sys
rows = [r for rs in json.load(open(sys.argv[1]))["responses"].values() for r in rs]
print(sum((r.get("repetition") or 0) >= 0.99 for r in rows) / len(rows))' "$1"
  }

  [ -f "$DIR/metrics.jsonl" ] || { echo "no metrics.jsonl in $DIR" >&2; return 1; }
  log "queue started: run $RUN, final step $FINAL${DRY_RUN:+ (dry run)}"
  local stopped=
  until ended; do
    local settled=$DIR/termination-$WATCH_LAST
    if [ -z "$stopped" ] && { [ -f "$settled.json" ] || [ -f "$settled.skipped" ]; }; then
      stop_watcher
      stopped=1
    fi
    sleep "$POLL"
  done
  log "the run has ended"
  stop_watcher
  mkdir -p "$(dirname "$LOCK")"
  flock "$LOCK" true

  probe "$FINAL"
  onpolicy "rollouts train" ""
  onpolicy evaluate dpo
  official ""
  if [ -f "$ONPOLICY/dpo/model.safetensors" ] || [ -n "${DRY_RUN:-}" ]; then official dpo "$ONPOLICY/dpo"; fi
  onpolicy evaluate "model rsft"

  local s
  for s in $LATER_PROBES; do probe "$s"; done
  for s in $VAL_STEPS; do
    saved "$s" || { log "sft-val $s: no checkpoint"; continue; }
    [ -f "$DIR/sft-val-$s.json" ] && { log "sft-val $s: done before"; continue; }
    item "sft-val $s" "$DIR/sft-val-$s.log" "$ROOT/scripts/container.sh" python -m eval.sft_val "$RUN" "$s"
  done
  local dpo_probe=$ONPOLICY/eval-dpo/termination.json share
  if [ -f "$DIR/termination-dry-$FINAL.json" ]; then
    log "dry $FINAL: done before"
  elif [ -f "$dpo_probe" ] && share=$(loop_share "$dpo_probe") &&
       python3 -c 'import sys; sys.exit(float(sys.argv[1]) >= float(sys.argv[2]))' "$share" "$DRY_BELOW"; then
    log "dry $FINAL: skipped, the DPO model loops in $share of its probe's responses"
  else
    item "dry $FINAL" "$DIR/termination-dry-$FINAL.log" \
      env VLLM_GPU_MEMORY_UTILIZATION="$PROBE_UTIL" "$ROOT/scripts/official_termination.sh" --dry "$RUN" "$FINAL"
  fi
  log "queue done"
}

main "$@"
