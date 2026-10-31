#!/bin/bash
# The GPU queue after an SFT run (docs/post-training.md §8.7). While the run trains it waits; once the run has written
# its end record (metrics.jsonl kind "end") and its container CONTAINER (default sft-run) is gone, it stops the run's
# termination watchers (scripts/official_termination_watch.sh, if any), lets a probe already started finish (the
# probes' lock), and runs two branches at once on the free GPU, their vLLM shares summing to 0.9 of its memory:
#   A. (SERVE_UTIL, default 0.3) the reasoning protocol of the final step (scripts/official_reasoning.sh): first its
#      natural-termination probe, the DPO stage's before-measurement, then the served benchmarks; then the export's
#      KodCode val pass@1 (onpolicy.sh evaluate model, the probe copied);
#   B. (ONPOLICY_UTIL, default 0.6) the on-policy stage (scripts/onpolicy.sh): rollouts, verdicts and selection, then
#      RSFT and DPO (beside A's server), then the DPO model's evaluation (the same probe on the same prompts and
#      seeds, KodCode val pass@1), then the reasoning protocol of the DPO model (MODEL/TAG dpo: official-<step>-dpo/,
#      on port DPO_PORT, default 8001, beside A's on 8000).
# The shares: KV cache per token 2 x 24 layers x 6 heads x 128 x 2 bytes = 72 KiB, so of 94.5 GiB, 0.3 holds 20
# sequences (EvalPlus's n) of about 18K tokens after the weights, and 0.6 about 47 rollouts at their 16,384-token cap;
# training gets the 0.7 left beside A. Then, alone at PROBE_UTIL (default 0.85): the final step's probe if A did not
# write it; the RSFT model's evaluation only if the DPO model's probe closes fewer responses than the SFT model's;
# the final step's DRY comparison (scripts/official_termination.sh --dry) unless the DPO model loops in fewer than
# DRY_BELOW (default 5%) of its probe's responses. Every item whose output exists is skipped; a failed item is logged
# and its branch goes on. The body is one function, parsed before it runs, so a checkout updated while the queue
# waits does not change the running script.
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
  local DRY_BELOW=${DRY_BELOW:-0.05}
  local PROBE_UTIL=${PROBE_UTIL:-0.85} SERVE_UTIL=${SERVE_UTIL:-0.3} ONPOLICY_UTIL=${ONPOLICY_UTIL:-0.6}
  local DPO_PORT=${DPO_PORT:-8001}
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
  onpolicy() {  # <util> <stages> <eval models>; training sizes its pairs beside A's server (SERVER_UTIL)
    item "onpolicy $2${3:+ ($3)}" "$DIR/onpolicy-$FINAL-${2// /-}${3:+-${3// /-}}.log" env STAGES="$2" \
      EVAL_MODELS="$3" VLLM_GPU_MEMORY_UTILIZATION="$1" SERVER_UTIL="$SERVE_UTIL" "$ROOT/scripts/onpolicy.sh" "$RUN" \
      "$FINAL"
  }
  official() {  # <util> <port> <tag or empty> [<model dir>]
    local out=$DIR/official-$FINAL${3:+-$3}
    [ -f "$out/summary.json" ] && { log "official ${3:-sft} $FINAL: done before"; return 0; }
    item "official ${3:-sft} $FINAL" "$out.log" env VLLM_GPU_MEMORY_UTILIZATION="$1" PORT="$2" \
      ${4:+MODEL="$4"} ${3:+TAG="$3"} "$ROOT/scripts/official_reasoning.sh" "$RUN" "$FINAL"
  }
  share() {  # <probe record> <closed|loop>: the share of its responses that closed, or that loop (never closed,
    # repetition >= 0.99)
    python3 -c 'import json, sys
rows = [r for rs in json.load(open(sys.argv[1]))["responses"].values() for r in rs]
loop = lambda r: not r["closed"] and (r.get("repetition") or 0) >= 0.99
print(sum(r["closed"] if sys.argv[2] == "closed" else loop(r) for r in rows) / len(rows))' "$1" "$2"
  }
  below() { python3 -c 'import sys; sys.exit(float(sys.argv[1]) >= float(sys.argv[2]))' "$1" "$2"; }

  [ -f "$DIR/metrics.jsonl" ] || { echo "no metrics.jsonl in $DIR" >&2; return 1; }
  log "queue started: run $RUN, final step $FINAL${DRY_RUN:+ (dry run)}"
  until ended; do sleep "$POLL"; done
  log "the run has ended"
  stop_watcher
  mkdir -p "$(dirname "$LOCK")"
  flock "$LOCK" true

  { official "$SERVE_UTIL" 8000 ""; onpolicy "$SERVE_UTIL" evaluate model; } &  # A
  local branch_a=$!
  onpolicy "$ONPOLICY_UTIL" "rollouts train" ""  # B
  onpolicy "$ONPOLICY_UTIL" evaluate dpo
  if [ -f "$ONPOLICY/dpo/model.safetensors" ] || [ -n "${DRY_RUN:-}" ]; then
    official "$ONPOLICY_UTIL" "$DPO_PORT" dpo "$ONPOLICY/dpo"
  fi
  wait "$branch_a"
  log "branches A and B done"
  probe "$FINAL"

  local sft_probe=$DIR/termination-$FINAL.json dpo_probe=$ONPOLICY/eval-dpo/termination.json closed loops
  if [ -n "${DRY_RUN:-}" ]; then
    log "dry run: RSFT evaluation if the DPO model's probe closes fewer responses than $sft_probe"
  elif [ -f "$dpo_probe" ] && [ -f "$sft_probe" ] && closed=$(share "$dpo_probe" closed) &&
       below "$closed" "$(share "$sft_probe" closed)"; then
    log "the DPO model closes $closed of its probe's responses, fewer than the SFT model: RSFT evaluation"
    onpolicy "$PROBE_UTIL" evaluate rsft
  fi
  if [ -f "$DIR/termination-dry-$FINAL.json" ]; then
    log "dry $FINAL: done before"
  elif [ -f "$dpo_probe" ] && loops=$(share "$dpo_probe" loop) && below "$loops" "$DRY_BELOW"; then
    log "dry $FINAL: skipped, the DPO model loops in $loops of its probe's responses"
  else
    item "dry $FINAL" "$DIR/termination-dry-$FINAL.log" \
      env VLLM_GPU_MEMORY_UTILIZATION="$PROBE_UTIL" "$ROOT/scripts/official_termination.sh" --dry "$RUN" "$FINAL"
  fi
  log "queue done"
}

main "$@"
