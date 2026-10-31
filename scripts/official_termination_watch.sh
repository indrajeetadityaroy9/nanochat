#!/bin/bash
# Runs scripts/official_termination.sh on every full-state checkpoint of a run (docs/post-training.md §5.1), in step
# order, one at a time, each once its file has stopped growing (a minute apart). While the run trains, only the steps
# in DURING (space-separated; default every step), each once the run has logged SETTLE (default 30) train steps past
# it; once the run has ended (its `end` record, or the docker container CONTAINER, if given, gone), every checkpoint
# left. One line per probe in LOG (default RUNS/<run>/termination.log); a probe skipped for GPU memory (exit 2) or
# failed is logged and not retried. Exits once the run has ended and every checkpoint is probed.
#
#   [DURING="<steps>"] [CONTAINER=<name>] scripts/official_termination_watch.sh <run name>
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN=$1 DIR=$ROOT/artifacts/runs/$1
LOG=${LOG:-$DIR/termination.log} SETTLE=${SETTLE:-30} DURING=${DURING:-} CONTAINER=${CONTAINER:-}

state() {  # the last train step, then "ended" if the run has written its end record
  python3 -c 'import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(max(r["step"] for r in rows if r.get("kind") == "train"), "ended" if any(r.get("kind") == "end" for r in rows)
      else "training")' "$DIR/metrics.jsonl"
}

summary() {
  python3 -c 'import json, sys
d = json.load(open(sys.argv[1])); s = d["sets"]; r = d.get("job_tokens_per_second") or {}
def one(k):
    x = s[k]; o = x["outcomes"]; c = x["closed_within"]
    return ("{}: {} closed-ended/closed-limit/unclosed-limit/loop-stopped/unclosed-ended {}/{}/{}/{}/{};"
            " closed by 1K..32K {}").format(
        k, x["responses"], o["closed_ended"], o["closed_limit"], o["unclosed_limit"],
        o.get("unclosed_loop_stopped", 0), o["unclosed_ended"],
        "/".join("{:.3f}".format(v) for v in c.values())) + (
        "; EvalPlus base/plus {:.3f}/{:.3f}".format(x["evalplus_pass"]["base"], x["evalplus_pass"]["plus"])
        if "evalplus_pass" in x else "")
lost = r.get("seconds_lost")
print(" | ".join(one(k) for k in ("evalplus", "val", "all")) + " | {} s; run tok/s {} -> {}, {} s lost".format(
    d["seconds"], r.get("before_median") and round(r["before_median"]), r.get("during_median") and
    round(r["during_median"]), None if lost is None else round(lost)))' "$1"
}

settled() { [ -f "$DIR/termination-$1.json" ] || [ -f "$DIR/termination-$1.skipped" ] && return 1; return 0; }

while true; do
  read -r last status < <(state)
  [ -n "$CONTAINER" ] && [ -z "$(docker ps -q -f name="^$CONTAINER\$")" ] && status=ended
  for checkpoint in $(ls "$DIR"/checkpoint-*.pt 2>/dev/null | sort -V); do
    step=${checkpoint##*-}; step=${step%.pt}
    settled "$step" || continue
    [ "$status" = ended ] || [ -z "$DURING" ] || [[ " $DURING " == *" $step "* ]] || continue
    size=$(stat -c %s "$checkpoint"); sleep 60
    [ "$size" = "$(stat -c %s "$checkpoint")" ] || continue
    [ "$status" = ended ] || [ "$last" -ge $((step + SETTLE)) ] || continue
    "$ROOT/scripts/official_termination.sh" "$RUN" "$step" > "$DIR/termination-$step.log" 2>&1
    code=$?
    if [ $code -eq 0 ]; then
      echo "$(date -u +%FT%TZ) step $step: $(summary "$DIR/termination-$step.json")" >> "$LOG"
    else
      echo "$(date -u +%FT%TZ) step $step: probe $([ $code -eq 2 ] && echo skipped || echo failed) (exit $code;" \
           "$DIR/termination-$step.log)" >> "$LOG"
      touch "$DIR/termination-$step.skipped"
    fi
  done
  if [ "$status" = ended ]; then
    pending=0
    for checkpoint in "$DIR"/checkpoint-*.pt; do
      step=${checkpoint##*-}; settled "${step%.pt}" && pending=1
    done
    [ $pending -eq 0 ] && { echo "$(date -u +%FT%TZ) the run has ended; every checkpoint probed" >> "$LOG"; exit 0; }
  fi
  sleep 60
done
