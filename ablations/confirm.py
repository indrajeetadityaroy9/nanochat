"""
Confirmation: every change ablations/{lr,mlp,weight_decay}.py adopted (RESULTS/<experiment>.json), together, against
production one scale up (d16 at Chinchilla's tokens per parameter). Production takes the bundle only if it is better
here by the same rule.

    python -m ablations.confirm    (ablations/harness.py; ablations.md)
"""
import json

from ablations.harness import CONFIRM, CONFIRM_SEEDS, main
from nanochat.paths import RESULTS

if __name__ == "__main__":
    bundle = {field: value for experiment in ("lr", "mlp", "weight_decay")
              for field, value in json.loads((RESULTS / f"{experiment}.json").read_text())["adopted"].items()}
    if not bundle:
        raise SystemExit("no experiment adopted a change: nothing to confirm")
    main(alternatives={"bundle": bundle}, setting=CONFIRM, seeds=CONFIRM_SEEDS)
