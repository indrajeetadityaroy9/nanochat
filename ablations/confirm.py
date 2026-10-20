"""
Confirmation. Adoption across every experiment's alternatives (RESULTS/<experiment>.json): an alternative is adopted iff
it is better by harness.better with its code p adjusted by Benjamini-Hochberg over the whole family (scipy
false_discovery_control: the expected share of false adoptions among adoptions at most ALPHA); per experiment, of its
adopted alternatives, the one with the lowest code. The adoptions go to RESULTS/adoption.json; together they are the
bundle, run against production one scale up (d16 at Chinchilla's tokens per parameter). Production takes the bundle
only if it is better there by the same rule (its own p: one test).

    python -m ablations.confirm    (ablations/harness.py; ablations.md)
"""
import json

import pandas as pd
from scipy.stats import false_discovery_control

from ablations.harness import CONFIRM, CONFIRM_SEEDS, DECISION, better, main
from nanochat.paths import RESULTS

EXPERIMENTS = ("lr", "clip", "rewarm", "rope", "qk_gains", "residual_scalars", "value_embeds", "windows", "attn_res",
               "key_offset", "mlp")


def adopt(family: dict[tuple[str, str], dict]) -> tuple[dict, dict]:
    """(each alternative's Benjamini-Hochberg-adjusted code p, {experiment: its adopted alternative's name})."""
    adjusted = dict(zip(family, false_discovery_control([a["p"][DECISION] for a in family.values()], method="bh"),
                        strict=True))
    adopted = {}
    for (experiment, name), alternative in family.items():
        table = pd.DataFrame({"difference": alternative["difference"], "p": alternative["p"]})
        if better(table, adjusted[experiment, name]) and (
                experiment not in adopted
                or alternative["difference"][DECISION] < family[experiment, adopted[experiment]]["difference"][DECISION]):
            adopted[experiment] = name
    return adjusted, adopted


if __name__ == "__main__":
    family = {(experiment, name): alternative for experiment in EXPERIMENTS
              for name, alternative in json.loads((RESULTS / f"{experiment}.json").read_text())["alternatives"].items()}
    adjusted, adopted = adopt(family)
    bundle = {field: value for experiment, name in adopted.items()
              for field, value in family[experiment, name]["fields"].items()}
    (RESULTS / "adoption.json").write_text(json.dumps(
        {"adjusted_p": {f"{e}: {n}": p for (e, n), p in adjusted.items()}, "adopted": adopted, "bundle": bundle},
        allow_nan=False), encoding="utf-8")
    if not bundle:
        raise SystemExit("no alternative adopted: nothing to confirm")
    main(alternatives={"bundle": bundle}, setting=CONFIRM, seeds=CONFIRM_SEEDS)
