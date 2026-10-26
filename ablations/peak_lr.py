"""
Peak-lr calibration (d24): the base model's multiplier of DeepSeek LLM's law eta = 0.3118 C^-0.125 (arXiv 2401.02954
§3.1 Eq. 1), the optimum of d24 at SHORT tokens per parameter; the law, with its batch, carries the compute dependence
to the base plan. The runs: conf/experiment/peak_lr_d24.yaml.
"""
import json
import math
from dataclasses import asdict

import numpy as np

import metrics
from corpus.mixture import FILE_LEVEL_STAGE, REPO_LEVEL_STAGE
from corpus.tokenizer import get_tokenizer
from paths import RESULTS
from run import RunConfig, plan_of, tokenizer_path
from train.stages import stage_mixture

EXPERIMENT = "peak_lr_d24"
# short horizon in tokens per parameter: the longest d24 run a sweep affords
SHORT = 5


def records() -> list[dict]:
    """The experiment's records, results/<EXPERIMENT>.jsonl (ablations/records.py)."""
    return [json.loads(line) for line in (RESULTS / f"{EXPERIMENT}.jsonl").read_text(encoding="utf-8").splitlines()]


def check(line: dict) -> None:
    """Raises unless the run trained the current recipe: its recorded plan, tokenizer data key and each stage's
    mixture are those the current code computes for its RunConfig."""
    config = RunConfig(**line["config"])
    run = plan_of(config)
    tokenizer = get_tokenizer(tokenizer_path(config))
    mixtures = {stage.seq_len: stage_mixture(tokenizer, stage, run)[0]
                for stage in (FILE_LEVEL_STAGE, REPO_LEVEL_STAGE)}
    for r in line["records"]:
        changed = r["kind"] == "run" and (r["plan"] != asdict(run) or r["data_key"] != tokenizer.data_key)
        if changed or (r["kind"] in ("run", "stage") and r["stage"]["mixture"] != mixtures[r["stage"]["seq_len"]]):
            raise RuntimeError(f"{line['run_name']}: not trained on the current recipe; re-run its experiment")


def fit(lines: list[dict]) -> dict:
    """The optimum: the vertex of the parabola in u = ln(multiplier) through the lowest-code run and its neighbours
    (as arXiv 2409.19913 §3.1), or that run itself if its upper neighbour diverged; a diverged run counts as worse than
    any evaluated one. Raises where the rule extends the grid x2: the lowest-code run at a grid end."""
    runs = sorted(lines, key=lambda line: line["config"]["learning_rate_scale"])
    u = [math.log(line["config"]["learning_rate_scale"]) for line in runs]
    code = [math.inf if line["measures"] is None else line["measures"]["code"] for line in runs]
    best = int(np.argmin(code))
    if best in (0, len(runs) - 1) or not math.isfinite(code[best - 1]):
        raise RuntimeError("the lowest-code run is at a grid end or its lower neighbour diverged: extend the grid")
    if not math.isfinite(code[best + 1]):
        return {"curvature": None, "vertex": math.exp(u[best]), "code": code}
    curvature, vertex = metrics.parabola_vertex(u[best - 1:best + 2], code[best - 1:best + 2])
    return {"curvature": curvature, "vertex": math.exp(vertex), "code": code}


def analysis(lines: list[dict]) -> dict:
    """The estimator over the experiment's records, each checked against the current recipe (`check`)."""
    for line in lines:
        check(line)
    d24 = fit(lines)
    return d24 | {"base_multiplier": d24["vertex"], "base_lr": d24["vertex"] * plan_of(RunConfig()).learning_rate}
