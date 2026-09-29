"""
Data-mixture search for code generation over per-corpus component models that continue one shared base model, as
DeMix (Li et al., arXiv 2602.00747; dev/papers/demix_2602.00747v3.md), with the X-MDE regressor features of Belenki et
al. (arXiv 2502.15950; dev/papers/mde_2502.15950v1.md) and the code validation loss of Xing et al. (arXiv 2604.07769;
dev/papers/discore_code_influence_2604.07769v1.md):

- proxy of the model trained on mixture alpha: the merge sum_i alpha_i Theta_i of the component models (DeMix Eq. 6);
- signal: each merged model's average rank over the code benchmarks (DeMix §2.4), each benchmark measured by the
  validation loss of its reference solutions (Xing §3.2.2: the benchmark's problems and ground-truth solutions, at
  most 200 problems; Table 2: this loss tracks pass@1);
- predictor: LightGBM (DeMix §2.4 Step 3) on the mixture weights and each benchmark's MDE loss, the ensemble
  sum_i alpha_i p_i of the components' cached next-token probabilities (MDE §3.1 Algorithm 1, §3.2);
- search: merged models over three rounds, the later rounds the predictor's best of a large uniform sample of the
  simplex, then the average of the predictor's top mixtures (DeMix §2.4, §3).

The components were trained on their corpus mixed half and half with the general corpora (DeMix §3, beta = 0.5), so a
merge alpha stands for the data 0.5 * general + 0.5 * sum_i alpha_i D_i (DeMix App. A.1: the searched subspace keeps
non-general data under half): the mixture written for base_train.

python -m scripts.demix --components stack_edu=demix-stack_edu,climbmix=demix-climbmix,... --general climbmix,...
python -m scripts.demix --evaluate demix-final,demix-uniform
"""

import os
import json
import argparse

import numpy as np
import torch
import lightgbm

from nanochat.common import compute_init, compute_cleanup, autodetect_device_type, get_base_dir, print0
from nanochat.checkpoint_manager import build_model, find_last_step
from nanochat.data.eval.humaneval import HumanEval
from nanochat.data.eval.mbpp import MBPP


def benchmarks(seed):
    """{benchmark: [(prompt, reference solution)]}: HumanEval's problems with their canonical solutions, and MBPP test
    problems with their reference code, prompted by the description and one test in a docstring (the InCoder MBPP
    prompt of Xing et al.'s evaluation)."""
    humaneval = HumanEval().ds
    mbpp = MBPP("test").ds
    chosen = np.random.default_rng(seed).choice(len(mbpp), 200, replace=False)  # Xing §3.2.2: "over 200 problems, we randomly sampled 200"
    return {
        "humaneval": [(humaneval[i]["prompt"], humaneval[i]["canonical_solution"]) for i in range(len(humaneval))],
        "mbpp": [(f'"""\n{row["text"]}\n{row["test_list"][0]}\n"""\n', row["code"]) for row in (mbpp[int(i)] for i in chosen)],
    }


def solution_rows(tokenizer, examples, tokens_per_batch, device):
    """Micro-batches (inputs, targets) of the examples, shortest first, each within the model's training micro-batch of
    tokens: BOS, prompt and solution tokens, the targets -1 everywhere but on the solution's tokens, so the loss covers
    the reference solution given its prompt."""
    bos = tokenizer.get_bos_token_id()
    seqs = sorted(((tokenizer.encode(prompt, prepend=bos), tokenizer.encode(solution)) for prompt, solution in examples),
                  key=lambda ps: len(ps[0]) + len(ps[1]))
    groups = [[]]
    for p, s in seqs:
        if groups[-1] and (len(groups[-1]) + 1) * (len(p) + len(s)) > tokens_per_batch:
            groups.append([])
        groups[-1].append((p, s))
    batches = []
    for group in groups:
        width = max(len(p) + len(s) for p, s in group) - 1
        x = torch.full((len(group), width), bos, dtype=torch.int32)
        y = torch.full((len(group), width), -1, dtype=torch.int64)
        for row, (p, s) in enumerate(group):
            seq = p + s
            x[row, :len(seq) - 1] = torch.tensor(seq[:-1])
            y[row, len(p) - 1:len(seq) - 1] = torch.tensor(s)
        batches.append((x.to(device), y.to(device)))
    return batches


@torch.no_grad()
def token_losses(model, batches):
    """Loss (nats) of every solution token, in the batches' order."""
    return torch.cat([model(x, y, loss_reduction="none").view(y.shape)[y >= 0].float() for x, y in batches])


def average_rank(losses):
    """Each model's rank (1 = lowest loss) on each benchmark, averaged over the benchmarks (DeMix §2.4)."""
    return np.mean([np.argsort(np.argsort(column)) + 1 for column in np.asarray(losses).T], axis=0)


def load(tag, device):
    """(model, tokenizer, meta) of a trained model tag's last checkpoint."""
    checkpoint_dir = os.path.join(get_base_dir(), "base_checkpoints", tag)
    return build_model(checkpoint_dir, find_last_step(checkpoint_dir), device, phase="eval")


def benchmark_rows(tokenizer, meta, seed, device):
    tokens_per_batch = meta["device_batch_size"] * meta["max_seq_len"]
    return {name: solution_rows(tokenizer, examples, tokens_per_batch, device) for name, examples in benchmarks(seed).items()}


def evaluate(tags, device, seed):
    """Validation loss of each benchmark's reference solutions for trained models."""
    results, rows = {}, None
    for tag in tags:
        model, tokenizer, meta = load(tag, device)
        rows = rows or benchmark_rows(tokenizer, meta, seed, device)
        results[tag] = {"dataset": meta["user_config"]["dataset"], **{name: token_losses(model, batches).mean().item() for name, batches in rows.items()}}
        print0(f"{tag}: " + " | ".join(f"{name} {results[tag][name]:.4f}" for name in rows))
    return results


def search(components, general, device, seed):
    names, tags = list(components), list(components.values())
    model, tokenizer, meta = load(tags[0], device)
    rows = benchmark_rows(tokenizer, meta, seed, device)
    stacks, probs = {}, {name: [] for name in rows}
    for tag in tags:
        component, _, _ = load(tag, device)
        for key, tensor in component.state_dict().items():
            stacks.setdefault(key, []).append(tensor)
        for name, batches in rows.items():  # MDE §3.1: each expert's cached probabilities of the validation tokens
            probs[name].append(torch.exp(-token_losses(component, batches)))
        del component
    stacks = {key: torch.stack(tensors) for key, tensors in stacks.items()}
    probs = {name: torch.stack(p) for name, p in probs.items()}  # (components, tokens)
    params = model.state_dict()  # shares storage with the model, so copying into it sets the merged weights

    def merged_losses(alpha):
        """Validation loss of each benchmark for the merge sum_i alpha_i Theta_i (DeMix Eq. 6)."""
        with torch.no_grad():
            for key, stack in stacks.items():
                params[key].copy_(torch.tensordot(torch.tensor(alpha, device=device, dtype=stack.dtype), stack, dims=1))
        return [token_losses(model, rows[name]).mean().item() for name in rows]

    def features(alphas):
        """The mixture weights and each benchmark's MDE loss (MDE §3.2: 'the MDE approximation as additional source of
        features')."""
        a = torch.tensor(alphas, device=device, dtype=torch.float32)
        mde = [(-torch.log(a @ probs[name]).mean(dim=1)).cpu().numpy() for name in rows]
        return np.column_stack([alphas, *mde])

    def fit(alphas, ranks):  # DeMix §2.4 Step 3; §3: "we set the learning rate to 0.02 and the number of iterations to 300"
        params = dict(objective="regression", learning_rate=0.02, verbose=-1)
        return lightgbm.train(params, lightgbm.Dataset(features(alphas), ranks), num_boost_round=300)

    def sample():  # DeMix §2.4 Step 1: uniformly from the simplex; its released predictor scores 200,000 mixtures
        return rng.dirichlet(np.ones(len(names)), 200_000)

    rng = np.random.default_rng(seed)
    evaluated, losses = np.empty((0, len(names))), []
    for round_index, count in enumerate([64, 32, 16]):  # DeMix §3: "we sample 64, 32, and 16 mixtures in each respective iteration"
        if round_index == 0:
            chosen = sample()[:count]
        else:  # DeMix §2.4 Step 4: the predictor's best of newly sampled mixtures
            candidates = sample()
            chosen = candidates[np.argsort(predictor.predict(features(candidates)))[:count]]
        for alpha in chosen:  # DeMix §2.4 Step 2
            losses.append(merged_losses(alpha))
        evaluated = np.vstack([evaluated, chosen])
        ranks = average_rank(losses)
        predictor = fit(evaluated, ranks)
        best = int(np.argmin(ranks))
        print0(f"round {round_index + 1}: {len(evaluated)} merged models; best average rank {ranks[best]:.1f}: "
               + ", ".join(f"{n} {w:.3f}" for n, w in zip(names, evaluated[best])))
    candidates = sample()
    alpha = candidates[np.argsort(predictor.predict(features(candidates)))[:128]].mean(axis=0)  # DeMix §3: "average the top 128 mixtures"
    mixture = {name: 0.5 * a + (0.5 / len(general) if name in general else 0.0) for name, a in zip(names, alpha)}
    print0("merge weights: " + ", ".join(f"{n} {w:.4f}" for n, w in zip(names, alpha)))
    print0("mixture: " + ",".join(f"{n}:{w:.6f}" for n, w in mixture.items()))
    return {"components": components, "general": general, "benchmarks": list(rows), "evaluated": evaluated.tolist(),
            "losses": losses, "ranks": average_rank(losses).tolist(), "alpha": dict(zip(names, alpha.tolist())),
            "mixture": mixture}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeMix data-mixture search with X-MDE features on code validation loss")
    parser.add_argument("--components", type=str, default=None, help="corpus=model_tag,... : the component model of each corpus")
    parser.add_argument("--general", type=str, default=None, help="comma-separated general corpora the components were mixed with at half")
    parser.add_argument("--evaluate", type=str, default=None, help="comma-separated model tags: report their code validation losses")
    parser.add_argument("--seed", type=int, default=42, help="seed of the MBPP problem sample and the mixture sampling")
    args = parser.parse_args()
    _, _, _, _, device = compute_init(autodetect_device_type())
    out_dir = os.path.join(get_base_dir(), "demix")
    os.makedirs(out_dir, exist_ok=True)
    if args.components:
        result = search(dict(part.split("=") for part in args.components.split(",")), args.general.split(","), device, args.seed)
        path = os.path.join(out_dir, "search.json")
    else:
        result = evaluate(args.evaluate.split(","), device, args.seed)
        path = os.path.join(out_dir, "evaluate.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print0(f"wrote {path}")
    compute_cleanup()
