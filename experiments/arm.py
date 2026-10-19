"""
One arm of the ablation study (experiments/design.py): scripts/base_train.train of a Variant at DEPTH and RATIO, its
initialization by SEED and its data order by DATA_SEED (one final checkpoint), then the bits per byte of each mixture
corpus's val split in rows of the file-level stage (bpb) and of each repository corpus's in rows of the repo-level
stage (bpb32: what the 32K stage adds; eval/loss_eval.val_bpb). Written with the run's summary (steps, final train
loss, median tok/s, fraction of steps whose gradient norm exceeded the clip) to <base_dir>/ablation/<tag>.json.

    python -m experiments.arm TAG DEPTH RATIO SEED DATA_SEED '<Variant fields as JSON>'
"""
import json
import os
import statistics
import sys

import torch

from nanochat.config import FILE_STAGE, MIXTURE, REPO_STAGE
from nanochat.data.loader import mixture_corpora
from nanochat.data.tokenizer import get_tokenizer
from nanochat.eval.loss_eval import val_bpb
from nanochat.paths import base_dir, write_json
from nanochat.train.optim import GRAD_CLIP
from nanochat.train.schedule import plan
from scripts.base_train import Variant, train

tag, depth, ratio = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
seed, data_seed, variant = int(sys.argv[4]), int(sys.argv[5]), Variant(**json.loads(sys.argv[6]))
tokenizer = get_tokenizer()
steps = plan(depth, ratio, tokenizer.vocab_size, variant.batch_scale).num_iterations
model = train(depth=depth, ratio=ratio, seed=seed, data_seed=data_seed, save_every=steps, grad_clip=GRAD_CLIP, tag=tag,
              variant=variant)
losses = torch.compile(model.losses, dynamic=False, fullgraph=True)
device = torch.device("cuda")
bpb = val_bpb(losses, tokenizer, mixture_corpora(MIXTURE), stage=FILE_STAGE, device=device)
bpb32 = val_bpb(losses, tokenizer, mixture_corpora(REPO_STAGE.mixture), stage=REPO_STAGE, device=device)
with open(os.path.join(base_dir(), "metrics", f"{tag}.jsonl"), encoding="utf-8") as f:
    records = [r for r in map(json.loads, f) if r["kind"] == "train"]
os.makedirs(os.path.join(base_dir(), "ablation"), exist_ok=True)
write_json(os.path.join(base_dir(), "ablation", f"{tag}.json"), {
    "tag": tag, "depth": depth, "ratio": ratio, "seed": seed, "data_seed": data_seed, "variant": json.loads(sys.argv[6]),
    "steps": steps, "bpb": bpb, "bpb32": bpb32, "final_loss": records[-1]["loss"],
    "tok_per_s": statistics.median(r["tok_per_s"] for r in records[1:]),
    "clipped": sum(r["grad_norm"] > GRAD_CLIP for r in records) / len(records)})
