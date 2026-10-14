# nanochat: code pretraining on a DGX Spark

Work in progress: a fork of [karpathy/nanochat](https://github.com/karpathy/nanochat) for pretraining a code-generation LLM on one NVIDIA DGX Spark (GB10).

## Setup

```bash
docker build -f docker/Dockerfile -t nanochat .
docker run --rm -it --device nvidia.com/gpu=all --ipc=host --init --ulimit memlock=-1 --ulimit stack=67108864 \
  --user $(id -u):$(id -g) -e HOME=/tmp -e HF_TOKEN -e WANDB_API_KEY \
  -v ~/nanochat-runs:/runs -v $PWD:/workspace/nanochat nanochat bash
```

## Pipeline

```bash
python -m nanochat.data.pretrain.fetch --dataset=stack_v3 --max-files=<N>
python -m nanochat.tokenizer --dataset=stack_v3
python -m nanochat.data.pretrain.compile --dataset=stack_v3 --max-files=<N>
python -m scripts.base_train --dataset=stack_v3 --depth=12
python -m scripts.base_eval
```

Corpora: `stack_v3`, `stack_edu`, `opc_fineweb_code`, `opc_fineweb_math`, `climbmix`. `--dataset=a:0.6,b:0.4` trains on a mixture.
