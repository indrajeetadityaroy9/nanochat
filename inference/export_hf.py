"""
A checkpoint as a Hugging Face model directory, which vLLM (inference/vllm_model.py) and transformers
(inference/modeling_nanochat.py, by trust_remote_code) load:
- model.safetensors: the training state dict's names under "model." (lm_head.weight as is); matrices and tables bf16,
  as the training ops round them; the per-dim and per-layer vectors and the value-embedding gates fp32, as they read
  them.
- config.json: model_type "nanochat", architectures NanochatForCausalLM, auto_map to modeling_nanochat.py (copied
  beside it); model.modeling.layer_windows as Gemma 3's layer_types + sliding_window (keys attended including the
  query's own: FA4's window_left + 1); model.modeling.has_ve's layers; eps FLT_EPSILON (model.modeling.norm).
- generation_config.json: BOS ends a generation (a document's and a FIM middle's end in training).
- tokenizer.json + tokenizer_config.json: transformers' TikTokenConverter of the tiktoken ranks and split pattern
  (ByteLevel BPE, the pattern as an Oniguruma Split; its possessive quantifiers are Oniguruma syntax), the special
  tokens after the mergeable ones in corpus.tokenizer.SPECIAL_TOKENS order (same ids), BOS prepended to an encoded
  text (TemplateProcessing).

    python -m inference.export_hf <checkpoint.pt> <out dir> [<tokenizer dir>]   (default corpus.tokenizer.TOKENIZER)
"""

import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import processors
from transformers import PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import TikTokenConverter

from corpus.tokenizer import BOS, SPECIAL_TOKENS, TOKENIZER, get_tokenizer
from model.config import HEAD_DIM, ROPE_THETA, VOCAB_SIZE
from model.modeling import ModelConfig, has_ve, layer_windows

# parameters kept fp32 (name suffixes)
FP32 = ("resid_lambdas", "x0_lambdas", "q_norm_weight", "k_norm_weight", "ve_gate.weight")


def export(checkpoint: Path, out: Path, tokenizer_dir: Path = TOKENIZER) -> None:
    out.mkdir(parents=True, exist_ok=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig(**state["config"])
    n, C, T = config.num_hidden_layers, config.hidden_size, config.max_position_embeddings
    save_file({(k if k == "lm_head.weight" else f"model.{k}"):
               v.to(torch.float32 if k.endswith(FP32) else torch.bfloat16).contiguous()
               for k, v in state["model"].items()}, out / "model.safetensors", metadata={"format": "pt"})

    tokenizer = get_tokenizer(tokenizer_dir)
    if tokenizer.vocab_size != VOCAB_SIZE:
        raise ValueError(f"{tokenizer_dir}: {tokenizer.vocab_size} ids, the model has {VOCAB_SIZE}")
    windows = layer_windows(config)
    (short,) = {w for w in windows if w < T}
    shutil.copy(Path(__file__).with_name("modeling_nanochat.py"), out / "modeling_nanochat.py")
    (out / "config.json").write_text(json.dumps({
        "architectures": ["NanochatForCausalLM"], "model_type": "nanochat",
        "auto_map": {"AutoConfig": "modeling_nanochat.NanochatConfig", "AutoModel": "modeling_nanochat.NanochatModel",
                     "AutoModelForCausalLM": "modeling_nanochat.NanochatForCausalLM"},
        "vocab_size": VOCAB_SIZE, "hidden_size": C, "intermediate_size": 4 * C, "num_hidden_layers": n,
        "num_attention_heads": C // HEAD_DIM, "num_key_value_heads": C // HEAD_DIM, "head_dim": HEAD_DIM,
        "max_position_embeddings": T, "rope_parameters": {"rope_type": "default", "rope_theta": float(ROPE_THETA)},
        "sliding_window": short + 1,
        "layer_types": ["sliding_attention" if w < T else "full_attention" for w in windows],
        "value_embedding_layers": [i for i in range(n) if has_ve(i, num_hidden_layers=n)],
        "final_logit_softcapping": config.final_logit_softcapping, "rms_norm_eps": torch.finfo(torch.float32).eps,
        "tie_word_embeddings": False, "torch_dtype": "bfloat16",
        "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.bos_token_id,
    }, indent=2) + "\n")
    (out / "generation_config.json").write_text(json.dumps(
        {"bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.bos_token_id}, indent=2) + "\n")

    fast = TikTokenConverter(vocab_file=str(tokenizer_dir / "tokenizer.tiktoken"), pattern=tokenizer.pattern,
                             extra_special_tokens=list(SPECIAL_TOKENS)).converted()
    if [fast.token_to_id(t) for t in SPECIAL_TOKENS] != [tokenizer.special_tokens[t] for t in SPECIAL_TOKENS]:
        raise ValueError("the special tokens' ids differ from corpus.tokenizer's")
    fast.post_processor = processors.TemplateProcessing(single=f"{BOS} $A", pair=f"{BOS} $A {BOS} $B",
                                                       special_tokens=[(BOS, tokenizer.bos_token_id)])
    PreTrainedTokenizerFast(tokenizer_object=fast, bos_token=BOS, eos_token=BOS,
                            model_max_length=T).save_pretrained(out)


if __name__ == "__main__":
    export(Path(sys.argv[1]), Path(sys.argv[2]), *map(Path, sys.argv[3:4]))
