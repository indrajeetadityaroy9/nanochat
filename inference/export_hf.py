"""A checkpoint (full-state checkpoint-<step>.pt or weights-only weights-<step>.pt, train/checkpoint.py) as a Hugging Face
model directory (weights, config, generation config, the corpus tokenizer), which transformers and vLLM's Transformers
backend load by trust_remote_code through inference/modeling_nanochat.py.

With --chat (a post-trained checkpoint, docs/post-training.md §11.1): the tokenizer's chat template renders BOS, the
messages' contents joined by "\\n\\n" and stripped, then THINK, the SFT context's text (posttrain/sft_data.py render:
the stripped prompt and THINK, encoded as one text); a system message, as lcb_runner's chat styles send one, joins the
user turn as its no-system-role style writes it (SYSTEM_MESSAGE_GENERIC, "\\n\\n", the question). The generation
config's sampling defaults are the reasoning protocol's T 0.6, top-p 0.95 (DeepSeek-R1, arXiv 2501.12948 v1 §3).

    python -m inference.export_hf <checkpoint.pt> <out dir> [--chat]
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
from inference.modeling_nanochat import NanochatPreTrainedModel
from model.config import HEAD_DIM, ROPE_THETA, VOCAB_SIZE
from model.modeling import ModelConfig, has_ve, layer_windows
from model.ops.linear import EPS
from posttrain.sft import THINK

# docs/post-training.md §11.1's context; Jinja's trim is Python's str.strip; THINK as a Jinja string literal, since a
# template's own trailing newline is dropped (Jinja's keep_trailing_newline is off)
CHAT_TEMPLATE = ("{{ bos_token }}{{ messages | map(attribute='content') | join('\\n\\n') | trim }}"
                 f"{{{{ {json.dumps(THINK)} }}}}")
REASONING_SAMPLING = {"temperature": 0.6, "top_p": 0.95}


def export(checkpoint: Path, out: Path, *, chat: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=True)
    state = torch.load(checkpoint, map_location="cpu", mmap=True)
    config = ModelConfig(**state["config"])
    n, C, T = config.num_hidden_layers, config.hidden_size, config.max_position_embeddings
    # bf16 what modeling.py reads in bf16 (the GEMM copies, the embedding cast, ve_gate.cu's table rounding); the
    # compiled forward's x0 is the fp32 embedding row's norm (Inductor drops the cast; decode.cu k_norm EMBED)
    fp32 = set(NanochatPreTrainedModel._keep_in_fp32_modules_strict)
    save_file({(k if k == "lm_head.weight" else f"model.{k}"):
               v.to(torch.float32 if fp32 & set(k.split(".")) else torch.bfloat16).contiguous()
               for k, v in state["model"].items()}, out / "model.safetensors", metadata={"format": "pt"})

    tokenizer = get_tokenizer()
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
        # transformers' window counts the query: S keys before it + 1
        "sliding_window": short + 1,
        "layer_types": ["sliding_attention" if w < T else "full_attention" for w in windows],
        "value_embedding_layers": [i for i in range(n) if has_ve(i, num_hidden_layers=n)],
        "final_logit_softcapping": config.final_logit_softcapping, "rms_norm_eps": EPS,
        # vLLM's head (LogitsProcessor._apply_head) in fp32, as the evaluation computes it; transformers ignores it
        "head_dtype": "float32", "tie_word_embeddings": False, "dtype": "bfloat16",
        "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.bos_token_id,
    }, indent=2) + "\n")
    (out / "generation_config.json").write_text(json.dumps(
        {"bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.bos_token_id}
        | (REASONING_SAMPLING if chat else {}), indent=2) + "\n")

    fast = TikTokenConverter(vocab_file=str(TOKENIZER / "tokenizer.tiktoken"), pattern=tokenizer.pattern,
                             extra_special_tokens=list(SPECIAL_TOKENS)).converted()
    fast.post_processor = processors.TemplateProcessing(single=f"{BOS} $A",
                                                       special_tokens=[(BOS, tokenizer.bos_token_id)])
    PreTrainedTokenizerFast(tokenizer_object=fast, bos_token=BOS, eos_token=BOS, model_max_length=T,
                            chat_template=CHAT_TEMPLATE if chat else None).save_pretrained(out)


if __name__ == "__main__":
    export(Path(sys.argv[1]), Path(sys.argv[2]), chat="--chat" in sys.argv[3:])
