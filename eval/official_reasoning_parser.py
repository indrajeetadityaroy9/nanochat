"""vLLM reasoning parser "nanochat" (`vllm serve --reasoning-parser nanochat`, registered in the server's processes by
eval/official_site/sitecustomize.py) for the post-trained model's format (docs/post-training.md §3): the prompt
ends with THINK, the model writes its reasoning, END_THINK and the answer. The markers are ordinary text (several
tokens each), so vLLM's token-based parsers (deepseek_r1 and the like look "<think>" up as one vocabulary entry)
cannot split them; this one splits the text at the last "</think>": the reasoning before it, the answer after it,
stripped. A response without "</think>" (it ran to its budget or ended early) is all reasoning, its answer "" (an
empty answer, which fails; None content crashes EvalPlus's sanitize). Non-streaming only (EvalPlus and lcb_runner do
not stream).
"""

from collections.abc import Sequence

from vllm.reasoning import ReasoningParser, ReasoningParserManager

END = "</think>"


@ReasoningParserManager.register_module("nanochat")
class NanochatReasoningParser(ReasoningParser):
    @property
    def reasoning_end_str(self) -> str:
        return END

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        return END in self.model_tokenizer.decode(list(input_ids))

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        _, end, answer = self.model_tokenizer.decode(input_ids).rpartition(END)
        return self.model_tokenizer.encode(answer, add_special_tokens=False) if end else []

    def extract_reasoning(self, model_output: str, request) -> tuple[str | None, str | None]:
        reasoning, end, answer = model_output.rpartition(END)
        return (reasoning, answer.strip()) if end else (model_output, "")

    def extract_reasoning_streaming(self, *args, **kwargs):
        raise NotImplementedError("the nanochat reasoning parser does not stream")
