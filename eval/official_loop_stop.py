"""Loop stopping for `vllm serve` in scripts/official_reasoning.sh: the natural-termination probe's stop rule
(eval/loop.py, as eval/official_termination.py `generate` applies it after each round) as a logits processor of
vLLM 0.32's model runner V2 (its default). At each check, every CHUNK generated tokens from REPEAT_TAIL on, a request
whose output holds no "</think>" and whose last REPEAT_TAIL tokens have `repetition` >= LOOP gets the eos, BOS, as
its next token. A stopped response is the probe's "unclosed_loop_stopped" at the same length; served, it ends with
finish reason "stop" and, having no "</think>", an empty answer (eval/official_reasoning_parser.py), which every
scorer fails, as it fails the same loop run on to max_tokens. Clients identify a stopped response from its token ids
(eval/loop.py `stopped`); each stop also appends a line {"tokens", "repetition"} to the file named by
NANOCHAT_LOOP_STOP_LOG, when set.

Per step, the check costs a host-side test of each request's output length; only a request at a check reads its
tokens back from the device.

    vllm serve ... --logits-processors eval.official_loop_stop:LoopStopLogitsProcessor
"""

import json
import os

from vllm.v1.worker.gpu.sample.logits_processor.interface import (LogitsContext, LogitsProcessor,
                                                                  LogitsProcRequestState)

from eval.loop import CHUNK, REPEAT_TAIL, looping

LOG = "NANOCHAT_LOOP_STOP_LOG"


class LoopStopLogitsProcessor(LogitsProcessor):
    def __init__(self, vllm_config, req_states: LogitsProcRequestState):
        from corpus.tokenizer import get_tokenizer

        tok = get_tokenizer()
        self.bos, self.decode = tok.bos_token_id, tok.enc.decode
        self.req_states = req_states
        self.log = os.environ.get(LOG)

    def apply(self, logits, ctx: LogitsContext):
        slots = ctx.idx_mapping_np
        seq_lens = ctx.seq_lens_upper_bound_np[:len(slots)]
        generated = seq_lens - self.req_states.prompt_len.np[slots]
        # a check: n generated tokens at a check length, and not a chunk inside a refill after preemption
        refilled = seq_lens >= self.req_states.prefill_len.np[slots]
        for i in ((generated >= REPEAT_TAIL) & (generated % CHUNK == 0) & refilled).nonzero()[0].tolist():
            slot, n = int(slots[i]), int(generated[i])
            if (r := looping(self.output(slot, n, ctx), self.decode)) is None:
                continue
            rows = (ctx.expanded_idx_mapping == slot).nonzero().squeeze(1)
            logits[rows] = float("-inf")
            logits[rows, self.bos] = 0.0
            if self.log:
                with open(self.log, "a") as f:
                    f.write(json.dumps({"tokens": n, "repetition": r}) + "\n")
        return logits

    def output(self, slot: int, n: int, ctx: LogitsContext) -> list[int]:
        """The request's n generated tokens: committed ones from the token history (valid up to total_len), the last
        from this step's input (its row's input id) when not yet committed."""
        start = int(self.req_states.prompt_len.np[slot])
        end = start + n
        committed = min(int(self.req_states.total_len.gpu[slot]), end)
        ids = self.req_states.all_token_ids.gpu[slot, start:committed].tolist()
        if committed < end:
            rows = (ctx.expanded_idx_mapping == slot).nonzero().squeeze(1)
            ids += ctx.input_ids[rows][:end - committed].tolist()
        return ids
