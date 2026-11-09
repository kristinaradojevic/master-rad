"""Provider for the locally fine-tuned model served by `mlx_lm.server`
(see ml/finetune/train_lora.py and the README's Fine-tuning section).

Why this isn't just OpenAIProvider with a different base_url: training
data (ml/preprocess/prepare_finetune_data.py) splits long verdicts into
several chunked examples, because full verdicts (median ~8800 tokens) don't
fit this machine's training memory limit (max_seq_length=4096). Each chunk
is still tokenized as a *complete* assistant turn — every chat-format
training example ends with the model's end-of-turn token, chunk 1 included.
That risks teaching the model to treat ~3000 tokens as a finished response,
well before a real verdict is actually done, if it's only ever called the
same way chunk-1 examples were (plain facts, no more).

So this provider mirrors the training-time continuation format instead: if
a response looks incomplete, it's extended with another call using the
exact "already written: <tail>, continue" prompt (CONTINUATION_TEMPLATE in
services/prompt.py) that the chunked training data also uses for every
chunk after the first — so the conditioning the model sees at generation
time matches what it was actually trained on.

"Looks incomplete" is a heuristic (see _looks_complete below), not a
guarantee — it's cheap to get wrong on an unusual verdict structure.
"""

import tiktoken
from openai import OpenAI

from .base import LLMProvider
from ..prompt import CONTINUATION_TEMPLATE

_ENCODING = tiktoken.get_encoding("o200k_base")
# Mirrors prepare_finetune_data.py's CONTEXT_OVERLAP_TOKENS / HARD_CAP_TOKENS
# — kept close to training-time chunk sizes so continuations stay in the
# same regime the model was fine-tuned on.
_TAIL_TOKENS = 500
_MAX_TOKENS_PER_CALL = 3200
_MAX_ROUNDS = 6
# mlx_lm.server defaults to greedy decoding (temperature=0.0) with no
# repetition penalty. On this model (LoRA on ~91 highly formulaic real
# verdicts) that combination reliably collapses into repeating a short
# phrase verbatim dozens or hundreds of times (seen in practice: e.g.
# "када је окривљени лишен слободе" ~100x, then after adding only a
# repetition penalty, "Republike Srbije" ~300x instead) — greedy decoding
# has no randomness to escape the loop once it starts, so the penalty alone
# wasn't enough. 0.7 went too far the other way: the model drifted into
# incoherent text, including an unrelated block in Chinese at one point —
# this LoRA (rank 8, ~3 epochs over ~460 chunked examples) doesn't seem
# trained enough to stay on-distribution at that temperature. 0.3 was the
# best of what was tried: no loops, stays in Serbian, coherent structure,
# though still not fully faithful to the input (invents incidental details
# like a date) — a real limitation of this fine-tuning run worth noting as
# such, not something sampling alone fixes. Set here rather than as server
# launch flags so it's visible/tunable in one place close to the problem.
_TEMPERATURE = 0.3
_TOP_P = 0.9
_REPETITION_PENALTY = 1.3
_REPETITION_CONTEXT_SIZE = 64

# A real verdict always ends with this mandatory closing section (see
# SYSTEM_PROMPT's required structure) — its presence is a much more
# reliable "is this verdict actually done" signal than trusting the model's
# own end-of-turn token, which chunked training data undermines (see
# module docstring). Checked across both scripts verdicts in this corpus
# are written in (Cyrillic and Latin).
_CLOSING_MARKERS = ("pravnom leku", "pravni lek", "правном леку", "правни лек")


def _looks_complete(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _CLOSING_MARKERS)


def _tail(text: str, max_tokens: int) -> str:
    tokens = _ENCODING.encode(text)
    return _ENCODING.decode(tokens[-max_tokens:])


class LocalMLXProvider(LLMProvider):
    provider_name = "local-mlx"

    def __init__(self, model: str, base_url: str):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key="not-needed")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        full_text = ""
        for _ in range(_MAX_ROUNDS):
            prompt = (
                user_prompt
                if not full_text
                else user_prompt + CONTINUATION_TEMPLATE.format(tail=_tail(full_text, _TAIL_TOKENS))
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=_MAX_TOKENS_PER_CALL,
                temperature=_TEMPERATURE,
                top_p=_TOP_P,
                extra_body={
                    "repetition_penalty": _REPETITION_PENALTY,
                    "repetition_context_size": _REPETITION_CONTEXT_SIZE,
                },
            )
            choice = response.choices[0]
            full_text += choice.message.content or ""
            # Check content first, regardless of finish_reason: a response
            # that already contains the closing section is done even if it
            # happened to also hit the token cap right around there. Only
            # keep going when it doesn't look done AND got cut by length —
            # a natural stop (finish_reason != "length") without the
            # closing section is unusual but trusted over looping forever.
            if _looks_complete(full_text) or choice.finish_reason != "length":
                break
        return full_text
