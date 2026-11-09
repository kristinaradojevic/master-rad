"""Build an OpenAI-style fine-tuning JSONL dataset from the structured
fields extracted by extract_structured_fields.py, paired with the original
verdict text as the target completion.

Real verdicts run long (median ~8k tokens, some over 40k) — far more than
fits in one training example on this machine (max_seq_length=4096 in
ml/finetune/train_lora.py, after repeated out-of-memory errors at higher
values). Letting mlx_lm.lora truncate long examples itself means the model
never sees how most verdicts end (sentencing rationale, legal remedy
notice) — a first attempt at this showed 87% of examples truncated, keeping
on average only ~40% of the verdict text. Instead, verdicts that don't fit
are split here into multiple sequential chunks, each its own training
example, so every chunk fits the budget and the model still sees the whole
document, just across several examples instead of one. Chunks after the
first include a short tail of the previously-written text as context, so
the model learns to continue coherently rather than restart.

Chunk sizes are computed with Qwen2.5's own tokenizer and its real chat
template (via transformers.AutoTokenizer, pulled in by ml/requirements.txt)
rather than estimated with a different model's tokenizer (an earlier
version used tiktoken as a stand-in, which still let a handful of chunks
through over mlx_lm.lora's max_seq_length — Serbian Cyrillic text doesn't
tokenize quite the same way under every tokenizer). Every chunk is checked
against the real tokenizer before being written, so mlx_lm.lora should
never need to truncate anything itself.

This does introduce a mismatch with how the model is actually prompted in
production (one call, no "already written" context) — continuation-style
examples teach it what mid/late-verdict prose looks like, but not under the
exact framing it'll see at inference. Worth noting as a limitation rather
than a free fix.

Each training example reuses the app's real system prompt and user-prompt
builder (backend/app/services/prompt.py) with use_rag=False and
num_examples=0, so the model is trained on exactly the prompt shape it will
see in production, without few-shot examples baked into the training input
itself (the fine-tuned weights are meant to replace those examples).

Requires ml/data/split.json and ml/data/structured/*.json (run
split_corpus.py and extract_structured_fields.py first).

Usage:
    python ml/preprocess/prepare_finetune_data.py
"""

import json
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.schemas import VerdictInput  # noqa: E402
from app.services.prompt import (  # noqa: E402
    CONTINUATION_TEMPLATE,
    SYSTEM_PROMPT,
    build_user_prompt,
)

EXTRACTED_DIR = REPO_ROOT / "ml" / "data" / "extracted"
STRUCTURED_DIR = REPO_ROOT / "ml" / "data" / "structured"
SPLIT_FILE = REPO_ROOT / "ml" / "data" / "split.json"
OUT_DIR = REPO_ROOT / "ml" / "data" / "finetune"

VALIDATION_FRACTION = 0.1
SEED = 42

# Must match train_lora.py's BASE_MODEL — only the tokenizer config gets
# downloaded here (a few MB), not the full model weights.
BASE_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"

# mlx_lm.lora trains at max_seq_length=4096 (see train_lora.py). Leaving a
# small margin below that absorbs rounding at chunk boundaries (decoding
# then re-encoding a token slice isn't always perfectly length-stable) and
# whatever separator mlx_lm's own chat-template formatting adds beyond what
# we measure below.
MAX_SEQ_LENGTH = 4096
SAFETY_MARGIN_TOKENS = 64
HARD_CAP_TOKENS = MAX_SEQ_LENGTH - SAFETY_MARGIN_TOKENS
# Tail of the previous chunk repeated as context so continuation chunks
# don't start from nothing.
CONTEXT_OVERLAP_TOKENS = 500

TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL)


def _chat_template_tokens(system_prompt: str, user_prompt: str) -> int:
    """Exact token count for everything except the assistant's own content
    — the real system/user text plus whatever role markers and separators
    Qwen's chat template adds around them."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    # add_generation_prompt appends the marker that would normally open the
    # assistant's turn, so this counts everything the assistant's content
    # budget actually has to fit after.
    # apply_chat_template returns a BatchEncoding (dict-like, with
    # "input_ids"/"attention_mask" keys) by default — len() on it counts
    # keys, not tokens. Pull out input_ids explicitly.
    return len(TOKENIZER.apply_chat_template(messages, add_generation_prompt=True)["input_ids"])


def _build_examples(name: str) -> list[dict]:
    stem = Path(name).stem
    fields = json.loads((STRUCTURED_DIR / f"{stem}.json").read_text(encoding="utf-8"))
    # model_construct skips the strict field validation VerdictInput otherwise
    # enforces (e.g. defendant_jmbg must be 13 digits). That validation exists
    # to catch bad input from a judge typing into the live form; it doesn't
    # apply here — the LLM extraction step sometimes can't find a full JMBG
    # in an old verdict (redacted, or genuinely absent from e.g. a dismissal
    # ruling), and the raw extracted value should still make it into training
    # data rather than be dropped over a field that hasn't caused any errors.
    verdict_input = VerdictInput.model_construct(**fields)
    verdict_text = (EXTRACTED_DIR / name).read_text(encoding="utf-8")
    base_user_prompt = build_user_prompt(verdict_input, use_rag=False, num_examples=0)

    verdict_tokens = TOKENIZER.encode(verdict_text)

    examples = []
    pos = 0
    tail_tokens: list[int] = []
    while pos < len(verdict_tokens):
        if pos == 0:
            user_prompt = base_user_prompt
        else:
            tail_text = TOKENIZER.decode(tail_tokens)
            user_prompt = base_user_prompt + CONTINUATION_TEMPLATE.format(tail=tail_text)

        overhead = _chat_template_tokens(SYSTEM_PROMPT, user_prompt)
        budget = max(200, HARD_CAP_TOKENS - overhead)

        chunk_tokens = verdict_tokens[pos : pos + budget]
        chunk_text = TOKENIZER.decode(chunk_tokens)

        # Re-check against the real chat template once assembled — decoding
        # then re-encoding a token slice isn't always perfectly
        # length-stable at the boundary, so trim further if needed instead
        # of trusting the estimate above.
        while True:
            total = len(
                TOKENIZER.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": chunk_text},
                    ]
                )["input_ids"]
            )
            if total <= MAX_SEQ_LENGTH or len(chunk_tokens) <= 1:
                break
            chunk_tokens = chunk_tokens[: -max(1, (total - MAX_SEQ_LENGTH))]
            chunk_text = TOKENIZER.decode(chunk_tokens)

        examples.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": chunk_text},
                ]
            }
        )

        pos += len(chunk_tokens)
        tail_tokens = verdict_tokens[max(0, pos - CONTEXT_OVERLAP_TOKENS) : pos]

    return examples


def main():
    if not SPLIT_FILE.exists():
        sys.exit(f"{SPLIT_FILE} not found — run split_corpus.py first.")
    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    finetune_files = split["finetune"]

    # Grouped by source verdict so a verdict's chunks all land in the same
    # train/valid split — otherwise a model validated on a chunk it saw the
    # neighboring chunk of during training would look better than it is.
    examples_by_file: dict[str, list[dict]] = {}
    missing = []
    for name in finetune_files:
        stem = Path(name).stem
        if not (STRUCTURED_DIR / f"{stem}.json").exists():
            missing.append(name)
            continue
        try:
            examples_by_file[name] = _build_examples(name)
        except Exception as e:
            print(f"SKIPPED (bad structured data): {name} ({e})")

    if missing:
        print(f"{len(missing)} file(s) have no extracted structured fields yet — "
              "run extract_structured_fields.py first. Skipping them:")
        for name in missing:
            print(f"  - {name}")

    if not examples_by_file:
        sys.exit("No usable training examples — nothing to write.")

    multi_chunk = sum(1 for rows in examples_by_file.values() if len(rows) > 1)
    print(f"{len(examples_by_file)} verdicts -> {sum(len(r) for r in examples_by_file.values())} "
          f"training examples ({multi_chunk} verdicts needed more than one chunk)")

    names = list(examples_by_file.keys())
    rng = random.Random(SEED)
    rng.shuffle(names)
    cutoff = max(1, round(len(names) * (1 - VALIDATION_FRACTION)))
    train_names, valid_names = names[:cutoff], names[cutoff:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, split_names in [("train", train_names), ("valid", valid_names)]:
        rows = [row for name in split_names for row in examples_by_file[name]]
        rng.shuffle(rows)
        out_path = OUT_DIR / f"{split_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{len(rows)} examples ({len(split_names)} verdicts) -> {out_path}")


if __name__ == "__main__":
    main()
