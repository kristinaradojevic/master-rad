"""Build an OpenAI fine-tuning JSONL dataset from the structured fields
extracted by extract_structured_fields.py, paired with the original verdict
text as the target completion.

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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.schemas import VerdictInput  # noqa: E402
from app.services.prompt import SYSTEM_PROMPT, build_user_prompt  # noqa: E402

EXTRACTED_DIR = REPO_ROOT / "ml" / "data" / "extracted"
STRUCTURED_DIR = REPO_ROOT / "ml" / "data" / "structured"
SPLIT_FILE = REPO_ROOT / "ml" / "data" / "split.json"
OUT_DIR = REPO_ROOT / "ml" / "data" / "finetune"

VALIDATION_FRACTION = 0.1
SEED = 42


def _build_example(name: str) -> dict:
    stem = Path(name).stem
    fields = json.loads((STRUCTURED_DIR / f"{stem}.json").read_text(encoding="utf-8"))
    verdict_input = VerdictInput(**fields)
    verdict_text = (EXTRACTED_DIR / name).read_text(encoding="utf-8")

    user_prompt = build_user_prompt(verdict_input, use_rag=False, num_examples=0)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": verdict_text},
        ]
    }


def main():
    if not SPLIT_FILE.exists():
        sys.exit(f"{SPLIT_FILE} not found — run split_corpus.py first.")
    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    finetune_files = split["finetune"]

    examples = []
    missing = []
    for name in finetune_files:
        stem = Path(name).stem
        if not (STRUCTURED_DIR / f"{stem}.json").exists():
            missing.append(name)
            continue
        try:
            examples.append(_build_example(name))
        except Exception as e:
            print(f"SKIPPED (bad structured data): {name} ({e})")

    if missing:
        print(f"{len(missing)} file(s) have no extracted structured fields yet — "
              "run extract_structured_fields.py first. Skipping them:")
        for name in missing:
            print(f"  - {name}")

    if not examples:
        sys.exit("No usable training examples — nothing to write.")

    rng = random.Random(SEED)
    rng.shuffle(examples)
    cutoff = max(1, round(len(examples) * (1 - VALIDATION_FRACTION)))
    train, valid = examples[:cutoff], examples[cutoff:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, rows in [("train", train), ("valid", valid)]:
        out_path = OUT_DIR / f"{split_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{len(rows)} examples -> {out_path}")


if __name__ == "__main__":
    main()
