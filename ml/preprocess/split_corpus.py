"""Split the real verdict corpus into a fine-tuning set and a RAG pool.

Keeping the two disjoint means fine-tuning and RAG never draw on the same
verdicts, which keeps the thesis comparison between generation methods
(plain model / RAG / fine-tuning) methodologically clean.

The split is deterministic (seeded by filename, not by OS directory order)
so re-running this after adding a few new files to ml/data/extracted/
reclassifies only the new files instead of reshuffling everything.

Usage:
    python ml/preprocess/split_corpus.py
"""

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXTRACTED_DIR = REPO_ROOT / "ml" / "data" / "extracted"
SPLIT_FILE = REPO_ROOT / "ml" / "data" / "split.json"

FINETUNE_FRACTION = 0.8
SEED = 42


def main():
    files = sorted(p.name for p in EXTRACTED_DIR.glob("*.txt"))
    if not files:
        sys.exit(f"No .txt files found in {EXTRACTED_DIR}")

    rng = random.Random(SEED)
    shuffled = files[:]
    rng.shuffle(shuffled)

    cutoff = round(len(shuffled) * FINETUNE_FRACTION)
    finetune = sorted(shuffled[:cutoff])
    rag_pool = sorted(shuffled[cutoff:])

    SPLIT_FILE.write_text(
        json.dumps({"finetune": finetune, "rag_pool": rag_pool}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"{len(finetune)} files -> fine-tuning, {len(rag_pool)} files -> RAG pool")
    print(f"Wrote {SPLIT_FILE}")


if __name__ == "__main__":
    main()
