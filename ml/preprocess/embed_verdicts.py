"""Embed real verdict texts for RAG-based few-shot retrieval.

Reads every .txt in ml/data/extracted/ that belongs to the RAG pool half of
the train/RAG split (see split_corpus.py — the other half is reserved for
fine-tuning, so the two methods are never compared on overlapping examples),
embeds it with OpenAI's text-embedding-3-small, and caches the result in
ml/data/embeddings.json. Rerun this whenever ml/data/extracted/ or
ml/data/split.json changes — backend/app/services/rag.py reads the cache,
it never embeds the corpus itself.

The API caps input at 8192 tokens but several real verdicts run much longer,
so long documents are split into multiple chunks, each embedded separately
(one cache entry per chunk, tagged with its source file). rag.py scores each
*document* by its single best-matching chunk, so nothing is truncated away —
long verdicts just cost more embedding calls to index.

Usage:
    python ml/preprocess/embed_verdicts.py
"""

import json
import sys
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(REPO_ROOT / "backend" / ".env")

EXTRACTED_DIR = REPO_ROOT / "ml" / "data" / "extracted"
OUT_FILE = REPO_ROOT / "ml" / "data" / "embeddings.json"
SPLIT_FILE = REPO_ROOT / "ml" / "data" / "split.json"
EMBEDDING_MODEL = "text-embedding-3-small"
# The API hard-caps input at 8192 tokens per call; leave some margin under
# that since token/char ratio varies a bit across documents.
MAX_TOKENS = 8000

_encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)


def _chunk_text(text: str) -> list[str]:
    tokens = _encoding.encode(text)
    chunks = [tokens[i : i + MAX_TOKENS] for i in range(0, len(tokens), MAX_TOKENS)]
    return [_encoding.decode(chunk) for chunk in chunks]


def main():
    files = sorted(EXTRACTED_DIR.glob("*.txt"))
    if not files:
        sys.exit(f"No .txt files found in {EXTRACTED_DIR}")

    if SPLIT_FILE.exists():
        rag_pool = set(json.loads(SPLIT_FILE.read_text(encoding="utf-8"))["rag_pool"])
        before = len(files)
        files = [f for f in files if f.name in rag_pool]
        print(f"Restricting to RAG pool ({SPLIT_FILE.name}): {len(files)}/{before} files")
    else:
        print(f"Warning: {SPLIT_FILE} not found, embedding the whole corpus. "
              "Run split_corpus.py first to keep RAG and fine-tuning data disjoint.")

    client = OpenAI()  # reads OPENAI_API_KEY from the environment
    entries = []
    failed = []

    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
            chunks = _chunk_text(text)
            file_entries = []
            for i, chunk in enumerate(chunks):
                response = client.embeddings.create(model=EMBEDDING_MODEL, input=chunk)
                file_entries.append(
                    {"file": path.name, "chunk": i, "embedding": response.data[0].embedding}
                )
            entries.extend(file_entries)
            note = f", {len(chunks)} chunks" if len(chunks) > 1 else ""
            print(f"{path.name} -> embedded ({len(text)} chars{note})")
        except Exception as e:
            print(f"FAILED: {path.name} ({e})")
            failed.append(path.name)

    OUT_FILE.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(entries)} embeddings to {OUT_FILE}")
    if failed:
        print(f"{len(failed)} file(s) failed to embed:")
        for name in failed:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
