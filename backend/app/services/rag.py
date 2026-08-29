"""Embedding-based retrieval of similar real verdicts, as an alternative to
the static few-shot example set in prompt.py.

Requires ml/data/embeddings.json, built offline by
ml/preprocess/embed_verdicts.py from ml/data/extracted/*.txt. Not built
automatically here — embedding the whole corpus on every import would be
slow and cost money on every backend restart.

Long verdicts are split into multiple chunks by the embedding script (each a
separate cache entry tagged with its source file), since a single embedding
call can't cover a whole long document. A document's relevance to a query is
its single best-matching chunk (max-pooled), so ranking is over documents,
not raw chunks — a long verdict doesn't get an unfair edge just for having
more chunks.
"""

import json
import math
from functools import lru_cache
from pathlib import Path

from openai import OpenAI

from ..schemas import VerdictInput

EMBEDDING_MODEL = "text-embedding-3-small"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXTRACTED_DIR = _REPO_ROOT / "ml" / "data" / "extracted"
_EMBEDDINGS_FILE = _REPO_ROOT / "ml" / "data" / "embeddings.json"


class RagUnavailable(Exception):
    pass


@lru_cache(maxsize=1)
def _load_corpus() -> list[dict]:
    if not _EMBEDDINGS_FILE.exists():
        raise RagUnavailable(
            f"{_EMBEDDINGS_FILE} not found — run "
            "'python ml/preprocess/embed_verdicts.py' first."
        )
    return json.loads(_EMBEDDINGS_FILE.read_text(encoding="utf-8"))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def _embed_query(text: str) -> list[float]:
    client = OpenAI()  # reads OPENAI_API_KEY from the environment
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def _query_text(data: VerdictInput) -> str:
    return (
        f"{data.criminal_offense}\n{data.charge_description}\n{data.established_facts}"
    )


def retrieve_similar(data: VerdictInput, k: int) -> list[str]:
    """Return the text of the k real verdicts most similar to this case."""
    corpus = _load_corpus()
    query_embedding = _embed_query(_query_text(data))

    best_per_file: dict[str, float] = {}
    for entry in corpus:
        sim = _cosine_similarity(query_embedding, entry["embedding"])
        if sim > best_per_file.get(entry["file"], -1.0):
            best_per_file[entry["file"]] = sim

    ranked = sorted(best_per_file.items(), key=lambda pair: pair[1], reverse=True)

    texts = []
    for filename, _ in ranked[:k]:
        texts.append((_EXTRACTED_DIR / filename).read_text(encoding="utf-8"))
    return texts
