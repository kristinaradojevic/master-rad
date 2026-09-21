# Generator presuda (master rad)

Web app in which a judge enters structured information about a case, an LLM
generates a full verdict draft in Serbian, the judge reviews and edits it,
and downloads the final document as a PDF. Multiple LLMs can be plugged in
and compared for the thesis evaluation.

## Structure

```
backend/    FastAPI — prompt building, LLM providers, PDF rendering (WeasyPrint)
frontend/   Next.js — the judge's form, verdict review/editing, PDF download
ml/         Data preparation (docx extraction) and model comparison
docs/       Thesis notes, prompt experiments
```

## Running locally

### Backend (Python 3.11+)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your API keys
uvicorn app.main:app --reload
```

API on http://localhost:8000 (interactive docs at /docs).

Note: WeasyPrint needs system libraries on macOS: `brew install pango`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

App on http://localhost:3000.

## Adding a model to compare

Add an entry in `backend/app/services/llm/registry.py`. Models whose API key
is missing from `.env` are hidden automatically.

## Working with real verdicts

1. Put `.docx` files in `ml/data/raw/` (git-ignored — never commit them).
2. `python ml/preprocess/extract_docx.py` → plain text in `ml/data/extracted/`.
3. `python ml/preprocess/split_corpus.py` → splits the corpus 80/20 into a
   fine-tuning set and a RAG pool (`ml/data/split.json`), so the two methods
   are never evaluated on overlapping examples.
4. Use the RAG pool as few-shot examples (`ml/preprocess/embed_verdicts.py`,
   see below) and the fine-tuning set to train a custom model (see
   Fine-tuning below).

## Fine-tuning (thesis experiments)

Trains a custom `gpt-4o-mini` on the fine-tuning half of the corpus
(`split_corpus.py`'s `finetune` list), so it can be compared against plain
prompting and RAG for the same cases.

Real verdicts only exist as full text, not as the structured fields a judge
would type into the form, so the pipeline first reverse-extracts those
fields with an LLM before building (input → verdict) training pairs:

```bash
python ml/preprocess/extract_structured_fields.py   # LLM extracts VerdictInput fields per verdict
python ml/preprocess/prepare_finetune_data.py        # builds train.jsonl / valid.jsonl
python ml/preprocess/run_finetune_job.py             # uploads + trains, polls until done
```

When the job succeeds, copy the printed model id into `FINETUNED_MODEL_ID`
in `backend/.env` — it then appears in `/api/models` as
"GPT-4o mini (fine-tuned)" like any other model, selectable from the
frontend the same way as the RAG toggle.

## Model comparison (thesis experiments)

Define inputs in `ml/evaluation/test_cases.json`, then:

```bash
python ml/evaluation/compare_models.py
```

Outputs land in `ml/evaluation/results/<timestamp>/`, one JSON per
(case, model) pair, including generation time.
