# Generator presuda (master rad)

Web app in which a judge enters structured information about a case, an LLM
generates a full verdict draft in Serbian, the judge reviews and edits it,
and downloads the final document as a PDF. Multiple LLMs can be plugged in
and compared for the thesis evaluation.

## Structure

```
backend/    FastAPI — prompt building, LLM providers, PDF rendering (WeasyPrint)
frontend/   Next.js — the judge's form, verdict review/editing, PDF download
ml/         Data preparation (docx extraction, anonymization) and model comparison
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
3. `python ml/preprocess/anonymize.py` → `ml/data/anonymized/` (review manually!).
4. Use anonymized texts as few-shot examples in `backend/app/services/prompt.py`,
   or later as fine-tuning data.

## Model comparison (thesis experiments)

Define inputs in `ml/evaluation/test_cases.json`, then:

```bash
python ml/evaluation/compare_models.py
```

Outputs land in `ml/evaluation/results/<timestamp>/`, one JSON per
(case, model) pair, including generation time.
