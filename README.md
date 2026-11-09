# Generator presuda (master rad)

Web app in which a judge enters structured information about a case, an LLM
generates a full verdict draft in Serbian, the judge reviews and edits it,
and downloads the final document as a PDF. Multiple LLMs can be plugged in
and compared for the thesis evaluation.

## Quickstart — sve komande po redu

Redosled od praznog checkout-a do radne aplikacije sa RAG-om i fine-tuned
modelom. Sve `python`/`pip` komande ispod pretpostavljaju da je
`backend/.venv` aktiviran (`source backend/.venv/bin/activate`) — to je
venv koji ima i FastAPI zavisnosti i sve ml/finetuning dodatke.

```bash
# 1) Backend — instalacija i konfiguracija
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # upiši OPENAI_API_KEY / ANTHROPIC_API_KEY
cd ..

# 2) Frontend — instalacija
cd frontend && npm install && cd ..

# 3) Pokretanje aplikacije (dva terminala)
cd backend && uvicorn app.main:app --reload     # http://localhost:8000
cd frontend && npm run dev                       # http://localhost:3000
```

To je dovoljno za generisanje presuda sa bilo kojim konfigurisanim modelom
(bez RAG-a i fine-tuninga). Za pun tok rada za tezu (RAG + fine-tuning),
nastavi redom:

```bash
# 4) Pripremi realne presude (jednom, ili kad se korpus promeni)
#    .docx fajlove prvo stavi u ml/data/raw/ (git-ignored)
python ml/preprocess/extract_docx.py
python ml/preprocess/split_corpus.py

# 5) RAG — embeduj samo RAG pool (23 od 114 presuda)
python ml/preprocess/embed_verdicts.py

# 6) Fine-tuning — vidi detaljno objašnjenje i hardverske napomene u
#    sekciji "Fine-tuning" ispod pre nego što ovo pokreneš
pip install -r ml/requirements.txt
python ml/preprocess/extract_structured_fields.py
python ml/preprocess/prepare_finetune_data.py
python ml/finetune/train_lora.py

# 7) Posluži fine-tuned model (ostavi da radi u svom terminalu)
#    mlx_lm.server je unutar venv-a, ne globalno — koristi puni put ili
#    aktiviraj venv (source backend/.venv/bin/activate) pre poziva
backend/.venv/bin/mlx_lm.server --model ml/data/finetune/fused_model --port 8081 --max-tokens 8000
#    zatim u backend/.env dodaj: LOCAL_MLX_BASE_URL=http://localhost:8081/v1
#    i restartuj uvicorn (korak 3) da se model pojavi u /api/models
```

### Svakodnevno pokretanje (model je već istreniran)

Koraci 4-6 su jednokratni (već odrađeni, ne treba ih ponavljati). Kad samo
želiš da koristiš aplikaciju sa već istreniranim fine-tuned modelom, treba
pokrenuti ova tri procesa, svaki u svom terminalu:

```bash
# terminal 1 — lokalni MLX server (fine-tuned model)
cd "master-rad" && backend/.venv/bin/mlx_lm.server --model ml/data/finetune/fused_model --port 8081 --max-tokens 8000

# terminal 2 — backend (LOCAL_MLX_BASE_URL mora već biti u backend/.env)
cd "master-rad/backend" && source .venv/bin/activate && uvicorn app.main:app --reload

# terminal 3 — frontend
cd "master-rad/frontend" && npm run dev
```

Zatim otvori http://localhost:3000 — "Qwen 2.5 7B (fine-tuned, local)" je
odmah dostupan u listi modela.

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

OpenAI shut down self-serve fine-tuning in 2026, so this trains locally
instead: LoRA fine-tuning of an open-weight model
(`mlx-community/Qwen2.5-7B-Instruct-4bit`) using Apple's MLX framework on
the fine-tuning half of the corpus (`split_corpus.py`'s `finetune` list).
**Requires Apple Silicon** — MLX uses the Mac's GPU directly, there's no
cloud/CUDA path here.

Real verdicts only exist as full text, not as the structured fields a judge
would type into the form, so the pipeline first reverse-extracts those
fields with an LLM before building (input → verdict) training pairs.
Verdicts also run far longer than fits in one training example on this
machine (median ~8800 tokens vs. a max_seq_length of 4096 — higher values
ran out of GPU memory), so long ones are split into several chunked
examples rather than truncated, each still covering the full verdict text
across chunks (see `prepare_finetune_data.py`'s docstring for the exact
numbers and tradeoffs). Because of that, this model isn't served with a
single call like the other providers — `LocalMLXProvider` (in
`backend/app/services/llm/local_mlx_provider.py`) automatically asks for a
continuation, using the same prompt format the chunked training data used,
whenever a generated verdict looks cut off:

```bash
pip install -r ml/requirements.txt                  # mlx-lm, Apple Silicon only

python ml/preprocess/extract_structured_fields.py    # LLM extracts VerdictInput fields per verdict
python ml/preprocess/prepare_finetune_data.py        # builds train.jsonl / valid.jsonl
python ml/finetune/train_lora.py                     # trains locally, then fuses the adapter
```

The last step downloads the ~4-5GB base model on first run, trains for a
while (minutes to a few hours depending on hyperparameters — see the
comments at the top of `train_lora.py`), and prints the fused model's path.
Serve it with:

```bash
backend/.venv/bin/mlx_lm.server --model ml/data/finetune/fused_model --port 8081 --max-tokens 8000
```

(leave this running in its own terminal), then add to `backend/.env`:

```
LOCAL_MLX_BASE_URL=http://localhost:8081/v1
```

It then appears in `/api/models` as "Qwen 2.5 7B (fine-tuned, local)" like
any other model, selectable from the frontend the same way as the RAG
toggle — no code changes needed, since `mlx_lm.server` speaks the same
OpenAI chat-completions protocol the app already uses.

## Model comparison (thesis experiments)

Define inputs in `ml/evaluation/test_cases.json`, then:

```bash
python ml/evaluation/compare_models.py
```

Outputs land in `ml/evaluation/results/<timestamp>/`, one JSON per
(case, model) pair, including generation time.
