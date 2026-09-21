"""Reverse-extract structured VerdictInput-like fields from real verdict
texts, for the fine-tuning set only (see split_corpus.py).

Real verdicts exist only as full text (ml/data/extracted/*.txt) — there's no
record of the structured facts a judge would have typed into the form. To
build (input -> full verdict) training pairs for fine-tuning, an LLM reads
each verdict and extracts the fields back out. This introduces some
extraction noise, but is the only way to get pairs at all without manually
re-keying ~90 verdicts.

Requires ml/data/split.json (run split_corpus.py first). Writes one JSON
file per verdict to ml/data/structured/, skipping files already extracted
so a failed or interrupted run can be resumed cheaply — pass --force to
redo everything.

Usage:
    python ml/preprocess/extract_structured_fields.py [--force]
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(REPO_ROOT / "backend" / ".env")

EXTRACTED_DIR = REPO_ROOT / "ml" / "data" / "extracted"
STRUCTURED_DIR = REPO_ROOT / "ml" / "data" / "structured"
SPLIT_FILE = REPO_ROOT / "ml" / "data" / "split.json"
EXTRACTION_MODEL = "gpt-4o-mini"

EXTRACTION_SYSTEM_PROMPT = """\
Iz teksta srpske krivične presude izvuci tražena polja u JSON formatu.
Koristi isključivo podatke koji su izričito navedeni u tekstu — ne izmišljaj
i ne dopunjuj ništa. Ako neko polje zaista nije navedeno u presudi (osim
defendant_additional_info, koje sme biti prazan string), upiši najbolju
razumnu vrednost izvedenu iz konteksta presude, nikad izmišljenu osobu ili
događaj."""

VERDICT_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "court_name": {"type": "string", "description": "Naziv suda, npr. 'Osnovni sud u Novom Sadu'"},
        "case_number": {"type": "string", "description": "Broj predmeta, npr. 'K 45/2026'"},
        "judge_name": {"type": "string", "description": "Ime i prezime sudije koji je izrekao presudu"},
        "court_reporter_name": {"type": "string", "description": "Ime i prezime zapisničara"},
        "verdict_date": {
            "type": "string",
            "description": "Datum donošenja presude u ISO obliku YYYY-MM-DD",
        },
        "defendant_name": {"type": "string", "description": "Ime i prezime okrivljenog"},
        "defendant_jmbg": {
            "type": "string",
            "description": "JMBG okrivljenog, tačno 13 cifara, bez razmaka i crtica",
        },
        "defendant_residence": {"type": "string", "description": "Prebivalište okrivljenog"},
        "defendant_additional_info": {
            "type": "string",
            "description": (
                "Dodatni lični podaci o okrivljenom (zanimanje, mesto rođenja, "
                "porodično stanje, ranija osuđivanost) ako se pominju, inače prazan string"
            ),
        },
        "criminal_offense": {
            "type": "string",
            "description": "Krivično delo i zakonska kvalifikacija, npr. 'čl. 246a st. 1 KZ'",
        },
        "charge_description": {
            "type": "string",
            "description": "Činjenični opis dela iz optužnog akta, kako je naveden u presudi",
        },
        "established_facts": {
            "type": "string",
            "description": "Utvrđeno činjenično stanje i ocena dokaza iz obrazloženja presude",
        },
        "decision_type": {
            "type": "string",
            "enum": ["osuđujuća", "oslobađajuća", "odbijajuća"],
        },
        "sentence": {
            "type": "string",
            "description": "Izrečena kazna/mera, ili obrazloženje oslobađanja/odbijanja",
        },
    },
    "required": [
        "court_name",
        "case_number",
        "judge_name",
        "court_reporter_name",
        "verdict_date",
        "defendant_name",
        "defendant_jmbg",
        "defendant_residence",
        "defendant_additional_info",
        "criminal_offense",
        "charge_description",
        "established_facts",
        "decision_type",
        "sentence",
    ],
    "additionalProperties": False,
}


def _extract_one(client: OpenAI, text: str) -> dict:
    response = client.chat.completions.create(
        model=EXTRACTION_MODEL,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "verdict_input", "schema": VERDICT_INPUT_SCHEMA, "strict": True},
        },
    )
    return json.loads(response.choices[0].message.content)


def main():
    force = "--force" in sys.argv

    if not SPLIT_FILE.exists():
        sys.exit(f"{SPLIT_FILE} not found — run split_corpus.py first.")
    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    finetune_files = split["finetune"]

    STRUCTURED_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI()  # reads OPENAI_API_KEY from the environment
    failed = []
    skipped = 0

    for name in finetune_files:
        out_path = STRUCTURED_DIR / (Path(name).stem + ".json")
        if out_path.exists() and not force:
            skipped += 1
            continue
        try:
            text = (EXTRACTED_DIR / name).read_text(encoding="utf-8")
            fields = _extract_one(client, text)
            out_path.write_text(json.dumps(fields, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{name} -> {out_path.name}")
        except Exception as e:
            print(f"FAILED: {name} ({e})")
            failed.append(name)

    print(f"\n{len(finetune_files) - len(failed) - skipped} extracted, {skipped} skipped (already done)")
    if failed:
        print(f"{len(failed)} file(s) failed to extract:")
        for name in failed:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
