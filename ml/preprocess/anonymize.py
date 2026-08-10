"""Anonymize extracted verdict texts before using them as examples,
fine-tuning data, or sending them to any external API.

IMPORTANT: real verdicts contain personal data (names, JMBG, addresses).
This step is mandatory before anything leaves ml/data/raw|extracted.

This is a starting point — regex rules for obviously structured data,
to be extended with a name/entity list or an NER model for Serbian.
Manually review the output before use.
"""

import re
from pathlib import Path

EXTRACTED_DIR = Path(__file__).resolve().parent.parent / "data" / "extracted"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "anonymized"

RULES: list[tuple[str, str]] = [
    (r"\b\d{13}\b", "[JMBG]"),                              # JMBG
    (r"\b\d{3}-\d{7,13}-\d{2}\b", "[BROJ RACUNA]"),         # bank account
    (r"\b(?:\+381|0)6\d[\s/-]?\d{3}[\s/-]?\d{3,4}\b", "[TELEFON]"),
    (r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]"),
]


def anonymize(text: str) -> str:
    for pattern, replacement in RULES:
        text = re.sub(pattern, replacement, text)
    return text


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for txt_path in sorted(EXTRACTED_DIR.glob("*.txt")):
        anonymized = anonymize(txt_path.read_text(encoding="utf-8"))
        out_path = OUT_DIR / txt_path.name
        out_path.write_text(anonymized, encoding="utf-8")
        print(f"{txt_path.name} -> {out_path}")


if __name__ == "__main__":
    main()
