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

# Name redaction anchored on the role keywords that consistently introduce a
# person's name in these verdicts. Matches 1-3 name-like words right after
# the keyword, in either "Title Case" or "ALL CAPS" form (Serbian verdicts
# commonly write party/judge names in all caps), including hyphenated
# surnames (e.g. TOMIĆ-JOKIĆ). Keywords are matched as stems (\w*) so a
# single rule covers all grammatical cases (okrivljenog/okrivljenom/...).
#
# This is a heuristic, not real NER — it only catches names immediately
# after a recognized keyword, so names introduced in other word orders
# (e.g. "X i Y, advokati") or without a nearby keyword will slip through.
# It reduces manual redaction work; it does not replace the manual review
# step described in the module docstring.
_UPPER = "А-ЯЂЈЉЊЋЏ"
_LOWER = "а-яђјљњћџ"
_NAME_WORD = rf"(?:[{_UPPER}][{_LOWER}]+|[{_UPPER}]{{2,}})(?:-(?:[{_UPPER}][{_LOWER}]+|[{_UPPER}]{{2,}}))?"
_NAME = rf"{_NAME_WORD}(?:\s{_NAME_WORD}){{0,2}}"
_ROLE_STEMS = [
    "окривљен",
    "оптужен",
    "оштећен",
    "сведок",
    "поротник",
    "бранил", "адвокат",
    "судиј",
    "записничар",
]
RULES += [
    # "(?i:...)" makes only the keyword stem case-insensitive (verdicts mix
    # Title Case body text with ALL CAPS headers like "ОПТУЖЕНИ:") — the name
    # part stays case-sensitive, since requiring a capital letter is exactly
    # what tells a name apart from an ordinary lowercase word.
    # ":?\s+" (not just a single space) so this also catches the header
    # style, where the name follows on its own line after blank lines.
    (rf"(\b(?i:{stem})\w{{0,4}}:?\s+){_NAME}", r"\1[IME]") for stem in _ROLE_STEMS
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
