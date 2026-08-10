"""Extract plain text from .docx verdict documents in ml/data/raw/.

Usage:
    python extract_docx.py

Writes one .txt file per verdict into ml/data/extracted/.
"""

from pathlib import Path

from docx import Document

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "extracted"


def extract_text(path: Path) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for docx_path in sorted(RAW_DIR.glob("*.docx")):
        text = extract_text(docx_path)
        out_path = OUT_DIR / (docx_path.stem + ".txt")
        out_path.write_text(text, encoding="utf-8")
        print(f"{docx_path.name} -> {out_path.name} ({len(text)} chars)")


if __name__ == "__main__":
    main()
