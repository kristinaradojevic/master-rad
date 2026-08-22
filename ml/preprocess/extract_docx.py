"""Extract plain text from .docx/.doc verdict documents in ml/data/raw/.

Usage:
    python extract_docx.py

Writes one .txt file per verdict into ml/data/extracted/.
"""

import subprocess
from pathlib import Path

from docx import Document

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "extracted"


def extract_text(path: Path) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def extract_legacy_doc(path: Path, out_path: Path) -> None:
    # python-docx can't open the old binary .doc format at all. macOS ships
    # `textutil`, which can convert it directly — no extra dependency needed.
    # (This step is macOS-only.)
    subprocess.run(
        ["textutil", "-convert", "txt", str(path), "-output", str(out_path)],
        check=True,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for docx_path in sorted(RAW_DIR.glob("*.docx")):
        text = extract_text(docx_path)
        out_path = OUT_DIR / (docx_path.stem + ".txt")
        out_path.write_text(text, encoding="utf-8")
        print(f"{docx_path.name} -> {out_path.name} ({len(text)} chars)")

    for doc_path in sorted(RAW_DIR.glob("*.doc")):
        out_path = OUT_DIR / (doc_path.stem + ".txt")
        extract_legacy_doc(doc_path, out_path)
        print(f"{doc_path.name} -> {out_path.name} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
