"""Load a PDF into per-page text, keeping page numbers for citations."""

from __future__ import annotations

import re
from pathlib import Path
from typing import BinaryIO

from pypdf import PdfReader

from pdfqa.models import Page


def _clean(text: str) -> str:
    text = text.replace("­", "")  # soft hyphens
    text = re.sub(r"-\n(\w)", r"\1", text)  # re-join words hyphenated across lines
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # join lines wrapped mid-paragraph
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(file: str | Path | BinaryIO, source: str | None = None) -> list[Page]:
    """Return one Page per non-empty PDF page (1-based page numbers).

    `file` can be a path or a binary file object (e.g. a Streamlit upload).
    """
    if source is None:
        source = Path(file).name if isinstance(file, (str, Path)) else getattr(file, "name", "document.pdf")
    reader = PdfReader(file)
    pages: list[Page] = []
    for number, pdf_page in enumerate(reader.pages, start=1):
        text = _clean(pdf_page.extract_text() or "")
        if text:
            pages.append(Page(text=text, page=number, source=str(source)))
    if not pages:
        raise ValueError(
            f"No extractable text in {source}. It may be a scanned PDF; run OCR first (e.g. ocrmypdf)."
        )
    return pages
