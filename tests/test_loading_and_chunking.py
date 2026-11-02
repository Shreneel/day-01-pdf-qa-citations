import pytest

from pdfqa.chunking import _fallback_split, chunk_pages
from pdfqa.loader import load_pdf
from pdfqa.models import Page


def test_load_pdf_keeps_one_based_page_numbers(sample_pdf):
    pages = load_pdf(sample_pdf)
    assert [p.page for p in pages] == [1, 2, 3, 4, 5, 6]
    assert "20 days of paid time off" in pages[1].text
    assert all(p.source == "handbook.pdf" for p in pages)


def test_load_pdf_accepts_file_objects(sample_pdf):
    with open(sample_pdf, "rb") as f:
        pages = load_pdf(f, source="upload.pdf")
    assert pages[0].source == "upload.pdf"


def test_chunks_never_cross_pages(sample_pdf):
    pages = load_pdf(sample_pdf)
    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=40)
    page_text = {p.page: p.text for p in pages}
    assert len(chunks) > len(pages)
    for chunk in chunks:
        # every word of the chunk comes from its own page
        assert chunk.text.split()[0] in page_text[chunk.page]
        assert len(chunk.text) <= 200


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_pages([Page("text", 1, "x.pdf")], chunk_size=100, chunk_overlap=100)


def test_fallback_splitter_respects_size_and_overlaps():
    text = " ".join(f"Sentence number {i} is here." for i in range(40))
    parts = _fallback_split(text, size=120, overlap=40)
    assert all(len(p) <= 120 for p in parts)
    assert len(parts) > 1
    # consecutive chunks share at least one sentence
    assert parts[0].split(".")[-2].strip() in parts[1]
