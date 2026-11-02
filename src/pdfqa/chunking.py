"""Split pages into chunks. Chunks are split *within* a page so each keeps one page number."""

from __future__ import annotations

import re

from pdfqa.models import Chunk, Page

_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _fallback_split(text: str, size: int, overlap: int) -> list[str]:
    """Greedy splitter used when langchain-text-splitters isn't installed.

    Packs sentences into chunks up to `size` characters, carrying roughly `overlap`
    characters of trailing context into the next chunk.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if s.strip()]
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for sentence in sentences:
        while len(sentence) > size:  # hard-wrap very long sentences
            chunks.append(sentence[:size])
            sentence = sentence[size - overlap :]
        if current and length + len(sentence) + 1 > size:
            chunks.append(" ".join(current))
            tail: list[str] = []
            tail_len = 0
            for prev in reversed(current):
                if tail_len + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev) + 1
            current, length = tail, tail_len
        current.append(sentence)
        length += len(sentence) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def _splitter(size: int, overlap: int):
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        return lambda text: _fallback_split(text, size, overlap)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, separators=_SEPARATORS
    )
    return splitter.split_text


def chunk_pages(pages: list[Page], chunk_size: int = 800, chunk_overlap: int = 120) -> list[Chunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    split = _splitter(chunk_size, chunk_overlap)
    chunks: list[Chunk] = []
    for page in pages:
        for i, text in enumerate(split(page.text)):
            if text.strip():
                chunks.append(
                    Chunk(text=text.strip(), page=page.page, source=page.source, chunk_id=f"p{page.page}-c{i}")
                )
    return chunks
