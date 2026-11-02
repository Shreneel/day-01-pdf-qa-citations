"""Prompt construction and citation parsing/validation."""

from __future__ import annotations

import re

from pdfqa.models import Chunk, Citation

SYSTEM_PROMPT = """You answer questions about a PDF using ONLY the context excerpts provided.

Rules:
- Every sentence that states a fact must end with a citation of the page it came from, written exactly as [p.N] (for example [p.4]). Use [p.2, 5] when a sentence draws on two pages.
- Cite only page numbers that appear in the context headers.
- If the context does not contain the answer, reply exactly: I couldn't find that in the document.
- Be concise: 1-4 sentences unless the question asks for a list."""

# Matches [p.3], [p. 3], [p3], [p.3, 5], [pp.3-4] style citations.
_CITATION = re.compile(r"\[p{1,2}\.?\s*(\d+(?:\s*[,\-–]\s*\d+)*)\]", re.IGNORECASE)


def format_context(contexts: list[Chunk]) -> str:
    return "\n\n".join(f"[p.{c.page}] ({c.source})\n{c.text}" for c in contexts)


def build_prompt(question: str, contexts: list[Chunk]) -> list[tuple[str, str]]:
    user = f"Context excerpts:\n\n{format_context(contexts)}\n\nQuestion: {question}"
    return [("system", SYSTEM_PROMPT), ("human", user)]


def parse_cited_pages(text: str) -> list[int]:
    """Return cited page numbers in first-seen order. Ranges like 3-5 expand to 3, 4, 5."""
    pages: list[int] = []
    for match in _CITATION.finditer(text):
        for part in re.split(r"\s*,\s*", match.group(1)):
            bounds = [int(x) for x in re.split(r"\s*[\-–]\s*", part)]
            span = range(bounds[0], bounds[-1] + 1) if len(bounds) == 2 and bounds[1] >= bounds[0] else bounds[:1]
            for p in span:
                if p not in pages:
                    pages.append(p)
    return pages


def resolve_citations(text: str, contexts: list[Chunk], snippet_chars: int = 240) -> tuple[list[Citation], list[int]]:
    """Map cited pages to retrieved chunks. Returns (valid citations, cited pages not in context)."""
    by_page: dict[int, Chunk] = {}
    for chunk in contexts:  # contexts are ranked; keep the best chunk per page
        by_page.setdefault(chunk.page, chunk)
    citations, invalid = [], []
    for page in parse_cited_pages(text):
        chunk = by_page.get(page)
        if chunk is None:
            invalid.append(page)
            continue
        snippet = chunk.text if len(chunk.text) <= snippet_chars else chunk.text[:snippet_chars].rsplit(" ", 1)[0] + "…"
        citations.append(Citation(page=page, source=chunk.source, snippet=snippet))
    return citations, invalid
