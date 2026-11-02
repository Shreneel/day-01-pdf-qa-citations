"""Plain data types shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Page:
    """Text of one PDF page. `page` is 1-based, matching what a reader sees."""

    text: str
    page: int
    source: str


@dataclass(frozen=True)
class Chunk:
    """A retrievable piece of a page. Chunks never span two pages,
    so every chunk maps to exactly one citable page number."""

    text: str
    page: int
    source: str
    chunk_id: str


@dataclass(frozen=True)
class Citation:
    page: int
    source: str
    snippet: str


@dataclass
class Answer:
    question: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    # Pages the model cited that were NOT in the retrieved context (hallucinated citations).
    invalid_pages: list[int] = field(default_factory=list)
    contexts: list[Chunk] = field(default_factory=list)

    @property
    def is_grounded(self) -> bool:
        """True when the answer cites at least one page and every cited page was retrieved."""
        return bool(self.citations) and not self.invalid_pages
