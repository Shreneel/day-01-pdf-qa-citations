"""End-to-end pipeline: PDF -> pages -> chunks -> retriever -> cited answer."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from pdfqa.chunking import chunk_pages
from pdfqa.citations import build_prompt, resolve_citations
from pdfqa.config import Settings
from pdfqa.llm import NOT_FOUND, AnswerModel, get_model
from pdfqa.loader import load_pdf
from pdfqa.models import Answer, Chunk, Page
from pdfqa.retrieval import Retriever, build_retriever


class PDFQA:
    def __init__(self, pages: list[Page], settings: Settings | None = None,
                 retriever: Retriever | None = None, model: AnswerModel | None = None):
        self.settings = settings or Settings.from_env()
        self.pages = pages
        self.chunks: list[Chunk] = chunk_pages(pages, self.settings.chunk_size, self.settings.chunk_overlap)
        self.retriever = retriever or build_retriever(
            self.chunks, self.settings.retriever, self.settings.persist_dir, self.settings.embedding_model
        )
        self.model = model or get_model(self.settings.llm_provider, self.settings.llm_model)

    @classmethod
    def from_pdf(cls, file: str | Path | BinaryIO, settings: Settings | None = None, **kwargs) -> "PDFQA":
        return cls(load_pdf(file), settings=settings, **kwargs)

    def ask(self, question: str, k: int | None = None) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("Question is empty")
        hits = self.retriever.search(question, k or self.settings.top_k)
        contexts = [chunk for chunk, _ in hits]
        if not contexts:
            return Answer(question=question, text=NOT_FOUND)
        text = self.model.answer(question, contexts, build_prompt(question, contexts))
        citations, invalid = resolve_citations(text, contexts)
        return Answer(question=question, text=text, citations=citations, invalid_pages=invalid, contexts=contexts)
