"""Settings read from environment variables (and a .env file if python-dotenv is installed)."""

from __future__ import annotations

import os
from dataclasses import dataclass

try:  # optional
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


@dataclass(frozen=True)
class Settings:
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4
    # "auto" uses Chroma when langchain-chroma is installed, otherwise a TF-IDF retriever.
    retriever: str = "auto"
    # "auto" picks Anthropic, then OpenAI, based on which API key is set; otherwise offline.
    llm_provider: str = "auto"
    llm_model: str | None = None
    embedding_model: str = "text-embedding-3-small"
    persist_dir: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            chunk_size=_int("CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_int("CHUNK_OVERLAP", cls.chunk_overlap),
            top_k=_int("TOP_K", cls.top_k),
            retriever=os.getenv("RETRIEVER", cls.retriever).lower(),
            llm_provider=os.getenv("LLM_PROVIDER", cls.llm_provider).lower(),
            llm_model=os.getenv("LLM_MODEL") or None,
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            persist_dir=os.getenv("CHROMA_DIR") or None,
        )
