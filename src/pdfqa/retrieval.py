"""Retrievers: Chroma vector search (LangChain) with a dependency-free TF-IDF fallback."""

from __future__ import annotations

import hashlib
import math
import os
import re
import uuid
from collections import Counter
from typing import Protocol

import numpy as np

from pdfqa.models import Chunk

STOPWORDS = frozenset(
    """a an and are as at be by can do does for from has have how i if in is it its many
    much of on or our should that the their this to was we what when where which who why
    will with you your""".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        """Return up to k (chunk, score) pairs, best first. Higher score = more relevant."""
        ...


class TfidfRetriever:
    """Keyword retriever with sublinear TF-IDF and cosine similarity. No network, no model."""

    name = "tfidf"

    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("No chunks to index")
        self.chunks = chunks
        docs = [Counter(tokenize(c.text)) for c in chunks]
        df = Counter(term for doc in docs for term in doc)
        self.vocab = {term: i for i, term in enumerate(sorted(df))}
        n = len(chunks)
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1 for t in sorted(df)])
        self.matrix = np.vstack([self._vector(doc) for doc in docs])

    def _vector(self, counts: Counter) -> np.ndarray:
        vec = np.zeros(len(self.vocab))
        for term, count in counts.items():
            idx = self.vocab.get(term)
            if idx is not None:
                vec[idx] = 1 + math.log(count)
        vec *= self.idf
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        q = self._vector(Counter(tokenize(query)))
        if not q.any():
            return []
        scores = self.matrix @ q
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]


def hash_embedding(text: str, dim: int = 384) -> list[float]:
    """Deterministic bag-of-words hashing embedding (unigrams + bigrams).

    Used as an offline stand-in for a real embedding model so the Chroma path
    works in CI without an API key. Not semantic, but keyword-faithful.
    """
    tokens = tokenize(text)
    vec = np.zeros(dim)
    for feature in tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]:
        h = int.from_bytes(hashlib.md5(feature.encode()).digest()[:8], "little")
        vec[h % dim] += 1.0 if (h >> 63) & 1 == 0 else -1.0
    norm = np.linalg.norm(vec)
    return (vec / norm if norm else vec).tolist()


def get_embeddings(model: str = "text-embedding-3-small"):
    """OpenAI embeddings when OPENAI_API_KEY is set, otherwise offline hashing embeddings."""
    from langchain_core.embeddings import Embeddings

    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model)

    class HashEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [hash_embedding(t) for t in texts]

        def embed_query(self, text: str) -> list[float]:
            return hash_embedding(text)

    return HashEmbeddings()


class ChromaRetriever:
    """Vector search over a Chroma collection built with LangChain."""

    name = "chroma"

    def __init__(self, chunks: list[Chunk], embeddings=None, persist_dir: str | None = None,
                 embedding_model: str = "text-embedding-3-small"):
        from langchain_chroma import Chroma
        from langchain_core.documents import Document

        if not chunks:
            raise ValueError("No chunks to index")
        self.embeddings = embeddings or get_embeddings(embedding_model)
        docs = [
            Document(page_content=c.text, metadata={"page": c.page, "source": c.source, "chunk_id": c.chunk_id})
            for c in chunks
        ]
        self.store = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            collection_name=f"pdfqa-{uuid.uuid4().hex[:8]}",
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        results = self.store.similarity_search_with_score(query, k=k)
        out = []
        for doc, distance in results:
            meta = doc.metadata
            chunk = Chunk(text=doc.page_content, page=int(meta["page"]), source=str(meta["source"]),
                          chunk_id=str(meta.get("chunk_id", "")))
            out.append((chunk, 1.0 - float(distance)))  # cosine distance -> similarity
        return out


def chroma_available() -> bool:
    try:
        import langchain_chroma  # noqa: F401
        import langchain_core  # noqa: F401
    except ImportError:
        return False
    return True


def build_retriever(chunks: list[Chunk], kind: str = "auto", persist_dir: str | None = None,
                    embedding_model: str = "text-embedding-3-small") -> Retriever:
    if kind == "auto":
        kind = "chroma" if chroma_available() else "tfidf"
    if kind == "chroma":
        return ChromaRetriever(chunks, persist_dir=persist_dir, embedding_model=embedding_model)
    if kind == "tfidf":
        return TfidfRetriever(chunks)
    raise ValueError(f"Unknown retriever '{kind}'. Use auto, chroma or tfidf.")
