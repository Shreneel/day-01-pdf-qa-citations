"""Answer generators: Ollama (local), Groq (free hosted), Claude or OpenAI via LangChain,
or an offline extractive fallback."""

from __future__ import annotations

import os
import re
import urllib.request
from typing import Protocol

from pdfqa.models import Chunk
from pdfqa.retrieval import tokenize

DEFAULT_MODELS = {
    "ollama": "llama3.2",
    "groq": "llama-3.3-70b-versatile",
    "anthropic": "claude-sonnet-5-5",
    "openai": "gpt-4.1-mini",
}
PROVIDERS = ("auto", "ollama", "groq", "anthropic", "openai", "offline")


def ollama_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def ollama_available(timeout: float = 1.0) -> bool:
    """True when an Ollama server answers at OLLAMA_BASE_URL."""
    try:
        with urllib.request.urlopen(f"{ollama_url()}/api/tags", timeout=timeout) as resp:
            return resp.status == 200
    except (OSError, ValueError):
        return False
NOT_FOUND = "I couldn't find that in the document."


class AnswerModel(Protocol):
    name: str

    def answer(self, question: str, contexts: list[Chunk], prompt: list[tuple[str, str]]) -> str: ...


class ExtractiveModel:
    """Offline fallback: returns the context sentences that best overlap the question,
    each tagged with its page. Lets the app, tests and CI run without an API key."""

    name = "extractive (offline)"

    def __init__(self, max_sentences: int = 2):
        self.max_sentences = max_sentences

    def answer(self, question: str, contexts: list[Chunk], prompt=None) -> str:
        q_terms = set(tokenize(question))
        if not q_terms:
            return NOT_FOUND
        scored: list[tuple[float, int, str, int]] = []
        for rank, chunk in enumerate(contexts):
            for sentence in re.split(r"(?<=[.!?])\s+", chunk.text):
                terms = set(tokenize(sentence))
                overlap = len(q_terms & terms)
                if overlap:
                    # Favor overlap, then earlier (better-ranked) chunks, then shorter sentences.
                    score = overlap / len(q_terms) - 0.01 * rank - 0.0005 * len(sentence)
                    scored.append((score, chunk.page, sentence.strip(), rank))
        if not scored:
            return NOT_FOUND
        scored.sort(key=lambda s: s[0], reverse=True)
        picked, seen = [], set()
        for _, page, sentence, _ in scored:
            if sentence not in seen:
                picked.append(f"{sentence} [p.{page}]")
                seen.add(sentence)
            if len(picked) == self.max_sentences:
                break
        return " ".join(picked)


class ChatModel:
    """Wraps a LangChain chat model: ChatOllama, ChatGroq, ChatAnthropic or ChatOpenAI."""

    def __init__(self, provider: str, model: str | None = None, temperature: float = 0.0):
        model = model or DEFAULT_MODELS[provider]
        if provider == "ollama":
            from langchain_ollama import ChatOllama

            self.llm = ChatOllama(model=model, base_url=ollama_url(), temperature=temperature)
        elif provider == "groq":
            from langchain_groq import ChatGroq

            self.llm = ChatGroq(model=model, temperature=temperature, max_tokens=800)
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            self.llm = ChatAnthropic(model=model, temperature=temperature, max_tokens=800)
        elif provider == "openai":
            from langchain_openai import ChatOpenAI

            self.llm = ChatOpenAI(model=model, temperature=temperature)
        else:
            raise ValueError(f"Unknown provider '{provider}'")
        self.name = f"{provider}:{model}"

    def answer(self, question: str, contexts: list[Chunk], prompt: list[tuple[str, str]]) -> str:
        content = self.llm.invoke(prompt).content
        if isinstance(content, list):  # some providers return content blocks
            content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
        return str(content).strip()


def resolve_provider(provider: str = "auto") -> str:
    """Pick a provider. "auto" order: Groq key, Anthropic key, OpenAI key, running Ollama, offline."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use one of: {', '.join(PROVIDERS)}")
    if provider != "auto":
        return provider
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if ollama_available():
        return "ollama"
    return "offline"


def get_model(provider: str = "auto", model: str | None = None) -> AnswerModel:
    provider = resolve_provider(provider)
    if provider == "offline":
        return ExtractiveModel()
    return ChatModel(provider, model)
