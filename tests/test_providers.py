import pytest

from pdfqa import llm

# Captured at import, before the autouse fixture replaces it with a stub.
REAL_OLLAMA_CHECK = llm.ollama_available


def test_auto_falls_back_to_offline_with_no_keys_and_no_ollama():
    assert llm.resolve_provider("auto") == "offline"


def test_auto_uses_ollama_when_it_is_running(monkeypatch):
    monkeypatch.setattr(llm, "ollama_available", lambda timeout=1.0: True)
    assert llm.resolve_provider("auto") == "ollama"


@pytest.mark.parametrize(
    "env,expected",
    [
        ({"GROQ_API_KEY": "x", "ANTHROPIC_API_KEY": "y"}, "groq"),
        ({"ANTHROPIC_API_KEY": "y", "OPENAI_API_KEY": "z"}, "anthropic"),
        ({"OPENAI_API_KEY": "z"}, "openai"),
    ],
)
def test_auto_prefers_keys_in_order(monkeypatch, env, expected):
    monkeypatch.setattr(llm, "ollama_available", lambda timeout=1.0: True)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert llm.resolve_provider("auto") == expected


def test_explicit_provider_wins(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    assert llm.resolve_provider("ollama") == "ollama"


def test_unknown_provider_rejected():
    with pytest.raises(ValueError):
        llm.resolve_provider("gemini")


def test_ollama_check_is_false_when_nothing_listens(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    assert REAL_OLLAMA_CHECK(timeout=0.5) is False
