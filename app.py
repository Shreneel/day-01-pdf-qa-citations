"""Streamlit UI. Run locally: streamlit run app.py · Deploys as-is to Streamlit Community Cloud."""

from __future__ import annotations

import io
import os
import sys
from dataclasses import replace
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

# On Streamlit Cloud, keys live in st.secrets. Copy them into the environment
# so the rest of the code reads one place. Local .env / environment variables still win.
try:
    for key in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LLM_PROVIDER", "LLM_MODEL", "OLLAMA_BASE_URL"):
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = str(st.secrets[key])
except Exception:  # no secrets.toml locally is fine
    pass

from pdfqa import PDFQA  # noqa: E402
from pdfqa.config import Settings  # noqa: E402

SAMPLE = Path(__file__).parent / "data" / "sample" / "northwind_handbook.pdf"
MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS_PER_SESSION", "20"))  # protects free API quota on a public demo

st.set_page_config(page_title="PDF Q&A with citations", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Reading and indexing the PDF…")
def build_index(pdf_bytes: bytes, name: str, top_k: int) -> PDFQA:
    file = io.BytesIO(pdf_bytes)
    file.name = name
    settings = replace(Settings.from_env(), top_k=top_k)
    return PDFQA.from_pdf(file, settings=settings)


with st.sidebar:
    st.header("Document")
    upload = st.file_uploader("Upload a PDF", type=["pdf"])
    top_k = st.slider("Chunks to retrieve", 2, 10, 4)
    if upload is None:
        st.caption("No upload yet, using the sample Northwind Robotics handbook.")

pdf_bytes, name = (upload.getvalue(), upload.name) if upload else (SAMPLE.read_bytes(), SAMPLE.name)
qa = build_index(pdf_bytes, name, top_k)

st.session_state.setdefault("asked", 0)

with st.sidebar:
    st.divider()
    st.metric("Pages indexed", len(qa.pages))
    st.metric("Chunks", len(qa.chunks))
    st.caption(f"Retriever: **{qa.retriever.name}** · Model: **{qa.model.name}**")
    if qa.model.name.startswith("extractive"):
        st.info(
            "Offline mode: answers are sentences copied from the PDF, not generated. "
            "Start Ollama locally, or add GROQ_API_KEY, for LLM answers."
        )
    st.caption(f"Questions this session: {st.session_state.asked} / {MAX_QUESTIONS}")

st.title("📄 Ask your PDF")
st.caption(f"Answers cite the page they came from · {name}")

history_key = f"history-{hash(pdf_bytes)}"
st.session_state.setdefault(history_key, [])

for turn in st.session_state[history_key]:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        for c in turn.get("citations", []):
            with st.expander(f"Page {c.page}"):
                st.write(c.snippet)

limit_reached = st.session_state.asked >= MAX_QUESTIONS
if limit_reached:
    st.info(f"This demo allows {MAX_QUESTIONS} questions per visit. Clone the repo to run it without limits.")

if question := st.chat_input("Ask a question about the document", disabled=limit_reached):
    st.session_state.asked += 1
    st.session_state[history_key].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching…"):
                answer = qa.ask(question)
        except Exception as exc:  # network, rate limit, model not pulled, etc.
            hint = ""
            if qa.model.name.startswith("ollama"):
                model = qa.model.name.split(":", 1)[1]
                hint = f" Check that Ollama is running and the model is pulled: `ollama pull {model}`."
            elif qa.model.name.startswith("groq"):
                hint = " The free Groq quota may be used up for now; try again in a minute."
            st.error(f"The model call failed: {exc}.{hint}")
            st.stop()
        st.markdown(answer.text)
        if answer.invalid_pages:
            st.warning(f"The model cited pages it wasn't shown: {answer.invalid_pages}. Treat those claims with caution.")
        for c in answer.citations:
            with st.expander(f"Page {c.page}"):
                st.write(c.snippet)
    st.session_state[history_key].append(
        {"role": "assistant", "content": answer.text, "citations": answer.citations}
    )
