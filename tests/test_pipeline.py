import pytest

from pdfqa import PDFQA
from pdfqa.config import Settings
from pdfqa.llm import NOT_FOUND, ExtractiveModel, get_model
from pdfqa.retrieval import TfidfRetriever, chroma_available

# A tiny gold set: question -> (page the answer is on, a string the answer must contain)
GOLD = [
    ("How many days of paid time off do employees get?", 2, "20 days"),
    ("How many unused PTO days can be carried over?", 2, "5 unused PTO days"),
    ("What is the home office stipend for remote employees?", 3, "$750"),
    ("What is the daily meal limit when travelling internationally?", 4, "$90"),
    ("What is the annual learning budget?", 5, "$2,000"),
    ("How quickly must a lost laptop be reported?", 6, "1 hour"),
]

OFFLINE = Settings(chunk_size=300, chunk_overlap=60, top_k=3, retriever="tfidf", llm_provider="offline")


@pytest.fixture(scope="module")
def qa(sample_pdf):
    return PDFQA.from_pdf(sample_pdf, settings=OFFLINE)


def test_offline_defaults_need_no_api_key():
    assert isinstance(get_model("auto"), ExtractiveModel)


@pytest.mark.parametrize("question,page,expected", GOLD)
def test_answers_are_correct_and_cite_the_right_page(qa, question, page, expected):
    answer = qa.ask(question)
    assert expected in answer.text
    assert page in [c.page for c in answer.citations]
    assert answer.is_grounded


def test_retrieval_hit_rate_on_gold_set(qa):
    hits = sum(page in [c.page for c, _ in qa.retriever.search(q, 3)] for q, page, _ in GOLD)
    assert hits / len(GOLD) == 1.0


def test_unrelated_question_returns_not_found(qa):
    answer = qa.ask("Quantum chromodynamics gluon")
    assert answer.text == NOT_FOUND
    assert not answer.citations


def test_hallucinated_citation_is_flagged(qa):
    class LyingModel:
        name = "liar"

        def answer(self, question, contexts, prompt):
            return "Employees get 20 days of PTO [p.2]. They also get a free car [p.42]."

    liar = PDFQA(qa.pages, settings=OFFLINE, retriever=qa.retriever, model=LyingModel())
    answer = liar.ask("How much PTO?")
    assert answer.invalid_pages == [42]
    assert not answer.is_grounded


def test_empty_question_rejected(qa):
    with pytest.raises(ValueError):
        qa.ask("   ")


def test_tfidf_ranks_exact_keyword_match_first(qa):
    best, score = TfidfRetriever(qa.chunks).search("multi-factor authentication", 1)[0]
    assert best.page == 6 and score > 0


@pytest.mark.skipif(not chroma_available(), reason="langchain-chroma not installed")
def test_chroma_pipeline_with_offline_embeddings(qa):
    from dataclasses import replace

    chroma_qa = PDFQA(qa.pages, settings=replace(OFFLINE, retriever="chroma"))
    assert chroma_qa.retriever.name == "chroma"
    answer = chroma_qa.ask("How many days of paid time off do employees get?")
    assert "20 days" in answer.text
    assert 2 in [c.page for c in answer.citations]
