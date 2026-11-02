from pdfqa.citations import build_prompt, parse_cited_pages, resolve_citations
from pdfqa.models import Chunk

CTX = [
    Chunk("Employees get 20 days of PTO.", 2, "h.pdf", "p2-c0"),
    Chunk("The meal limit is $75.", 4, "h.pdf", "p4-c0"),
]


def test_parse_common_citation_styles():
    assert parse_cited_pages("A [p.2]. B [p. 4]. C [p7].") == [2, 4, 7]
    assert parse_cited_pages("Both [p.2, 4] and again [p.2].") == [2, 4]
    assert parse_cited_pages("Range [pp.3-5].") == [3, 4, 5]
    assert parse_cited_pages("No citations here.") == []


def test_resolve_flags_pages_not_in_context():
    citations, invalid = resolve_citations("PTO is 20 days [p.2]. Parking is free [p.9].", CTX)
    assert [c.page for c in citations] == [2]
    assert citations[0].snippet.startswith("Employees get 20 days")
    assert invalid == [9]


def test_prompt_labels_every_excerpt_with_its_page():
    system, human = build_prompt("How much PTO?", CTX)
    assert system[0] == "system" and "[p.N]" in system[1]
    assert "[p.2] (h.pdf)" in human[1] and "[p.4] (h.pdf)" in human[1]
    assert human[1].endswith("Question: How much PTO?")
