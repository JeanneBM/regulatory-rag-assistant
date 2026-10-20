import pytest
from rag import parse_articles, retrieve, validate_claims, answer

def fixture():
    return [{"id": "DORA-art-6-part-1", "act": "DORA", "article": "6",
             "text": "Financial entities shall have an ICT risk management framework."}]

def test_retrieves_relevant_evidence():
    assert retrieve("ICT risk management", fixture())[0]["article"] == "6"

def test_unrelated_query_abstains():
    assert retrieve("banana pineapple recipe", fixture()) == []

def test_filter_prevents_cross_act_results():
    assert retrieve("ICT risk", fixture(), "AI_ACT") == []

def test_fabricated_citation_rejected():
    with pytest.raises(ValueError):
        validate_claims({"claims": [{"source_id": "invented", "quote": "a fabricated quotation"}]}, fixture())

def test_fabricated_quote_rejected():
    with pytest.raises(ValueError):
        validate_claims({"claims": [{"source_id": fixture()[0]["id"],
                                    "quote": "This quotation is not in the evidence"}]}, fixture())

def test_valid_quote():
    assert validate_claims({"claims": [{"source_id": fixture()[0]["id"],
                                      "statement": "The article addresses ICT risk.",
                                      "quote": fixture()[0]["text"]}]}, fixture())

def test_no_evidence_never_calls_provider():
    assert answer("Anything", [])["status"] == "insufficient_evidence"

def test_parser_preserves_article_and_excludes_annex():
    html = "<div id='document1'><p>Article 1</p><p>" + ("Scope text. " * 15) + "</p><p>ANNEX I</p><p>Excluded material</p></div>"
    chunks = parse_articles(html, "AI_ACT", "https://eur-lex.europa.eu/", "2026-10-20")
    assert chunks[0]["article"] == "1"
    assert "Excluded material" not in chunks[0]["text"]
    assert len(chunks[0]["sha256"]) == 64

def test_invalid_source_rejected():
    with pytest.raises(ValueError):
        parse_articles("<p>Please verify you are human</p>", "DORA", "url", "date")
