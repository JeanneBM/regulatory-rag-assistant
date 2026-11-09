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

@pytest.mark.parametrize("complete", [True, False])
def test_saved_eurlex_html_uses_coverage_guard(monkeypatch, tmp_path, complete):
    import rag

    def saved_act(name, count):
        path = tmp_path / name
        path.write_text("".join(f"<p>Article {i}</p><p>{'Article text. ' * 10}</p>"
                                for i in range(1, count + 1)), encoding="utf-8")
        return path

    def no_network(*args, **kwargs):
        pytest.fail("Local HTML import attempted a network request")

    corpus = tmp_path / "corpus.json"
    corpus.write_text("previous corpus", encoding="utf-8")
    monkeypatch.setattr(rag, "DATA", corpus)
    monkeypatch.setattr(rag.requests, "get", no_network)
    ai_act = saved_act("ai-act.html", 113)
    dora = saved_act("dora.html", 64 if complete else 63)
    if not complete:
        with pytest.raises(ValueError, match="DORA: incomplete article coverage"):
            rag.ingest(ai_act_html=ai_act, dora_html=dora)
        assert corpus.read_text(encoding="utf-8") == "previous corpus"
        return

    assert rag.ingest(ai_act_html=ai_act, dora_html=dora) == 177
    chunk = rag.load_corpus()[-1]
    assert chunk["import_method"] == "local_html"
    assert chunk["url"] == rag.SOURCES["DORA"]
    assert "imported_at" in chunk

def test_first_load_tries_official_fallback_and_reuses_corpus(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import rag

    calls = []

    def get(url, **kwargs):
        calls.append(url)
        if url == rag.SOURCES["AI_ACT"]:
            return SimpleNamespace(status_code=202)
        count = 113 if "/2024/" in url else 64
        html = "".join(f"<p>Article {i}</p><p>{'Article text. ' * 10}</p>"
                       for i in range(1, count + 1))
        return SimpleNamespace(status_code=200, text=html)

    monkeypatch.setattr(rag, "DATA", tmp_path / "corpus.json")
    monkeypatch.setattr(rag.requests, "get", get)
    chunks = rag.load_corpus()
    assert len(chunks) == 177
    assert calls == [
        rag.SOURCES["AI_ACT"],
        "https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng",
        rag.SOURCES["DORA"],
    ]
    assert chunks[0]["url"] == rag.SOURCES["AI_ACT"]
    assert chunks[0]["fetched_from"] == calls[1]
    assert rag.load_corpus() == chunks
    assert len(calls) == 3


def test_official_download_blocks_external_redirect(monkeypatch):
    from types import SimpleNamespace
    import rag

    calls = []

    def get(url, **kwargs):
        calls.append(url)
        return SimpleNamespace(status_code=302, headers={"Location": "https://example.com/file"})

    monkeypatch.setattr(rag.requests, "get", get)
    with pytest.raises(ValueError, match="outside the allowed"):
        rag.fetch_official_html(rag.SOURCES["DORA"])
    assert calls == [rag.SOURCES["DORA"]]

def test_http_failure_uses_browser_and_validates_coverage(monkeypatch):
    import rag

    def blocked(url):
        raise ValueError("HTTP 202")

    def rendered(url):
        html = "".join(f"<p>Article {i}</p><p>{'Article text. ' * 10}</p>"
                       for i in range(1, 65))
        return html, url

    monkeypatch.setattr(rag, "fetch_official_html", blocked)
    monkeypatch.setattr(rag, "fetch_browser_html", rendered)
    chunks = rag.download_act("DORA", rag.SOURCES["DORA"], "date")
    assert len(chunks) == 64
    assert chunks[-1]["import_method"] == "browser"

    monkeypatch.setattr(rag, "fetch_browser_html", lambda url: ("<p>Verify access</p>", url))
    with pytest.raises(RuntimeError, match="browser: No articles parsed"):
        rag.download_act("DORA", rag.SOURCES["DORA"], "date")
