import json
from types import SimpleNamespace

import pytest

import rag
from rag import parse_articles, validate_claims

BODY = "Scope text. " * 10


def parse(html, act="AI_ACT"):
    return parse_articles(f"<div id='document1'>{html}</div>", act, "url", "date")


def claim(source, quote):
    return {"claims": [{"source_id": source["id"], "statement": "A claim.", "quote": quote}]}


def test_inline_markup_stays_in_its_sentence_and_quotes_match():
    chunk = parse("<p>Article 10</p><p>Providers shall, in accordance with "
                  "<span>the obligations</span> laid\n   down here, keep logs. "
                  f"<!-- editorial note --> {BODY}</p>")[0]
    assert "Providers shall, in accordance with the obligations laid down here, keep logs." \
        in chunk["text"]
    assert "editorial note" not in chunk["text"]
    quote = "Providers shall, in accordance with the obligations laid down here, keep logs."
    assert validate_claims(claim(chunk, quote), [chunk])


def test_cross_reference_link_is_not_an_article_heading():
    chunks = parse(f"<p>Article 6</p><p>{BODY}</p>"
                   f"<p>Article 7</p><p>As referred to in <a href='#'>Article 6</a>, {BODY}</p>")
    assert [c["article"] for c in chunks] == ["6", "7"]
    assert "As referred to in Article 6, Scope" in chunks[1]["text"]


def test_long_article_splits_between_words_and_keeps_heading():
    paragraphs = "".join(f"<p>{'regulation ' * 150}</p>" for _ in range(6))
    chunks = parse(f"<p>Article 1</p>{paragraphs}", "DORA")
    assert len(chunks) > 1
    assert [c["id"] for c in chunks] == [f"DORA-art-1-part-{i}" for i in range(1, len(chunks) + 1)]
    for chunk in chunks:
        assert chunk["text"].startswith("Article 1\n")
        assert len(chunk["text"]) <= rag.CHUNK_CHARS
        assert set(chunk["text"].split("\n", 1)[1].split()) == {"regulation"}
    words = sum(len(c["text"].split("\n", 1)[1].split()) for c in chunks)
    assert words == 6 * 150


def test_quote_matching_ignores_only_presentation_differences():
    chunk = parse("<p>Article 5</p><p>1.   The ‘AI system’ shall "
                  f"be placed\non the market only if compliant. {BODY}</p>")[0]
    assert validate_claims(claim(chunk, "The 'AI system' shall be placed on the market"), [chunk])
    with pytest.raises(ValueError, match="Quote"):
        validate_claims(claim(chunk, "The 'AI system' may be placed on the market"), [chunk])


@pytest.mark.parametrize("payload", [
    ["not", "an", "object"],
    {"claims": ["not a claim object"]},
    {"claims": [{"source_id": ["unhashable"], "statement": "x", "quote": "long enough quote"}]},
])
def test_malformed_model_output_raises_value_error(payload):
    chunk = parse(f"<p>Article 1</p><p>{BODY}</p>")[0]
    with pytest.raises(ValueError):
        validate_claims(payload, [chunk])


@pytest.fixture
def azure(monkeypatch):
    import openai

    response = {}

    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(
                create=lambda **kwargs: response["value"]))

    for name in ("AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"):
        monkeypatch.setenv(name, "test")
    monkeypatch.setattr(openai, "AzureOpenAI", Client)

    def respond(content, finish_reason="stop"):
        message = SimpleNamespace(content=content)
        response["value"] = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason=finish_reason, message=message)], usage=None)
    return respond


def test_generation_path_validates_quotes(azure):
    chunk = parse(f"<p>Article 1</p><p>Providers shall keep logs. {BODY}</p>")[0]
    azure(json.dumps(claim(chunk, "Providers shall keep logs.")))
    assert rag.answer("Logs?", [chunk])["status"] == "answered"
    azure(json.dumps({"claims": []}))
    assert rag.answer("Logs?", [chunk])["status"] == "insufficient_evidence"


@pytest.mark.parametrize("content, finish_reason", [
    ('{"claims": [{"statement": "Trunc', "length"),
    (None, "content_filter"),
])
def test_truncated_or_filtered_reply_is_an_error(azure, content, finish_reason):
    azure(content, finish_reason)
    chunk = parse(f"<p>Article 1</p><p>{BODY}</p>")[0]
    with pytest.raises(ValueError, match=finish_reason):
        rag.answer("question", [chunk])
