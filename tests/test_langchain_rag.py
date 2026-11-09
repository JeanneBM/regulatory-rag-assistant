import json

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda

from langchain_rag import answer, build_index, load_index, retrieve, split_documents


class LocalEmbeddings(Embeddings):
    """Deterministic test double, not a semantic embedding model."""
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [1.0, 0.0] if "ICT" in text else [0.0, 1.0]


@pytest.fixture
def chunks():
    return [
        {"id": "DORA-6", "act": "DORA", "article": "6",
         "text": "Financial entities shall have an ICT risk management framework."},
        {"id": "AI-1", "act": "AI_ACT", "article": "1",
         "text": "This Regulation lays down harmonised rules on artificial intelligence."},
    ]


def test_persisted_index_reuses_vectors_and_filters(tmp_path, chunks):
    path = tmp_path / "index.json"
    embedding = LocalEmbeddings()
    build_index(chunks, embedding, path, "test")
    embedding.embed_documents = lambda _: pytest.fail("Must not re-embed saved documents")
    store = load_index(embedding, path, "test")
    assert retrieve("ICT risk", store)[0]["id"] == "DORA-6-lc-1"
    assert retrieve("ICT risk", store, act="AI_ACT") == []
    with pytest.raises(ValueError, match="changed"):
        load_index(embedding, path, "different")


def test_stale_index_rejected_after_corpus_refresh(tmp_path, chunks):
    path = tmp_path / "index.json"
    embedding = LocalEmbeddings()
    build_index(chunks, embedding, path, "test")
    assert load_index(embedding, path, "test", corpus=chunks).index.ntotal == 2
    refreshed = [dict(chunks[0], text=chunks[0]["text"] + " Amended."), chunks[1]]
    with pytest.raises(ValueError, match="Corpus changed"):
        load_index(embedding, path, "test", corpus=refreshed)


def test_empty_evidence_skips_generation():
    model = RunnableLambda(lambda _: pytest.fail("No provider call expected"))
    assert answer("question", [], model)["status"] == "insufficient_evidence"


def test_lcel_answer_validates_actual_quote(chunks):
    payload = {"claims": [{"statement": "An ICT framework is required.",
                           "source_id": "DORA-6", "quote": chunks[0]["text"]}],
               "refusal_reason": ""}
    model = RunnableLambda(lambda _: json.dumps(payload))
    assert answer("ICT?", chunks, model)["status"] == "answered"
    payload["claims"][0]["quote"] = "A fabricated quotation absent from the source."
    with pytest.raises(ValueError, match="Quote"):
        answer("ICT?", chunks, model)
    payload["claims"][0]["source_id"] = "invented"
    with pytest.raises(ValueError, match="source"):
        answer("ICT?", chunks, model)


def test_model_can_abstain_and_malformed_output_fails(chunks):
    model = RunnableLambda(lambda _: '{"claims": [], "refusal_reason": "Not supported."}')
    assert answer("unanswerable", chunks, model)["status"] == "insufficient_evidence"
    with pytest.raises(Exception):
        answer("question", chunks, RunnableLambda(lambda _: '{"claims": "invalid"}'))


def test_splitter_preserves_provenance(chunks):
    chunks[0]["text"] *= 80
    documents = split_documents(chunks[:1])
    assert len(documents) > 1
    assert all(len(d.page_content) <= 1200 for d in documents)
    assert all(d.metadata["article"] == "6" for d in documents)
    assert len({d.metadata["id"] for d in documents}) == len(documents)
