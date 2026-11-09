import json
from types import SimpleNamespace

import pytest

import rag

XHTML = "application/xhtml+xml;charset=UTF-8"


def act_html(count):
    return "".join(f"<p class='oj-ti-art'>Article {i}</p><p>{'Article text. ' * 10}</p>"
                   for i in range(1, count + 1))


class Cellar:
    """Fake Publications Office: 303 from the CELEX resource to a plain-HTTP item."""

    def __init__(self, counts=None, status=200, location=None, content_type=XHTML):
        self.counts = counts or {a: m["articles"] for a, m in rag.ACTS.items()}
        self.status, self.location, self.content_type = status, location, content_type
        self.calls = []

    def item(self, celex):
        return f"https://{rag.CELLAR_HOST}/resource/cellar/{celex}.0006.03/DOC_1"

    def __call__(self, url, headers=None, **kwargs):
        self.calls.append(url)
        assert headers["Accept"] == "application/xhtml+xml" and headers["Accept-Language"] == "eng"
        for act, meta in rag.ACTS.items():
            celex = meta["celex"]
            if url == rag.CELLAR_URL.format(celex=celex):
                target = self.location or self.item(celex).replace("https://", "http://")
                return SimpleNamespace(status_code=303, headers={"Location": target}, text="")
            if url == self.item(celex):
                return SimpleNamespace(status_code=self.status, text=act_html(self.counts[act]),
                                       headers={"Content-Type": self.content_type})
        raise AssertionError(f"unexpected request {url}")


@pytest.fixture
def corpus(monkeypatch, tmp_path):
    path = tmp_path / "corpus.json"
    monkeypatch.setattr(rag, "DATA", path)
    return path


def install(monkeypatch, cellar):
    monkeypatch.setattr(rag.requests, "get", cellar)
    return cellar


def test_ingest_downloads_every_act_from_cellar_over_https(monkeypatch, corpus):
    cellar = install(monkeypatch, Cellar())
    assert rag.ingest() == 113 + 64
    assert all(call.startswith(f"https://{rag.CELLAR_HOST}/") for call in cellar.calls)
    chunks = json.loads(corpus.read_text(encoding="utf-8"))
    dora = next(c for c in chunks if c["act"] == "DORA")
    assert dora["url"] == rag.SOURCES["DORA"]  # citation link stays on EUR-Lex
    assert dora["fetched_from"] == cellar.item("32022R2554")
    assert dora["celex"] == "32022R2554"


@pytest.mark.parametrize("cellar, error", [
    (Cellar(location="https://example.com/DOC_1"), "redirect outside"),
    (Cellar(status=202), "HTTP 202"),
    (Cellar(content_type="application/pdf"), "content type"),
    (Cellar(counts={"AI_ACT": 113, "DORA": 63}), "DORA: incomplete article coverage"),
])
def test_failed_import_keeps_previous_corpus(monkeypatch, corpus, cellar, error):
    corpus.write_text("previous corpus", encoding="utf-8")
    install(monkeypatch, cellar)
    with pytest.raises((ValueError, RuntimeError), match=error):
        rag.ingest()
    assert corpus.read_text(encoding="utf-8") == "previous corpus"


def test_first_use_downloads_once_and_new_acts_join_automatically(monkeypatch, corpus):
    cellar = install(monkeypatch, Cellar())
    assert len(rag.load_corpus()) == 177
    requests_made = len(cellar.calls)
    assert len(rag.load_corpus()) == 177
    assert len(cellar.calls) == requests_made  # reuses the stored corpus

    monkeypatch.setitem(rag.ACTS, "NIS2", {"celex": "32022L2555", "articles": 46})
    cellar.counts["NIS2"] = 46
    chunks = rag.load_corpus()
    assert {c["act"] for c in chunks} == {"AI_ACT", "DORA", "NIS2"}
    assert len(cellar.calls) > requests_made
