# Regulatory RAG Assistant

An evidence-grounded research PoC for exploring the EU AI Act and DORA. Retrieves relevant article excerpts locally and optionally generates cited answers using Azure OpenAI.

**Status:** Demonstrated PoC. The EU Publications Office import runs live in CI weekly. The [Windows smoke report](evidence/windows-retrieval-smoke-2026-11-09.md) records retrieval and a live Azure OpenAI answer with verified source quotations. Broader answer quality and legal compliance have not been evaluated.

## Scope

- AI Act: Regulation (EU) 2024/1689.
- DORA: Regulation (EU) 2022/2554.
- Official English original published acts, downloaded automatically from the EU Publications Office (Cellar); citations link to EUR-Lex.
- Article-level parsing, paragraph-aligned chunks, source URLs, retrieval timestamps and SHA256 fingerprints.
- Local TF-IDF retrieval and source filtering.
- Optional Azure OpenAI generation with exact-quote and source-ID validation.
- Streamlit interface, CLI and offline unit tests.

An optional LangChain semantic pipeline is available below. Azure AI Search, Bicep, managed identity, automatic refresh of an existing local corpus, annex indexing, paragraph-level identifiers, delegated acts and a completed evaluation dataset remain future work.

## Quick start

Python 3.11+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python rag.py ask "What does DORA say about ICT risk management?"
streamlit run app.py --server.address 127.0.0.1
```

PowerShell activation: ` .venv/Scripts/Activate.ps1 `.

The first `ask`, `streamlit run` or `langchain_rag.py index` downloads both acts automatically (about 2 MB) into `.local/` next to the code. No manual download or saved HTML is needed.

Without Azure configuration the application returns retrieved evidence only. No cloud resources are created. English queries work best because retrieval uses English lexical matching.

For generated answers, set environment variables shown in `.env.example` in your shell. The template is not loaded automatically. Use an existing Azure OpenAI chat deployment supporting JSON object responses and the configured API version. Never commit keys. Requests send the question and retrieved public excerpts to your configured Azure endpoint.

## Data flow

Publications Office Cellar (official XHTML) → validated article extraction → local JSON corpus → lexical retrieval → optional model response → source/quote checks → UI or CLI.

EUR-Lex pages answer scripts with an AWS WAF JavaScript challenge (HTTP 202), so the importer does not scrape them. It asks the Cellar for the act's CELEX resource with `Accept: application/xhtml+xml` and `Accept-Language: eng`. Cellar answers 303 with the English XHTML manifestation. Cellar redirects to plain HTTP; every hop is upgraded to HTTPS and must stay on `publications.europa.eu`. Each chunk keeps the EUR-Lex link as its citation URL, plus `celex` and `fetched_from`.

The corpus is downloaded on first use, and again whenever `ACTS` in `rag.py` lists an act the stored corpus lacks. Adding an act means adding its CELEX number and article count there. `python rag.py ingest` refreshes all acts explicitly. Any HTTP error, foreign redirect, unexpected content type or missing article fails closed and leaves the existing corpus intact. The [Live ingest workflow](.github/workflows/live-ingest.yml) runs the real import weekly and on changes to `rag.py`, and publishes the corpus as a build artifact. Original HTML is fingerprinted after text decoding; the digest identifies the fetched response text, not a legal version identifier.

The parser excludes annexes and signatures from the final article. Text is extracted as one whitespace-normalised line per block element, so links, spans and footnote markers stay inside their sentence and cannot be mistaken for article headings. Long articles are split between paragraphs (or between words inside an oversized paragraph) into chunks of about 3000 characters, and each continuation chunk repeats the article heading. Paragraphs are not assigned identifiers. Each response can be traced to a corpus chunk.

## Evidence boundaries

In the standard Python CLI and Streamlit, the model selects a numbered excerpt and the application copies the quotation directly from the retrieved text. Unknown excerpt IDs are rejected. Exact citation validation checks provenance and quotation integrity. Quotes are compared after Unicode NFKC normalisation, straight-quote substitution and whitespace collapsing, so layout differences do not reject a genuine quote; wording, punctuation and word order must still match, and one invalid claim rejects the whole answer. Truncated or content-filtered model replies are errors, not refusals. Validation does **not** prove that a generated statement follows from the quote, nor prevent every prompt injection. Retrieved documents and questions are treated as untrusted input.

A positive retrieval score is not confidence. The 0.05 threshold is a provisional lexical cutoff requiring calibration. The application abstains for missing evidence and rejects malformed generated citations.

Original published acts are not a current consolidated legal corpus. Retrieval timestamps are not application dates. Amendments, applicability, exceptions and cross-references require expert review. The assistant supports research, not certification of compliance or legal advice.

## Verification

```bash
python -m pytest -q
```

Offline tests cover relevant retrieval, unrelated queries, source filtering, fabricated citations/quotes, empty-evidence abstention, parsing of inline markup and long articles, quote normalisation, the generation path with a stubbed client (including truncated and filtered replies), stale semantic indexes, and the Cellar import against a fake server (HTTPS upgrade, foreign redirects, WAF status, content type, coverage, first-use and new-act downloads). GitHub Actions runs tests on pushes and pull requests, with actions pinned to commit SHAs and Dependabot proposing updates.

Before calling this a validated RAG system:
1. Inspect article boundaries in the live-ingested corpus (coverage is already checked in CI).
2. Extend the [live Azure smoke test](evidence/windows-retrieval-smoke-2026-11-09.md) to additional questions and verify refusal behavior in the generation path.
3. Build 30–50 expert-reviewed questions with expected source articles.
4. Measure retrieval recall, citation validity, answer support, abstention, latency and token use.
5. Add adversarial document and question tests; do not infer security from prompt wording.

## Cost and shutdown

Local retrieval has no Azure service cost. Generation uses a bounded output and one retry, but provider charges still apply. Stop Streamlit with Ctrl+C and unset credentials when finished. Delete `.local/` to remove downloaded data. This repository deploys no persistent cloud infrastructure.

## Roadmap

- Curated evaluation set and per-run evidence reports.
- Embeddings and hybrid retrieval with Azure AI Search.
- Explicit source-version review and consolidated texts.
- Annexes, DORA delegated acts, GDPR and NIS2 as separately versioned sources.
- Azure identity, telemetry and optional disposable Bicep deployment.

## LangChain semantic RAG (optional)

The existing TF-IDF CLI and Streamlit interface remain the lexical baseline.
The separate semantic CLI uses LangChain Documents, RecursiveCharacterTextSplitter
(1200 characters, 200 overlap), AzureOpenAIEmbeddings, a FAISS retriever,
ChatPromptTemplate, LCEL and PydanticOutputParser. Source and exact-quote validation
runs after parsing and rejects invalid output. This checks citation integrity,
not whether a claim logically follows from its quote.

```bash
pip install -r requirements-langchain.txt
# Set AZURE_OPENAI_EMBEDDING_DEPLOYMENT to an existing Azure embedding deployment.
# Also set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY and the API version.
python langchain_rag.py index
python langchain_rag.py ask "What does DORA say about ICT risk management?" --act DORA
```

Set AZURE_OPENAI_DEPLOYMENT to an existing chat deployment for generated answers;
without it the CLI returns evidence only. Environment files are not auto-loaded.
Indexing sends corpus text to the embedding provider and incurs embedding charges.
Asking embeds the question and optionally calls the chat model. No resources are
provisioned. Only public source excerpts should be indexed.

The index stores vectors, chunk text, metadata, deployment name and corpus hash in
.local/semantic-index.json. It uses JSON rather than pickle. Documents are embedded
once per explicit index command; query commands reconstruct FAISS from saved vectors.
Use only your own index artifacts. Rebuild after corpus refresh, or any embedding
model, deployment or endpoint change. Deployment-name and corpus changes are checked
automatically: `ask` refuses an index built from a different corpus snapshot.

Retrieval normalizes vectors and uses squared L2 distance, converted to cosine
similarity. The default maximum distance 0.8 (cosine similarity at least 0.6) is
provisional, not a calibrated confidence score. Override with --max-distance.
An act filter applies before selecting results; at most five excerpts are returned.
No evidence skips the chat call; the model can also return an explicit refusal.
Provider, parsing and citation failures remain errors, not successful refusals.

```bash
python -m pytest -q
```

Offline tests use deterministic embedding and model doubles, exercise real FAISS
persistence/retrieval and the LCEL pipeline, and make no provider calls. They do not
establish semantic retrieval quality. Live EUR-Lex, Azure embeddings and chat still
require integration validation. A curated 20–30 question pilot and recall,
answer-support, refusal, latency and token measurements remain to be completed.
The CLI reports elapsed time; token accounting is not yet implemented in this path.

API references: [FAISS implementation](https://github.com/langchain-ai/langchain-community/blob/main/libs/community/langchain_community/vectorstores/faiss.py)
and [AzureChatOpenAI](https://reference.langchain.com/python/langchain-openai/chat_models/azure/AzureChatOpenAI).
