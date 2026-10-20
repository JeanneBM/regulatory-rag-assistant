# Regulatory RAG Assistant

An evidence-grounded research PoC for exploring the EU AI Act and DORA. Retrieves relevant article excerpts locally and optionally generates cited answers using Azure OpenAI.

**Status:** Initial implementation. Live EUR-Lex ingestion and Azure generation require an integration run; no accuracy or legal-compliance claim is made.

## Scope

- AI Act: Regulation (EU) 2024/1689.
- DORA: Regulation (EU) 2022/2554.
- Official English original published acts from EUR-Lex.
- Article-level parsing, bounded overlapping chunks, source URLs, retrieval timestamps and SHA256 fingerprints.
- Local TF-IDF retrieval and source filtering.
- Optional Azure OpenAI generation with exact-quote and source-ID validation.
- Streamlit interface, CLI and offline unit tests.

This release does not include semantic/vector search, Azure AI Search, Bicep, managed identity, automatic updates, annex indexing, paragraph-level identifiers, delegated acts or a completed evaluation dataset. These are explicit next steps.

## Quick start

Python 3.11+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python rag.py ingest
python rag.py ask "What does DORA say about ICT risk management?"
streamlit run app.py --server.address 127.0.0.1
```

PowerShell activation: ` .venv/Scripts/Activate.ps1 `.

Without Azure configuration the application returns retrieved evidence only. No cloud resources are created. English queries work best because retrieval uses English lexical matching.

For generated answers, set environment variables shown in `.env.example` in your shell. The template is not loaded automatically. Use an existing Azure OpenAI chat deployment supporting JSON object responses and the configured API version. Never commit keys. Requests send the question and retrieved public excerpts to your configured Azure endpoint.

## Data flow

Official allowlisted source URLs → validated article extraction → local JSON corpus → lexical retrieval → optional model response → source/quote checks → UI or CLI.

`python rag.py ingest` explicitly refreshes both source snapshots. A failed import leaves the existing corpus intact. Redirects and incomplete article coverage fail closed; EUR-Lex challenge pages may require investigation. Original HTML is fingerprinted after text decoding; the digest identifies the fetched response text, not a legal version identifier.

The parser excludes annexes and signatures from the final article. It retains paragraphs as text but does not assign paragraph identifiers. Each response can be traced to a corpus chunk.

## Evidence boundaries

Exact citation validation checks provenance and quotation integrity. It does **not** prove that a generated statement follows from the quote, nor prevent every prompt injection. Retrieved documents and questions are treated as untrusted input.

A positive retrieval score is not confidence. The 0.05 threshold is a provisional lexical cutoff requiring calibration. The application abstains for missing evidence and rejects malformed generated citations.

Original published acts are not a current consolidated legal corpus. Retrieval timestamps are not application dates. Amendments, applicability, exceptions and cross-references require expert review. The assistant supports research, not certification of compliance or legal advice.

## Verification

```bash
python -m pytest -q
```

Offline tests cover relevant retrieval, unrelated queries, source filtering, fabricated citations/quotes, empty-evidence abstention and parsing. GitHub Actions runs tests on pushes and pull requests.

Before calling this a validated RAG system:
1. Run live ingestion and inspect coverage and article boundaries.
2. Run an Azure request and inspect quotes, usage and refusal behavior.
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
