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

An optional LangChain semantic pipeline is available below. Azure AI Search, Bicep, managed identity, automatic updates, annex indexing, paragraph-level identifiers, delegated acts and a completed evaluation dataset remain future work.

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

`python rag.py ingest` explicitly refreshes both source snapshots. A failed import leaves the existing corpus intact. Redirects and incomplete article coverage fail closed. If EUR-Lex returns HTTP 202 or blocks scripted access, open the official English original in a browser and save the actual document as HTML (not a challenge page or PDF). Pass either or both saved files, as needed:

```bash
python rag.py ingest --ai-act-html "$HOME/Downloads/ai-act.html" --dora-html "$HOME/Downloads/dora.html"
```

[AI Act HTML](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1689) · [DORA HTML](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32022R2554)

Local imports retain the official citation URL, record `import_method: local_html`, and pass the same article coverage checks. Use unedited copies saved from the official pages. A local import does not authenticate the file against EUR-Lex.

Original HTML is fingerprinted after text decoding; the digest identifies the fetched response text, not a legal version identifier.

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

## LangChain semantic RAG (optional)

The existing TF-IDF CLI and Streamlit interface remain the lexical baseline.
The separate semantic CLI uses LangChain Documents, RecursiveCharacterTextSplitter
(1200 characters, 200 overlap), AzureOpenAIEmbeddings, a FAISS retriever,
ChatPromptTemplate, LCEL and PydanticOutputParser. Source and exact-quote validation
runs after parsing and rejects invalid output. This checks citation integrity,
not whether a claim logically follows from its quote.

```bash
pip install -r requirements-langchain.txt
python rag.py ingest
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
model, deployment or endpoint change (deployment-name changes are checked automatically).

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
