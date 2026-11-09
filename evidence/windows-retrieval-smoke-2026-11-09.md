# Windows retrieval smoke test

Date: 9 November 2026  
Tested revision: [`b5303a7`](https://github.com/JeanneBM/regulatory-rag-assistant/commit/b5303a7769b21cf8cadaa39d2ed5226a6f5f15ae) on `main`  
Environment: Windows, Git Bash, Python CLI, retrieval-only mode.

| Query | Expected | Observed |
| --- | --- | --- |
| What does DORA say about the ICT risk management framework? | Retrieve DORA Article 6 | `retrieval_only`; top hit `DORA-art-6-part-1`; TF-IDF cosine score 0.5391 |
| How do I bake a chocolate cake? | Return no supporting evidence | `insufficient_evidence`; empty `claims` |

The top hit links to the [official EUR-Lex text](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32022R2554) and records the [Cellar download source](https://publications.europa.eu/resource/cellar/0caf473a-85bd-11ed-9887-01aa75ed71a1.0006.03/DOC_1). The score is a retrieval ranking signal, not a confidence estimate.

The separate [GitHub Actions live-ingest run](https://github.com/JeanneBM/regulatory-rag-assistant/actions/runs/42643926005) verified download and article coverage for both acts: 113/113 AI Act articles and 64/64 DORA articles, with 260 total chunks. The Windows terminal output establishes the observed query behavior; it does not independently establish cold-cache state or full corpus coverage. Azure generation and validation of generated citations were outside this smoke test.

# Azure OpenAI generation follow-up

Date: 9 November 2026  
Tested revision: [`8ff0723`](https://github.com/JeanneBM/regulatory-rag-assistant/commit/8ff07233ed512ee5b541576fbc7d78ee660bc014) on `main`  
Environment: Windows, Git Bash, Python CLI, Azure OpenAI GPT-4.1 mini.

- **Query:** What does DORA say about the ICT risk management framework?
- **Observed:** `answered` with three claims. All citations resolved to `DORA-art-6-part-1`; the CLI copied the quotation text from the selected source excerpts.
- **Manual check:** The three selected excerpts support the main points about the framework, protection of ICT assets and review cadence. The response is a summary and omits some details in Article 6.
- **Usage:** 3,200 input tokens and 154 output tokens, 3,354 total.

This is a smoke test of one Azure-backed answer. It does not measure answer quality across a question set. The out-of-scope result above was tested in retrieval-only mode.
