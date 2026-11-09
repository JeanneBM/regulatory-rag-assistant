# Regulatory RAG validation

## Windows retrieval smoke test

Date: 9 November 2026  
Tested revision: [`b5303a7`](https://github.com/JeanneBM/regulatory-rag-assistant/commit/b5303a7769b21cf8cadaa39d2ed5226a6f5f15ae) on `main`  
Environment: Windows, Git Bash, Python CLI, retrieval-only mode.

| Query | Expected | Observed |
| --- | --- | --- |
| What does DORA say about the ICT risk management framework? | Retrieve DORA Article 6 | `retrieval_only`; top hit `DORA-art-6-part-1`; TF-IDF cosine score 0.5391 |
| How do I bake a chocolate cake? | Return no supporting evidence | `insufficient_evidence`; empty `claims` |

The top hit links to the [official EUR-Lex text](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32022R2554) and records the [Cellar download source](https://publications.europa.eu/resource/cellar/0caf473a-85bd-11ed-9887-01aa75ed71a1.0006.03/DOC_1). The score is a retrieval ranking signal, not a confidence estimate.

The separate [GitHub Actions live-ingest run](https://github.com/JeanneBM/regulatory-rag-assistant/actions/runs/42643926005) verified download and article coverage for both acts: 113/113 AI Act articles and 64/64 DORA articles, with 260 total chunks. The Windows terminal output establishes the observed query behavior; it does not independently establish cold-cache state or full corpus coverage. Azure generation and validation of generated citations were outside this smoke test.

## Azure OpenAI generation follow-up

Date: 9 November 2026  
Tested revision: [`8ff0723`](https://github.com/JeanneBM/regulatory-rag-assistant/commit/8ff07233ed512ee5b541576fbc7d78ee660bc014) on `main`  
Environment: Windows, Git Bash, Python CLI, Azure OpenAI GPT-4.1 mini.

- **Query:** What does DORA say about the ICT risk management framework?
- **Observed:** `answered` with three claims. All citations resolved to `DORA-art-6-part-1`; the CLI copied the quotation text from the selected source excerpts.
- **Manual check:** The three selected excerpts support the main points about the framework, protection of ICT assets and review cadence. The response is a summary and omits some details in Article 6.
- **Usage:** 3,200 input tokens and 154 output tokens, 3,354 total.

This is a smoke test of one Azure-backed answer. It does not measure answer quality across a question set. The out-of-scope result above was tested in retrieval-only mode.

## Ten-question Azure pilot

Date: 9 November 2026. Eight questions with predefined target articles and two out-of-scope controls, run with [`eval_smoke.py`](../eval_smoke.py) against the official AI Act and DORA corpus. One run per question.

| Check | Result |
| --- | --- |
| Expected article retrieved in the top five | 8/8 |
| Expected article cited in the answer | 8/8 |
| Verbatim quotations found in the cited retrieved chunk | 23/23 |
| Correct abstentions | 2/2: one with no retrieved evidence and one with related but insufficient evidence |
| Reported token usage | 30,153 input + 1,153 output = 31,306 total |

Manual review compared each statement with its selected quote and, where necessary, surrounding text in the supplied corpus. Finding a quotation verbatim is a provenance check. It does not establish support for every part of the statement.

| Case | Manual review |
| --- | --- |
| D01 · DORA Art. 5 | The second quote supports ultimate responsibility but does not itself support the additional statement about data protection policies. |
| D02 · DORA Art. 6 | The cited paragraphs support the main points. The summary omits testing conclusions as an additional review trigger. |
| D03 · DORA Art. 11 | The third quote establishes the general continuity policy. The detailed objectives in the claim need the following paragraph. |
| D04 · DORA Art. 17 | The main points are supported. The third claim omits the qualifier “as appropriate” for information provided to counterparties. |
| A01 · AI Act Art. 4 | The primary answer is supported. The extra Article 95 quote names an AI literacy goal without establishing the actors attributed in the statement. |
| A02 · AI Act Art. 5 | The first quote stops before the condition it relies on. The third statement overgeneralizes the prohibition by omitting both harmful-treatment conditions. |
| A03 · AI Act Art. 50 | The main disclosure point is supported. The answer adds other duties and omits relevant scope and exception details. |
| A04 · AI Act Art. 53 | The documentation claim is supported. The selected quote for the second claim stops before the Annex XII requirement, and the answer does not address the applicable open-source exception. |
| N01 and N02 | Both returned no claims. N02 reached the abstention path after retrieval. |

Anonymized input and run files retain official source identifiers and reduce exact timestamps to UTC dates. SHA-256 of the anonymized corpus: `3c26effe2d604733ecf84c82c4f36b58ae439b39f6fb9216225a26a72992692f`. SHA-256 of the anonymized run: `66af62b055c6daadd75cebe1a3ab34cec2098a65eb09d50ad3081a782882c33c`.

This pilot separates article retrieval, quotation provenance and semantic support. Its small curated sample does not establish production quality or legal compliance.
