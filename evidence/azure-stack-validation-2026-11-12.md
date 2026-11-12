# Azure stack validation (pre-deployment)

Date: 12 November 2026  
Tested revision: [`8108675`](https://github.com/JeanneBM/regulatory-rag-assistant/commit/810867569a2cbc06e7a93af5bd95cc024d5ebc88) on `feat/azure-native-stack`  
Environment: GitHub Actions, `ubuntu-latest`, Python 3.12.

This report covers what can be checked without an Azure subscription. Nothing below was measured on Azure AI Search, Azure OpenAI or Container Apps.

## Offline checks ([CI run](https://github.com/JeanneBM/regulatory-rag-assistant/actions/runs/43212877571))

| Check | Result |
| --- | --- |
| Unit tests (fakes for Cellar, Azure OpenAI, Azure AI Search) | 58 passed |
| `ruff check` | clean |
| `pip-audit` on the hash-pinned `requirements.lock` | no known vulnerabilities |
| `az bicep build` and `az bicep lint` on `infra/main.bicep` | no errors or warnings |
| Docker image | builds, runs as UID 10001, `/healthz` answers, CSP header present |

## Live import of the official texts ([live-ingest run](https://github.com/JeanneBM/regulatory-rag-assistant/actions/runs/43212877546/job/131164672499))

The acts were downloaded automatically from the Publications Office Cellar and parsed into citable units.

| Act | Articles covered | Citable units | Point-level units | With article title | With EUR-Lex anchor |
| --- | --- | --- | --- | --- | --- |
| AI Act (32024R1689) | 113/113 | 877 | 419 | 877 | 877 |
| DORA (32022R2554) | 64/64 | 601 | 384 | 601 | 601 |

- 1,478 units in total. The longest is AI Act Article 26(10) at 2,730 characters, so no unit needed splitting.
- The unit for AI Act Article 5(1), point (c), includes sub-points (i) and (ii). These are the conditions missing from the quotation in pilot case A02 on 9 November.
- Unusual shapes, all checked: two second lists of points (AI Act Article 43(1)), each quoted with its own lead-in; two quoted amendment blocks without point markers (AI Act Article 110, DORA Article 63); 28 text runs after a list of points. No unit contains a chapter or section heading.

## Retrieval-only pilot (local TF-IDF backend)

| Check | Result |
| --- | --- |
| Expected article in the top 8 | 8/8 |
| Expected article ranked first | 8/8 |
| N01 (out of scope) | no evidence retrieved |
| N02 (related but unanswerable) | DORA Article 6 retrieved; abstention needs the generation step |

This measures the offline fallback backend only. Azure AI Search hybrid and semantic retrieval is a different ranker and must be measured separately.

## Not yet validated

- Deployment with `azd up` and the resulting RBAC, managed identity and network configuration.
- Azure AI Search retrieval quality and the provisional reranker threshold (1.2).
- Structured-output generation, claim verification labels, token usage and latency.
- Manual review of generated claims against their quotations.

Next step: deploy, run `regrag eval` against the deployment, complete the manual-review column of the generated Markdown report and add it here.
