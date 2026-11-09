"""Run a small, fixed regulatory RAG pilot and save local JSON evidence."""
import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from rag import answer, load_corpus, retrieve


CASES = (
    ("D01", "Under DORA, what is the role of the management body in ICT risk governance?", ("DORA", "5")),
    ("D02", "What does DORA say about the ICT risk management framework?", ("DORA", "6")),
    ("D03", "What does DORA require for ICT business continuity and response and recovery plans?", ("DORA", "11")),
    ("D04", "What does DORA require for the ICT-related incident management process?", ("DORA", "17")),
    ("A01", "What does the AI Act say about AI literacy for staff of providers and deployers?", ("AI_ACT", "4")),
    ("A02", "What does the AI Act prohibit regarding AI social scoring of natural persons?", ("AI_ACT", "5")),
    ("A03", "Under the AI Act, what transparency obligations apply to AI systems intended to interact directly with natural persons?", ("AI_ACT", "50")),
    ("A04", "Under the AI Act, what technical documentation must providers of general-purpose AI models prepare?", ("AI_ACT", "53")),
    ("N01", "How do I bake a chocolate cake?", None),
    ("N02", "What VAT rate does DORA set for the ICT risk management framework of financial entities?", None),
)


def run_case(case, corpus, retrieval_only):
    case_id, question, expected = case
    start = perf_counter()
    evidence = retrieve(question, corpus)
    sources = {c["id"]: (c["act"], str(c["article"])) for c in evidence}
    row = {
        "id": case_id,
        "question": question,
        "expected_article": expected,
        "retrieved_ids": list(sources),
        "expected_in_top_5": expected in sources.values() if expected else None,
    }
    if retrieval_only:
        row["check"] = (
            "retrieval_hit" if row["expected_in_top_5"] else "retrieval_miss"
        ) if expected else ("distractors_retrieved" if evidence else "no_evidence")
    else:
        try:
            result = answer(question, evidence)
            claims = result["claims"]
            usage = result.get("usage") or {}
            row.update(status=result["status"], claims=claims, usage={
                k: usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                if k in usage
            })
            if expected:
                cited = any(sources.get(claim["source_id"]) == expected for claim in claims)
                row["check"] = (
                    "citation_hit_needs_review" if result["status"] == "answered" and cited
                    else "expected_answer_missing"
                )
            else:
                row["check"] = (
                    ("abstained_after_retrieval" if evidence else "abstained_no_retrieval")
                    if result["status"] == "insufficient_evidence" and not claims
                    else "expected_abstention_missing"
                )
        except Exception as exc:
            row.update(check="error", error_type=type(exc).__name__)
            if isinstance(exc, ValueError):
                row["validation_error"] = str(exc)
            if isinstance(getattr(exc, "status_code", None), int):
                row["provider_http_status"] = exc.status_code
    row["elapsed_ms"] = round((perf_counter() - start) * 1000)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-only", action="store_true", help="Skip all model calls")
    args = parser.parse_args()
    if not args.retrieval_only:
        required = ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_API_KEY")
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            parser.error("Missing Azure configuration: " + ", ".join(missing))
    corpus = load_corpus()
    rows = []
    for case in CASES:
        row = run_case(case, corpus, args.retrieval_only)
        rows.append(row)
        print(f"{row['id']}: {row['check']}", flush=True)
    now = datetime.now(timezone.utc)
    path = Path(__file__).resolve().parent / ".local" / f"eval-10-{now:%Y%m%dT%H%M%S%fZ}.json"
    totals = Counter(r["check"] for r in rows)
    tokens = {k: sum(r.get("usage", {}).get(k, 0) for r in rows)
              for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    report = {
        "utc": now.isoformat(),
        "mode": "retrieval_only" if args.retrieval_only else "azure",
        "checks": dict(totals),
        "reported_tokens": tokens,
        "cases": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Checks: {dict(totals)}")
    print(f"Reported tokens: {tokens}")
    print(f"Report: {path}")
    failed = {"retrieval_miss", "expected_answer_missing", "expected_abstention_missing", "error"}
    return int(bool(failed & totals.keys()))


if __name__ == "__main__":
    raise SystemExit(main())
