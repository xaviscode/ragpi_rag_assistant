from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import requests


DEFAULT_BASE_URL = "http://localhost:8000"
QUESTIONS_PATH = Path(__file__).parent / "questions.json"


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def post_query(base_url: str, case: dict[str, Any], timeout: int = 120) -> tuple[dict[str, Any], float]:
    payload = {
        "question": case["question"],
        "return_sources": case.get("return_sources", True),
        "return_evidence": case.get("return_evidence", True),
    }

    if case.get("document_ids"):
        payload["document_ids"] = case["document_ids"]

    start = time.perf_counter()
    r = requests.post(f"{base_url.rstrip('/')}/query", json=payload, timeout=timeout)
    elapsed = time.perf_counter() - start

    r.raise_for_status()
    return r.json(), elapsed


def normalize(text: Any) -> str:
    return str(text or "").strip()


def contains_all(text: str, expected: list[str]) -> bool:
    text_lower = text.lower()
    return all(item.lower() in text_lower for item in expected)


def contains_any(text: str, expected: list[str]) -> bool:
    text_lower = text.lower()
    return any(item.lower() in text_lower for item in expected)


def evidence_text(response: dict[str, Any]) -> str:
    evidence = response.get("evidence") or []
    return "\n".join(normalize(item.get("quote")) for item in evidence)


def source_text(response: dict[str, Any]) -> str:
    sources = response.get("sources") or []
    return "\n".join(normalize(item.get("source")) for item in sources)


def evaluate_case(case: dict[str, Any], response: dict[str, Any], elapsed: float) -> dict[str, Any]:
    answer = normalize(response.get("answer"))
    sources = response.get("sources") or []
    evidence = response.get("evidence") or []

    checks: list[tuple[str, bool, str]] = []

    if case.get("answer_must_not_be_empty", True):
        checks.append(("answer_not_empty", bool(answer), "Answer should not be empty."))

    if case.get("must_refuse"):
        checks.append(
            (
                "must_refuse",
                "i don't know based on the provided documents" in answer.lower()
                or "not_found" in answer.lower(),
                "Answer should refuse unsupported information.",
            )
        )

    if "expected_answer_contains" in case:
        expected = case["expected_answer_contains"]
        checks.append(
            (
                "expected_answer_contains",
                contains_all(answer, expected),
                f"Answer should contain all of: {expected}",
            )
        )

    if "expected_answer_contains_any" in case:
        expected = case["expected_answer_contains_any"]
        checks.append(
            (
                "expected_answer_contains_any",
                contains_any(answer, expected),
                f"Answer should contain at least one of: {expected}",
            )
        )

    if "expected_answer_regex" in case:
        pattern = case["expected_answer_regex"]
        checks.append(
            (
                "expected_answer_regex",
                bool(re.search(pattern, answer)),
                f"Answer should match regex: {pattern}",
            )
        )

    if case.get("must_have_sources"):
        checks.append(("must_have_sources", len(sources) > 0, "Sources should be returned."))

    if case.get("must_have_evidence"):
        checks.append(("must_have_evidence", len(evidence) > 0, "Evidence should be returned."))

    if "expected_source_contains" in case:
        expected = case["expected_source_contains"]
        src_text = source_text(response)
        checks.append(
            (
                "expected_source_contains",
                contains_all(src_text, expected),
                f"Sources should contain all of: {expected}",
            )
        )

    if "expected_source_contains_any" in case:
        expected = case["expected_source_contains_any"]
        src_text = source_text(response)
        checks.append(
            (
                "expected_source_contains_any",
                contains_any(src_text, expected),
                f"Sources should contain at least one of: {expected}",
            )
        )

    if "expected_evidence_contains" in case:
        expected = case["expected_evidence_contains"]
        ev_text = evidence_text(response)
        checks.append(
            (
                "expected_evidence_contains",
                contains_all(ev_text, expected),
                f"Evidence should contain all of: {expected}",
            )
        )

    if "expected_evidence_contains_any" in case:
        expected = case["expected_evidence_contains_any"]
        ev_text = evidence_text(response)
        checks.append(
            (
                "expected_evidence_contains_any",
                contains_any(ev_text, expected),
                f"Evidence should contain at least one of: {expected}",
            )
        )

    failed = [
        {
            "check": name,
            "message": message,
        }
        for name, passed, message in checks
        if not passed
    ]

    return {
        "id": case["id"],
        "category": case.get("category"),
        "passed": len(failed) == 0,
        "elapsed_seconds": round(elapsed, 3),
        "answer": answer,
        "num_sources": len(sources),
        "num_evidence": len(evidence),
        "failed_checks": failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--questions", default=str(QUESTIONS_PATH))
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--save-results", default="")
    args = parser.parse_args()

    cases = load_cases(Path(args.questions))
    results = []

    print(f"Running {len(cases)} RAG evaluation cases against {args.base_url}")
    print("-" * 80)

    for case in cases:
        print(f"[RUN] {case['id']}")

        try:
            response, elapsed = post_query(args.base_url, case, timeout=args.timeout)
            result = evaluate_case(case, response, elapsed)
        except Exception as exc:
            result = {
                "id": case["id"],
                "category": case.get("category"),
                "passed": False,
                "elapsed_seconds": None,
                "answer": "",
                "num_sources": 0,
                "num_evidence": 0,
                "failed_checks": [
                    {
                        "check": "request_error",
                        "message": str(exc),
                    }
                ],
            }

        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {case['id']} | {result.get('elapsed_seconds')}s")
        print(f"Answer: {result.get('answer')}")
        if result["failed_checks"]:
            for failure in result["failed_checks"]:
                print(f"  - {failure['check']}: {failure['message']}")
        print("-" * 80)

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    pass_rate = passed / total if total else 0.0

    avg_time_values = [
        r["elapsed_seconds"]
        for r in results
        if isinstance(r.get("elapsed_seconds"), (int, float))
    ]
    avg_time = sum(avg_time_values) / len(avg_time_values) if avg_time_values else 0.0

    summary = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(pass_rate, 3),
        "average_elapsed_seconds": round(avg_time, 3),
        "results": results,
    }

    print("\nSUMMARY")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.save_results:
        out_path = Path(args.save_results)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()