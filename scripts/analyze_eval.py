from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize failed evaluation traces for the next tuning iteration.")
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, default=Path("eval/trace-review-latest.json"))
    args = parser.parse_args()
    data = json.loads(args.result.read_text(encoding="utf-8"))
    failures = [item for item in data.get("cases", []) if not item.get("passed")]
    reasons = Counter()
    for item in failures:
        response = item.get("response") or {}
        assertion_codes = [str(assertion.get("code")) for assertion in item.get("assertions", []) if assertion.get("code")]
        reasons.update(assertion_codes or [response.get("reasonCode") or item.get("actual_result") or item.get("error") or "UNKNOWN"])
    report: dict[str, Any] = {
        "evalRunId": data.get("eval_run_id"),
        "summary": data.get("summary", {}),
        "failureCount": len(failures),
        "failureReasons": dict(reasons),
        "failures": [
            {
                "caseId": item.get("case_id"),
                "category": item.get("category"),
                "question": item.get("question"),
                "expected": item.get("expected_result"),
                "actual": item.get("actual_result"),
                "traceId": item.get("trace_id"),
                "error": item.get("error"),
                "reasonCode": (item.get("response") or {}).get("reasonCode"),
                "assertions": item.get("assertions", []),
                "answer": ((item.get("response") or {}).get("answer") or {}).get("text"),
                "citations": [
                    {
                        "title": citation.get("title"),
                        "chunkId": citation.get("chunkId"),
                        "verificationStatus": citation.get("verificationStatus"),
                    }
                    for citation in (((item.get("response") or {}).get("grounding") or {}).get("sourceReferences") or [])
                ],
            }
            for item in failures
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"failureCount": len(failures), "failureReasons": dict(reasons), "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
