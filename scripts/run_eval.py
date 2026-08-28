from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import httpx


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 200-case RAG regression matrix through the HTTP API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--dataset", type=Path, required=True, help="External JSON evaluation dataset; it is not shipped with the framework.")
    parser.add_argument("--token", default=os.environ.get("RAG_BEARER_TOKEN", "dev-local-token"))
    parser.add_argument("--no-generation", action="store_true")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("eval/results-latest.json"))
    parser.add_argument("--deadline-seconds", type=int, default=2400)
    parser.add_argument("--min-accuracy", type=float, default=0.0)
    args = parser.parse_args()
    cases = json.loads(args.dataset.read_text(encoding="utf-8"))
    headers = {"Authorization": f"Bearer {args.token}", "X-Contract-Version": "1.0"}
    async with httpx.AsyncClient(base_url=args.base_url.rstrip("/"), headers=headers, timeout=120) as client:
        run_response = await client.post(
            "/rag-admin-api/v1/evaluations/run",
            json={
                "requestId": f"cli-eval-200-{int(time.time())}",
                "cases": cases,
                "concurrency": args.concurrency,
                "useGeneration": not args.no_generation,
            },
        )
        run_response.raise_for_status()
        run_id = run_response.json()["evalRunId"]
        deadline = time.monotonic() + args.deadline_seconds
        last_completed = -1
        while time.monotonic() < deadline:
            await asyncio.sleep(2)
            response = await client.get(f"/rag-admin-api/v1/evaluations/{run_id}")
            response.raise_for_status()
            data = response.json()
            completed = data.get("completed_cases", 0)
            if completed != last_completed:
                print(f"{completed}/{data.get('total_cases', len(cases))}", flush=True)
                last_completed = completed
            if data.get("status") != "RUNNING":
                break
        else:
            raise SystemExit(f"evaluation deadline exceeded: {run_id}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = data.get("summary", {})
    print(json.dumps({"evalRunId": run_id, **summary, "output": str(args.output)}, ensure_ascii=False, indent=2))
    if data.get("status") != "COMPLETED" or data.get("completed_cases") != data.get("total_cases"):
        raise SystemExit(2)
    if float(summary.get("accuracy", 0)) < args.min_accuracy:
        raise SystemExit(2)


if __name__ == "__main__":
    asyncio.run(main())
