"""Summarize LangSmith traces into P50/P95/P99, TTFT, tokens, and USD cost.

Run the app (or eval) with LANGSMITH_TRACING=true first, then:

    python eval/latency_report.py
    python eval/latency_report.py --project multi-source-rag --limit 500

TTFT is only present when the trace contains ``first_token_time``; streaming
runs from this app may not populate that field in every LangSmith SDK version.
Token totals and cost are read from each root ``rag-*`` run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

from observability import langsmith_enabled, project_name  # noqa: E402
from pricing import estimate_cost, model_name  # noqa: E402


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser(description="Latency report from LangSmith traces.")
    parser.add_argument("--project", default=project_name())
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "eval" / "results" / "latency.json"))
    args = parser.parse_args()

    if not langsmith_enabled():
        raise SystemExit(
            "LangSmith is not configured. Set LANGSMITH_API_KEY and "
            "LANGSMITH_TRACING=true in .env, then run some queries."
        )

    try:
        from langsmith import Client
    except ImportError as error:
        raise SystemExit(f"Install langsmith first: {error}") from error

    client = Client()
    start_time = datetime.now(timezone.utc) - timedelta(days=args.days)
    runs = client.list_runs(
        project_name=args.project,
        run_type="chain",
        start_time=start_time,
        limit=args.limit,
    )

    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {
            "latency": [],
            "total_tokens": [],
            "input_tokens": [],
            "output_tokens": [],
            "cost": [],
            "ttft": [],
        }
    )
    for run in runs:
        if not (run.name or "").startswith("rag-"):
            continue
        if run.end_time is None:
            continue
        duration = (run.end_time - run.start_time).total_seconds()
        bucket = grouped[run.name]
        bucket["latency"].append(duration)

        input_tokens = int(run.prompt_tokens or 0)
        output_tokens = int(run.completion_tokens or 0)
        total_tokens = int(run.total_tokens or (input_tokens + output_tokens))
        model = (run.metadata or {}).get("model") or model_name()
        bucket["input_tokens"].append(float(input_tokens))
        bucket["output_tokens"].append(float(output_tokens))
        bucket["cost"].append(estimate_cost(model, input_tokens, output_tokens))
        if total_tokens:
            bucket["total_tokens"].append(float(total_tokens))
        first_token_time = getattr(run, "first_token_time", None)
        if first_token_time is not None:
            try:
                ttft = (first_token_time - run.start_time).total_seconds()
            except TypeError:
                try:
                    ttft = float(first_token_time)
                except (TypeError, ValueError):
                    ttft = None
            if ttft is not None:
                bucket["ttft"].append(ttft)

    report = {}
    for name, values in sorted(grouped.items()):
        latencies = values["latency"]
        costs = values["cost"]
        ttfts = values["ttft"]
        report[name] = {
            "runs": len(latencies),
            "latency_s": {
                "mean": sum(latencies) / len(latencies),
                "p50": percentile(latencies, 0.50),
                "p95": percentile(latencies, 0.95),
                "p99": percentile(latencies, 0.99),
            },
            "ttft_s": {
                "mean": sum(ttfts) / len(ttfts) if ttfts else None,
                "p50": percentile(ttfts, 0.50),
                "p95": percentile(ttfts, 0.95),
                "p99": percentile(ttfts, 0.99),
            },
            "avg_input_tokens": (
                sum(values["input_tokens"]) / len(values["input_tokens"])
                if values["input_tokens"]
                else None
            ),
            "avg_output_tokens": (
                sum(values["output_tokens"]) / len(values["output_tokens"])
                if values["output_tokens"]
                else None
            ),
            "avg_total_tokens": (
                sum(values["total_tokens"]) / len(values["total_tokens"])
                if values["total_tokens"]
                else None
            ),
            "avg_cost_usd": sum(costs) / len(costs) if costs else None,
            "total_cost_usd": sum(costs) if costs else None,
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
