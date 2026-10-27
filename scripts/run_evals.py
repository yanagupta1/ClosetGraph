#!/usr/bin/env python
"""Reusable eval runner.

A thin CLI over the existing harness functions - it does not implement any new
evaluation logic. It selects a suite, optionally repeats it (to surface model
nondeterminism), flushes Langfuse, and writes one structured JSON report with
the run configuration, per-case results, and any errors.

Examples:
    python scripts/run_evals.py --suite pairwise --repeats 3 --delay 60
    python scripts/run_evals.py --suite golden --delay 60 --out eval_reports/golden.json
    GROQ_MODEL=openai/gpt-oss-120b python scripts/run_evals.py --suite pairwise --repeats 2 --delay 60

Live runs make Groq calls, so use --delay to throttle under the free-tier limit.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running as `python scripts/run_evals.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from langfuse import get_client

from closetgraph.eval_harness import run_golden_set, run_pairwise_set


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ClosetGraph evals (golden and/or pairwise) and save a JSON report."
    )
    parser.add_argument(
        "--suite", choices=["golden", "pairwise", "both"], default="both",
        help="which suite to run (default: both)",
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="repeat count: golden runs N full passes; pairwise repeats each case N times (default: 1)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.0,
        help="seconds between model calls to throttle for free-tier rate limits (default: 0)",
    )
    parser.add_argument("--golden-path", default="tests/eval/golden_set.yaml")
    parser.add_argument("--pairwise-path", default="tests/eval/pairwise_set.yaml")
    parser.add_argument("--wardrobe-path", default="docs/wardrobe.json")
    parser.add_argument(
        "--out", default=None,
        help="output JSON path (default: eval_reports/report_<UTC timestamp>.json)",
    )
    args = parser.parse_args()

    load_dotenv()

    report = {
        "config": {
            "suite": args.suite,
            "repeats": args.repeats,
            "delay_seconds": args.delay,
            "golden_path": args.golden_path,
            "pairwise_path": args.pairwise_path,
            "wardrobe_path": args.wardrobe_path,
            "groq_model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            "judge_model": os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b"),
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
        "golden": [],
        "pairwise": [],
        "errors": [],
    }

    if args.suite in ("golden", "both"):
        for i in range(args.repeats):
            try:
                report["golden"].append(
                    run_golden_set(args.golden_path, args.wardrobe_path, delay_seconds=args.delay)
                )
            except Exception as error:  # record and continue; don't lose other repeats
                report["errors"].append({"suite": "golden", "repeat": i, "error": repr(error)})
                print(f"golden repeat {i} failed: {error!r}")

    if args.suite in ("pairwise", "both"):
        try:
            report["pairwise"] = run_pairwise_set(
                args.pairwise_path, args.wardrobe_path,
                repeats=args.repeats, delay_seconds=args.delay,
            )
        except Exception as error:
            report["errors"].append({"suite": "pairwise", "error": repr(error)})
            print(f"pairwise failed: {error!r}")

    try:
        get_client().flush()
    except Exception:
        pass  # tracing is best-effort; never fail the report over it

    report["config"]["finished_at"] = datetime.now(timezone.utc).isoformat()

    out_path = args.out or (
        f"eval_reports/report_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"wrote {out_path} ({len(report['errors'])} error(s))")


if __name__ == "__main__":
    main()
