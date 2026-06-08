"""`caliper-calibrate` — read paired LLM + human scores and print/save a report.

Usage:
    caliper-calibrate <path/to/eval_config.yaml>
    python -m caliper.cli.calibrate <path/to/eval_config.yaml>
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from caliper.calibration import (
    build_score_pairs,
    fetch_all_scores_for_campaign,
    fetch_traces_for_campaign,
    print_report,
    report,
    write_csv,
)
from caliper.human_review import LangfuseAnnotationClient
from caliper.schemas import EvalConfig


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: caliper-calibrate <path/to/eval_config.yaml>", file=sys.stderr)
        sys.exit(2)

    load_dotenv()
    cfg_path = Path(sys.argv[1])
    with cfg_path.open(encoding="utf-8") as f:
        config = EvalConfig.model_validate(yaml.safe_load(f))

    client = LangfuseAnnotationClient.from_env()
    try:
        print(f"[caliper] fetching traces for campaign {config.name!r}")
        traces = fetch_traces_for_campaign(client, config.name)
        if not traces:
            print(f"[caliper] no traces found tagged campaign:{config.name}")
            sys.exit(0)
        print(f"[caliper] found {len(traces)} trace(s)")

        print("[caliper] fetching scores")
        scores = fetch_all_scores_for_campaign(client, config.name)
        print(f"[caliper] found {len(scores)} score(s) total")

        by_name = build_score_pairs(scores, set(traces.keys()))
        reports = report(by_name)
        print_report(reports)

        ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        out = Path("results") / f"calibration-{config.name}-{ts}.csv"
        write_csv(reports, by_name, out)
        print(f"[caliper] CSV written to {out}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
