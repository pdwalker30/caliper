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
    derive_pass_and_overall,
    fetch_all_scores_for_campaign,
    fetch_traces_for_campaign,
    print_report,
    report,
    write_csv,
)
from caliper.dataset_bootstrap import load_rubrics
from caliper.human_review import LangfuseAnnotationClient
from caliper.schemas import EvalConfig, Rubric


def _resolve_calibration_rubric(
    cfg_path: Path,
    config: EvalConfig,
) -> Rubric | None:
    """Pick the rubric whose thresholds drive human pass/overall derivation.

    Mirrors the runner's POC assumption that one rubric is representative for a
    campaign: prefer `default_rubric`, else the sole rubric, else the first with
    a WARN. Returns None (skip derivation) if no rubric can be loaded.
    """
    rubrics_dir = (cfg_path.parent / config.rubrics_dir).resolve()
    try:
        rubrics = load_rubrics(rubrics_dir)
    except Exception as e:
        print(f"[caliper] WARN: could not load rubrics ({e}); "
              f"skipping human pass/overall derivation", file=sys.stderr)
        return None
    if not rubrics:
        print("[caliper] WARN: no rubrics found; skipping human pass/overall derivation",
              file=sys.stderr)
        return None
    if config.default_rubric and config.default_rubric in rubrics:
        return rubrics[config.default_rubric]
    if len(rubrics) == 1:
        return next(iter(rubrics.values()))
    name = sorted(rubrics)[0]
    print(f"[caliper] WARN: multiple rubrics and no usable default_rubric; "
          f"using {name!r} for human pass/overall derivation", file=sys.stderr)
    return rubrics[name]


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: caliper-calibrate <path/to/eval_config.yaml>", file=sys.stderr)
        sys.exit(2)

    load_dotenv()
    cfg_path = Path(sys.argv[1])
    with cfg_path.open(encoding="utf-8") as f:
        config = EvalConfig.model_validate(yaml.safe_load(f))

    rubric = _resolve_calibration_rubric(cfg_path, config)

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
        if rubric is not None:
            # Only numeric scores are written to Langfuse; derive the pass flags
            # and overall (both sides) from those before reporting agreement.
            derive_pass_and_overall(by_name, rubric)
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
