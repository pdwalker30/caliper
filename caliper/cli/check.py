"""`caliper-check` — probe the stack and print a PASS/WARN/FAIL readiness report.

Usage:
    caliper-check                                 # probe stack-only checks
    caliper-check <path/to/eval_config.yaml>      # also probe config-specific items
    python -m caliper.cli.check <config>          # same, via -m
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

from caliper.diagnostics import CheckResult, run_checks


def _print_summary(results: list[CheckResult]) -> int:
    print()
    print("=" * 72)
    print("CALIPER STACK READINESS PROBE")
    print("=" * 72)
    width = max(len(r.name) for r in results)
    for r in results:
        symbol = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL"}[r.status]
        print(f"  [{symbol}]  {r.name.ljust(width)}  {r.detail}")
    fails = sum(1 for r in results if r.status == "FAIL")
    warns = sum(1 for r in results if r.status == "WARN")
    print()
    if fails:
        print(f"{fails} FAIL, {warns} WARN. Fix FAILs before running caliper-eval.")
        return 1
    if warns:
        print(f"All probes passed; {warns} WARN(s) — see details above.")
        return 0
    print("All probes passed cleanly. You're ready to run an eval pass.")
    return 0


def main() -> None:
    load_dotenv()
    config_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else None
    results = run_checks(config_path)
    sys.exit(_print_summary(results))


if __name__ == "__main__":
    main()
