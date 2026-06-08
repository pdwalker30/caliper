"""`caliper-eval` — run an eval pass.

Usage:
    caliper-eval <path/to/eval_config.yaml>
    python -m caliper.cli.eval <path/to/eval_config.yaml>
"""

from __future__ import annotations

import sys
from pathlib import Path

from caliper.runner import run_eval


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: caliper-eval <path/to/eval_config.yaml>", file=sys.stderr)
        sys.exit(2)
    run_eval(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
