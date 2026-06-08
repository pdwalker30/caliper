"""`caliper-eval` — run an eval pass.

Usage:
    caliper-eval [--force] <path/to/eval_config.yaml>
    python -m caliper.cli.eval [--force] <path/to/eval_config.yaml>

Flags:
    --force   Ignore `idempotent: true` in the config for this run. Every
              Cartesian cell is executed regardless of whether a matching
              trace already exists in the campaign. Useful for "I just want
              fresh results this once" without editing the config file.
"""

from __future__ import annotations

import sys
from pathlib import Path

from caliper.runner import run_eval


def _usage_and_exit() -> None:
    print(
        "Usage: caliper-eval [--force] <path/to/eval_config.yaml>",
        file=sys.stderr,
    )
    sys.exit(2)


def main() -> None:
    args = sys.argv[1:]
    force = False
    if args and args[0] == "--force":
        force = True
        args = args[1:]
    if len(args) != 1:
        _usage_and_exit()
    run_eval(Path(args[0]), force=force)


if __name__ == "__main__":
    main()
