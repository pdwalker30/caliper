"""`caliper-eval` — run an eval pass.

Usage:
    caliper-eval [--force] [--concurrency N] <path/to/eval_config.yaml>
    python -m caliper.cli.eval [--force] [--concurrency N] <path/to/eval_config.yaml>

Flags:
    --force            Ignore `idempotent: true` in the config for this run.
                       Every Cartesian cell is executed regardless of whether
                       a matching trace already exists.

    --concurrency N    Override the `concurrency` value in the config (default
                       10 in the schema). N=1 forces serial execution; raise
                       for faster passes against generous rate limits.
"""

from __future__ import annotations

import sys
from pathlib import Path

from caliper.runner import run_eval


def _usage_and_exit() -> None:
    print(
        "Usage: caliper-eval [--force] [--concurrency N] <path/to/eval_config.yaml>",
        file=sys.stderr,
    )
    sys.exit(2)


def main() -> None:
    args = sys.argv[1:]
    force = False
    concurrency: int | None = None

    # Manual flag parsing — keeping deps slim; argparse would be fine too.
    while args and args[0].startswith("--"):
        flag = args[0]
        if flag == "--force":
            force = True
            args = args[1:]
        elif flag == "--concurrency":
            if len(args) < 2:
                _usage_and_exit()
            try:
                concurrency = int(args[1])
            except ValueError:
                _usage_and_exit()
            if concurrency < 1:
                _usage_and_exit()
            args = args[2:]
        else:
            _usage_and_exit()

    if len(args) != 1:
        _usage_and_exit()
    run_eval(Path(args[0]), force=force, concurrency_override=concurrency)


if __name__ == "__main__":
    main()
