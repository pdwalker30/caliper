"""Caliper CLI entry points.

This subpackage contains thin command-line shims that wire argv to the
library functions in `caliper.runner`, `caliper.calibration`, and
`caliper.diagnostics`. Console scripts declared in pyproject.toml route
to the `main()` callables here:

    caliper-eval        -> caliper.cli.eval:main
    caliper-check       -> caliper.cli.check:main
    caliper-calibrate   -> caliper.cli.calibrate:main

Each shim parses argv, sets up environment (load_dotenv), and delegates.
No business logic lives here.
"""
