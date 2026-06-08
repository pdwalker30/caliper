"""Judge adapter protocol.

A judge takes (test_case_input, test_case_metadata, llm_output) plus optional
trajectory data, and returns a JudgeVerdict — per-dimension scores plus an
aggregate. The protocol is intentionally narrow so adding new judges (e.g. a
programmatic tool-call diff judge for agent evals) doesn't require any
framework changes — just a new class implementing this shape.
"""

from __future__ import annotations

from typing import Any, Protocol

from caliper.schemas import JudgeVerdict, TestCaseMetadata


class JudgeAdapter(Protocol):
    """A pluggable judge."""

    eval_type: str
    """The eval_type discriminator this judge handles. Use '*' to handle any."""

    def evaluate(
        self,
        test_case_input: str,
        test_case_metadata: TestCaseMetadata,
        llm_output: str,
        trace_observations: list[Any] | None = None,
    ) -> JudgeVerdict:
        ...
