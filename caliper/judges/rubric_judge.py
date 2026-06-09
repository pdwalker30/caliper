"""Generic LLM-as-judge driven by the rubric on each test case.

The judge prompt is a TEMPLATE — Caliper substitutes rubric dimensions, the
test case description, the expected reference data, and the LLM output into
placeholder slots. The judge model returns structured JSON that the parser
maps back into per-dimension DimensionScores and an aggregated overall.

Placeholders supported in judge_prompt.txt:

  {test_case_input}        — the raw content being judged
  {test_case_description}  — TestCaseMetadata.description
  {expected_json}          — TestCaseMetadata.expected, pretty-printed JSON
  {rubric_dimensions}      — multi-line block of rubric dimensions
  {llm_output}             — what the system under test produced
  {response_format_example} — example JSON shape the judge must return

Why not Jinja2? For a POC, str.format with .replace fallbacks is cheaper to
reason about and one fewer dep. Swap to Jinja later if templates get richer.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from caliper.judges.base import JudgeAdapter
from caliper.litellm_client import LiteLLMProxyClient
from caliper.schemas import (
    DimensionScore,
    JudgePromptMetadata,
    JudgeVerdict,
    Rubric,
    TestCaseMetadata,
)

JudgeMode = Literal["anchored", "blind"]


class RubricJudge(JudgeAdapter):
    """LLM-as-judge that scores against the rubric on the test case metadata."""

    eval_type = "*"  # matches any eval_type — the rubric is the contract

    def __init__(
        self,
        client: LiteLLMProxyClient,
        judge_model: str,
        judge_prompt_template: str,
        judge_prompt_metadata: JudgePromptMetadata | None = None,
    ):
        self.client = client
        self.judge_model = judge_model
        self.judge_prompt_template = judge_prompt_template
        self.judge_prompt_metadata = judge_prompt_metadata

    def evaluate(
        self,
        test_case_input: str,
        test_case_metadata: TestCaseMetadata,
        llm_output: str,
        trace_observations: list[Any] | None = None,
        mode: JudgeMode = "anchored",
    ) -> JudgeVerdict:
        rendered = self._render_prompt(
            test_case_input, test_case_metadata, llm_output, mode=mode
        )

        result = self.client.complete(
            model=self.judge_model,
            messages=[{"role": "user", "content": rendered}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        try:
            data = json.loads(result.output)
        except json.JSONDecodeError as e:
            # Surface raw response on parse failure so debugging is possible
            raise ValueError(
                f"Judge returned non-JSON response: {e}\n\nRaw:\n{result.output[:2000]}"
            ) from e

        return self._parse_verdict(data, test_case_metadata.rubric, raw=result.output)

    # ------------------------------------------------------------------ render

    def _render_prompt(
        self,
        test_case_input: str,
        test_case_metadata: TestCaseMetadata,
        llm_output: str,
        mode: JudgeMode = "anchored",
    ) -> str:
        rubric = test_case_metadata.rubric

        dimensions_block = "\n".join(
            f"- {d.name} (weight {d.weight}, pass when score >= {d.pass_threshold}, scale {d.scale}): {d.description}"
            for d in rubric.dimensions
        )

        example_obj = {
            d.name: {"score": 0.0, "reasoning": "<your reasoning>"}
            for d in rubric.dimensions
        }
        response_format_example = json.dumps(example_obj, indent=2)

        # In blind mode, the judge does NOT see the ground-truth reference.
        # It scores using only the code + the LLM's review + the rubric
        # dimension descriptions. Running both modes side-by-side surfaces
        # how much the reference is biasing the anchored scores upward.
        if mode == "blind":
            expected_str = (
                "(reference deliberately withheld — judge based solely on the "
                "code, the rubric dimensions, and the review below)"
            )
        else:
            expected_str = json.dumps(test_case_metadata.expected, indent=2)

        # Use sequential .replace() rather than .format() — test case content
        # often contains literal {curly_braces} (code, JSON examples) that
        # would break str.format.
        rendered = self.judge_prompt_template
        replacements = {
            "{test_case_input}": test_case_input,
            "{test_case_description}": test_case_metadata.description,
            "{expected_json}": expected_str,
            "{rubric_dimensions}": dimensions_block,
            "{llm_output}": llm_output,
            "{response_format_example}": response_format_example,
        }
        for placeholder, value in replacements.items():
            rendered = rendered.replace(placeholder, value)
        return rendered

    # ------------------------------------------------------------------- parse

    def _parse_verdict(
        self,
        data: dict[str, Any],
        rubric: Rubric,
        raw: str,
    ) -> JudgeVerdict:
        dim_scores: dict[str, DimensionScore] = {}
        for dim in rubric.dimensions:
            entry = data.get(dim.name) or {}
            raw_score = entry.get("score", 0.0)
            try:
                value = max(0.0, min(1.0, float(raw_score)))
            except (TypeError, ValueError):
                value = 0.0
            reasoning = str(entry.get("reasoning", ""))[:2000]
            dim_scores[dim.name] = DimensionScore(
                name=dim.name,
                value=value,
                passed=value >= dim.pass_threshold,
                reasoning=reasoning,
            )

        overall = _aggregate(rubric, dim_scores)
        return JudgeVerdict(
            dimensions=dim_scores,
            overall_value=overall,
            overall_passed=overall >= rubric.overall_pass_threshold,
            raw_response=raw[:5000],
        )


def _aggregate(rubric: Rubric, scores: dict[str, DimensionScore]) -> float:
    if rubric.aggregation == "weighted_mean":
        total_weight = sum(d.weight for d in rubric.dimensions)
        if total_weight == 0:
            return 0.0
        return sum(scores[d.name].value * d.weight for d in rubric.dimensions) / total_weight
    if rubric.aggregation == "min":
        return min(scores[d.name].value for d in rubric.dimensions)
    if rubric.aggregation == "max":
        return max(scores[d.name].value for d in rubric.dimensions)
    return 0.0
