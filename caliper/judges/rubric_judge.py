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
    SCORE_MAX,
    SCORE_MIN,
    DimensionScore,
    JudgePromptMetadata,
    JudgeVerdict,
    Rubric,
    TestCaseMetadata,
    aggregate_scores,
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
        rubric: Rubric,
        llm_output: str,
        trace_observations: list[Any] | None = None,
        mode: JudgeMode = "anchored",
    ) -> JudgeVerdict:
        rendered = self._render_prompt(
            test_case_input, test_case_metadata, rubric, llm_output, mode=mode
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

        verdict = self._parse_verdict(data, rubric, raw=result.output)
        # Stash the fully-substituted prompt so the runner can stamp it on the
        # judge generation — that's what makes the judge's question visible to a
        # human annotator opening the trace.
        verdict.rendered_prompt = rendered
        return verdict

    # ------------------------------------------------------------------ render

    def _render_prompt(
        self,
        test_case_input: str,
        test_case_metadata: TestCaseMetadata,
        rubric: Rubric,
        llm_output: str,
        mode: JudgeMode = "anchored",
    ) -> str:
        dimensions_block = "\n".join(
            f"- {d.name} (weight {d.weight}, pass when score >= {d.pass_threshold}, scale {d.scale}): {d.description}"
            for d in rubric.dimensions
        )

        # Mid-scale integer in the example signals the expected shape: an integer
        # on the configured scale, not a 0-1 float.
        example_score = (SCORE_MIN + SCORE_MAX) // 2
        example_obj = {
            d.name: {"score": example_score, "reasoning": "<your reasoning>"}
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
            value = _coerce_score(entry.get("score"))
            reasoning = str(entry.get("reasoning", ""))[:2000]
            dim_scores[dim.name] = DimensionScore(
                name=dim.name,
                value=value,
                passed=value >= dim.pass_threshold,
                reasoning=reasoning,
            )

        overall = aggregate_scores(rubric, {n: s.value for n, s in dim_scores.items()})
        return JudgeVerdict(
            dimensions=dim_scores,
            overall_value=overall,
            overall_passed=overall >= rubric.overall_pass_threshold,
            raw_response=raw[:5000],
        )


def _coerce_score(raw_score: Any) -> int:
    """Coerce a judge-returned score to an integer on the 1-5 scale.

    Rounds (the judge is asked for integers but may return 4.0 or "4"), clamps
    into range, and floors a missing/garbled score to the worst grade so a
    malformed judge response reads as a fail, never a silent pass.
    """
    try:
        return round(max(float(SCORE_MIN), min(float(SCORE_MAX), float(raw_score))))
    except (TypeError, ValueError):
        return SCORE_MIN
