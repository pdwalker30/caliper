"""Unit tests for the unified 1-5 scoring scale and human-aggregate derivation.

Pure logic only — no Langfuse / LiteLLM network calls. Covers:
  - judge score coercion (round/clamp/floor to integer 1-5)
  - shared rubric aggregation
  - verdict parsing (pass derivation + overall)
  - human-review score-config shape (one numeric per dim, nothing derived)
  - calibration's derivation of human pass + overall from dimension scores
"""

from __future__ import annotations

import pytest

from caliper.calibration import ScorePair, derive_pass_and_overall
from caliper.human_review import score_configs_for_rubric
from caliper.judges.rubric_judge import RubricJudge, _coerce_score
from caliper.schemas import (
    SCORE_MAX,
    SCORE_MIN,
    Rubric,
    RubricDimension,
    aggregate_scores,
)


def _rubric(aggregation: str = "weighted_mean") -> Rubric:
    return Rubric(
        dimensions=[
            RubricDimension(name="a", description="", weight=0.5, pass_threshold=4),
            RubricDimension(name="b", description="", weight=0.5, pass_threshold=3),
        ],
        aggregation=aggregation,  # type: ignore[arg-type]
        overall_pass_threshold=4,
    )


# --------------------------------------------------------------------------- coerce


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (4, 4),
        (4.0, 4),
        ("4", 4),
        (3.6, 4),   # rounds up
        (3.4, 3),   # rounds down
        (9, SCORE_MAX),    # clamps high
        (0, SCORE_MIN),    # clamps low
        (-3, SCORE_MIN),
        (None, SCORE_MIN),     # garbled -> floor (reads as a fail, never a silent pass)
        ("nonsense", SCORE_MIN),
        ({}, SCORE_MIN),
    ],
)
def test_coerce_score(raw: object, expected: int) -> None:
    assert _coerce_score(raw) == expected


def test_coerce_score_stays_in_range() -> None:
    for raw in range(-5, 12):
        assert SCORE_MIN <= _coerce_score(raw) <= SCORE_MAX


# ----------------------------------------------------------------------- aggregate


def test_aggregate_weighted_mean() -> None:
    assert aggregate_scores(_rubric(), {"a": 5, "b": 3}) == pytest.approx(4.0)


def test_aggregate_min_max() -> None:
    assert aggregate_scores(_rubric("min"), {"a": 5, "b": 3}) == 3
    assert aggregate_scores(_rubric("max"), {"a": 5, "b": 3}) == 5


def test_aggregate_zero_weight_floors_to_min() -> None:
    rubric = Rubric(
        dimensions=[RubricDimension(name="a", description="", weight=0, pass_threshold=4)],
        aggregation="weighted_mean",
    )
    assert aggregate_scores(rubric, {"a": 5}) == float(SCORE_MIN)


# -------------------------------------------------------------------------- verdict


def _judge() -> RubricJudge:
    # _parse_verdict never touches the client, so None is fine for a unit test.
    return RubricJudge(client=None, judge_model="x", judge_prompt_template="")  # type: ignore[arg-type]


def test_parse_verdict_derives_pass_and_overall() -> None:
    data = {"a": {"score": 5, "reasoning": "good"}, "b": {"score": 2, "reasoning": "meh"}}
    verdict = _judge()._parse_verdict(data, _rubric(), raw="{}")

    assert verdict.dimensions["a"].value == 5
    assert verdict.dimensions["a"].passed is True   # 5 >= 4
    assert verdict.dimensions["b"].value == 2
    assert verdict.dimensions["b"].passed is False  # 2 < 3
    assert verdict.overall_value == pytest.approx(3.5)  # 0.5*5 + 0.5*2
    assert verdict.overall_passed is False              # 3.5 < 4


def test_parse_verdict_missing_dimension_floors_to_fail() -> None:
    verdict = _judge()._parse_verdict({"a": {"score": 5}}, _rubric(), raw="{}")
    assert verdict.dimensions["b"].value == SCORE_MIN
    assert verdict.dimensions["b"].passed is False


# ---------------------------------------------------------------------- score configs


def test_score_configs_one_numeric_per_dim_nothing_derived() -> None:
    specs = score_configs_for_rubric(_rubric())

    assert [s.name for s in specs] == ["a", "b"]            # no __pass, no overall
    assert all(s.data_type == "NUMERIC" for s in specs)
    assert all(s.min_value == float(SCORE_MIN) for s in specs)
    assert all(s.max_value == float(SCORE_MAX) for s in specs)


# ----------------------------------------------------- pass/overall derivation


def test_derive_both_sides_pass_and_overall() -> None:
    # Only numeric scores exist on the trace; judge overall numeric is emitted.
    by_name = {
        "a": [ScorePair(trace_id="t", name="a", llm_value=5, human_value=5)],
        "b": [ScorePair(trace_id="t", name="b", llm_value=2, human_value=2)],
        "overall": [ScorePair(trace_id="t", name="overall", llm_value=3.5)],
    }
    derive_pass_and_overall(by_name, _rubric())

    # Pass derived identically for both sides: a 5>=4 -> pass, b 2<3 -> fail.
    assert by_name["a__pass"][0].llm_value == 1.0
    assert by_name["a__pass"][0].human_value == 1.0
    assert by_name["b__pass"][0].llm_value == 0.0
    assert by_name["b__pass"][0].human_value == 0.0
    # Overall: judge's emitted value preserved; human's derived to match.
    assert by_name["overall"][0].llm_value == pytest.approx(3.5)
    assert by_name["overall"][0].human_value == pytest.approx(3.5)
    assert by_name["overall"][0].paired is True
    assert by_name["overall__pass"][0].llm_value == 0.0   # 3.5 < 4
    assert by_name["overall__pass"][0].human_value == 0.0
    assert by_name["overall__pass"][0].paired is True


def test_derive_overall_skipped_for_incomplete_side() -> None:
    # Human scored only dim a; judge scored both.
    by_name = {
        "a": [ScorePair(trace_id="t", name="a", llm_value=5, human_value=5)],
        "b": [ScorePair(trace_id="t", name="b", llm_value=2, human_value=None)],
        "overall": [ScorePair(trace_id="t", name="overall", llm_value=3.5)],
    }
    derive_pass_and_overall(by_name, _rubric())

    assert by_name["a__pass"][0].human_value == 1.0      # per-dim still derived
    assert by_name["b__pass"][0].human_value is None     # b unscored by human
    assert by_name["overall"][0].human_value is None     # incomplete -> no overall
    assert by_name["overall__pass"][0].human_value is None
    # Judge is complete, so its overall pass is derived.
    assert by_name["overall__pass"][0].llm_value == 0.0


def test_derive_judge_overall_when_not_emitted() -> None:
    # No "overall" entry at all; judge scored every dim -> aggregate + pass.
    by_name = {
        "a": [ScorePair(trace_id="t", name="a", llm_value=5)],
        "b": [ScorePair(trace_id="t", name="b", llm_value=5)],
    }
    derive_pass_and_overall(by_name, _rubric())

    assert by_name["overall"][0].llm_value == pytest.approx(5.0)
    assert by_name["overall__pass"][0].llm_value == 1.0   # 5 >= 4
