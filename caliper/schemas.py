"""Pydantic data contracts shared across Caliper.

These models define the on-disk format for test cases, prompts, judge prompts,
and the per-pass eval configuration — plus the in-memory shape of judge verdicts
and per-dimension scores. Every other module in Caliper consumes these types;
this file is the schema source of truth.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Rubric — the contract between a test case and the judge
# ---------------------------------------------------------------------------


class RubricDimension(BaseModel):
    """One scored dimension within a rubric."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    weight: float = Field(ge=0, le=1)
    pass_threshold: float = Field(ge=0, le=1)
    scale: str = "0-1"


class Rubric(BaseModel):
    """A rubric — multiple dimensions plus aggregation policy.

    Rubrics are first-class reusable artifacts that live in rubrics_dir as
    `rubrics/<name>/rubric.json`. Test cases reference them by name (string)
    via `TestCaseMetadata.rubric`, or fall through to the config's
    `default_rubric`. Inline Rubric objects on test cases are still accepted
    (deprecated) for backward compat.
    """

    model_config = ConfigDict(extra="forbid")

    description: str = ""
    dimensions: list[RubricDimension]
    aggregation: Literal["weighted_mean", "min", "max"] = "weighted_mean"
    overall_pass_threshold: float = Field(ge=0, le=1, default=0.7)


# ---------------------------------------------------------------------------
# On-disk metadata.json shapes (per asset folder)
# ---------------------------------------------------------------------------


class TestCaseMetadata(BaseModel):
    """`metadata.json` next to a test-case content file.

    Test cases hold PURE INPUT DATA — what's being evaluated and its ground
    truth. They do NOT declare which rubric to use; that's an eval-pass
    orchestration concern that lives in EvalConfig. The `eval_type` field
    is the only contract between test data and evaluation logic — the
    EvalConfig maps eval_types to rubrics via default_rubric or
    rubric_by_eval_type.

    `expected` holds eval-type-specific reference data (seeded bugs,
    required tools, expected outcomes, etc.). Extra fields are allowed
    for forward-compat.
    """

    model_config = ConfigDict(extra="allow")

    eval_type: str
    expected: dict[str, Any] = Field(default_factory=dict)
    description: str = ""
    tags: list[str] = Field(default_factory=list)


class PromptMetadata(BaseModel):
    """`metadata.json` next to a prompt.txt."""

    model_config = ConfigDict(extra="allow")

    name: str
    version: str = "1"
    description: str = ""
    tags: list[str] = Field(default_factory=list)


class JudgePromptMetadata(BaseModel):
    """`metadata.json` next to a judge_prompt.txt."""

    model_config = ConfigDict(extra="allow")

    name: str
    version: str = "1"
    description: str = ""
    # Optional pin — if set, eval_config can default to this judge model
    default_judge_model: str | None = None


# ---------------------------------------------------------------------------
# Judge verdict — the in-memory result of one judging pass
# ---------------------------------------------------------------------------


class DimensionScore(BaseModel):
    """The judge's verdict for one rubric dimension on one trace."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float = Field(ge=0, le=1)
    passed: bool
    reasoning: str = ""


class JudgeVerdict(BaseModel):
    """Aggregated judge output for one (prompt × model × test_case × iter)."""

    model_config = ConfigDict(extra="forbid")

    dimensions: dict[str, DimensionScore]
    overall_value: float = Field(ge=0, le=1)
    overall_passed: bool
    raw_response: str = ""


# ---------------------------------------------------------------------------
# Eval configuration — the YAML driver for one eval pass
# ---------------------------------------------------------------------------


class RetryConfig(BaseModel):
    """Retry / backoff for LLM calls through the LiteLLM proxy.

    Applied to every model-under-test call AND every judge call. Default values
    are tuned for typical hosted-LLM rate limits — sustained 429s back off
    to ~60s waits within 5 attempts, which is enough to clear most quota
    refreshes without giving up on the call.
    """

    model_config = ConfigDict(extra="forbid")

    max_attempts: int = Field(ge=1, default=5)
    initial_wait_seconds: float = Field(ge=0, default=2.0)
    max_wait_seconds: float = Field(ge=0, default=60.0)
    exponential_base: float = Field(ge=1, default=2.0)
    jitter_seconds: float = Field(ge=0, default=1.0)
    # Status codes that trigger a retry. 429 is rate-limit; 5xx is upstream
    # transient. 408 / 425 also worth retrying on if your provider returns them.
    retry_on_statuses: list[int] = Field(
        default_factory=lambda: [408, 425, 429, 500, 502, 503, 504]
    )


class HumanReviewConfig(BaseModel):
    """Optional human-in-the-loop calibration of the LLM judge.

    When enabled, the eval runner samples N traces and enqueues them into a
    Langfuse Annotation Queue for manual scoring. Caliper's calibration
    module then reads paired (LLM, human) scores and reports agreement
    metrics so you know how much to trust the LLM judge.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    queue_name: str = "caliper-human-review"

    # How traces are selected for human review:
    #   auto:       Caliper picks target N from min/max samples + min/max pct
    #               below, distributed evenly across Runs. RECOMMENDED.
    #   stratified: explicit N samples per Run via `samples_per_run`
    #   random:     coin flip per trace with `sample_rate` probability
    #   all:        every trace goes to the queue (don't do this with humans)
    sample_strategy: Literal["auto", "stratified", "random", "all"] = "auto"

    # Auto strategy: caps and floors. Caliper picks an effective sample
    # count that's:
    #   - at least max(min_samples, ceil(min_pct * total_cells))
    #   - at most min(max_samples, ceil(max_pct * total_cells), total_cells)
    # When constraints conflict (big matrices), ceiling wins to protect
    # human time. Defaults derived from inter-rater-reliability literature:
    # below n=20 the CI on κ and r becomes uninformative; past n=100 the CIs
    # are already narrow enough that more samples don't change conclusions.
    min_samples: int = Field(ge=0, default=20)
    min_pct: float = Field(ge=0, le=1, default=0.05)
    max_samples: int = Field(ge=1, default=100)
    max_pct: float = Field(ge=0, le=1, default=0.20)

    # Legacy / explicit overrides — respected when sample_strategy is set to
    # the matching value. The auto strategy ignores these.
    samples_per_run: int = Field(ge=1, default=2)
    sample_rate: float = Field(ge=0, le=1, default=0.15)

    # When True, Caliper will attempt to create the queue and any missing
    # ScoreConfigs via the Langfuse REST API. When False (or if the API call
    # fails), the queue + configs must already exist in Langfuse — the eval
    # pass will WARN but not fail.
    auto_create: bool = True


class EvalConfig(BaseModel):
    """The top-level YAML file Caliper reads to drive one eval pass.

    Folder paths (`test_cases_dir`, `prompts_dir`, `judge_prompts_dir`) are
    resolved relative to the config file's directory, so configs are portable
    when whole examples/ folders move.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    dataset_name: str
    test_cases_dir: str
    prompts_dir: str
    judge_prompts_dir: str
    # Folder containing rubrics, parallel to prompts/ and judge_prompts/.
    # Each rubric is a sub-folder with rubric.json. Test cases do NOT
    # reference rubrics — assignment lives entirely in this config.
    rubrics_dir: str = "rubrics"
    # Default rubric used for every test case unless rubric_by_eval_type
    # carries a more specific mapping for that test case's eval_type.
    # Required in practice — without it, test cases have no rubric.
    default_rubric: str | None = None
    # Optional per-eval-type rubric override. When set, a test case whose
    # eval_type matches a key here uses the mapped rubric; otherwise it
    # falls through to default_rubric. Useful when a single eval pass mixes
    # multiple eval_types (e.g., code_review + agent_tool_call in one
    # campaign) and each type wants its own scoring scheme.
    rubric_by_eval_type: dict[str, str] = Field(default_factory=dict)
    judge_prompt: str
    judge_model: str
    models: list[str] = Field(min_length=1)
    iterations: int = Field(ge=1, default=1)
    # Number of Cartesian cells executed concurrently. Each cell does 2 LLM
    # calls (model under test + judge) so wall-clock speed-up is ~Nx for an
    # N-worker pool, up to provider rate limits. The retry block above will
    # back off on 429s — concurrency interacts cleanly with that.
    # Cap at 1 to force serial execution; raise for faster passes.
    concurrency: int = Field(ge=1, default=10)
    extra_run_metadata: dict[str, Any] = Field(default_factory=dict)
    human_review: HumanReviewConfig | None = None
    retry: RetryConfig = Field(default_factory=RetryConfig)
    # Judge modes to run per cell. "anchored" gives the judge the test case's
    # expected/ground-truth reference; "blind" hides it so the judge scores
    # using only the code + the LLM's review. Running both side-by-side
    # surfaces reference bias: if anchored consistently scores higher than
    # blind, the LLM under test is being artificially rewarded by the
    # judge already knowing the answer.
    #
    # Scores from the blind mode get a __blind suffix on every score name
    # (e.g., `finds_bug__blind`, `finds_bug__pass__blind`) so they coexist
    # with the anchored scores on the same trace.
    judge_modes: list[Literal["anchored", "blind"]] = Field(
        default_factory=lambda: ["anchored"]
    )
    # Optional subsetting. When non-empty, only folder ids in the list
    # participate in the Cartesian. Empty / omitted = use ALL folders in
    # the respective dir (current behavior).
    #
    # The Langfuse Dataset still gets bootstrapped with ALL test cases — the
    # subset only affects which cells get executed this pass. So toggling
    # subsets doesn't shrink or churn the persistent dataset.
    #
    # Missing ids (in the list but not present on disk) produce a WARN but
    # don't fail the pass.
    prompt_ids: list[str] = Field(default_factory=list)
    test_case_ids: list[str] = Field(default_factory=list)
    # Map LiteLLM-returned model names to names Langfuse's built-in pricing
    # map recognizes. Needed when models are hosted behind providers Langfuse
    # doesn't know about by default (e.g., Databricks-served Llama, internal
    # fine-tuned endpoints) — without this, the generation's `model` field
    # doesn't match Langfuse's pricing table and cost stays at 0.
    #
    # Format: {litellm_returned_name: langfuse_recognized_name, ...}
    # Example:
    #   databricks-llama-3-70b-instruct: llama-3-70b-instruct
    #   databricks/claude-sonnet-via-bedrock: claude-sonnet-4-5
    #
    # Alternative: register the custom model in the Langfuse UI under
    # Settings -> Models with the LiteLLM-returned name and your own prices.
    langfuse_model_mapping: dict[str, str] = Field(default_factory=dict)
    # Opt-in idempotency: when True, Caliper computes a hash per Cartesian cell
    # (campaign + prompt text + judge prompt text + model + judge model + snippet
    # content + expected + rubric + iteration) and skips any cell whose hash
    # already appears on a trace in this campaign. Edit a prompt -> hash changes
    # -> cell re-runs. No version bumping required.
    idempotent: bool = False
