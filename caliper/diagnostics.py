"""Stack readiness probes — library entry points.

Each probe returns a CheckResult (status + detail). `run_checks(config_path)`
runs the full battery and returns the list. The CLI shim that prints the
human-readable summary lives in `caliper.cli.check`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml

from caliper.human_review import (
    LangfuseAnnotationClient,
    score_configs_for_rubric,
)
from caliper.schemas import EvalConfig


@dataclass
class CheckResult:
    name: str
    status: str  # "PASS" | "WARN" | "FAIL"
    detail: str = ""


# ---------------------------------------------------------------------------
# Individual probes
# ---------------------------------------------------------------------------


def probe_langfuse_reachable() -> CheckResult:
    host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
    try:
        resp = httpx.get(f"{host.rstrip('/')}/api/public/health", timeout=5.0)
        if resp.is_success:
            return CheckResult(
                "Langfuse reachable",
                "PASS",
                f"{host} responded {resp.status_code}",
            )
        return CheckResult(
            "Langfuse reachable",
            "FAIL",
            f"{host} returned HTTP {resp.status_code}: {(resp.text or '')[:200]}",
        )
    except httpx.HTTPError as e:
        return CheckResult(
            "Langfuse reachable",
            "FAIL",
            f"Could not reach {host}: {e}",
        )


def probe_langfuse_auth() -> CheckResult:
    pub = os.environ.get("LANGFUSE_PUBLIC_KEY")
    sec = os.environ.get("LANGFUSE_SECRET_KEY")
    if not pub or not sec:
        return CheckResult(
            "Langfuse credentials",
            "FAIL",
            "LANGFUSE_PUBLIC_KEY and/or LANGFUSE_SECRET_KEY missing from env. "
            "Get them from Langfuse UI: Project Settings -> API Keys.",
        )
    try:
        client = LangfuseAnnotationClient.from_env()
        try:
            _ = client.list_score_configs()
            return CheckResult(
                "Langfuse credentials",
                "PASS",
                "Basic auth accepted on REST API",
            )
        finally:
            client.close()
    except Exception as e:
        return CheckResult(
            "Langfuse credentials",
            "FAIL",
            f"Auth probe failed: {e}",
        )


def probe_litellm_reachable() -> CheckResult:
    base = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000")
    try:
        resp = httpx.get(f"{base.rstrip('/')}/health/liveliness", timeout=5.0)
        if resp.is_success:
            return CheckResult(
                "LiteLLM proxy reachable",
                "PASS",
                f"{base} responded {resp.status_code}",
            )
        return CheckResult(
            "LiteLLM proxy reachable",
            "FAIL",
            f"{base} returned HTTP {resp.status_code}",
        )
    except httpx.HTTPError as e:
        return CheckResult(
            "LiteLLM proxy reachable",
            "FAIL",
            f"Could not reach {base}: {e}",
        )


def probe_litellm_master_key() -> CheckResult:
    key = os.environ.get("LITELLM_MASTER_KEY")
    if not key:
        return CheckResult(
            "LiteLLM master key",
            "FAIL",
            "LITELLM_MASTER_KEY missing from env (must start with 'sk-')",
        )
    if not key.startswith("sk-"):
        return CheckResult(
            "LiteLLM master key",
            "WARN",
            "Key set but doesn't start with 'sk-' — LiteLLM expects this prefix",
        )
    return CheckResult("LiteLLM master key", "PASS", "Present and well-formed")


def probe_eval_config_paths(config_path: Path) -> list[CheckResult]:
    results: list[CheckResult] = []
    try:
        with config_path.open(encoding="utf-8") as f:
            config = EvalConfig.model_validate(yaml.safe_load(f))
    except Exception as e:
        return [CheckResult("Eval config parse", "FAIL", str(e))]

    results.append(CheckResult("Eval config parse", "PASS", f"{config_path}"))

    cfg_dir = config_path.parent
    for label, attr in [
        ("test_cases_dir", "test_cases_dir"),
        ("prompts_dir", "prompts_dir"),
        ("judge_prompts_dir", "judge_prompts_dir"),
        ("rubrics_dir", "rubrics_dir"),
    ]:
        path = (cfg_dir / getattr(config, attr)).resolve()
        if path.is_dir():
            results.append(CheckResult(f"{label} exists", "PASS", str(path)))
        else:
            results.append(CheckResult(f"{label} exists", "FAIL", f"missing: {path}"))

    # Validate every test case's metadata.json against TestCaseMetadata AND
    # resolve its rubric — unconditional, not gated by human_review. This
    # catches malformed metadata (typos in field names, wrong shapes) BEFORE
    # any LLM call fires.
    from caliper.dataset_bootstrap import (
        load_rubrics,
        load_test_cases,
        resolve_rubric_for_eval_type,
    )

    try:
        test_cases_dir = (cfg_dir / config.test_cases_dir).resolve()
        rubrics_dir = (cfg_dir / config.rubrics_dir).resolve()
        cases = load_test_cases(test_cases_dir)
        results.append(
            CheckResult(
                "Test cases parse",
                "PASS",
                f"{len(cases)} test case(s) validated against TestCaseMetadata",
            )
        )
        try:
            rubrics = load_rubrics(rubrics_dir)
            unresolved: list[str] = []
            for case_id, _, meta in cases:
                try:
                    resolve_rubric_for_eval_type(
                        meta.eval_type,
                        rubrics,
                        config.default_rubric,
                        config.rubric_by_eval_type,
                    )
                except Exception as exc:
                    unresolved.append(f"{case_id} -> {exc}")
            if unresolved:
                results.append(
                    CheckResult(
                        "Rubric resolution",
                        "FAIL",
                        f"{len(unresolved)} test case(s) cannot resolve a rubric: "
                        + "; ".join(unresolved[:3]),
                    )
                )
            else:
                results.append(
                    CheckResult(
                        "Rubric resolution",
                        "PASS",
                        f"every test case's eval_type resolves to a rubric in {rubrics_dir.name}/",
                    )
                )
        except Exception as e:
            results.append(CheckResult("Rubric resolution", "FAIL", str(e)))
    except Exception as e:
        results.append(CheckResult("Test cases parse", "FAIL", str(e)))

    return results


def probe_human_review(config: EvalConfig, config_path: Path) -> list[CheckResult]:
    if not config.human_review or not config.human_review.enabled:
        return [
            CheckResult(
                "Human review configured",
                "PASS",
                "Disabled — skipping queue and score-config checks",
            )
        ]

    results: list[CheckResult] = []
    try:
        client = LangfuseAnnotationClient.from_env()
    except Exception as e:
        results.append(CheckResult("Human review setup", "FAIL", str(e)))
        return results

    try:
        from caliper.dataset_bootstrap import (
            load_rubrics,
            load_test_cases,
            resolve_rubric_for_eval_type,
        )

        test_cases_dir = (config_path.parent / config.test_cases_dir).resolve()
        rubrics_dir = (config_path.parent / config.rubrics_dir).resolve()
        rubrics = load_rubrics(rubrics_dir)
        cases = load_test_cases(test_cases_dir)
        if not cases:
            results.append(
                CheckResult(
                    "Rubric available for queue check", "FAIL", "No test cases loaded"
                )
            )
            return results
        # Use first case's eval_type to derive a representative rubric for
        # the ScoreConfig check.
        rubric = resolve_rubric_for_eval_type(
            cases[0][2].eval_type,
            rubrics,
            config.default_rubric,
            config.rubric_by_eval_type,
        )
        specs = score_configs_for_rubric(rubric)
        existing = {cfg["name"]: cfg for cfg in client.list_score_configs()}
        missing = [s.name for s in specs if s.name not in existing]
        if missing:
            results.append(
                CheckResult(
                    "Score configs exist",
                    "WARN",
                    f"Missing in Langfuse: {missing}. "
                    f"Caliper will create them at run time if auto_create=true.",
                )
            )
        else:
            results.append(
                CheckResult(
                    "Score configs exist", "PASS", f"{len(specs)} configs present"
                )
            )

        queue = client.find_queue(config.human_review.queue_name)
        if queue:
            results.append(
                CheckResult(
                    "Annotation queue exists",
                    "PASS",
                    f"Found queue {config.human_review.queue_name!r} (id={queue['id']})",
                )
            )
        else:
            results.append(
                CheckResult(
                    "Annotation queue exists",
                    "WARN",
                    f"Queue {config.human_review.queue_name!r} not found. "
                    f"Caliper will create it at run time if auto_create=true, "
                    f"or create manually in the Langfuse UI under Annotation Queues.",
                )
            )
    finally:
        client.close()
    return results


# ---------------------------------------------------------------------------
# Aggregate runner
# ---------------------------------------------------------------------------


def run_checks(config_path: Path | None) -> list[CheckResult]:
    """Run the full probe battery. Pass `config_path` to also probe config-
    specific items (asset folders, score configs, queue lookup)."""
    results: list[CheckResult] = []
    results.append(probe_langfuse_reachable())
    results.append(probe_langfuse_auth())
    results.append(probe_litellm_reachable())
    results.append(probe_litellm_master_key())
    if config_path is not None:
        results.extend(probe_eval_config_paths(config_path))
        try:
            with config_path.open(encoding="utf-8") as f:
                config = EvalConfig.model_validate(yaml.safe_load(f))
            results.extend(probe_human_review(config, config_path))
        except Exception:
            pass
    return results
