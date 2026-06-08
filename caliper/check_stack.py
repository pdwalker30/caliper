"""Caliper stack readiness probe.

Run BEFORE an eval pass to verify the full dependency chain is reachable
and configured correctly. Prints PASS/WARN/FAIL per check with reasons so
you can fix issues before burning LLM API budget on a doomed run.

    python -m caliper.check_stack                       # probe everything default
    python -m caliper.check_stack path/to/eval_config.yaml  # also probe config-specific items
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml
from dotenv import load_dotenv

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
            _ = client.list_score_configs()  # cheap call that requires auth
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
    ]:
        path = (cfg_dir / getattr(config, attr)).resolve()
        if path.is_dir():
            results.append(CheckResult(f"{label} exists", "PASS", str(path)))
        else:
            results.append(CheckResult(f"{label} exists", "FAIL", f"missing: {path}"))
    return results


def probe_human_review(config: EvalConfig) -> list[CheckResult]:
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
        # Score configs for first test case's rubric (POC assumption: all rubrics share dims)
        from caliper.dataset_bootstrap import load_test_cases

        cfg_dir = Path(".")  # set externally before calling; passed from caller in main()
        cases = load_test_cases(Path(config.test_cases_dir))
        if not cases:
            results.append(
                CheckResult("Rubric available for queue check", "FAIL", "No test cases loaded")
            )
            return results
        rubric = cases[0][2].rubric
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
                CheckResult("Score configs exist", "PASS", f"{len(specs)} configs present")
            )

        # Queue
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
# CLI
# ---------------------------------------------------------------------------


def run_checks(config_path: Path | None) -> list[CheckResult]:
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
            results.extend(probe_human_review(config))
        except Exception:
            # parse failure already surfaced above
            pass
    return results


def print_summary(results: list[CheckResult]) -> int:
    print()
    print("=" * 72)
    print("CALIPER STACK READINESS PROBE")
    print("=" * 72)
    width = max(len(r.name) for r in results)
    for r in results:
        symbol = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL"}[r.status]
        print(f"  [{symbol}]  {r.name.ljust(width)}  {r.detail}")
    fails = sum(1 for r in results if r.status == "FAIL")
    warns = sum(1 for r in results if r.status == "WARN")
    print()
    if fails:
        print(f"{fails} FAIL, {warns} WARN. Fix FAILs before running eval_runner.")
        return 1
    if warns:
        print(f"All probes passed; {warns} WARN(s) — see details above.")
        return 0
    print("All probes passed cleanly. You're ready to run an eval pass.")
    return 0


def main() -> None:
    load_dotenv()
    config_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else None
    results = run_checks(config_path)
    sys.exit(print_summary(results))


if __name__ == "__main__":
    main()
