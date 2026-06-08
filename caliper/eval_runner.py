"""Cartesian eval runner.

Reads a YAML EvalConfig, walks (prompt x model x test_case x iteration), and for
each combination opens a Langfuse trace with three observations:

  1. parent root span     — bound to the Langfuse Dataset Run (run_name)
  2. child generation     — the LLM call (token usage attached, server-side cost)
  3. child generation     — the judge call

Then it attaches one Score per rubric dimension plus boolean pass scores plus
an overall, all stamped on the parent trace so the Experiments comparison view
can aggregate them per (prompt, model) run.

Usage:

    python -m caliper.eval_runner path/to/eval_config.yaml
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from itertools import product
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langfuse import Langfuse

from caliper.dataset_bootstrap import bootstrap_dataset
from caliper.judges.rubric_judge import RubricJudge
from caliper.litellm_client import LiteLLMProxyClient
from caliper.schemas import (
    EvalConfig,
    JudgePromptMetadata,
    PromptMetadata,
    TestCaseMetadata,
)


# ---------------------------------------------------------------------------
# Asset loading
# ---------------------------------------------------------------------------


def load_eval_config(path: Path) -> EvalConfig:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return EvalConfig.model_validate(data)


def load_prompts(prompts_dir: Path) -> list[tuple[str, str, PromptMetadata]]:
    if not prompts_dir.is_dir():
        raise FileNotFoundError(f"prompts_dir not found: {prompts_dir}")
    out: list[tuple[str, str, PromptMetadata]] = []
    for sub in sorted(prompts_dir.iterdir()):
        if not sub.is_dir():
            continue
        prompt_path = sub / "prompt.txt"
        meta_path = sub / "metadata.json"
        if not prompt_path.exists() or not meta_path.exists():
            raise ValueError(f"Prompt folder {sub} must contain prompt.txt and metadata.json")
        text = prompt_path.read_text(encoding="utf-8")
        meta = PromptMetadata.model_validate_json(meta_path.read_text(encoding="utf-8"))
        out.append((sub.name, text, meta))
    if not out:
        raise ValueError(f"No prompt sub-folders found in {prompts_dir}")
    return out


def load_judge_prompt(
    judge_prompts_dir: Path,
    judge_prompt_name: str,
) -> tuple[str, JudgePromptMetadata]:
    sub = judge_prompts_dir / judge_prompt_name
    if not sub.is_dir():
        raise FileNotFoundError(f"Judge prompt folder not found: {sub}")
    text = (sub / "judge_prompt.txt").read_text(encoding="utf-8")
    meta = JudgePromptMetadata.model_validate_json(
        (sub / "metadata.json").read_text(encoding="utf-8")
    )
    return text, meta


# ---------------------------------------------------------------------------
# Eval pass
# ---------------------------------------------------------------------------


def run_eval(config_path: Path) -> None:
    load_dotenv()
    config = load_eval_config(config_path)

    cfg_dir = config_path.parent
    test_cases_dir = (cfg_dir / config.test_cases_dir).resolve()
    prompts_dir = (cfg_dir / config.prompts_dir).resolve()
    judge_prompts_dir = (cfg_dir / config.judge_prompts_dir).resolve()

    langfuse = Langfuse()
    client = LiteLLMProxyClient()

    # 1. Bootstrap dataset
    print(f"[caliper] bootstrapping dataset {config.dataset_name!r} from {test_cases_dir}")
    n_items = bootstrap_dataset(
        langfuse=langfuse,
        dataset_name=config.dataset_name,
        test_cases_dir=test_cases_dir,
        description=f"Caliper eval campaign: {config.name}",
    )
    print(f"[caliper] dataset has {n_items} item(s)")

    # 2. Load assets
    prompts = load_prompts(prompts_dir)
    judge_prompt_text, judge_prompt_meta = load_judge_prompt(
        judge_prompts_dir, config.judge_prompt
    )

    judge = RubricJudge(
        client=client,
        judge_model=config.judge_model,
        judge_prompt_template=judge_prompt_text,
        judge_prompt_metadata=judge_prompt_meta,
    )

    # 3. Fetch dataset (now that it's bootstrapped, items are linkable to runs)
    dataset = langfuse.get_dataset(name=config.dataset_name)

    # 4. Cartesian loop
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    total = len(prompts) * len(config.models) * len(dataset.items) * config.iterations
    print(
        f"[caliper] running {total} traces "
        f"({len(prompts)} prompt(s) x {len(config.models)} model(s) x "
        f"{len(dataset.items)} item(s) x {config.iterations} iter(s))"
    )

    counter = 0
    for (prompt_id, prompt_text, prompt_meta), model in product(prompts, config.models):
        run_name = f"{config.name}__{prompt_id}__{model}__{timestamp}"
        run_metadata = {
            "campaign": config.name,
            "prompt_id": prompt_id,
            "prompt_version": prompt_meta.version,
            "model": model,
            "judge_model": config.judge_model,
            "iterations": config.iterations,
            "judge_prompt": config.judge_prompt,
            **config.extra_run_metadata,
        }

        for item in dataset.items:
            test_case_meta = TestCaseMetadata.model_validate(item.metadata)

            for iteration in range(config.iterations):
                counter += 1
                print(
                    f"[caliper] {counter}/{total}  run={run_name}  "
                    f"item={item.id}  iter={iteration + 1}"
                )
                _run_one(
                    item=item,
                    run_name=run_name,
                    run_metadata=run_metadata,
                    prompt_text=prompt_text,
                    prompt_id=prompt_id,
                    model=model,
                    iteration=iteration,
                    test_case_meta=test_case_meta,
                    client=client,
                    judge=judge,
                    config=config,
                )

    langfuse.flush()
    print(f"[caliper] done. {counter} trace(s) written to Langfuse.")


def _run_one(
    *,
    item,  # langfuse DatasetItem
    run_name: str,
    run_metadata: dict,
    prompt_text: str,
    prompt_id: str,
    model: str,
    iteration: int,
    test_case_meta: TestCaseMetadata,
    client: LiteLLMProxyClient,
    judge: RubricJudge,
    config: EvalConfig,
) -> None:
    """Single (prompt, model, item, iteration) trace.

    Opens the parent span via item.run(...) so the trace is auto-linked to a
    Langfuse Dataset Run (run_name). All child observations + scores nest under
    that parent.
    """
    test_case_text = item.input["content"]
    rendered_prompt = prompt_text.replace("{test_case}", test_case_text)

    with item.run(
        run_name=run_name,
        run_metadata=run_metadata,
        run_description=f"Caliper eval pass: {config.name}",
    ) as parent:
        # Tags are how cross-cutting Trace-view filters work — they let you
        # ask questions that span Runs (e.g. "show every trace that scored
        # poorly on snippet X regardless of which combo produced it").
        # Run-level questions stay in run_metadata; per-trace questions in tags.
        parent.update(
            input={
                "prompt": prompt_text,
                "test_case": item.input,
                "iteration": iteration,
            },
            tags=[
                f"prompt:{prompt_id}",
                f"model:{model}",
                f"code_snippet:{item.id}",
                f"iteration:{iteration}",
                f"eval_type:{test_case_meta.eval_type}",
                f"campaign:{config.name}",
            ],
            metadata={
                "prompt_id": prompt_id,
                "model": model,
                "code_snippet": item.id,
                "iteration": iteration,
                "eval_type": test_case_meta.eval_type,
                "campaign": config.name,
            },
        )

        # ----- 1. The model under test -----
        with parent.start_as_current_generation(
            name=f"llm-call:{model}",
            model=model,
            input=[{"role": "user", "content": rendered_prompt}],
        ) as gen:
            result = client.complete(
                model=model,
                messages=[{"role": "user", "content": rendered_prompt}],
            )
            gen.update(
                output=result.output,
                usage_details=result.usage,
                model=result.model,
            )

        # ----- 2. The judge -----
        with parent.start_as_current_generation(
            name="judge",
            model=config.judge_model,
            input=[{"role": "system", "content": "Caliper rubric judge"}],
        ) as judge_gen:
            verdict = judge.evaluate(
                test_case_input=test_case_text,
                test_case_metadata=test_case_meta,
                llm_output=result.output,
            )
            judge_gen.update(
                output=verdict.model_dump(exclude={"raw_response"}),
            )

        parent.update(
            output={
                "generated": result.output,
                "verdict": verdict.model_dump(exclude={"raw_response"}),
            },
        )

        # ----- 3. Scores -----
        for dim_name, dim_score in verdict.dimensions.items():
            parent.score_trace(
                name=dim_name,
                value=dim_score.value,
                comment=dim_score.reasoning[:1000],
            )
            parent.score_trace(
                name=f"{dim_name}__pass",
                value=1 if dim_score.passed else 0,
                data_type="BOOLEAN",
            )
        parent.score_trace(name="overall", value=verdict.overall_value)
        parent.score_trace(
            name="overall__pass",
            value=1 if verdict.overall_passed else 0,
            data_type="BOOLEAN",
        )


def main() -> None:
    if len(sys.argv) != 2:
        print(
            "Usage: python -m caliper.eval_runner <path/to/eval_config.yaml>",
            file=sys.stderr,
        )
        sys.exit(2)
    run_eval(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
