"""Cartesian eval runner — library entry points.

This module is pure library code. It exposes `run_eval(config_path)` and
the helpers it calls. The CLI shim lives in `caliper.cli.eval` so this
module has zero side effects on import and can be embedded in notebooks,
CI scripts, or larger orchestrators without dragging argv parsing along.

For each (prompt, model, test_case, iteration) combination, `run_eval` opens
a Langfuse trace with:

  1. parent root span     — bound to the Langfuse Dataset Run (run_name)
  2. child generation     — the LLM call (token usage attached, server-side cost)
  3. child generation     — the judge call

Then it attaches one Score per rubric dimension plus boolean pass scores
plus an overall, all stamped on the parent trace so the Experiments
comparison view can aggregate them per (prompt, model) run.
"""

from __future__ import annotations

import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from itertools import product
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langfuse import Langfuse

from caliper.dataset_bootstrap import bootstrap_dataset, load_test_cases
from caliper.human_review import (
    LangfuseAnnotationClient,
    score_configs_for_rubric,
)
from caliper.idempotency import compute_cell_hash, fetch_existing_cell_hashes
from caliper.judges.rubric_judge import RubricJudge
from caliper.litellm_client import LiteLLMProxyClient
from caliper.schemas import (
    EvalConfig,
    JudgePromptMetadata,
    PromptMetadata,
    TestCaseMetadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _langfuse_model(name: str, mapping: dict[str, str]) -> str:
    """Apply the LiteLLM -> Langfuse model-name rewrite, or passthrough.

    Used so Langfuse's built-in pricing map can compute server-side cost even
    when LiteLLM returns model names Langfuse doesn't natively know about
    (Databricks-served models, internal fine-tunes, etc.).
    """
    return mapping.get(name, name)


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
# Eval pass — top-level orchestration
# ---------------------------------------------------------------------------


def run_eval(
    config_path: Path,
    force: bool = False,
    concurrency_override: int | None = None,
) -> None:
    """Run one complete eval pass from a YAML EvalConfig.

    Idempotent on the Langfuse Dataset (re-runs upsert items, don't duplicate).
    Each invocation produces a fresh timestamp-tagged Dataset Run per
    (prompt, model) combo so eval cycles don't collide.

    Args:
        config_path: Path to the YAML EvalConfig.
        force: If True, ignore `config.idempotent` and run every Cartesian cell
            regardless of whether a matching trace already exists. Equivalent
            to setting `idempotent: false` for this one invocation.
        concurrency_override: If set, overrides `config.concurrency` for this
            run only. CLI surface for the --concurrency N flag.
    """
    load_dotenv()
    config = load_eval_config(config_path)
    if concurrency_override is not None:
        # Mutate post-validation to honor the CLI override without rewriting
        # the YAML on disk. Pydantic v2 allows attribute assignment by default.
        config.concurrency = concurrency_override

    cfg_dir = config_path.parent
    test_cases_dir = (cfg_dir / config.test_cases_dir).resolve()
    prompts_dir = (cfg_dir / config.prompts_dir).resolve()
    judge_prompts_dir = (cfg_dir / config.judge_prompts_dir).resolve()

    langfuse = Langfuse()
    client = LiteLLMProxyClient(retry_config=config.retry)

    print(f"[caliper] bootstrapping dataset {config.dataset_name!r} from {test_cases_dir}")
    n_items = bootstrap_dataset(
        langfuse=langfuse,
        dataset_name=config.dataset_name,
        test_cases_dir=test_cases_dir,
        description=f"Caliper eval campaign: {config.name}",
    )
    print(f"[caliper] dataset has {n_items} item(s)")

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

    dataset = langfuse.get_dataset(name=config.dataset_name)

    annotation_client, queue_id = _maybe_setup_human_review(
        config=config, test_cases_dir=test_cases_dir
    )

    # Hash-based idempotency: fetch existing cell hashes for this campaign upfront.
    # In-memory set check during the loop -> zero per-cell Langfuse calls.
    existing_hashes = _maybe_fetch_existing_hashes(config=config, force=force)

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    total = len(prompts) * len(config.models) * len(dataset.items) * config.iterations
    print(
        f"[caliper] expanding Cartesian: {total} cell(s) "
        f"({len(prompts)} prompt(s) x {len(config.models)} model(s) x "
        f"{len(dataset.items)} item(s) x {config.iterations} iter(s))"
    )

    # Build the work list upfront.
    # Pre-compute sample flags deterministically (no threaded mutation of Sampler).
    submissions, skipped = _build_submissions(
        config=config,
        prompts=prompts,
        dataset=dataset,
        judge_prompt_text=judge_prompt_text,
        timestamp=timestamp,
        existing_hashes=existing_hashes,
        queue_id=queue_id,
    )

    print(
        f"[caliper] {len(submissions)} cell(s) to run, {skipped} skipped (idempotent), "
        f"concurrency={config.concurrency}"
    )

    ran = 0
    failed: list[tuple[str, str, str, int, str]] = []

    if not submissions:
        print("[caliper] nothing to do; all cells already exist in this campaign")
    else:
        # ThreadPool is correct here: every cell does HTTP I/O (LLM + judge + Langfuse).
        # The GIL doesn't block I/O-bound work. Shared clients (LiteLLM, Langfuse,
        # annotation HTTP) are documented thread-safe.
        with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
            future_to_sub = {
                pool.submit(
                    _run_one_wrapped,
                    sub=sub,
                    client=client,
                    judge=judge,
                    config=config,
                    annotation_client=annotation_client,
                    queue_id=queue_id,
                ): sub
                for sub in submissions
            }
            for i, future in enumerate(as_completed(future_to_sub), 1):
                sub = future_to_sub[future]
                result = future.result()  # never raises — wrapper catches
                if result["ok"]:
                    ran += 1
                    print(
                        f"[caliper] {i}/{len(submissions)}  OK    "
                        f"{sub['run_name']} / {sub['item'].id} / iter={sub['iteration'] + 1}  "
                        f"hash={sub['cell_hash'][:8]}"
                    )
                else:
                    failed.append(
                        (
                            sub["run_name"],
                            sub["item"].id,
                            sub["model"],
                            sub["iteration"],
                            result["error"],
                        )
                    )
                    print(
                        f"[caliper] {i}/{len(submissions)}  FAIL  "
                        f"{sub['run_name']} / {sub['item'].id} / iter={sub['iteration'] + 1}  "
                        f"{result['exc_type']}: {result['error'][:160]}",
                        file=sys.stderr,
                    )

    langfuse.flush()
    if annotation_client is not None:
        annotation_client.close()

    print()
    print(f"[caliper] done. ran={ran} skipped={skipped} failed={len(failed)} (of {total})")
    if failed:
        print("[caliper] failed cells:")
        for run_name, item_id, model, iteration, err in failed:
            print(f"  - {run_name} / {item_id} / iter={iteration} -> {err[:160]}")


def _build_submissions(
    *,
    config: EvalConfig,
    prompts: list[tuple[str, str, PromptMetadata]],
    dataset,
    judge_prompt_text: str,
    timestamp: str,
    existing_hashes: set[str],
    queue_id: str | None,
) -> tuple[list[dict], int]:
    """Expand the Cartesian into a list of submission dicts, pre-computing
    sample flags so the parallel section has no contended state.

    Returns (submissions_to_run, skipped_count).
    """
    skipped = 0
    by_run: dict[str, list[dict]] = {}

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
                cell_hash = compute_cell_hash(
                    campaign=config.name,
                    prompt_text=prompt_text,
                    judge_prompt_text=judge_prompt_text,
                    model=model,
                    judge_model=config.judge_model,
                    snippet_content=item.input["content"],
                    expected=test_case_meta.expected,
                    rubric=test_case_meta.rubric.model_dump(),
                    iteration=iteration,
                )

                if cell_hash in existing_hashes:
                    skipped += 1
                    continue

                by_run.setdefault(run_name, []).append(
                    {
                        "run_name": run_name,
                        "run_metadata": run_metadata,
                        "prompt_text": prompt_text,
                        "prompt_id": prompt_id,
                        "model": model,
                        "iteration": iteration,
                        "item": item,
                        "test_case_meta": test_case_meta,
                        "cell_hash": cell_hash,
                        "should_sample": False,  # filled in below
                    }
                )

    # Pre-compute sample flags (deterministic; no threaded mutation).
    if (
        config.human_review
        and config.human_review.enabled
        and queue_id is not None
    ):
        strategy = config.human_review.sample_strategy
        rng = random.Random(0xC4117E5)
        for run_subs in by_run.values():
            if strategy == "all":
                for s in run_subs:
                    s["should_sample"] = True
            elif strategy == "stratified":
                n = config.human_review.samples_per_run
                for s in run_subs[:n]:
                    s["should_sample"] = True
            elif strategy == "random":
                for s in run_subs:
                    s["should_sample"] = rng.random() < config.human_review.sample_rate

    submissions: list[dict] = []
    for run_subs in by_run.values():
        submissions.extend(run_subs)
    return submissions, skipped


def _run_one_wrapped(
    *,
    sub: dict,
    client: LiteLLMProxyClient,
    judge: RubricJudge,
    config: EvalConfig,
    annotation_client: LangfuseAnnotationClient | None,
    queue_id: str | None,
) -> dict:
    """Run one cell inside a worker thread. Never raises — wraps the result.

    Per-cell isolation: an exception from _run_one becomes a dict result that
    the main thread aggregates into the failed list. The thread pool keeps
    running other cells regardless.
    """
    try:
        _run_one(
            item=sub["item"],
            run_name=sub["run_name"],
            run_metadata=sub["run_metadata"],
            prompt_text=sub["prompt_text"],
            prompt_id=sub["prompt_id"],
            model=sub["model"],
            iteration=sub["iteration"],
            test_case_meta=sub["test_case_meta"],
            client=client,
            judge=judge,
            config=config,
            annotation_client=annotation_client,
            queue_id=queue_id,
            should_sample=sub["should_sample"],
            cell_hash=sub["cell_hash"],
        )
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e), "exc_type": type(e).__name__}


def _maybe_fetch_existing_hashes(*, config: EvalConfig, force: bool) -> set[str]:
    """Fetch the set of cell hashes already present in this campaign's traces.

    Returns an empty set when idempotency is disabled or force=True.
    Best-effort: if the upfront fetch fails for any reason, log a WARN and
    return an empty set — the eval pass proceeds as if no prior runs existed.
    """
    if not config.idempotent:
        return set()
    if force:
        print("[caliper] --force: ignoring idempotent setting for this run")
        return set()

    try:
        client = LangfuseAnnotationClient.from_env()
    except Exception as e:
        print(
            f"[caliper] WARN: idempotency disabled ({e}); proceeding without skip check",
            file=sys.stderr,
        )
        return set()

    try:
        print(f"[caliper] fetching existing cell hashes for campaign {config.name!r}")
        hashes = fetch_existing_cell_hashes(client, config.name)
        print(f"[caliper] {len(hashes)} existing cell(s) on record")
        return hashes
    except Exception as e:
        print(
            f"[caliper] WARN: idempotency lookup failed ({e}); proceeding without skip check",
            file=sys.stderr,
        )
        return set()
    finally:
        client.close()


def _maybe_setup_human_review(
    *,
    config: EvalConfig,
    test_cases_dir: Path,
) -> tuple[LangfuseAnnotationClient | None, str | None]:
    """Set up human-review queue + score configs if configured.

    Never fails the eval pass. On any error, prints a WARN to stderr and
    returns (None, None) — the runner will proceed without enqueueing.
    """
    if not config.human_review or not config.human_review.enabled:
        return None, None

    try:
        client = LangfuseAnnotationClient.from_env()
    except Exception as e:
        print(f"[caliper] WARN: human review setup skipped — {e}", file=sys.stderr)
        return None, None

    try:
        cases = load_test_cases(test_cases_dir)
        rubric = cases[0][2].rubric
    except Exception as e:
        print(
            f"[caliper] WARN: could not derive rubric for queue setup ({e}); "
            f"enqueueing disabled for this pass",
            file=sys.stderr,
        )
        client.close()
        return None, None

    queue_id: str | None = None
    if config.human_review.auto_create:
        try:
            specs = score_configs_for_rubric(rubric)
            config_ids = [client.ensure_score_config(s) for s in specs]
            queue_id = client.ensure_queue(
                name=config.human_review.queue_name,
                score_config_ids=config_ids,
                description=f"Caliper human review: {config.name}",
            )
            print(
                f"[caliper] human review queue ready: "
                f"{config.human_review.queue_name!r} (id={queue_id})"
            )
        except Exception as e:
            print(
                f"[caliper] WARN: auto-create of score configs / queue failed: {e}\n"
                f"[caliper]       Eval pass will continue WITHOUT enqueueing for human review.\n"
                f"[caliper]       Create the queue manually in Langfuse UI and re-run, or\n"
                f"[caliper]       set human_review.auto_create=false to suppress this attempt.",
                file=sys.stderr,
            )
            queue_id = None
    else:
        queue = client.find_queue(config.human_review.queue_name)
        if queue:
            queue_id = queue["id"]
            print(
                f"[caliper] using pre-existing queue {config.human_review.queue_name!r}"
            )
        else:
            print(
                f"[caliper] WARN: queue {config.human_review.queue_name!r} not found "
                f"(auto_create=false). Enqueueing disabled for this pass.",
                file=sys.stderr,
            )

    return client, queue_id


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
    annotation_client: LangfuseAnnotationClient | None = None,
    queue_id: str | None = None,
    should_sample: bool = False,
    cell_hash: str = "",
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
        # cell_hash tag is what idempotency checks against on later runs.
        # Stamp it both as a tag (for fast filter) and in metadata (for query).
        tags = [
            f"prompt:{prompt_id}",
            f"model:{model}",
            f"code_snippet:{item.id}",
            f"iteration:{iteration}",
            f"eval_type:{test_case_meta.eval_type}",
            f"campaign:{config.name}",
        ]
        metadata: dict[str, object] = {
            "prompt_id": prompt_id,
            "model": model,
            "code_snippet": item.id,
            "iteration": iteration,
            "eval_type": test_case_meta.eval_type,
            "campaign": config.name,
        }
        if cell_hash:
            tags.append(f"cell_hash:{cell_hash}")
            metadata["cell_hash"] = cell_hash

        parent.update(
            input={
                "prompt": prompt_text,
                "test_case": item.input,
                "iteration": iteration,
            },
            tags=tags,
            metadata=metadata,
        )

        # ----- 1. The model under test -----
        with parent.start_as_current_generation(
            name=f"llm-call:{model}",
            model=_langfuse_model(model, config.langfuse_model_mapping),
            input=[{"role": "user", "content": rendered_prompt}],
        ) as gen:
            result = client.complete(
                model=model,
                messages=[{"role": "user", "content": rendered_prompt}],
            )
            gen.update(
                output=result.output,
                usage_details=result.usage,
                # Map LiteLLM-returned name -> Langfuse pricing-map name.
                # Passthrough if no mapping entry exists for this name.
                model=_langfuse_model(result.model, config.langfuse_model_mapping),
            )

        # ----- 2. The judge -----
        with parent.start_as_current_generation(
            name="judge",
            model=_langfuse_model(config.judge_model, config.langfuse_model_mapping),
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

        # ----- 4. Optional: enqueue for human review -----
        # should_sample was pre-computed in _build_submissions so the parallel
        # section has no contended Sampler state.
        if should_sample and annotation_client is not None and queue_id is not None:
            try:
                annotation_client.add_trace_to_queue(
                    queue_id=queue_id,
                    trace_id=parent.trace_id,
                )
            except Exception as e:
                print(
                    f"[caliper] WARN: failed to enqueue trace {parent.trace_id} "
                    f"for human review: {e}",
                    file=sys.stderr,
                )
