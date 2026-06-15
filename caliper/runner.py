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
from typing import Any

import yaml
from dotenv import load_dotenv
from langfuse import Langfuse

from caliper.dataset_bootstrap import (
    bootstrap_dataset,
    load_rubrics,
    load_test_cases,
    resolve_rubric_for_eval_type,
)
from caliper.human_review import (
    LangfuseAnnotationClient,
    score_configs_for_rubric,
)
from caliper.idempotency import compute_cell_hash, fetch_existing_cell_hashes
from caliper.judges.rubric_judge import RubricJudge
from caliper.litellm_client import LiteLLMProxyClient
from caliper.schemas import (
    EvalConfig,
    HumanReviewConfig,
    JudgePromptMetadata,
    PromptMetadata,
    Rubric,
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


def _filter_prompts(
    prompts: list[tuple[str, str, PromptMetadata]],
    allow_ids: list[str],
) -> list[tuple[str, str, PromptMetadata]]:
    """Subset prompts by id allowlist. Empty allowlist = no filter."""
    if not allow_ids:
        return prompts
    selected = set(allow_ids)
    available = {p[0] for p in prompts}
    missing = selected - available
    if missing:
        print(
            f"[caliper] WARN: prompt_ids not found in prompts dir: "
            f"{sorted(missing)} (available: {sorted(available)})",
            file=sys.stderr,
        )
    filtered = [p for p in prompts if p[0] in selected]
    print(
        f"[caliper] prompt_ids filter: {len(filtered)} of {len(prompts)} "
        f"prompt(s) selected"
    )
    return filtered


def _test_case_key(item) -> str:
    """The user-facing test-case key (the on-disk folder name).

    Bootstrap namespaces the Langfuse item id as `<dataset>::<folder>` because
    Langfuse dataset-item ids are unique per *project*, not per dataset — a bare
    folder name collides across datasets. The bare folder name is stashed in
    item metadata as `test_case_key`; everything user-facing (tags, the
    `test_case_ids` allowlist, logs) keys off that, never the namespaced id.
    Falls back to `item.id` for items created before this scheme existed.
    """
    md = getattr(item, "metadata", None)
    if isinstance(md, dict):
        key = md.get("test_case_key")
        if key:
            return str(key)
    return item.id


def _dedupe_items_by_key(items: list) -> list:
    """Collapse items that share a test_case_key.

    A dataset can end up with both a bare-id item and a namespaced one for the
    same test case if it was partially populated before the id-namespacing
    change. Keep the namespaced item (id != bare key) and warn, so a mixed-scheme
    dataset doesn't silently double every Cartesian cell.
    """
    by_key: dict[str, object] = {}
    for it in items:
        key = _test_case_key(it)
        prev = by_key.get(key)
        if prev is None:
            by_key[key] = it
            continue
        keep = it if it.id != key else prev
        print(
            f"[caliper] WARN: duplicate dataset items for test case {key!r} "
            f"(ids: {prev.id!r}, {it.id!r}); using {keep.id!r}",
            file=sys.stderr,
        )
        by_key[key] = keep
    return list(by_key.values())


def _filter_dataset_items(items: list, allow_ids: list[str]) -> list:
    """Subset Langfuse dataset items by test-case-key allowlist. Empty = no filter."""
    if not allow_ids:
        return items
    selected = set(allow_ids)
    available = {_test_case_key(i) for i in items}
    missing = selected - available
    if missing:
        print(
            f"[caliper] WARN: test_case_ids not found in dataset: "
            f"{sorted(missing)} (available: {sorted(available)})",
            file=sys.stderr,
        )
    filtered = [i for i in items if _test_case_key(i) in selected]
    print(
        f"[caliper] test_case_ids filter: {len(filtered)} of {len(items)} "
        f"test case(s) selected"
    )
    return filtered


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
    rubrics_dir = (cfg_dir / config.rubrics_dir).resolve()

    langfuse = Langfuse()
    client = LiteLLMProxyClient(retry_config=config.retry)

    # Load rubrics + validate every test case's eval_type resolves cleanly.
    # Failing fast here beats failing on the first cell that hits a missing
    # rubric mapping mid-pass.
    rubrics = load_rubrics(rubrics_dir)
    raw_cases = load_test_cases(test_cases_dir)
    case_rubrics: dict[str, Rubric] = {}
    for case_id, _, meta in raw_cases:
        case_rubrics[case_id] = resolve_rubric_for_eval_type(
            meta.eval_type,
            rubrics,
            config.default_rubric,
            config.rubric_by_eval_type,
        )

    print(f"[caliper] bootstrapping dataset {config.dataset_name!r} from {test_cases_dir}")
    # Dataset create + item upsert go through the raw REST client (not the SDK)
    # to match Caliper's other Langfuse management calls and avoid SDK drift on
    # the dataset-item upsert path. Live trace ingestion below stays on the SDK.
    lf_rest = LangfuseAnnotationClient.from_env()
    try:
        n_items = bootstrap_dataset(
            client=lf_rest,
            dataset_name=config.dataset_name,
            test_cases_dir=test_cases_dir,
            description=f"Caliper eval campaign: {config.name}",
        )
    finally:
        lf_rest.close()
    print(f"[caliper] dataset has {n_items} item(s)")
    print(f"[caliper] resolved rubric for {len(case_rubrics)} test case(s)")

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

    # Apply subsetting filters BEFORE building the Cartesian. The dataset
    # itself still contains every test case (idempotent bootstrap), so toggling
    # subsets doesn't churn the persistent Langfuse state.
    prompts = _filter_prompts(prompts, config.prompt_ids)
    dataset_items = _filter_dataset_items(
        _dedupe_items_by_key(list(dataset.items)), config.test_case_ids
    )

    annotation_client, queue_id = _maybe_setup_human_review(
        config=config,
        case_rubrics=case_rubrics,
    )

    # Hash-based idempotency: fetch existing cell hashes for this campaign upfront.
    # In-memory set check during the loop -> zero per-cell Langfuse calls.
    existing_hashes = _maybe_fetch_existing_hashes(config=config, force=force)

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    total = len(prompts) * len(config.models) * len(dataset_items) * config.iterations
    print(
        f"[caliper] expanding Cartesian: {total} cell(s) "
        f"({len(prompts)} prompt(s) x {len(config.models)} model(s) x "
        f"{len(dataset_items)} item(s) x {config.iterations} iter(s))"
    )

    # Build the work list upfront.
    # Pre-compute sample flags deterministically (no threaded mutation of Sampler).
    submissions, skipped = _build_submissions(
        config=config,
        prompts=prompts,
        dataset_items=dataset_items,
        case_rubrics=case_rubrics,
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
                        f"{sub['run_name']} / {_test_case_key(sub['item'])} / iter={sub['iteration'] + 1}  "
                        f"hash={sub['cell_hash'][:8]}"
                    )
                else:
                    failed.append(
                        (
                            sub["run_name"],
                            _test_case_key(sub["item"]),
                            sub["model"],
                            sub["iteration"],
                            result["error"],
                        )
                    )
                    print(
                        f"[caliper] {i}/{len(submissions)}  FAIL  "
                        f"{sub['run_name']} / {_test_case_key(sub['item'])} / iter={sub['iteration'] + 1}  "
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
    dataset_items: list,
    case_rubrics: dict[str, Rubric],
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
        # Lead with the two comparison axes (prompt, then model) so the Dataset
        # Runs list sorts/scans by what you're actually comparing; campaign +
        # timestamp trail behind purely to keep the run name unique per cycle.
        run_name = f"{prompt_id}__{model}__{config.name}__{timestamp}"
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

        for item in dataset_items:
            test_case_meta = TestCaseMetadata.model_validate(item.metadata)
            # The resolved rubric for this test case (looked up by eval_type
            # at run_eval time). Test cases no longer carry their own rubric;
            # the eval_config owns the assignment. case_rubrics is keyed by the
            # bare folder name, so resolve via the test-case key (item.id is the
            # dataset-namespaced id, which would miss).
            rubric = case_rubrics[_test_case_key(item)]

            for iteration in range(config.iterations):
                cell_hash = compute_cell_hash(
                    campaign=config.name,
                    prompt_text=prompt_text,
                    judge_prompt_text=judge_prompt_text,
                    model=model,
                    judge_model=config.judge_model,
                    snippet_content=item.input["content"],
                    expected=test_case_meta.expected,
                    rubric=rubric.model_dump(),
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
                        "rubric": rubric,
                        "cell_hash": cell_hash,
                        "should_sample": False,  # filled in below
                    }
                )

    # Pre-compute sample flags (deterministic; no threaded mutation).
    submissions: list[dict] = []
    for run_subs in by_run.values():
        submissions.extend(run_subs)

    if (
        config.human_review
        and config.human_review.enabled
        and queue_id is not None
    ):
        _apply_sample_flags(submissions, by_run, config.human_review)

    return submissions, skipped


def _compute_auto_target(total: int, cfg: HumanReviewConfig) -> int:
    """Compute the effective sample count from caps + floors.

    floor   = max(min_samples, ceil(min_pct * total))
    ceiling = min(max_samples, ceil(max_pct * total))
    target  = clamp(ceiling, [0, total])
    if target < floor and total > target:
        target = min(floor, total)

    When the floor exceeds the ceiling (big matrices with high min_pct
    + low max_samples), the ceiling wins to protect human time.
    """
    import math

    floor = max(cfg.min_samples, math.ceil(cfg.min_pct * total))
    ceiling = min(cfg.max_samples, math.ceil(cfg.max_pct * total))
    target = min(ceiling, total)
    if target < floor and total > target:
        target = min(floor, total)
    return target


def _apply_sample_flags(
    submissions: list[dict],
    by_run: dict[str, list[dict]],
    cfg: HumanReviewConfig,
) -> None:
    """Mark which submissions get sampled into the human review queue.

    Modifies submission dicts in place. Strategies:
      auto       — caps + floors; even distribution across Runs
      stratified — explicit `samples_per_run` per Run
      random     — coin flip per cell with `sample_rate` probability
      all        — every cell goes to the queue
    """
    import math

    strategy = cfg.sample_strategy
    rng = random.Random(0xC4117E5)

    if strategy == "all":
        for s in submissions:
            s["should_sample"] = True
        return

    if strategy == "random":
        for s in submissions:
            s["should_sample"] = rng.random() < cfg.sample_rate
        return

    if strategy == "stratified":
        for run_subs in by_run.values():
            for s in run_subs[: cfg.samples_per_run]:
                s["should_sample"] = True
        return

    # strategy == "auto"
    total = len(submissions)
    if total == 0:
        return
    target = _compute_auto_target(total, cfg)
    n_runs = len([rs for rs in by_run.values() if rs])
    if n_runs == 0:
        return
    per_run = math.ceil(target / n_runs)
    # Allocate per_run to each Run, then trim from the tail to hit target exactly
    allocated = 0
    for run_subs in by_run.values():
        if not run_subs:
            continue
        take = min(per_run, len(run_subs), max(0, target - allocated))
        for s in run_subs[:take]:
            s["should_sample"] = True
        allocated += take
        if allocated >= target:
            break
    print(
        f"[caliper] human review auto-sampling: target={target} "
        f"(floor={max(cfg.min_samples, math.ceil(cfg.min_pct * total))}, "
        f"ceiling={min(cfg.max_samples, math.ceil(cfg.max_pct * total))}, "
        f"total_cells={total}, runs={n_runs})"
    )


def _run_one_wrapped(
    *,
    sub: dict,
    client: LiteLLMProxyClient,
    judge: RubricJudge,
    config: EvalConfig,
    annotation_client: LangfuseAnnotationClient | None,
    queue_id: str | None,
) -> dict:
    """Run one cell inside a worker thread. Never raises — wraps the result."""
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
            rubric=sub["rubric"],
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
    case_rubrics: dict[str, Rubric],
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

    if not case_rubrics:
        print(
            "[caliper] WARN: no resolved rubrics; enqueueing disabled",
            file=sys.stderr,
        )
        client.close()
        return None, None

    # POC assumption: all test cases in this pass share rubric dim names,
    # so the first one is representative for ScoreConfig derivation.
    rubric = next(iter(case_rubrics.values()))

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
    rubric: Rubric,
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
    #rendered_prompt = prompt_text.replace("{test_case}", test_case_text)

    if "{test_case}" in prompt_text:
        rendered_prompt = prompt_text.replace("{test_case}", test_case_text)
    else:
        rendered_prompt = f"{prompt_text}\n\nTest case:\n{test_case_text}"

    with item.run(
        run_name=run_name,
        run_metadata=run_metadata,
        run_description=f"Caliper eval pass: {config.name}",
    ) as parent:
        # cell_hash tag is what idempotency checks against on later runs.
        # Stamp it both as a tag (for fast filter) and in metadata (for query).
        # `test_case:<id>` is the generic tag that scales beyond the original
        # code-review use case (agent eval inputs, customer-service queries,
        # etc.). The `code_snippet:` tag was a domain leak from the original
        # sample; this is the corrected generic shape.
        # User-facing key = bare folder name (item.id is namespaced by dataset).
        tc_key = _test_case_key(item)
        tags = [
            f"prompt:{prompt_id}",
            f"model:{model}",
            f"test_case:{tc_key}",
            f"iteration:{iteration}",
            f"eval_type:{test_case_meta.eval_type}",
            f"campaign:{config.name}",
        ]
        metadata: dict[str, object] = {
            "prompt_id": prompt_id,
            "model": model,
            "test_case": tc_key,
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

        # ----- 2. The judge (per configured mode) -----
        # Anchored mode = judge sees expected/reference data (current default).
        # Blind mode    = judge sees only the code + review (reference withheld).
        # Running both surfaces reference bias: gap between anchored and blind
        # scores tells you how much the cheat sheet is propping up the judge.
        verdicts_by_mode: dict[str, Any] = {}
        for mode in config.judge_modes:
            gen_name = "judge" if mode == "anchored" else f"judge:{mode}"
            with parent.start_as_current_generation(
                name=gen_name,
                model=_langfuse_model(config.judge_model, config.langfuse_model_mapping),
            ) as judge_gen:
                verdict = judge.evaluate(
                    test_case_input=test_case_text,
                    test_case_metadata=test_case_meta,
                    rubric=rubric,
                    llm_output=result.output,
                    mode=mode,
                )
                # Stamp the actual rendered judge prompt as the generation input so
                # a human annotator can see exactly what the judge was asked.
                judge_gen.update(
                    input=[{"role": "user", "content": verdict.rendered_prompt}],
                    output=verdict.model_dump(exclude={"raw_response", "rendered_prompt"}),
                )
            verdicts_by_mode[mode] = verdict

        # The anchored verdict is the primary one (backward compat with the
        # existing report shape). Fall back to the first mode's verdict if
        # anchored isn't in the list.
        primary_verdict = verdicts_by_mode.get(
            "anchored", next(iter(verdicts_by_mode.values()))
        )

        parent.update(
            output={
                "generated": result.output,
                "verdict": primary_verdict.model_dump(
                    exclude={"raw_response", "rendered_prompt"}
                ),
                "verdict_modes": list(verdicts_by_mode.keys()),
            },
        )

        # ----- 3. Scores -----
        # Emit NUMERIC scores only. Pass/fail is DERIVED at calibration time
        # from the numeric score + the rubric threshold (see
        # caliper.calibration.derive_pass_and_overall) — emitting boolean
        # `__pass` columns too would just double the columns in the Langfuse
        # comparison view for no added information.
        #
        # Anchored mode emits unsuffixed names; other modes append `__<mode>`
        # (e.g. `finds_bug` vs `finds_bug__blind`). Variants of one dimension
        # are emitted consecutively so they stay adjacent in Langfuse versions
        # that order score columns by creation order (newer versions sort
        # alphabetically, where a shared name prefix already groups them).
        modes = list(verdicts_by_mode.keys())
        for dim_name in primary_verdict.dimensions:
            for mode in modes:
                suffix = "" if mode == "anchored" else f"__{mode}"
                dim_score = verdicts_by_mode[mode].dimensions[dim_name]
                parent.score_trace(
                    name=f"{dim_name}{suffix}",
                    value=dim_score.value,
                    comment=dim_score.reasoning[:1000],
                )
        for mode in modes:
            suffix = "" if mode == "anchored" else f"__{mode}"
            parent.score_trace(
                name=f"overall{suffix}",
                value=verdicts_by_mode[mode].overall_value,
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
