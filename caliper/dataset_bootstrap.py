"""Bootstrap a Langfuse Dataset from a folder of test cases.

Folder convention:

    test_cases_dir/
    +-- sql_injection_42/
    |   +-- code_snippet.py         <- content file (any name except metadata.json)
    |   +-- metadata.json           <- TestCaseMetadata
    +-- path_traversal_example/
    |   +-- snippet.go
    |   +-- metadata.json
    +-- ...

Each sub-folder becomes one Langfuse DatasetItem whose `id` is the folder name.
That makes the operation idempotent — re-running upserts on the same id rather
than creating duplicates.
"""

from __future__ import annotations

from pathlib import Path

from caliper.human_review import LangfuseAnnotationClient
from caliper.schemas import Rubric, TestCaseMetadata


# ---------------------------------------------------------------------------
# Rubric loading + resolution
# ---------------------------------------------------------------------------


def load_rubrics(rubrics_dir: Path) -> dict[str, Rubric]:
    """Load all rubrics from `rubrics_dir/<name>/rubric.json` into {name: Rubric}.

    Empty / non-existent rubrics_dir returns {}. That's fine as long as no
    test case references rubrics by name AND no config default_rubric is set;
    if either of those conditions ARE met, resolve_rubric will fail loudly.
    """
    if not rubrics_dir.is_dir():
        return {}

    rubrics: dict[str, Rubric] = {}
    for sub in sorted(rubrics_dir.iterdir()):
        if not sub.is_dir():
            continue
        rubric_path = sub / "rubric.json"
        if not rubric_path.exists():
            raise ValueError(
                f"Rubric folder {sub} must contain rubric.json"
            )
        rubric = Rubric.model_validate_json(
            rubric_path.read_text(encoding="utf-8")
        )
        rubrics[sub.name] = rubric
    return rubrics


def resolve_rubric_for_eval_type(
    eval_type: str,
    rubrics: dict[str, Rubric],
    default_rubric_name: str | None,
    rubric_by_eval_type: dict[str, str],
) -> Rubric:
    """Resolve which Rubric applies given a test case's eval_type.

    Resolution order:
      1. rubric_by_eval_type[eval_type] — if eval_type is keyed here
      2. default_rubric_name from EvalConfig
      3. ValueError if neither is set
    """
    # Per-eval-type override wins if set
    name = rubric_by_eval_type.get(eval_type) or default_rubric_name
    if name is None:
        raise ValueError(
            f"No rubric available for eval_type {eval_type!r}. Set one of: "
            f"EvalConfig.default_rubric, or "
            f"EvalConfig.rubric_by_eval_type[{eval_type!r}]."
        )
    if name not in rubrics:
        available = sorted(rubrics.keys()) or "(none)"
        raise ValueError(
            f"Rubric {name!r} (requested for eval_type {eval_type!r}) was "
            f"not found in rubrics_dir. Available: {available}"
        )
    return rubrics[name]


def attach_rubrics_to_cases(
    cases: list[tuple[str, str, TestCaseMetadata]],
    rubrics: dict[str, Rubric],
    default_rubric_name: str | None,
    rubric_by_eval_type: dict[str, str],
) -> list[tuple[str, str, TestCaseMetadata, Rubric]]:
    """Walk loaded test cases and attach the resolved Rubric.

    Returns a list of (case_id, content, meta, resolved_rubric) tuples.
    Downstream code reads the resolved Rubric from the 4th element rather
    than from TestCaseMetadata (which no longer carries it).
    """
    out: list[tuple[str, str, TestCaseMetadata, Rubric]] = []
    for case_id, content, meta in cases:
        rubric = resolve_rubric_for_eval_type(
            meta.eval_type, rubrics, default_rubric_name, rubric_by_eval_type
        )
        out.append((case_id, content, meta, rubric))
    return out


def load_test_cases(
    test_cases_dir: Path,
) -> list[tuple[str, str, TestCaseMetadata]]:
    """Read the folder tree above into [(id, content, metadata), ...].

    Validates each metadata.json against TestCaseMetadata as it loads — failing
    fast on schema drift beats discovering it 200 traces into an eval pass.
    """
    if not test_cases_dir.is_dir():
        raise FileNotFoundError(f"test_cases_dir not found: {test_cases_dir}")

    cases: list[tuple[str, str, TestCaseMetadata]] = []
    for sub in sorted(test_cases_dir.iterdir()):
        if not sub.is_dir():
            continue

        meta_path = sub / "metadata.json"
        if not meta_path.exists():
            raise ValueError(f"Missing metadata.json in {sub}")

        content_files = [
            f for f in sub.iterdir() if f.is_file() and f.name != "metadata.json"
        ]
        if not content_files:
            raise ValueError(f"No content file in {sub}")
        if len(content_files) > 1:
            names = ", ".join(f.name for f in content_files)
            raise ValueError(
                f"Multiple content files in {sub} ({names}); expected exactly one"
            )

        content = content_files[0].read_text(encoding="utf-8")
        metadata = TestCaseMetadata.model_validate_json(meta_path.read_text(encoding="utf-8"))
        cases.append((sub.name, content, metadata))

    if not cases:
        raise ValueError(f"No test-case sub-folders found in {test_cases_dir}")

    return cases


def bootstrap_dataset(
    client: LangfuseAnnotationClient,
    dataset_name: str,
    test_cases_dir: Path,
    description: str = "",
) -> int:
    """Create the dataset (if needed) and upsert one item per test case.

    Test case metadata is sent as-is — no rubric. Rubrics live entirely in
    eval_config land and are resolved per cell by the runner based on each
    test case's eval_type.

    Idempotent: the Langfuse REST endpoints `POST /api/public/datasets` and
    `POST /api/public/dataset-items` upsert on name and on id respectively, so
    re-running with the same inputs produces no duplicates. We go straight to
    REST (rather than the SDK's create_dataset / create_dataset_item) to match
    the rest of Caliper's Langfuse management calls and to stay clear of SDK
    version drift on the dataset-item upsert path.

    Item-id namespacing: Langfuse dataset-item ids are unique per *project*, not
    per dataset. A bare folder name (e.g. `regex_issue`) therefore collides the
    moment the same test case appears in two datasets in one project — Langfuse
    rejects the second create ("already exists in a dataset other than ...",
    surfaced as a misleading "not found" on older builds). We dodge that by
    namespacing the stored id as `<dataset_name>::<folder>` and keeping the bare
    folder name as the user-facing `test_case_key` in metadata. Downstream code
    keys off `test_case_key` (see runner._test_case_key), so tags, the
    `test_case_ids` allowlist, and the UI all stay clean folder names.
    """
    client.ensure_dataset(name=dataset_name, description=description)

    cases = load_test_cases(test_cases_dir)
    for case_id, content, metadata in cases:
        meta = metadata.model_dump()
        # Bare folder name kept as the stable, user-facing key; the Langfuse id
        # itself is namespaced below to stay project-unique across datasets.
        meta["test_case_key"] = case_id
        client.upsert_dataset_item(
            dataset_name=dataset_name,
            item_id=f"{dataset_name}::{case_id}",
            input={"content": content},
            metadata=meta,
        )
    return len(cases)
