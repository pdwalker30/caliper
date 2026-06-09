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

import sys
from pathlib import Path

from langfuse import Langfuse

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


def resolve_rubric(
    test_case_id: str,
    test_case_meta: TestCaseMetadata,
    rubrics: dict[str, Rubric],
    default_rubric_name: str | None,
) -> Rubric:
    """Resolve which Rubric applies to this test case.

    Priority:
      1. test_case_meta.rubric as Rubric object (inline; deprecated)
      2. test_case_meta.rubric as string (named reference)
      3. default_rubric_name from EvalConfig
    """
    r = test_case_meta.rubric
    if isinstance(r, Rubric):
        print(
            f"[caliper] WARN: test case {test_case_id!r} uses an inline rubric "
            f"in metadata.json — this is deprecated. Extract it into "
            f"rubrics/<name>/rubric.json and reference by name for DRY rubric "
            f"definitions.",
            file=sys.stderr,
        )
        return r
    if isinstance(r, str):
        if r not in rubrics:
            available = sorted(rubrics.keys()) or "(none)"
            raise ValueError(
                f"Test case {test_case_id!r} references rubric {r!r} but it "
                f"was not found in rubrics_dir. Available: {available}"
            )
        return rubrics[r]
    # r is None
    if default_rubric_name is None:
        raise ValueError(
            f"Test case {test_case_id!r} has no rubric and EvalConfig has no "
            f"default_rubric. Set one of: TestCaseMetadata.rubric or "
            f"EvalConfig.default_rubric."
        )
    if default_rubric_name not in rubrics:
        available = sorted(rubrics.keys()) or "(none)"
        raise ValueError(
            f"EvalConfig.default_rubric is {default_rubric_name!r} but it "
            f"was not found in rubrics_dir. Available: {available}"
        )
    return rubrics[default_rubric_name]


def resolve_rubrics_in_cases(
    cases: list[tuple[str, str, TestCaseMetadata]],
    rubrics: dict[str, Rubric],
    default_rubric_name: str | None,
) -> list[tuple[str, str, TestCaseMetadata]]:
    """Walk all loaded test cases and inline the resolved Rubric.

    After this runs, every TestCaseMetadata.rubric is a Rubric object —
    downstream code (judge, score emission, hash computation, dataset
    metadata) doesn't have to handle the union shape.
    """
    for case_id, _, meta in cases:
        resolved = resolve_rubric(case_id, meta, rubrics, default_rubric_name)
        meta.rubric = resolved
    return cases


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
    langfuse: Langfuse,
    dataset_name: str,
    test_cases_dir: Path,
    rubrics_dir: Path,
    default_rubric_name: str | None,
    description: str = "",
) -> int:
    """Create the dataset (if needed) and upsert one item per test case.

    Resolves each test case's rubric reference into an inlined Rubric BEFORE
    sending to Langfuse, so the DatasetItem metadata in Langfuse always
    contains the full Rubric (rather than just a name string).

    Idempotent: Langfuse's create_dataset / create_dataset_item APIs are upsert-
    on-name and upsert-on-id respectively. Re-running with the same inputs
    produces no duplicates.
    """
    langfuse.create_dataset(name=dataset_name, description=description)

    rubrics = load_rubrics(rubrics_dir)
    cases = load_test_cases(test_cases_dir)
    cases = resolve_rubrics_in_cases(cases, rubrics, default_rubric_name)

    for case_id, content, metadata in cases:
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            id=case_id,
            input={"content": content},
            metadata=metadata.model_dump(),
        )
    return len(cases)
