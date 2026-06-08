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

from langfuse import Langfuse

from caliper.schemas import TestCaseMetadata


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
    description: str = "",
) -> int:
    """Create the dataset (if needed) and upsert one item per test case.

    Returns the number of items upserted.

    Idempotent: Langfuse's create_dataset / create_dataset_item APIs are upsert-
    on-name and upsert-on-id respectively. Re-running with the same inputs
    produces no duplicates.
    """
    langfuse.create_dataset(name=dataset_name, description=description)

    cases = load_test_cases(test_cases_dir)
    for case_id, content, metadata in cases:
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            id=case_id,
            input={"content": content},
            metadata=metadata.model_dump(),
        )
    return len(cases)
