"""Hash-based idempotency for the eval Cartesian.

When `idempotent: true` is set in EvalConfig, the runner builds a SHA-256
hash of every input that observably influences a single cell's output and
skips cells whose hash already appears on a trace in the campaign. Editing
a prompt automatically forces re-runs because the hash changes — no version
field to remember to bump.

What goes into the hash:

    campaign            — namespacing (two campaigns same prompt -> different)
    prompt_text         — verbatim prompt.txt content
    judge_prompt_text   — verbatim judge_prompt.txt content
    model               — model alias (LiteLLM-side name)
    judge_model         — judge model alias
    snippet_content     — verbatim test-case content file
    expected            — TestCaseMetadata.expected, JSON-serialized
    rubric              — Rubric, JSON-serialized (sort_keys for stability)
    iteration           — iteration index within this Run

What does NOT go in:
    prompt_metadata.version (now redundant — hash detects content change)
    prompt_metadata.tags / description (flavor only)
    snippet_metadata.tags / description (flavor only)
    extra_run_metadata (labeling only)
    iterations count config (irrelevant to one cell)
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from caliper.human_review import LangfuseAnnotationClient


def compute_cell_hash(
    *,
    campaign: str,
    prompt_text: str,
    judge_prompt_text: str,
    model: str,
    judge_model: str,
    snippet_content: str,
    expected: dict[str, Any],
    rubric: dict[str, Any],
    iteration: int,
) -> str:
    """SHA-256 of every input that influences this cell, truncated to 16 hex chars.

    2^64 possibilities ⇒ collision-free at any realistic eval scale.
    NUL-byte separator between parts prevents `abc|def` colliding with `ab|cdef`.
    """
    parts = [
        campaign,
        prompt_text,
        judge_prompt_text,
        model,
        judge_model,
        snippet_content,
        json.dumps(expected, sort_keys=True, ensure_ascii=False),
        json.dumps(rubric, sort_keys=True, ensure_ascii=False),
        str(iteration),
    ]
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def fetch_existing_cell_hashes(
    client: LangfuseAnnotationClient,
    campaign: str,
) -> set[str]:
    """Walk every trace tagged `campaign:<x>`, extract its `cell_hash:<x>` tag.

    One paginated query against /api/public/traces (`tags=campaign:X` filter).
    The runner calls this once before the Cartesian and checks against the
    returned set in-memory — no per-cell API calls.
    """
    hashes: set[str] = set()
    page = 1
    target_tag = f"campaign:{campaign}"
    while True:
        resp = client._client.get(
            "/api/public/traces",
            params={"page": page, "limit": 100, "tags": target_tag},
        )
        client._raise_with_context(resp, "list_traces_for_idempotency")
        data = resp.json()
        for trace in data.get("data", []):
            for tag in trace.get("tags", []) or []:
                if tag.startswith("cell_hash:"):
                    hashes.add(tag[len("cell_hash:"):])
        meta = data.get("meta", {})
        if page >= meta.get("totalPages", 1):
            break
        page += 1
    return hashes
