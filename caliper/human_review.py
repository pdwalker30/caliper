"""Human review integration — Annotation Queues + Score Configs.

The Langfuse Python SDK has historically had spotty coverage of Annotation
Queue and ScoreConfig CRUD. To avoid SDK-version-drift problems, this module
talks to the documented Langfuse REST API directly via httpx.

REST API references (Langfuse v3):
    https://langfuse.com/docs/api
    POST /api/public/score-configs       — create a score config
    GET  /api/public/score-configs       — list (paginated)
    POST /api/public/annotation-queues   — create a queue
    GET  /api/public/annotation-queues   — list (paginated)
    POST /api/public/annotation-queues/{queueId}/items  — enqueue a trace

If Langfuse changes the API shape or you hit auth/scope errors, swap the
HTTP calls here — nothing else in Caliper depends on this module's internals.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from caliper.schemas import HumanReviewConfig, Rubric

# ---------------------------------------------------------------------------
# Score Config — Langfuse-native definition of "what this score means"
# ---------------------------------------------------------------------------


@dataclass
class ScoreConfigSpec:
    """Caliper-side description of a ScoreConfig we want to exist in Langfuse."""

    name: str
    data_type: Literal["NUMERIC", "BOOLEAN", "CATEGORICAL"]
    description: str = ""
    min_value: float | None = None
    max_value: float | None = None
    categories: list[dict[str, Any]] | None = None  # [{label, value}, ...]


def score_configs_for_rubric(rubric: Rubric) -> list[ScoreConfigSpec]:
    """Derive the full set of ScoreConfigSpecs implied by a rubric.

    Names MUST match what eval_runner emits: each rubric dimension contributes
    one NUMERIC config (the score) plus one BOOLEAN config (the pass flag).
    Plus the aggregated overall pair. The names are what the human sees in
    the annotation UI, so they need to read naturally.
    """
    specs: list[ScoreConfigSpec] = []
    for dim in rubric.dimensions:
        specs.append(
            ScoreConfigSpec(
                name=dim.name,
                data_type="NUMERIC",
                description=dim.description,
                min_value=0.0,
                max_value=1.0,
            )
        )
        specs.append(
            ScoreConfigSpec(
                name=f"{dim.name}__pass",
                data_type="BOOLEAN",
                description=f"Did this trace pass the {dim.name} threshold ({dim.pass_threshold})?",
            )
        )
    specs.append(
        ScoreConfigSpec(
            name="overall",
            data_type="NUMERIC",
            description="Aggregated overall score across all rubric dimensions.",
            min_value=0.0,
            max_value=1.0,
        )
    )
    specs.append(
        ScoreConfigSpec(
            name="overall__pass",
            data_type="BOOLEAN",
            description="Did this trace pass the overall threshold?",
        )
    )
    return specs


# ---------------------------------------------------------------------------
# Langfuse HTTP client — direct REST API access for queue + config ops
# ---------------------------------------------------------------------------


class LangfuseAnnotationClient:
    """Thin HTTP client for Langfuse score-config + annotation-queue endpoints.

    Uses Basic auth with the public + secret API keys (same auth Langfuse's
    REST docs describe).
    """

    def __init__(
        self,
        host: str,
        public_key: str,
        secret_key: str,
        timeout: float = 30.0,
    ):
        self.host = host.rstrip("/")
        token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        self._client = httpx.Client(
            base_url=self.host,
            headers={
                "Authorization": f"Basic {token}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    @classmethod
    def from_env(cls) -> LangfuseAnnotationClient:
        host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
        if not public_key or not secret_key:
            raise RuntimeError(
                "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set "
                "(see .env.example for setup)"
            )
        return cls(host=host, public_key=public_key, secret_key=secret_key)

    def close(self) -> None:
        self._client.close()

    # --- Score Configs -----------------------------------------------------

    def list_score_configs(self) -> list[dict[str, Any]]:
        """Returns all score configs in the project. Paginated; we walk all pages."""
        out: list[dict[str, Any]] = []
        page = 1
        while True:
            resp = self._client.get(
                "/api/public/score-configs",
                params={"page": page, "limit": 100},
            )
            self._raise_with_context(resp, "list_score_configs")
            data = resp.json()
            out.extend(data.get("data", []))
            meta = data.get("meta", {})
            if page >= meta.get("totalPages", 1):
                break
            page += 1
        return out

    def find_score_config(self, name: str) -> dict[str, Any] | None:
        for cfg in self.list_score_configs():
            if cfg.get("name") == name:
                return cfg
        return None

    def create_score_config(self, spec: ScoreConfigSpec) -> dict[str, Any]:
        body: dict[str, Any] = {
            "name": spec.name,
            "dataType": spec.data_type,
            "description": spec.description,
        }
        if spec.data_type == "NUMERIC":
            if spec.min_value is not None:
                body["minValue"] = spec.min_value
            if spec.max_value is not None:
                body["maxValue"] = spec.max_value
        elif spec.data_type == "CATEGORICAL":
            body["categories"] = spec.categories or []
        # BOOLEAN doesn't need extra fields
        resp = self._client.post("/api/public/score-configs", json=body)
        self._raise_with_context(resp, f"create_score_config({spec.name!r})")
        return resp.json()

    def ensure_score_config(self, spec: ScoreConfigSpec) -> str:
        """Find-or-create; returns the config's id."""
        existing = self.find_score_config(spec.name)
        if existing:
            return existing["id"]
        created = self.create_score_config(spec)
        return created["id"]

    # --- Annotation Queues -------------------------------------------------

    def list_annotation_queues(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        page = 1
        while True:
            resp = self._client.get(
                "/api/public/annotation-queues",
                params={"page": page, "limit": 100},
            )
            self._raise_with_context(resp, "list_annotation_queues")
            data = resp.json()
            out.extend(data.get("data", []))
            meta = data.get("meta", {})
            if page >= meta.get("totalPages", 1):
                break
            page += 1
        return out

    def find_queue(self, name: str) -> dict[str, Any] | None:
        for q in self.list_annotation_queues():
            if q.get("name") == name:
                return q
        return None

    def create_queue(
        self,
        name: str,
        score_config_ids: list[str],
        description: str = "",
    ) -> dict[str, Any]:
        body = {
            "name": name,
            "description": description,
            "scoreConfigIds": score_config_ids,
        }
        resp = self._client.post("/api/public/annotation-queues", json=body)
        self._raise_with_context(resp, f"create_queue({name!r})")
        return resp.json()

    def ensure_queue(
        self,
        name: str,
        score_config_ids: list[str],
        description: str = "",
    ) -> str:
        existing = self.find_queue(name)
        if existing:
            return existing["id"]
        created = self.create_queue(name, score_config_ids, description)
        return created["id"]

    def add_trace_to_queue(self, queue_id: str, trace_id: str) -> dict[str, Any]:
        """Enqueue a single trace for human annotation."""
        body = {"objectId": trace_id, "objectType": "TRACE"}
        resp = self._client.post(
            f"/api/public/annotation-queues/{queue_id}/items",
            json=body,
        )
        self._raise_with_context(
            resp, f"add_trace_to_queue(queue={queue_id}, trace={trace_id})"
        )
        return resp.json()

    # --- Datasets ----------------------------------------------------------

    def ensure_dataset(self, name: str, description: str = "") -> dict[str, Any]:
        """Create-or-upsert a dataset by name.

        `POST /api/public/datasets` is upsert-on-name: re-running with the same
        name updates in place rather than erroring or duplicating. This replaces
        the SDK's `create_dataset`, which is the management surface most exposed
        to SDK-version drift.
        """
        body: dict[str, Any] = {"name": name}
        if description:
            body["description"] = description
        resp = self._client.post("/api/public/datasets", json=body)
        self._raise_with_context(resp, f"ensure_dataset({name!r})")
        return resp.json()

    def upsert_dataset_item(
        self,
        dataset_name: str,
        item_id: str,
        input: Any,
        metadata: Any | None = None,
        expected_output: Any | None = None,
    ) -> dict[str, Any]:
        """Create-or-update one dataset item, keyed on `item_id`.

        `POST /api/public/dataset-items` upserts on `id` — passing the test-case
        folder name as the id makes re-runs idempotent (update in place) instead
        of erroring or producing duplicates. This is the documented REST
        behavior the SDK's `create_dataset_item` was supposed to wrap.
        """
        body: dict[str, Any] = {
            "datasetName": dataset_name,
            "id": item_id,
            "input": input,
        }
        if metadata is not None:
            body["metadata"] = metadata
        if expected_output is not None:
            body["expectedOutput"] = expected_output
        resp = self._client.post("/api/public/dataset-items", json=body)
        self._raise_with_context(
            resp, f"upsert_dataset_item({dataset_name!r}/{item_id!r})"
        )
        return resp.json()

    # --- Helpers -----------------------------------------------------------

    def _raise_with_context(self, resp: httpx.Response, operation: str) -> None:
        """Raise with the full response body on error so debugging is fast.

        The user runs this at work against their Langfuse install and pastes
        errors back — clear context here is the whole iteration loop.
        """
        if resp.is_success:
            return
        body_excerpt = (resp.text or "<no body>")[:1000]
        raise RuntimeError(
            f"Langfuse REST API call failed during {operation}\n"
            f"  URL:    {resp.request.method} {resp.request.url}\n"
            f"  Status: {resp.status_code}\n"
            f"  Body:   {body_excerpt}"
        )


# ---------------------------------------------------------------------------
# Sampling — which traces go to the human queue
# ---------------------------------------------------------------------------


class Sampler:
    """Decides which traces to enqueue for human review.

    Deterministic for `stratified` and `all` strategies — same eval pass run
    twice would enqueue the same traces, which is what you want for
    reproducibility. `random` uses a fixed seed for the same reason.
    """

    def __init__(self, config: HumanReviewConfig, seed: int = 0xC4117E5):
        import random

        self._cfg = config
        self._rng = random.Random(seed)
        self._counts_per_run: dict[str, int] = {}

    def should_sample(
        self,
        run_name: str,
        item_id: str,
        iteration: int,
        overall_score: float,
    ) -> bool:
        if self._cfg.sample_strategy == "all":
            return True
        if self._cfg.sample_strategy == "random":
            return self._rng.random() < self._cfg.sample_rate
        # stratified
        current = self._counts_per_run.get(run_name, 0)
        if current < self._cfg.samples_per_run:
            self._counts_per_run[run_name] = current + 1
            return True
        return False
