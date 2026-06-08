"""Thin client over the LiteLLM Proxy.

Caliper does NOT import `litellm` directly. Instead it speaks OpenAI's HTTP
API to the LiteLLM proxy container, which handles vendor routing, retries, and
rate limits. This keeps Caliper's runtime deps minimal and — importantly —
avoids LiteLLM's auto-instrumentation of Langfuse so that *we* explicitly own
every Langfuse write from the eval runner.

If you later swap the proxy out for direct vendor SDK calls, the wrapper here
is the only Caliper module that changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


@dataclass
class LlmResult:
    """Normalized result from one LLM call.

    `model` is whatever the proxy reported back — usually the canonical
    upstream model name (e.g. resolves an alias like `claude-sonnet` to the
    dated upstream id). We pass that to Langfuse so its server-side cost
    map can compute cost.
    """

    output: str
    model: str
    usage: dict[str, int]
    raw_response: Any


class LiteLLMProxyClient:
    """OpenAI-compatible client pointed at the local LiteLLM proxy."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.base_url = base_url or os.environ.get(
            "LITELLM_BASE_URL", "http://localhost:4000"
        )
        api_key_resolved = api_key or os.environ.get("LITELLM_MASTER_KEY")
        if not api_key_resolved:
            raise RuntimeError(
                "LITELLM_MASTER_KEY is required (set in .env or pass api_key=)"
            )
        self.api_key = api_key_resolved
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LlmResult:
        """Make a chat-completion call through the proxy."""
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        )

        output = response.choices[0].message.content or ""
        usage = {
            "input": response.usage.prompt_tokens if response.usage else 0,
            "output": response.usage.completion_tokens if response.usage else 0,
            "total": response.usage.total_tokens if response.usage else 0,
        }

        return LlmResult(
            output=output,
            model=response.model,
            usage=usage,
            raw_response=response,
        )
