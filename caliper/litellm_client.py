"""Thin client over the LiteLLM Proxy.

Caliper does NOT import `litellm` directly. Instead it speaks OpenAI's HTTP
API to the LiteLLM proxy container, which handles vendor routing, retries, and
rate limits. This keeps Caliper's runtime deps minimal and — importantly —
avoids LiteLLM's auto-instrumentation of Langfuse so that *we* explicitly own
every Langfuse write from the eval runner.

The client wraps `chat.completions.create` with tenacity-driven exponential
backoff: 429/5xx/timeout/connection errors retry per the RetryConfig on the
EvalConfig (or the default if none supplied). Per-cell retries are bounded;
when they exhaust, the caller catches the exception and skips that cell so
the matrix continues.

If you later swap the proxy out for direct vendor SDK calls, the wrapper here
is the only Caliper module that changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import tenacity
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
)
from tenacity import (
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from caliper.schemas import RetryConfig


@dataclass
class LlmResult:
    """Normalized result from one LLM call.

    `model` is whatever the proxy reported back — usually the canonical
    upstream model name (e.g. resolves an alias to the dated upstream id).
    We pass that to Langfuse so its server-side cost map can compute cost.
    """

    output: str
    model: str
    usage: dict[str, int]
    raw_response: Any


class LiteLLMProxyClient:
    """OpenAI-compatible client pointed at the local LiteLLM proxy, with retry."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        retry_config: RetryConfig | None = None,
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

        self.retry_config = retry_config or RetryConfig()
        self._retry_statuses = set(self.retry_config.retry_on_statuses)

    def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LlmResult:
        """Make a chat-completion call through the proxy, with retry/backoff."""
        retrying = tenacity.Retrying(
            stop=stop_after_attempt(self.retry_config.max_attempts),
            wait=wait_exponential_jitter(
                initial=self.retry_config.initial_wait_seconds,
                max=self.retry_config.max_wait_seconds,
                exp_base=self.retry_config.exponential_base,
                jitter=self.retry_config.jitter_seconds,
            ),
            retry=retry_if_exception(self._is_retryable),
            reraise=True,
            before_sleep=self._log_retry,
        )
        return retrying(self._do_complete, model, messages, **kwargs)

    def _do_complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LlmResult:
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

    def _is_retryable(self, exc: BaseException) -> bool:
        """Classify an exception as transient (retry) or fatal (raise)."""
        if isinstance(exc, (APITimeoutError, APIConnectionError)):
            return True
        if isinstance(exc, APIStatusError):
            return exc.status_code in self._retry_statuses
        return False

    @staticmethod
    def _log_retry(retry_state: tenacity.RetryCallState) -> None:
        """Print one line per retry attempt so users see progress, not silence."""
        import sys

        attempt = retry_state.attempt_number
        wait = retry_state.next_action.sleep if retry_state.next_action else 0
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        exc_summary = type(exc).__name__ if exc else "unknown"
        status = getattr(exc, "status_code", "-") if exc else "-"
        print(
            f"[caliper] retry: attempt {attempt} hit {exc_summary} "
            f"(status={status}); sleeping {wait:.1f}s",
            file=sys.stderr,
        )
