"""
Optional LLM call tracer backed by LangFuse. No-op when not configured.

Usage:
    tracer = GeminiTracer()  # no-op if env vars not set
    if tracer.enabled:
        tracer.observe(run_id="...", step_name="verify_object", model="gemini-2.0-flash",
                       prompt="...", response="...", usage={"input_tokens": 100, "output_tokens": 50})
    tracer.flush()
"""

from __future__ import annotations

import os
from typing import Any


def _langfuse_configured() -> bool:
    """LangFuse is active only when both key and host are set."""
    return bool(os.environ.get("LANGFUSE_SECRET_KEY", "").strip()) and bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    )


def _langfuse_host() -> str:
    return os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com").strip()


class GeminiTracer:
    """Records LLM generation events to LangFuse. Fully no-op when disabled.

    One trace per ``run_id``; each ``observe()`` call creates a span
    inside that trace.
    """

    def __init__(self) -> None:
        self._client: Any = None
        self._traces: dict[str, Any] = {}
        self._enabled = False
        self._init_client()

    # ── public ─────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    def observe(
        self,
        *,
        run_id: str,
        step_name: str,
        model: str,
        prompt: str,
        response: str,
        usage: dict[str, int] | None = None,
    ) -> None:
        """Record one LLM generation as a span under the run_id trace."""
        if not self._enabled or self._client is None:
            return

        trace = self._get_or_create_trace(run_id)
        trace.generation(
            name=step_name,
            model=model,
            input=prompt,
            output=response,
            usage=self._langfuse_usage(usage) if usage else None,
        )

    def flush(self) -> None:
        """Force-flush pending events to LangFuse."""
        if self._enabled and self._client is not None:
            self._client.flush()

    # ── internal ───────────────────────────────────────────────────

    def _init_client(self) -> None:
        if not _langfuse_configured():
            self._enabled = False
            return
        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                secret_key=os.environ["LANGFUSE_SECRET_KEY"],
                public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
                host=_langfuse_host(),
            )
            self._enabled = True
        except ImportError:
            self._enabled = False

    def _get_or_create_trace(self, run_id: str) -> Any:
        if run_id not in self._traces:
            self._traces[run_id] = self._client.trace(id=run_id, name="pipeline_run")
        return self._traces[run_id]

    @staticmethod
    def _langfuse_usage(usage: dict[str, int]) -> dict[str, Any]:
        return {
            "input": usage.get("input_tokens", 0),
            "output": usage.get("output_tokens", 0),
            "unit": "TOKENS",
        }
