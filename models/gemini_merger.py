"""
GeminiMerger — thin wrapper around deterministic MergeEngine.

Previously sent an execution log to Gemini for merge decisions. Now delegates
to runtime.merge_engine.MergeEngine — pure code, no LLM call.

The class is kept for backward compatibility (imported by container, engine,
executor, tests). New code should use MergeEngine directly.
"""

from __future__ import annotations

from typing import Any

from schemas.template_spec import ParsedTaskSpec


class GeminiMerger:
    """Final annotation merge — delegates to deterministic MergeEngine.

    Maintains the same public interface as before (async merge() method with
    the same signature) so existing callers work without changes.
    """

    def __init__(self, tracer: Any | None = None) -> None:
        # tracer is accepted for backward compatibility but no longer used
        # (merge is now fully deterministic — no LLM to trace)
        # Lazy import to avoid circular dep (runtime → models → runtime)
        from runtime.merge_engine import MergeEngine
        self._engine = MergeEngine()

    async def merge(
        self,
        *,
        image_path: str,
        parsed: ParsedTaskSpec,
        candidates_data: list[dict[str, Any]],
        scene_pure_negative: bool = False,
        scene_fallback: dict[str, Any] | None = None,
        run_id: str = "",
        execution_log_text: str = "",
    ) -> dict[str, Any]:
        return self._engine.merge(
            image_path=image_path,
            parsed=parsed,
            candidates_data=candidates_data,
            scene_pure_negative=scene_pure_negative,
            scene_fallback=scene_fallback,
        )
