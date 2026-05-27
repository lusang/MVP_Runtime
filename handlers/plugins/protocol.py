"""
Attribute plugin protocol — registered handlers implement this async interface.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from schemas.bbox import BBox
from schemas.template_spec import ParsedTaskSpec, TemplateAttributeSpec


@runtime_checkable
class AttributeHandlerPlugin(Protocol):
    """Pluggable per-template-slot attribute analyzer."""

    async def analyze(
        self,
        *,
        image_path: str,
        bbox: BBox,
        parsed_template: ParsedTaskSpec,
        object_id: str,
        spec: TemplateAttributeSpec,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        """Return a JSON-serializable fragment for the given attribute slot.

        Args:
            model_id: Optional model override from Planner resolution. When None,
                      the handler's default model (e.g. GEMINI_MODEL env var) is used.
        """
        ...
