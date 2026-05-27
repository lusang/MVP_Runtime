"""
Gemini plugin for semantic attributes (scope=semantic).
"""

from __future__ import annotations

from typing import Any

from models.gemini_verifier import GeminiVerifier
from schemas.bbox import BBox
from schemas.template_spec import ParsedTaskSpec, TemplateAttributeSpec


class GeminiAttributePlugin:
    """Per-attribute semantic verification via GeminiVerifier."""

    def __init__(self, verifier: GeminiVerifier | None = None) -> None:
        self._verifier = verifier or GeminiVerifier()

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
        return await self._verifier.verify_attribute(
            image_path=image_path,
            bbox=bbox,
            parsed=parsed_template,
            object_id=object_id,
            attribute_name=spec.name,
            attribute_type=spec.type,
            options=spec.options,
            description=spec.description,
            scope=spec.scope,
            model_id=model_id,
        )
