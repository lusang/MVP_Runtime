"""
Validated shapes for `resource/Template.json` consumed by `TemplateParser`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

AttributeScope = Literal["semantic", "quality", "negative"]

HANDLER_BY_SCOPE: dict[AttributeScope, str] = {
    "semantic": "gemini",
    "quality": "opencv_quality",
    "negative": "gemini_negative",
}


class TemplateAttributeSpec(BaseModel):
    """One executable attribute slot (handler assigned by parser from scope).

    ``layer`` controls pipeline routing:
      1 = core features (Gemini per-candidate, training signal)
      2 = visibility (OpenCV numeric, no LLM)
      3 = confusion factors (Gemini scene-level, guard)
      4 = scene semantics (metadata only, no training)
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    type: str = Field(default="unknown")
    layer: int = Field(default=1, ge=1, le=4)
    options: list[Any] = Field(default_factory=list)
    description: str = Field(default="")
    handler: str = Field(..., min_length=1)
    key: str = Field(..., min_length=1)
    scope: AttributeScope
    params: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = Field(default=True)
    analysis_scope: Literal["crop", "full_image"] = Field(
        default="crop",
        description="crop=assess within detection crop; full_image=needs full-scene context (e.g. background, person behavior).",
    )
    quality_requirements: dict[str, Any] | None = Field(
        default=None,
        description="Per-attribute quality thresholds (max_occlusion, max_blur, max_lighting_issue).",
    )

    @field_validator("name", "handler", "key")
    @classmethod
    def _strip_required(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("must be non-empty")
        return s


class ParsedTaskSpec(BaseModel):
    """Normalized task template for RuntimeEngine (resource/Template.json shape)."""

    model_config = ConfigDict(extra="forbid")

    object_name: str = Field(..., description="objects[0].name — detection / verification target.")
    description: str = Field(default="")
    include: str = Field(default="")
    exclude: str = Field(default="")
    geometry: str = Field(default="bbox")

    semantic_attributes: list[TemplateAttributeSpec] = Field(default_factory=list)
    quality_attributes: list[TemplateAttributeSpec] = Field(default_factory=list)
    negative_attributes: list[TemplateAttributeSpec] = Field(default_factory=list)

    quality_block: dict[str, Any] = Field(default_factory=dict)
    negative_block: dict[str, Any] = Field(default_factory=dict)

    extras: dict[str, Any] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def all_attribute_slots(self) -> list[TemplateAttributeSpec]:
        """Execution order: semantic → quality → negative."""
        return [
            *self.semantic_attributes,
            *self.quality_attributes,
            *self.negative_attributes,
        ]


# Backward-compatible alias
ParsedTemplate = ParsedTaskSpec
