"""
Dynamic attribute stage — template-driven plugins via `AttributeHandlerRegistry`.

Results are partitioned by scope into semantic attributes, quality, and negative maps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from handlers.registry import AttributeHandlerRegistry
from schemas.bbox import BBox
from schemas.template_spec import AttributeScope, ParsedTaskSpec


@dataclass
class AttributeStageResult:
    """Per-bbox handler outputs partitioned by template scope."""

    attributes: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    negative: dict[str, Any] = field(default_factory=dict)


class AttributeHandler:
    """
    Orchestrates pluggable handlers for each template attribute slot.

    Execution order: semantic → quality → negative (see `ParsedTaskSpec.all_attribute_slots`).
    """

    def __init__(self, registry: AttributeHandlerRegistry) -> None:
        self._registry = registry

    async def analyze_by_scopes(
        self,
        *,
        image_path: str,
        bbox: BBox,
        parsed: ParsedTaskSpec,
        object_id: str,
        full_image_path: str | None = None,
        full_bbox: BBox | None = None,
        skip_keys: frozenset[str] | None = None,
        include_keys: frozenset[str] | None = None,
        scopes: set[str] | None = None,
    ) -> AttributeStageResult:
        result = AttributeStageResult()
        buckets: dict[AttributeScope, dict[str, Any]] = {
            "semantic": result.attributes,
            "quality": result.quality,
            "negative": result.negative,
        }

        for spec in parsed.all_attribute_slots:
            if not spec.enabled:
                continue
            if scopes is not None and spec.scope not in scopes:
                continue
            if skip_keys and spec.key in skip_keys:
                continue
            if include_keys is not None and spec.key not in include_keys:
                continue

            # Negative-scope plugins receive the FULL image + original bbox for context
            if spec.scope == "negative" and full_image_path is not None:
                effective_image = full_image_path
                effective_bbox = full_bbox or bbox
            else:
                effective_image = image_path
                effective_bbox = bbox

            plugin = self._registry.resolve(spec.handler)
            fragment = await plugin.analyze(
                image_path=effective_image,
                bbox=effective_bbox,
                parsed_template=parsed,
                object_id=object_id,
                spec=spec,
            )

            bucket = buckets[spec.scope]
            if spec.key in bucket:
                raise ValueError(
                    f"duplicate attribute key {spec.key!r} in scope {spec.scope!r}"
                )
            bucket[spec.key] = fragment

        return result
