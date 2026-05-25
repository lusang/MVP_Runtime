"""
`TemplateParser` — load and validate resource/Template.json into `ParsedTaskSpec`.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import ValidationError

from schemas.template_spec import (
    HANDLER_BY_SCOPE,
    AttributeScope,
    ParsedTaskSpec,
    TemplateAttributeSpec,
)


class TemplateParser:
    """
    Parses `objects[]` template (MVP: first object only).

    Auto-maps attribute scopes to registry handler ids:
    semantic → gemini, quality → opencv_quality, negative → gemini_negative.
    """

    def parse(self, raw: dict[str, Any]) -> ParsedTaskSpec:
        if not isinstance(raw, dict):
            raise ValueError("template root must be a JSON object")

        raw_copy = deepcopy(raw)
        objects_raw = raw_copy.pop("objects", None)
        if objects_raw is None:
            raise ValueError("template must contain `objects` array")
        if not isinstance(objects_raw, list) or not objects_raw:
            raise ValueError("`objects` must be a non-empty array")
        if not isinstance(objects_raw[0], dict):
            raise ValueError("`objects[0]` must be an object")

        obj = dict(objects_raw[0])
        extras = dict(raw_copy)

        name = obj.get("name")
        if not name or not str(name).strip():
            raise ValueError("`objects[0].name` is required")
        object_name = str(name).strip()

        semantic = self._parse_attribute_list(
            obj.get("attributes"),
            scope="semantic",
            label="objects[0].attributes",
        )
        quality_block = obj.get("quality") if isinstance(obj.get("quality"), dict) else {}
        quality = self._parse_attribute_list(
            quality_block.get("attributes"),
            scope="quality",
            label="objects[0].quality.attributes",
        )
        negative_block = obj.get("negative") if isinstance(obj.get("negative"), dict) else {}
        negative = self._parse_attribute_list(
            negative_block.get("attributes"),
            scope="negative",
            label="objects[0].negative.attributes",
        )

        return ParsedTaskSpec(
            object_name=object_name,
            description=str(obj.get("description", "")),
            include=str(obj.get("include", "")),
            exclude=str(obj.get("exclude", "")),
            geometry=str(obj.get("geometry", "bbox")),
            semantic_attributes=semantic,
            quality_attributes=quality,
            negative_attributes=negative,
            quality_block=dict(quality_block),
            negative_block=dict(negative_block),
            extras=extras,
            raw=deepcopy(raw),
        )

    def _parse_attribute_list(
        self,
        items: Any,
        *,
        scope: AttributeScope,
        label: str,
    ) -> list[TemplateAttributeSpec]:
        if items is None:
            return []
        if not isinstance(items, list):
            raise ValueError(f"`{label}` must be an array")

        handler = HANDLER_BY_SCOPE[scope]
        result: list[TemplateAttributeSpec] = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"`{label}[{i}]` must be an object")
            attr_name = item.get("name")
            if not attr_name or not str(attr_name).strip():
                raise ValueError(f"`{label}[{i}].name` is required")
            name = str(attr_name).strip()
            params = {
                "attribute_name": name,
                "attribute_type": item.get("type", "unknown"),
                "options": list(item.get("options") or []),
                "description": str(item.get("description", "")),
                "scope": scope,
            }
            try:
                spec = TemplateAttributeSpec(
                    name=name,
                    type=str(item.get("type", "unknown")),
                    options=list(item.get("options") or []),
                    description=str(item.get("description", "")),
                    handler=handler,
                    key=name,
                    scope=scope,
                    params=params,
                    enabled=True,
                )
            except ValidationError as exc:
                raise ValueError(f"invalid `{label}[{i}]`: {exc}") from exc
            result.append(spec)
        return result
