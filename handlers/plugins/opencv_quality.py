"""
OpenCV plugin for quality attributes (scope=quality).
"""

from __future__ import annotations

from typing import Any

from models.opencv_analyzer import OpenCVAnalyzer
from schemas.bbox import BBox
from schemas.template_spec import ParsedTaskSpec, TemplateAttributeSpec


class OpenCVQualityPlugin:
    """Per-attribute quality analysis via OpenCVAnalyzer."""

    def __init__(self, analyzer: OpenCVAnalyzer | None = None) -> None:
        self._analyzer = analyzer or OpenCVAnalyzer()

    async def analyze(
        self,
        *,
        image_path: str,
        bbox: BBox,
        parsed_template: ParsedTaskSpec,
        object_id: str,
        spec: TemplateAttributeSpec,
    ) -> dict[str, Any]:
        if spec.scope != "quality":
            raise ValueError(f"OpenCVQualityPlugin expected quality scope, got {spec.scope!r}")
        return await self._analyzer.analyze_quality(
            image_path=image_path,
            bbox=bbox,
            parsed=parsed_template,
            object_id=object_id,
            attribute_name=spec.name,
            attribute_type=spec.type,
            options=spec.options,
            description=spec.description,
        )
