"""
Protocol definitions for model adapters (YOLO, VLM verifier, OpenCV analysis).
"""

from typing import Any, Protocol

from schemas.bbox import BBox
from schemas.detection import DetectionCandidate
from schemas.template_spec import ParsedTaskSpec


class DetectorProtocol(Protocol):
    """Anything that can propose bounding boxes for an image path and target object."""

    async def detect(
        self,
        image_path: str,
        *,
        target_object: str,
        parsed: ParsedTaskSpec,
        run_id: str = "default",
    ) -> list[DetectionCandidate]:
        """Return zero or more detection candidates."""
        ...


class VerifierProtocol(Protocol):
    """Anything that can verify a crop / bbox against task semantics."""

    async def verify_object(
        self,
        *,
        image_path: str,
        bbox: BBox,
        parsed: ParsedTaskSpec,
        object_id: str,
    ) -> dict[str, Any]:
        """Return object-level verification payload."""
        ...

    async def verify_attribute(
        self,
        *,
        image_path: str,
        bbox: BBox,
        parsed: ParsedTaskSpec,
        object_id: str,
        attribute_name: str,
        attribute_type: str,
        options: list[Any],
        description: str,
        scope: str,
    ) -> dict[str, Any]:
        """Return per-attribute verification / value payload."""
        ...


class VisualAnalyzerProtocol(Protocol):
    """Low-level CV analysis behind OpenCV-named adapter."""

    async def analyze_quality(
        self,
        *,
        image_path: str,
        bbox: BBox,
        parsed: ParsedTaskSpec,
        object_id: str,
        attribute_name: str,
        attribute_type: str,
        options: list[Any],
        description: str,
    ) -> dict[str, Any]:
        ...
