"""
CandidateState — per-detection state object that drives step execution routing.
Replaces the ad-hoc `intermediate_results` list of dicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from schemas.bbox import BBox


@dataclass
class CandidateState:
    """Mutable per-candidate state accumulated across pipeline steps."""

    object_id: str
    bbox: BBox
    analysis_path: str = ""
    analysis_bbox: BBox | None = None
    crop_path: str | None = None

    # Detection
    detector_score: float = 0.0

    # Verification routing flag
    exists: bool = True
    verify_score: float = 0.0

    # Quality visibility (populated by quality step)
    visibility: dict[str, Any] = field(default_factory=dict)

    # Continuous quality metrics (preserved from OpenCV)
    metrics: dict[str, Any] = field(default_factory=dict)

    # Feasibility (computed after quality step) — None = unknown
    attribute_feasibility: dict[str, bool | None] = field(default_factory=dict)
    missing_attributes: list[str] = field(default_factory=list)

    # Pipeline stage outputs
    verification: dict[str, Any] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    negative_flags: dict[str, Any] = field(default_factory=dict)

    # Audit trail
    analysis_history: list[dict[str, str]] = field(default_factory=list)

    # Confidence decomposition — merge is final authority
    detection_confidence: float = 0.0
    verification_confidence: float = 0.0
    merge_confidence: float | None = None
    confidence: float = 0.0

    def record(self, step: str, decision: str) -> None:
        self.analysis_history.append({"step": step, "decision": decision})

    def compute_confidence(self, merge_confidence: float | None = None) -> float:
        """Compute final confidence. merge_confidence overrides mechanical average."""
        self.detection_confidence = self.detector_score
        self.verification_confidence = self.verify_score
        self.merge_confidence = merge_confidence
        if merge_confidence is not None:
            self.confidence = max(0.0, min(1.0, merge_confidence))
        else:
            self.confidence = max(0.0, min(1.0, (self.detector_score + self.verify_score) / 2.0))
        return self.confidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "detector_score": self.detector_score,
            "bbox": self.bbox.model_dump(),
            "crop_path": self.crop_path,
            "exists": self.exists,
            "verify_score": self.verify_score,
            "confidence": self.confidence,
            "detection_confidence": self.detection_confidence,
            "verification_confidence": self.verification_confidence,
            "merge_confidence": self.merge_confidence,
            "visibility": self.visibility,
            "metrics": self.metrics,
            "attribute_feasibility": self.attribute_feasibility,
            "missing_attributes": self.missing_attributes,
            "verification": self.verification,
            "attributes": self.attributes,
            "quality": self.quality,
            "negative_flags": self.negative_flags,
            "analysis_history": self.analysis_history,
        }
