"""
Core domain model: unified per-object annotation state.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from schemas.bbox import BBox


class ObjectState(BaseModel):
    """
    Single object lifecycle container inside the runtime.

    Partitions:
    - attributes: semantic (Gemini)
    - quality: quality block (OpenCV)
    - negative_attributes: negative checks (Gemini)
    """

    model_config = ConfigDict(extra="allow")

    object_id: str = Field(..., description="Stable id within one runtime invocation.")
    object_name: str = Field(..., description="Template target object name (e.g. Package).")
    bbox: BBox = Field(..., description="Detection box from YOLO (full-image coordinates).")
    preselection_crop_path: str | None = Field(
        default=None,
        description="Cropped 预选 image path for this candidate; downstream stages prefer this.",
    )
    verification: dict[str, Any] = Field(
        default_factory=dict,
        description="Gemini object-level verification.",
    )
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Semantic attributes keyed by template attribute name.",
    )
    quality: dict[str, Any] = Field(
        default_factory=dict,
        description="Quality attributes keyed by template attribute name.",
    )
    negative_attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Negative-sample attribute results keyed by template name.",
    )
    negative: bool = Field(
        default=False,
        description="Whether this detection is treated as hard-negative / distractor.",
    )
    negative_category: str | None = Field(
        default=None,
        description="Category label when negative (Pure Negative, Hard Negative, Ambiguous, Open-set Negative).",
    )
    is_positive: bool | None = Field(
        default=None,
        description="Final merge judgment: True = positive annotation, False = negative.",
    )
    annotation_panel: dict[str, Any] | None = Field(
        default=None,
        description="Per-object merge panel from Gemini final merge step.",
    )
    visibility: dict[str, Any] = Field(
        default_factory=dict,
        description="Quality visibility scores: {occlusion, blur, lighting} each with value/confidence.",
    )
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Continuous quality metrics from OpenCV (laplacian_var, edge_density, histogram_mean).",
    )
    attribute_feasibility: dict[str, bool | None] = Field(
        default_factory=dict,
        description="Per-attribute feasibility flags computed from quality visibility. None = unknown.",
    )
    detection_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Detector confidence score.",
    )
    verification_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Verification confidence score.",
    )
    merge_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Merge step confidence — the final authority.",
    )
    missing_attributes: list[str] = Field(
        default_factory=list,
        description="Attributes that were skipped due to insufficient quality.",
    )
    analysis_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Step-by-step audit trail of routing decisions.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Combined confidence in [0, 1].",
    )
