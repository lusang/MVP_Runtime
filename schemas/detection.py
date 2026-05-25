"""
Structures produced by the detection adapter before full `ObjectState` materialization.
"""

from pydantic import BaseModel, Field

from schemas.bbox import BBox


class DetectionCandidate(BaseModel):
    """
    One bounding box hypothesis from the detector stage.

    Serializes to: bbox [x1,y1,x2,y2], confidence, label (typically "candidate").
    Pre-selection crop path is written under storage/preselection/{run_id}/.
    """

    bbox: BBox = Field(..., description="Candidate box in image pixel space.")
    label: str = Field(default="candidate", description="Detector label / role.")
    score: float = Field(default=0.5, ge=0.0, le=1.0, description="Detector confidence.")
    target_object: str = Field(default="", description="Template object name used for detection.")
    crop_path: str | None = Field(
        default=None,
        description="Filesystem path to cropped pre-selection image for this candidate.",
    )

    def to_api_dict(self) -> dict:
        """Unified list-item shape for detection consumers."""
        return {
            "bbox": [self.bbox.x1, self.bbox.y1, self.bbox.x2, self.bbox.y2],
            "confidence": self.score,
            "label": self.label,
            "target_object": self.target_object,
            "crop_path": self.crop_path,
        }
