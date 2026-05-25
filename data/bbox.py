"""
Bounding-box geometry primitive.
"""

from pydantic import BaseModel, Field


class BBox(BaseModel):
    """Axis-aligned box in pixel coordinates: top-left (x1,y1), bottom-right (x2,y2)."""

    x1: float = Field(..., description="Left / min x")
    y1: float = Field(..., description="Top / min y")
    x2: float = Field(..., description="Right / max x")
    y2: float = Field(..., description="Bottom / max y")

    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def compute_iou(a: BBox, b: BBox) -> float:
    """Intersection over Union of two bounding boxes."""
    x_left = max(a.x1, b.x1)
    y_top = max(a.y1, b.y1)
    x_right = min(a.x2, b.x2)
    y_bottom = min(a.y2, b.y2)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    inter = (x_right - x_left) * (y_bottom - y_top)
    union = a.area() + b.area() - inter
    return inter / union if union > 0 else 0.0
