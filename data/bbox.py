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
