"""
Backward-compatibility re-export — canonical type lives in data.bbox.
"""

from data.bbox import BBox, compute_iou

__all__ = ["BBox", "compute_iou"]
