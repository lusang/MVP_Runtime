"""
Public I/O contracts — the only types users of this system need to know about.

Input:  AnnotationRunRequest
Output: AnnotationRunResponse → AnnotationResult + RuntimeTrace
"""

from data.bbox import BBox
from data.io import (
    AnnotationObject,
    AnnotationResult,
    AnnotationRunRequest,
    AnnotationRunResponse,
    RuntimeTrace,
)

__all__ = [
    "AnnotationObject",
    "AnnotationResult",
    "AnnotationRunRequest",
    "AnnotationRunResponse",
    "BBox",
    "RuntimeTrace",
]
