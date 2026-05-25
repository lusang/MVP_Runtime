"""
Backward-compatibility re-exports — canonical types live in data.io.
"""

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
    "RuntimeTrace",
]
