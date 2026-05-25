"""
Pydantic schemas shared across API, runtime, and handlers.
"""

from schemas.api import AnnotationObject, AnnotationResult, AnnotationRunRequest, AnnotationRunResponse, RuntimeTrace
from schemas.bbox import BBox
from schemas.object_state import ObjectState
from schemas.template_spec import ParsedTaskSpec, ParsedTemplate, TemplateAttributeSpec

__all__ = [
    "AnnotationObject",
    "AnnotationResult",
    "AnnotationRunRequest",
    "AnnotationRunResponse",
    "BBox",
    "ObjectState",
    "ParsedTaskSpec",
    "ParsedTemplate",
    "RuntimeTrace",
    "TemplateAttributeSpec",
]
