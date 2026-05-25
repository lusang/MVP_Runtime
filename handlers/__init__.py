"""
Handler layer: orchestration glue between runtime steps and model adapters.
"""

from handlers.attribute_handler import AttributeHandler, AttributeStageResult
from handlers.registry import AttributeHandlerRegistry
from handlers.verification_handler import VerificationHandler

__all__ = [
    "VerificationHandler",
    "AttributeHandler",
    "AttributeStageResult",
    "AttributeHandlerRegistry",
]
