"""Pluggable attribute handler implementations."""

from handlers.plugins.gemini_attribute import GeminiAttributePlugin
from handlers.plugins.gemini_negative import GeminiNegativePlugin
from handlers.plugins.opencv_quality import OpenCVQualityPlugin
from handlers.plugins.protocol import AttributeHandlerPlugin

__all__ = [
    "AttributeHandlerPlugin",
    "GeminiAttributePlugin",
    "GeminiNegativePlugin",
    "OpenCVQualityPlugin",
]
