"""
Model *adapters* — pluggable backends (mock or real) behind stable async interfaces.

This package intentionally mirrors the name in the project spec; it is not an ORM layer.
"""

from models.gemini_verifier import GeminiVerifier
from models.opencv_analyzer import OpenCVAnalyzer
from models.yolo_detector import YOLODetector

__all__ = ["YOLODetector", "GeminiVerifier", "OpenCVAnalyzer"]
