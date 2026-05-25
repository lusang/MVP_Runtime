"""
Application composition root — wires adapters, registries, and `RuntimeEngine`.
"""

from __future__ import annotations

from typing import Any

from dataclasses import dataclass

from handlers.attribute_handler import AttributeHandler
from handlers.plugins.gemini_attribute import GeminiAttributePlugin
from handlers.plugins.gemini_negative import GeminiNegativePlugin
from handlers.plugins.opencv_quality import OpenCVQualityPlugin
from handlers.registry import AttributeHandlerRegistry
from handlers.verification_handler import VerificationHandler
from models.gemini_merger import GeminiMerger
from models.gemini_verifier import GeminiVerifier
from models.opencv_analyzer import OpenCVAnalyzer
from models.yolo_detector import YOLODetector
from runtime.engine import RuntimeEngine
from runtime.performance_tracker import PerformanceTracker
from runtime.planner import Planner
from runtime.template_parser import TemplateParser


@dataclass(frozen=True, slots=True)
class AppContainer:
    attribute_registry: AttributeHandlerRegistry
    runtime_engine: RuntimeEngine
    tracer: Any = None


def build_default_attribute_registry(
    *,
    verifier: GeminiVerifier | None = None,
    analyzer: OpenCVAnalyzer | None = None,
) -> AttributeHandlerRegistry:
    gemini = verifier or GeminiVerifier()
    cv = analyzer or OpenCVAnalyzer()
    registry = AttributeHandlerRegistry()
    registry.register("gemini", lambda: GeminiAttributePlugin(gemini))
    registry.register("gemini_negative", lambda: GeminiNegativePlugin(gemini))
    registry.register("opencv_quality", lambda: OpenCVQualityPlugin(cv))
    return registry


def build_container(*, tracer: Any = None) -> AppContainer:
    from runtime.tracer import GeminiTracer

    effective_tracer = tracer if tracer is not None else GeminiTracer()
    verifier = GeminiVerifier(tracer=effective_tracer)
    analyzer = OpenCVAnalyzer()
    registry = build_default_attribute_registry(verifier=verifier, analyzer=analyzer)
    tracker = PerformanceTracker()
    tracker.ensure_schema()

    engine = RuntimeEngine(
        detector=YOLODetector(),
        verifier=verifier,
        verification_handler=VerificationHandler(verifier),
        attribute_registry=registry,
        attribute_handler=AttributeHandler(registry),
        template_parser=TemplateParser(),
        merger=GeminiMerger(tracer=effective_tracer),
        planner=Planner(),
        tracker=tracker,
    )
    return AppContainer(
        attribute_registry=registry,
        runtime_engine=engine,
        tracer=effective_tracer,
    )


__all__ = ["AppContainer", "build_container", "build_default_attribute_registry"]
