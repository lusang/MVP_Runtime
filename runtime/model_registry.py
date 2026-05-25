"""
ModelRegistry — catalog of available models with capabilities, cost, and latency profiles.

Read-only after startup. Models are defined as module-level constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

StepName = Literal["detect", "verify", "attribute", "quality", "negative", "merge"]
ScopeName = Literal["semantic", "quality", "negative"]


@dataclass(frozen=True, slots=True)
class ModelCapability:
    """One capability of a model — which step/scope it supports."""

    step: StepName
    scopes: tuple[ScopeName, ...] = ()
    estimated_latency_ms: float = 100.0
    cost_tier: Literal["free", "low", "medium", "high"] = "free"
    default_confidence_threshold: float = 0.0
    notes: str = ""


@dataclass(frozen=True, slots=True)
class ModelEntry:
    """One model in the catalog."""

    model_id: str
    display_name: str
    model_type: Literal["yolo", "gemini", "opencv", "mechanical"]
    capabilities: tuple[ModelCapability, ...] = ()
    weight_path: str | None = None
    gemini_model_id: str | None = None
    env_var: str | None = None
    notes: str = ""


_MODELS: tuple[ModelEntry, ...] = (
    ModelEntry(
        model_id="yolo-world-v2-x",
        display_name="YOLO-World-v2 X-Large",
        model_type="yolo",
        weight_path="weights/yolov8x-worldv2.pt",
        env_var="YOLO_MODEL_PATH",
        capabilities=(
            ModelCapability(step="detect", estimated_latency_ms=450, cost_tier="free", default_confidence_threshold=0.10),
        ),
        notes="Most accurate, ~200MB, ~450ms on GPU.",
    ),
    ModelEntry(
        model_id="yolo-world-v2-s",
        display_name="YOLO-World-v2 Small",
        model_type="yolo",
        weight_path="weights/yolov8s-worldv2.pt",
        env_var="YOLO_MODEL_PATH_S",
        capabilities=(
            ModelCapability(step="detect", estimated_latency_ms=120, cost_tier="free", default_confidence_threshold=0.15),
        ),
        notes="Faster, less accurate. ~70MB, ~120ms on GPU. Download via ultralytics.",
    ),
    ModelEntry(
        model_id="gemini-2.0-flash",
        display_name="Gemini 2.0 Flash",
        model_type="gemini",
        gemini_model_id="gemini-2.0-flash",
        env_var="GEMINI_MODEL",
        capabilities=(
            ModelCapability(step="verify", estimated_latency_ms=800, cost_tier="low"),
            ModelCapability(step="attribute", scopes=("semantic",), estimated_latency_ms=600, cost_tier="low"),
            ModelCapability(step="attribute", scopes=("negative",), estimated_latency_ms=800, cost_tier="low"),
            ModelCapability(step="negative", scopes=("negative",), estimated_latency_ms=900, cost_tier="low"),
            ModelCapability(step="merge", estimated_latency_ms=2500, cost_tier="low"),
        ),
        notes="Fast, cheap. Good default for most steps.",
    ),
    ModelEntry(
        model_id="gemini-2.5-pro",
        display_name="Gemini 2.5 Pro",
        model_type="gemini",
        gemini_model_id="gemini-2.5-pro",
        env_var="GEMINI_MODEL_PRO",
        capabilities=(
            ModelCapability(step="verify", estimated_latency_ms=2000, cost_tier="medium"),
            ModelCapability(step="attribute", scopes=("semantic",), estimated_latency_ms=1500, cost_tier="medium"),
            ModelCapability(step="attribute", scopes=("negative",), estimated_latency_ms=1800, cost_tier="medium"),
            ModelCapability(step="negative", scopes=("negative",), estimated_latency_ms=2200, cost_tier="medium"),
            ModelCapability(step="merge", estimated_latency_ms=5000, cost_tier="medium"),
        ),
        notes="Higher quality, slower. For complex/high-stakes templates.",
    ),
    ModelEntry(
        model_id="opencv-heuristics",
        display_name="OpenCV Heuristics",
        model_type="opencv",
        capabilities=(
            ModelCapability(step="quality", scopes=("quality",), estimated_latency_ms=30, cost_tier="free"),
        ),
        notes="Blur/lighting/occlusion via Laplacian/histogram/Canny. No API cost.",
    ),
    ModelEntry(
        model_id="mechanical-merge",
        display_name="Mechanical Merge",
        model_type="mechanical",
        capabilities=(
            ModelCapability(step="merge", estimated_latency_ms=5, cost_tier="free"),
        ),
        notes="Averages YOLO + verification scores. Free fallback.",
    ),
)


def get_model_catalog() -> tuple[ModelEntry, ...]:
    return _MODELS


def get_model_entry(model_id: str) -> ModelEntry | None:
    for m in _MODELS:
        if m.model_id == model_id:
            return m
    return None


def get_models_for_step(step: str) -> list[ModelEntry]:
    return [m for m in _MODELS for c in m.capabilities if c.step == step]


def catalog_as_text_for_prompt() -> str:
    """Render the catalog as a compact text table for the Planner prompt."""
    lines = ["Available Models:"]
    for m in _MODELS:
        caps = ", ".join(
            f"{c.step}"
            + (f"({','.join(c.scopes)})" if c.scopes else "")
            + f" ~{c.estimated_latency_ms}ms ${c.cost_tier}"
            for c in m.capabilities
        )
        lines.append(f"  {m.model_id}: [{m.model_type}] {caps}")
    return "\n".join(lines)
