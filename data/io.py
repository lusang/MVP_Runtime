"""
Public API I/O types — the single contract between callers and the runtime.

Input:  AnnotationRunRequest  {"image_path": "...", "template_path": "..."}
Output: AnnotationRunResponse
        ├── annotation_result: AnnotationResult  (clean → import into annotation platforms)
        └── runtime_trace:     RuntimeTrace      (debug → badcase analysis, evaluator)
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class AnnotationRunRequest(BaseModel):
    """Body for `POST /run_annotation`."""

    image_path: str = Field(..., description="Filesystem path to the source image.")
    template_path: str = Field(..., description="Filesystem path to template.json configuration.")


class AnnotationObject(BaseModel):
    """Clean per-object output for annotation platforms — no trace/debug data."""

    bbox: list[float] = Field(..., description="Bounding box as [x1, y1, x2, y2] in full-image coordinates.")
    category: str = Field(..., description="Target object name from template (e.g. Package).")
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Semantic attributes with non-null values (excludes quality, negative, infeasible).",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Layer 4 analytics attributes (scene semantics, not for training).",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    status: Literal["accepted", "rejected", "pending"] = Field(
        default="pending",
        description="Final verdict: accepted (positive annotation), rejected (negative/distractor), pending (ambiguous).",
    )


class AnnotationResult(BaseModel):
    """Clean annotation output ready for import into annotation platforms."""

    image: str = Field(..., description="Source image path.")
    objects: list[AnnotationObject] = Field(default_factory=list)


class RuntimeTrace(BaseModel):
    """Debug/trace data for system analysis, badcase review, and evaluator consumption."""

    steps: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Executed step IDs with model info (step:model_id).",
    )
    candidate_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Full per-candidate state across all pipeline steps.",
    )
    planner_decisions: dict[str, Any] = Field(
        default_factory=dict,
        description="Plan metadata: plan_id, planner_model, steps list, early_exit_rules, skip_conditions.",
    )
    quality_scores: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per-candidate quality visibility scores and continuous metrics.",
    )
    merge_reasoning: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Reasoning traces from the merge/audit step.",
    )
    resolved_attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Conflict-resolved attributes across positive candidates (highest confidence per key).",
    )
    annotation_panel: dict[str, Any] | None = Field(
        default=None,
        description="Per-object merge panels keyed by object_id (raw merge output before ObjectStateBuilder).",
    )
    meta: dict[str, Any] = Field(
        default_factory=dict,
        description="Pipeline metadata (run_id, engine version, elapsed_ms, object_name, scene_pure_negative).",
    )


class AnnotationRunResponse(BaseModel):
    """Unified response from the runtime engine — split into annotation result and trace."""

    annotation_result: AnnotationResult
    runtime_trace: RuntimeTrace


# ─────────────────────────────────────────────────────────────────────────────
# Async batch annotation models  (POST /run_annotation_async)
# ─────────────────────────────────────────────────────────────────────────────


class AsyncFrameRequest(BaseModel):
    """A single frame within an async task."""

    frame_id: str = Field(..., description="Frame-level identifier from caller.")
    url: str = Field(..., description="File URL (file:// or http(s)://).")
    timestamp_ms: int = Field(default=0, description="Timestamp in video (ms); 0 for still images.")


class AsyncTaskItem(BaseModel):
    """A single task within an async batch — each task gets one callback."""

    task_id: str = Field(..., description="Caller-side unique task identifier, echoed in callback.")
    media_type: Literal["image", "video_clip"] = Field(default="image", description="image or video_clip.")
    frames: list[AsyncFrameRequest] = Field(..., description="One or more frames to annotate.")
    fps: float | None = Field(default=None, description="Video framerate (suggested for video_clip).")


class AsyncAnnotationRequest(BaseModel):
    """Body for ``POST /run_annotation_async``."""

    template: dict[str, Any] = Field(..., description="Template JSON with ``objects[]`` matching Template.json structure.")
    callback_url: str = Field(..., description="MVP calls this URL after finishing EACH task.")
    tasks: list[AsyncTaskItem] = Field(..., description="Task list — each task gets an independent callback.")


class AsyncAnnotationResponse(BaseModel):
    """Immediate acceptance response for ``POST /run_annotation_async``."""

    run_id: str = Field(..., description="MVP-side run identifier for tracking.")
    status: Literal["accepted"] = Field(default="accepted", description="Fixed: accepted.")
    task_count: int = Field(..., description="Number of tasks accepted.")

