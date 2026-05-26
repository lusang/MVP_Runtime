"""
Background batch orchestrator for ``POST /run_annotation_async``.

Flow per request:
  1. Parse template JSON (from request body) → ``ParsedTaskSpec``
  2. Compile plan once via ``Planner.compile(parsed)``
  3. Create temp directory for http:// downloads
  4. For each task → for each frame:
       - Resolve URL (file:// direct, http:// download)
       - Run ``RuntimeEngine.run_with_plan``
       - Collect per-frame result
  5. Per task: POST callback (with retry)
  6. Clean up temp directory
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

from runtime.batch_registry import is_cancelled, mark_processed, unregister
from runtime.callback import post_callback
from runtime.engine import RuntimeEngine
from runtime.planner import Planner
from runtime.template_parser import TemplateParser
from data.io import (
    AsyncAnnotationRequest,
    AsyncTaskItem,
)
from schemas.template_spec import ParsedTaskSpec
from storage.url_resolver import clean_temp_dir, make_temp_dir, resolve_url

logger = logging.getLogger("mvp.async_worker")


TRACES_DIR = Path(__file__).resolve().parent.parent / "storage" / "traces"


def _run_annotation_result_to_dict(obj: Any) -> dict[str, Any]:
    """Convert a Pydantic model to a plain dict, recursing into nested models."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj


def _save_trace(run_id: str, task_id: str, frame_id: str, trace: dict[str, Any]) -> None:
    """Append frame trace to run-level file at storage/traces/{run_id}.json."""
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    path = TRACES_DIR / f"{run_id}.json"

    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {"run_id": run_id, "tasks": []}
    else:
        existing = {"run_id": run_id, "tasks": []}

    task_entry = None
    for t in existing["tasks"]:
        if t.get("task_id") == task_id:
            task_entry = t
            break
    if task_entry is None:
        task_entry = {"task_id": task_id, "frames": []}
        existing["tasks"].append(task_entry)

    task_entry["frames"].append({
        "frame_id": frame_id,
        "trace": trace,
    })

    try:
        path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        logger.error("Failed to write trace %s: %s", path, exc)


def build_operator_summary(trace: Any, ann: Any) -> dict[str, Any]:
    """Human-readable summary for operators/annotators.

    Returns: decision, reason, confidence, risk.
    """
    meta = trace.meta if hasattr(trace, "meta") else {}
    object_name = meta.get("object_name", "object")

    # -- decision --
    accepted = [o for o in (ann.objects or []) if o.status == "accepted"]
    rejected = [o for o in (ann.objects or []) if o.status == "rejected"]

    if meta.get("scene_pure_negative"):
        decision = f"Pure negative scene, no {object_name}"
    elif len(accepted) == 1:
        cat = accepted[0].category or object_name
        decision = f"Accepted as {cat}"
    elif len(accepted) > 1:
        decision = f"{len(accepted)} {object_name}(s) accepted"
    elif len(rejected) == len(ann.objects or []):
        decision = f"All candidates rejected"
    else:
        decision = f"No {object_name} accepted"

    # -- reason: pick most informative merge_reasoning line --
    merge_reasoning = list(trace.merge_reasoning or [])
    reason = ""
    if merge_reasoning:
        # Priority: accept/reject > confirm > detect > first
        high_prio = []
        mid_prio = []
        low_prio = []
        for line in merge_reasoning:
            text = ""
            if isinstance(line, dict):
                text = line.get("output") or line.get("reasoning") or str(line)
            else:
                text = str(line)
            t = text.lower()
            if any(kw in t for kw in ("accept", "reject")):
                high_prio.append(text)
            elif "confirm" in t:
                mid_prio.append(text)
            elif "detect" in t:
                low_prio.append(text)
        if high_prio:
            reason = high_prio[0]
        elif mid_prio:
            reason = mid_prio[0]
        elif low_prio:
            reason = low_prio[0]
        else:
            entry = merge_reasoning[0]
            reason = entry.get("output", str(entry)) if isinstance(entry, dict) else str(entry)
    elif trace.candidate_history:
        reason = f"{len(trace.candidate_history)} candidate(s) detected"

    # -- confidence: max across candidates --
    confidence = None
    max_conf = 0.0
    for c in (trace.candidate_history or []):
        c_conf = c.get("confidence", 0) if isinstance(c, dict) else 0
        if c_conf and c_conf > max_conf:
            max_conf = c_conf
    if max_conf > 0:
        confidence = round(max_conf, 4)

    # -- risk --
    risk = []
    if meta.get("scene_pure_negative"):
        risk.append("scene_pure_negative")
    for qs in (trace.quality_scores or []):
        if isinstance(qs, dict) and qs.get("visibility", 1) < 0.5:
            risk.append(f"low_visibility:{qs.get('object_id', '?')}")

    return {
        "decision": decision,
        "reason": reason,
        "confidence": confidence,
        "risk": risk,
    }


def build_runtime_summary(trace: Any) -> dict[str, Any]:
    """Technical runtime summary for developers.

    Returns: step_flow, planner, merge_strategy, objects_detected,
    scene_pure_negative, elapsed_ms, executed_steps.
    """
    meta = trace.meta if hasattr(trace, "meta") else {}
    planner_decisions = trace.planner_decisions if hasattr(trace, "planner_decisions") else {}

    step_flow = meta.get("executed_steps", []) or [
        s["step_id"] if isinstance(s, dict) else str(s)
        for s in (trace.steps or [])
    ]

    return {
        "step_flow": step_flow,
        "planner": planner_decisions.get("planner_model", meta.get("planner_model", "")),
        "planner_version": planner_decisions.get("planner_version", meta.get("version", "")),
        "merge_strategy": meta.get("merge_adapter", ""),
        "objects_detected": len(trace.candidate_history or []),
        "scene_pure_negative": meta.get("scene_pure_negative", False),
        "elapsed_ms": meta.get("elapsed_ms", 0),
    }


async def process_batch(
    engine: RuntimeEngine,
    request: AsyncAnnotationRequest,
    run_id: str,
) -> None:
    """Process an entire async batch in the background.

    This function is designed to be spawned via ``asyncio.create_task``.
    It runs to completion regardless of individual frame or callback failures.
    """
    logger.info("Batch %s: starting with %d task(s)", run_id, len(request.tasks))

    # 1. Parse template (the template dict is from the request body)
    try:
        parsed = TemplateParser().parse(deepcopy(request.template))
    except (ValueError, KeyError) as exc:
        logger.error("Batch %s: template parse failed: %s", run_id, exc)
        # All tasks fail if template is unparseable
        for task in request.tasks:
            await post_callback(
                request.callback_url,
                {
                    "run_id": run_id,
                    "task_id": task.task_id,
                    "status": "failed",
                    "error": "TemplateParseError",
                    "detail": str(exc),
                },
                run_id=run_id,
                task_id=task.task_id,
            )
        return

    # 2. Compile plan once (shared for all frames)
    plan = Planner.compile(parsed)

    # 3. Process each task
    temp_dir = make_temp_dir()
    n_tasks = len(request.tasks)
    cancelled_early = False
    try:
        for idx, task in enumerate(request.tasks, start=1):
            # Check cancellation before each task
            if is_cancelled(run_id):
                logger.warning(
                    "Batch %s: cancelled before task %d/%d task_id=%s",
                    run_id, idx, n_tasks, task.task_id,
                )
                cancelled_early = True
                break

            logger.info(
                "Batch %s: processing task %d/%d task_id=%s frames=%d",
                run_id, idx, n_tasks, task.task_id, len(task.frames),
            )
            await _process_task(
                engine=engine,
                task=task,
                parsed=parsed,
                plan=plan,
                run_id=run_id,
                callback_url=request.callback_url,
                temp_dir=temp_dir,
            )
            logger.info(
                "Batch %s: task %d/%d done task_id=%s",
                run_id, idx, n_tasks, task.task_id,
            )
    finally:
        clean_temp_dir(temp_dir)
        unregister(run_id)

    if cancelled_early:
        logger.info("Batch %s: cancelled (%d/%d tasks processed)", run_id, idx - 1, n_tasks)
    else:
        logger.info("Batch %s: completed (%d tasks)", run_id, n_tasks)


async def _process_task(
    *,
    engine: RuntimeEngine,
    task: AsyncTaskItem,
    parsed: ParsedTaskSpec,
    plan: Any,
    run_id: str,
    callback_url: str,
    temp_dir: str,
) -> None:
    """Process a single task: run all frames, then POST callback."""
    frame_results: list[dict[str, Any]] = []
    first_success: tuple[Any, Any] | None = None  # (annotation_result, trace)

    n_frames = len(task.frames)
    for f_idx, frame in enumerate(task.frames, start=1):
        logger.info(
            "Batch %s task=%s frame %d/%d frame_id=%s url=%s",
            run_id, task.task_id, f_idx, n_frames,
            frame.frame_id, frame.url,
        )
        try:
            local_path = resolve_url(frame.url, temp_dir=temp_dir)
        except (FileNotFoundError, ValueError, Exception) as exc:
            logger.warning(
                "Batch %s task=%s frame=%s: URL resolve failed: %s",
                run_id, task.task_id, frame.frame_id, exc,
            )
            frame_results.append({
                "frame_id": frame.frame_id,
                "status": "failed",
                "error": type(exc).__name__,
                "detail": str(exc),
            })
            continue

        try:
            response = await engine.run_with_plan(
                image_path=local_path,
                plan=plan,
                parsed=parsed,
            )

            ann = response.annotation_result
            trace = response.runtime_trace

            n_objects = len(ann.objects)
            elapsed = trace.meta.get("elapsed_ms", 0)
            logger.info(
                "Batch %s task=%s frame=%s: completed objects=%d elapsed=%.0fms",
                run_id, task.task_id, frame.frame_id, n_objects, elapsed,
            )

            # Save full trace locally (aggregated per-run)
            _save_trace(run_id, task.task_id, frame.frame_id, _run_annotation_result_to_dict(trace))

            frame_results.append({
                "frame_id": frame.frame_id,
                "status": "completed",
            })

            # Keep first successful result for the callback payload
            if first_success is None:
                first_success = (ann, trace)

        except FileNotFoundError as exc:
            logger.warning(
                "Batch %s task=%s frame=%s: file not found: %s",
                run_id, task.task_id, frame.frame_id, exc,
            )
            frame_results.append({
                "frame_id": frame.frame_id,
                "status": "failed",
                "error": "FileNotFoundError",
                "detail": str(exc),
            })
        except Exception as exc:
            logger.error(
                "Batch %s task=%s frame=%s: pipeline error: %s",
                run_id, task.task_id, frame.frame_id, exc,
                exc_info=True,
            )
            frame_results.append({
                "frame_id": frame.frame_id,
                "status": "failed",
                "error": type(exc).__name__,
                "detail": str(exc),
            })

    # Build callback payload with flat structure (no frames wrapper)
    any_succeeded = any(f.get("status") == "completed" for f in frame_results)
    if any_succeeded and first_success is not None:
        ann, trace = first_success
        callback_data: dict[str, Any] = {
            "run_id": run_id,
            "task_id": task.task_id,
            "status": "completed",
            "operator_summary": build_operator_summary(trace, ann),
            "runtime_summary": build_runtime_summary(trace),
            "annotation_result": _run_annotation_result_to_dict(ann),
            "trace_ref": f"storage/traces/{run_id}.json",
        }
    else:
        first_error = frame_results[0] if frame_results else {}
        callback_data = {
            "run_id": run_id,
            "task_id": task.task_id,
            "status": "failed",
            "error": first_error.get("error", "UnknownError"),
            "detail": first_error.get("detail", "all frames failed"),
        }

    logger.info(
        "Batch %s task=%s: sending callback (status=%s)",
        run_id, task.task_id, callback_data.get("status"),
    )
    cb_ok = await post_callback(
        callback_url,
        callback_data,
        run_id=run_id,
        task_id=task.task_id,
    )
    if cb_ok:
        logger.info("Batch %s task=%s: callback succeeded", run_id, task.task_id)
    else:
        logger.warning(
            "Batch %s task=%s: callback failed (dead letter written)",
            run_id, task.task_id,
        )

    # Track progress for cancel_run status reporting
    is_failed = callback_data.get("status") != "completed"
    mark_processed(run_id, failed=is_failed)
