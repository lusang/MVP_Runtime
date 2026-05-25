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

import logging
from copy import deepcopy
from typing import Any

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


def _run_annotation_result_to_dict(obj: Any) -> dict[str, Any]:
    """Convert a Pydantic model to a plain dict, recursing into nested models."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj


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
    try:
        for task in request.tasks:
            await _process_task(
                engine=engine,
                task=task,
                parsed=parsed,
                plan=plan,
                run_id=run_id,
                callback_url=request.callback_url,
                temp_dir=temp_dir,
            )
    finally:
        clean_temp_dir(temp_dir)

    logger.info("Batch %s: completed", run_id)


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
    task_ok = True

    for frame in task.frames:
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
            task_ok = False
            continue

        try:
            response = await engine.run_with_plan(
                image_path=local_path,
                plan=plan,
                parsed=parsed,
            )

            ann = response.annotation_result
            trace = response.runtime_trace

            frame_results.append({
                "frame_id": frame.frame_id,
                "status": "completed",
                "annotation_result": _run_annotation_result_to_dict(ann),
                "runtime_trace": _run_annotation_result_to_dict(trace),
            })
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
            task_ok = False
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
            task_ok = False

    # Build callback payload
    if task_ok:
        callback_data: dict[str, Any] = {
            "run_id": run_id,
            "task_id": task.task_id,
            "status": "completed",
            "frames": frame_results,
        }
    else:
        # If at least one frame succeeded, still send "completed" with
        # per-frame status. Only send "failed" if all frames failed.
        any_succeeded = any(f.get("status") == "completed" for f in frame_results)
        if any_succeeded:
            callback_data = {
                "run_id": run_id,
                "task_id": task.task_id,
                "status": "completed",
                "frames": frame_results,
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

    await post_callback(
        callback_url,
        callback_data,
        run_id=run_id,
        task_id=task.task_id,
    )
