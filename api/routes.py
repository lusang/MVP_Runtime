"""
HTTP interface — kept free of pipeline logic (only wiring + validation).
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_container, get_runtime_engine
from di.container import AppContainer
from runtime.async_worker import process_batch
from runtime.batch_registry import cancel as cancel_batch, register as register_batch
from runtime.engine import RuntimeEngine
from data.io import (
    AnnotationRunRequest,
    AnnotationRunResponse,
    AsyncAnnotationRequest,
    AsyncAnnotationResponse,
)

router = APIRouter(prefix="", tags=["annotation"])


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint for datahub / k8s probes."""
    return {
        "status": "ok",
        "service": "mvp-runtime",
    }


@router.post("/run_annotation", response_model=AnnotationRunResponse)
async def run_annotation(
    body: AnnotationRunRequest,
    engine: Annotated[RuntimeEngine, Depends(get_runtime_engine)],
) -> AnnotationRunResponse:
    """
    Run the full annotation pipeline for a single image + template pair.

    Returns a JSON-serializable `annotation_result` shape (`AnnotationRunResponse`).
    """
    try:
        return await engine.run(image_path=body.image_path, template_path=body.template_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/run_annotation_async", status_code=202, response_model=AsyncAnnotationResponse)
async def run_annotation_async(
    body: AsyncAnnotationRequest,
    container: Annotated[AppContainer, Depends(get_container)],
) -> AsyncAnnotationResponse:
    """
    Submit a batch annotation task.

    Accepts a template + list of tasks. Returns immediately with a ``run_id``.
    Results are POSTed back to ``callback_url`` once per task via the configured
    retry strategy (see API_INTEGRATION.md §2.3).
    """
    # Capture incoming request for test fixtures (dev only)
    _save_incoming_fixture(body)

    if not body.tasks:
        raise HTTPException(status_code=400, detail="tasks list must not be empty")
    if not body.callback_url:
        raise HTTPException(status_code=400, detail="callback_url is required")
    if not body.template:
        raise HTTPException(status_code=400, detail="template is required")

    run_id = str(uuid.uuid4())

    task = asyncio.create_task(
        process_batch(
            engine=container.runtime_engine,
            request=body,
            run_id=run_id,
        )
    )
    register_batch(run_id, task, task_count=len(body.tasks))

    return AsyncAnnotationResponse(
        run_id=run_id,
        status="accepted",
        task_count=len(body.tasks),
    )


@router.post("/cancel_run/{run_id}")
async def cancel_run(run_id: str) -> dict:
    """Cancel a running async batch.

    Best-effort cancellation: completed tasks still send callbacks,
    unstarted tasks are skipped. Returns 404 if run_id is unknown.
    """
    result = cancel_batch(run_id)
    if not result["found"]:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found or already completed")

    logger = logging.getLogger("mvp.api")
    logger.info("Cancel run=%s: cancelled=%d processed=%d failed=%d",
                run_id, result["cancelled_tasks"], result["processed_count"], result["failed_count"])

    return {
        "status": "cancelled",
        "run_id": run_id,
        "cancelled_tasks": result["cancelled_tasks"],
    }


def _save_incoming_fixture(body: AsyncAnnotationRequest) -> None:
    """Save incoming request JSON to test/fixtures/ for replay later.

    Only writes when the fixtures directory exists (dev mode).
    Each request gets a unique timestamped file.
    """
    fixtures_dir = Path(__file__).resolve().parent.parent / "test" / "fixtures"
    if not fixtures_dir.is_dir():
        print("[fixture] fixtures dir not found, skipping")
        return
    data = body.model_dump()
    ts = datetime.now().strftime("%H%M%S")
    task_count = len(data.get("tasks", []))
    path = fixtures_dir / f"request_{ts}_{task_count}tasks.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[fixture] saved {task_count} tasks to {path}")
