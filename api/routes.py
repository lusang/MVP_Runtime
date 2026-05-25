"""
HTTP interface — kept free of pipeline logic (only wiring + validation).
"""

import asyncio
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_container, get_runtime_engine
from di.container import AppContainer
from runtime.async_worker import process_batch
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
    if not body.tasks:
        raise HTTPException(status_code=400, detail="tasks list must not be empty")
    if not body.callback_url:
        raise HTTPException(status_code=400, detail="callback_url is required")
    if not body.template:
        raise HTTPException(status_code=400, detail="template is required")

    run_id = str(uuid.uuid4())

    asyncio.create_task(
        process_batch(
            engine=container.runtime_engine,
            request=body,
            run_id=run_id,
        )
    )

    return AsyncAnnotationResponse(
        run_id=run_id,
        status="accepted",
        task_count=len(body.tasks),
    )
