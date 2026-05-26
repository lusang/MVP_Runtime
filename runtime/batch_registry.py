"""
In-memory registry for running async batches with cancellation support.

Tracks asyncio.Tasks spawned by POST /run_annotation_async so they
can be cancelled via POST /cancel_run/{run_id}.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger("mvp.batch_registry")

_registry: dict[str, dict[str, Any]] = {}


def register(
    run_id: str,
    task: asyncio.Task[None],
    task_count: int,
) -> None:
    """Register a running batch."""
    cancel_event = asyncio.Event()
    _registry[run_id] = {
        "task": task,
        "cancel": cancel_event,
        "task_count": task_count,
        "processed_count": 0,
        "failed_count": 0,
    }
    logger.info(
        "Batch %s: registered (%d tasks)", run_id, task_count,
    )


def unregister(run_id: str) -> None:
    """Remove a completed batch from the registry."""
    _registry.pop(run_id, None)
    logger.info("Batch %s: unregistered", run_id)


def is_cancelled(run_id: str) -> bool:
    """Check if cancellation has been requested for this batch."""
    entry = _registry.get(run_id)
    if entry is None:
        return False
    return entry["cancel"].is_set()


def cancel(run_id: str) -> dict[str, Any]:
    """Request cancellation of a running batch.

    Returns a dict with cancellation status suitable for the API response.
    """
    entry = _registry.get(run_id)
    if entry is None:
        return {"found": False}

    entry["cancel"].set()
    # Best-effort: cancel the asyncio Task itself
    task: asyncio.Task | None = entry.get("task")
    if task and not task.done():
        task.cancel()

    remaining = entry["task_count"] - entry["processed_count"]
    logger.info(
        "Batch %s: cancelled (%d processed, %d remaining)",
        run_id, entry["processed_count"], remaining,
    )
    return {
        "found": True,
        "cancelled_tasks": max(0, remaining),
        "processed_count": entry["processed_count"],
        "failed_count": entry["failed_count"],
    }


def mark_processed(run_id: str, failed: bool = False) -> None:
    """Increment processed count for a batch."""
    entry = _registry.get(run_id)
    if entry is None:
        return
    entry["processed_count"] = entry.get("processed_count", 0) + 1
    if failed:
        entry["failed_count"] = entry.get("failed_count", 0) + 1


def get_status(run_id: str) -> dict[str, Any] | None:
    """Get current status of a running batch, or None if not found."""
    entry = _registry.get(run_id)
    if entry is None:
        return None
    return {
        "run_id": run_id,
        "task_count": entry["task_count"],
        "processed_count": entry["processed_count"],
        "failed_count": entry["failed_count"],
        "cancelled": entry["cancel"].is_set(),
    }
