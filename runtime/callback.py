"""
Callback posting with retry + dead letter queue.

Retry strategy per API_INTEGRATION.md §2.3:
  - 1st attempt: immediate
  - 2nd attempt: 5s delay
  - 3rd attempt: 30s delay
  - After 3 failures: write to dead letter queue
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger("mvp.callback")

DEAD_LETTER_ROOT = Path(__file__).resolve().parent.parent / "storage" / "dead_letter"
CALLBACK_TIMEOUT = 30.0

_RETRY_DELAYS = [0.0, 5.0, 30.0]


def _write_dead_letter(run_id: str, task_id: str, payload: dict[str, Any]) -> None:
    """Write a failed callback payload to the dead letter queue."""
    dir_path = DEAD_LETTER_ROOT / run_id
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / f"{task_id}.json"
    try:
        file_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.warning("Dead letter written: %s", file_path)
    except OSError as exc:
        logger.error("Failed to write dead letter %s: %s", file_path, exc)


async def post_callback(url: str, data: dict[str, Any], *, run_id: str, task_id: str) -> bool:
    """POST *data* to *url* with retries.

    Returns ``True`` if the callback succeeded, ``False`` if all retries
    were exhausted (dead letter written in that case).
    """
    last_error: str = ""

    for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
        if delay > 0:
            await asyncio.sleep(delay)

        try:
            async with httpx.AsyncClient(timeout=CALLBACK_TIMEOUT) as client:
                resp = await client.post(url, json=data)
            if resp.is_success:
                logger.info(
                    "Callback succeeded [%d/3] run=%s task=%s status=%d",
                    attempt, run_id, task_id, resp.status_code,
                )
                return True
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            logger.warning(
                "Callback attempt %d/3 failed run=%s task=%s: %s",
                attempt, run_id, task_id, last_error,
            )

        except httpx.TimeoutException:
            last_error = "timeout"
            logger.warning(
                "Callback attempt %d/3 timeout run=%s task=%s",
                attempt, run_id, task_id,
            )
        except httpx.ConnectError:
            last_error = "connection refused"
            logger.warning(
                "Callback attempt %d/3 connection refused run=%s task=%s",
                attempt, run_id, task_id,
            )
        except httpx.RequestError as exc:
            last_error = str(exc)
            logger.warning(
                "Callback attempt %d/3 error run=%s task=%s: %s",
                attempt, run_id, task_id, exc,
            )

    # All retries exhausted
    logger.error(
        "Callback failed after %d attempts run=%s task=%s: %s",
        len(_RETRY_DELAYS), run_id, task_id, last_error,
    )
    _write_dead_letter(run_id, task_id, data)
    return False
