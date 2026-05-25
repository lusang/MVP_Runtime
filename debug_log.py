"""NDJSON debug logging for agent diagnostics (session ea8e9c)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

_LOG_PATH = Path(__file__).resolve().parent / "debug-ea8e9c.log"
_SESSION_ID = "ea8e9c"


def agent_log(
    *,
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any] | None = None,
    run_id: str = "default",
) -> None:
    # #region agent log
    try:
        payload = {
            "sessionId": _SESSION_ID,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000),
        }
        with _LOG_PATH.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        pass
    # #endregion
