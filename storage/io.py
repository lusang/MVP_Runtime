"""
Async filesystem helpers — template loading, lightweight validation.

No database: everything is path / JSON oriented for MVP.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiofiles


async def read_json_dict(path: str | Path) -> dict[str, Any]:
    """Read a UTF-8 JSON object into a dictionary."""
    p = Path(path)
    async with aiofiles.open(p, mode="r", encoding="utf-8") as fp:
        raw = await fp.read()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object, got {type(data).__name__}")
    return data


def assert_path_exists(path: str | Path, *, kind: str) -> Path:
    """Ensure `path` exists; raise `FileNotFoundError` with a helpful message."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{kind} not found: {p.resolve()}")
    return p
