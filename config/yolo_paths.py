"""
Resolve YOLO `.pt` paths: prefer `weights/` under project root, then project root.
"""

from __future__ import annotations

import os
from pathlib import Path

_CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _CONFIG_DIR.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"


def resolve_yolo_model_path(raw: str | None = None) -> Path:
    """
    Return an existing weights file, or the path where Ultralytics should place a download.

    Lookup order for relative paths:
      1. `{PROJECT_ROOT}/weights/{name}`
      2. `{PROJECT_ROOT}/{name}`
    """
    value = (raw if raw is not None else os.environ.get("YOLO_MODEL_PATH", "")).strip()
    if not value:
        return WEIGHTS_DIR / "yolov8s-worldv2.pt"

    path = Path(value)
    if path.is_file():
        return path.resolve()

    if path.is_absolute():
        return path

    for base in (WEIGHTS_DIR, PROJECT_ROOT):
        candidate = base / path
        if candidate.is_file():
            return candidate.resolve()

    return (WEIGHTS_DIR / path.name).resolve()


def is_world_weights(path: str | Path) -> bool:
    return "world" in Path(path).stem.lower()
