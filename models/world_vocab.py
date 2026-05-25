"""
Build YOLO-World `set_classes` vocabulary from template `ObjectTarget`.
"""

from __future__ import annotations

import os
import re

from models.object_target import ObjectTarget

_DEFAULT_EN = (
    "package",
    "cardboard box",
    "delivery parcel",
    "shipping box",
    "mail bag",
    "plastic bag",
    "envelope",
)


def _parse_include_bullets(include: str) -> list[str]:
    if not include.strip():
        return []
    parts = re.split(r"\s*-\s+", include.replace("\n", " "))
    out: list[str] = []
    for part in parts:
        text = part.strip().strip("-").strip()
        if len(text) >= 2 and text not in out:
            out.append(text)
    return out


def world_classes_from_target(target: ObjectTarget, *, max_classes: int = 12) -> list[str]:
    """
    Open-vocabulary prompts for YOLO-World, derived from template object fields.
    """
    limit = max(1, int(os.environ.get("YOLO_WORLD_MAX_CLASSES", str(max_classes))))
    classes: list[str] = []

    name = target.name.strip()
    if name:
        classes.append(name)
        if name.lower() != "package":
            classes.append("package")

    classes.extend(_parse_include_bullets(target.include))

    extra = os.environ.get("YOLO_WORLD_EXTRA_CLASSES", "").strip()
    if extra:
        for item in extra.split(","):
            text = item.strip()
            if text and text not in classes:
                classes.append(text)

    for phrase in _DEFAULT_EN:
        if len(classes) >= limit:
            break
        if phrase not in classes:
            classes.append(phrase)

    return classes[:limit]
