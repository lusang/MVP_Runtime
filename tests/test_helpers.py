"""Shared test utilities."""

from pathlib import Path

from PIL import Image


def write_minimal_jpeg(path: Path, *, width: int = 320, height: int = 240) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color=(128, 128, 128)).save(path, format="JPEG")
