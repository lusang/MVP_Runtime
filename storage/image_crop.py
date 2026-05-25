"""
Crop detection regions from source images for pre-selection (预选) pipelines.
"""

from __future__ import annotations

from pathlib import Path

from schemas.bbox import BBox

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[misc, assignment]


def crop_bbox_to_file(
    *,
    image_path: str | Path,
    bbox: BBox,
    output_path: str | Path,
    padding_ratio: float = 0.05,
) -> Path:
    """
    Crop `bbox` from `image_path` and write JPEG to `output_path`.

    Applies small relative padding; clamps to image bounds.
    """
    if Image is None:
        raise RuntimeError("Pillow is required for image cropping. Run: pip install Pillow")

    src = Path(image_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        bw = bbox.x2 - bbox.x1
        bh = bbox.y2 - bbox.y1
        pad_x = bw * padding_ratio
        pad_y = bh * padding_ratio
        x1 = max(0, int(bbox.x1 - pad_x))
        y1 = max(0, int(bbox.y1 - pad_y))
        x2 = min(w, int(bbox.x2 + pad_x))
        y2 = min(h, int(bbox.y2 + pad_y))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"invalid crop after clamp: ({x1},{y1})-({x2},{y2}) for image {w}x{h}")
        crop = im.crop((x1, y1, x2, y2))
        crop.save(out, format="JPEG", quality=95)
    return out.resolve()


def bbox_for_full_crop(crop_path: str | Path) -> BBox:
    """BBox covering the entire pre-selection crop (for downstream analyzers)."""
    if Image is None:
        return BBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0)
    with Image.open(crop_path) as im:
        w, h = im.size
    return BBox(x1=0.0, y1=0.0, x2=float(w), y2=float(h))
