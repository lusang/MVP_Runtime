"""
OpenCV quality analyzer — blur (Laplacian variance), lighting (histogram mean),
and occlusion (Canny edge density) on image crops.

Falls back to mock when cv2 is unavailable or image read fails.
"""

from __future__ import annotations

from typing import Any

from schemas.bbox import BBox
from schemas.template_spec import ParsedTaskSpec

try:
    import cv2
    import numpy as np

    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[misc]
    np = None  # type: ignore[misc]
    _HAS_CV2 = False


def _read_gray_region(image_path: str, bbox: BBox) -> np.ndarray | None:
    """Return grayscale image region, or full image if bbox covers it entirely."""
    if not _HAS_CV2:
        return None
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape[:2]
    x1 = max(0, int(bbox.x1))
    y1 = max(0, int(bbox.y1))
    x2 = min(w, int(bbox.x2))
    y2 = min(h, int(bbox.y2))
    if x2 <= x1 or y2 <= y1:
        return img  # fallback to full image
    return img[y1:y2, x1:x2]


# --------------- blur (discrete) ---------------

def _analyze_blur(gray: np.ndarray, options: list[str]) -> dict[str, Any]:
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if var > 150:
        value = options[0] if len(options) > 0 else "clear"
        confidence = min(1.0, var / 500.0)
    elif var > 50:
        value = options[1] if len(options) > 1 else "slight"
        confidence = 0.5 + (var - 50) / 200.0
    else:
        value = options[2] if len(options) > 2 else "heavy"
        confidence = max(0.0, var / 100.0)
    return {"value": value, "confidence": round(min(1.0, max(0.0, confidence)), 3), "laplacian_var": round(var, 1)}


# --------------- lighting ---------------

def _analyze_lighting(gray: np.ndarray, options: list[str]) -> dict[str, Any]:
    mean = float(np.mean(gray))
    if mean > 180:
        value = options[2] if len(options) > 2 else "harsh"
        confidence = min(1.0, (mean - 180) / 75.0)
    elif mean < 60:
        value = options[1] if len(options) > 1 else "dim"
        confidence = min(1.0, (60 - mean) / 60.0)
    else:
        value = options[0] if len(options) > 0 else "normal"
        distance = abs(mean - 127) / 67.0
        confidence = max(0.0, 1.0 - distance)
    return {"value": value, "confidence": round(min(1.0, max(0.0, confidence)), 3), "histogram_mean": round(mean, 1)}


# --------------- occlusion ---------------

def _analyze_occlusion(gray: np.ndarray, options: list[str]) -> dict[str, Any]:
    edges = cv2.Canny(gray, 50, 150)
    density = float(np.sum(edges > 0)) / max(1, edges.size)
    if density > 0.08:
        value = options[0] if len(options) > 0 else "none"
        confidence = min(1.0, density / 0.15)
    elif density > 0.03:
        value = options[1] if len(options) > 1 else "partial"
        confidence = 0.3 + (density - 0.03) / 0.1
    else:
        value = options[2] if len(options) > 2 else "heavy"
        confidence = max(0.0, density / 0.06)
    return {"value": value, "confidence": round(min(1.0, max(0.0, confidence)), 3), "edge_density": round(density, 4)}


# --------------- dispatcher ---------------

_ANALYZERS = {
    "blur": _analyze_blur,
    "lighting": _analyze_lighting,
    "occlusion": _analyze_occlusion,
}


# ── Numeric (continuous score) analyzers ─────────────────────────────

_NUMERIC_ANALYZERS: dict[str, Any] = {}


def _analyze_blur_numeric(gray: np.ndarray) -> dict[str, Any]:
    """Return blur as continuous score 0.0 (heavily blurred) → 1.0 (sharp).

    Based on normalized Laplacian variance. Typical ranges:
      < 0.15:  heavy blur
      0.15-0.40: slight blur
      > 0.40:   sharp/clear
    """
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    score = min(1.0, var / 500.0)
    return {"value": round(score, 3), "confidence": 1.0, "laplacian_var": round(var, 1), "score": round(score, 3)}


def _analyze_lighting_numeric(gray: np.ndarray) -> dict[str, Any]:
    """Return exposure as continuous score 0.0 (dark) → 1.0 (bright).

    Based on normalized histogram mean. Typical ranges:
      < 0.25:  dim/underexposed
      0.25-0.75: normal
      > 0.75:  bright/overexposed
    """
    mean = float(np.mean(gray))
    score = mean / 255.0
    return {"value": round(score, 3), "confidence": 1.0, "histogram_mean": round(mean, 1), "score": round(score, 3)}


def _analyze_occlusion_numeric(gray: np.ndarray) -> dict[str, Any]:
    """Return occlusion as continuous score 0.0 (heavily occluded) → 1.0 (none).

    Based on edge density. Low edge density → likely occluded.
    Typical ranges:
      < 0.03: heavy occlusion
      0.03-0.08: partial occlusion
      > 0.08: no occlusion
    """
    edges = cv2.Canny(gray, 50, 150)
    density = float(np.sum(edges > 0)) / max(1, edges.size)
    # Invert: high edge density → low occlusion (high score)
    score = min(1.0, density / 0.15)
    return {"value": round(score, 3), "confidence": 1.0, "edge_density": round(density, 4), "score": round(score, 3)}


for _name, _func in [
    ("blur", _analyze_blur_numeric),
    ("lighting", _analyze_lighting_numeric),
    ("occlusion", _analyze_occlusion_numeric),
]:
    _NUMERIC_ANALYZERS[_name] = _func


def _mock_result(
    attribute_name: str,
    attribute_type: str,
    options: list[Any],
    object_id: str,
    image_path: str,
    parsed: ParsedTaskSpec,
    description: str,
    bbox: BBox,
) -> dict[str, Any]:
    value = options[0] if options else "none"
    return {
        "adapter": "OpenCVAnalyzerMock",
        "channel": "quality",
        "attribute_name": attribute_name,
        "attribute_type": attribute_type,
        "value": value,
        "confidence": 0.8,
        "object_id": object_id,
        "object_name": parsed.object_name,
        "bbox_area": max(0.0, (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)),
        "image_path": image_path,
        "description": description,
    }


class OpenCVAnalyzer:
    """Quality-channel only — semantic attributes are handled by Gemini plugins."""

    async def analyze_quality(
        self,
        *,
        image_path: str,
        bbox: BBox,
        parsed: ParsedTaskSpec,
        object_id: str,
        attribute_name: str,
        attribute_type: str,
        options: list[Any],
        description: str,
    ) -> dict[str, Any]:
        analyzer = _ANALYZERS.get(attribute_name)
        if analyzer is None or not _HAS_CV2:
            return _mock_result(
                attribute_name, attribute_type, options, object_id, image_path, parsed, description, bbox
            )

        gray = _read_gray_region(image_path, bbox)
        if gray is None or gray.size == 0:
            return _mock_result(
                attribute_name, attribute_type, options, object_id, image_path, parsed, description, bbox
            )

        try:
            analysis = analyzer(gray, [str(o) for o in options])
        except Exception:
            return _mock_result(
                attribute_name, attribute_type, options, object_id, image_path, parsed, description, bbox
            )

        return {
            "adapter": "OpenCVAnalyzer",
            "channel": "quality",
            "attribute_name": attribute_name,
            "attribute_type": attribute_type,
            "value": analysis["value"],
            "confidence": analysis["confidence"],
            "object_id": object_id,
            "object_name": parsed.object_name,
            "bbox_area": max(0.0, (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)),
            "image_path": image_path,
            "description": description,
            "metrics": {k: v for k, v in analysis.items() if k not in ("value", "confidence")},
        }

    async def analyze_numeric(
        self,
        *,
        image_path: str,
        bbox: BBox,
        parsed: ParsedTaskSpec,
        object_id: str,
        attribute_name: str,
        description: str,
    ) -> dict[str, Any]:
        """Analyze a numeric attribute — returns continuous score 0.0-1.0.

        Unlike analyze_quality(), this does NOT discretize into options.
        Returns raw normalized measurement for business-defined thresholds.
        """
        analyzer = _NUMERIC_ANALYZERS.get(attribute_name)
        if analyzer is None or not _HAS_CV2:
            return _mock_result(
                attribute_name, "numeric", [], object_id, image_path, parsed, description, bbox
            )

        gray = _read_gray_region(image_path, bbox)
        if gray is None or gray.size == 0:
            return _mock_result(
                attribute_name, "numeric", [], object_id, image_path, parsed, description, bbox
            )

        try:
            analysis = analyzer(gray)
        except Exception:
            return _mock_result(
                attribute_name, "numeric", [], object_id, image_path, parsed, description, bbox
            )

        return {
            "adapter": "OpenCVAnalyzer",
            "channel": "numeric_quality",
            "attribute_name": attribute_name,
            "attribute_type": "numeric",
            "value": analysis["value"],
            "confidence": analysis["confidence"],
            "score": analysis["score"],
            "object_id": object_id,
            "object_name": parsed.object_name,
            "bbox_area": max(0.0, (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)),
            "image_path": image_path,
            "description": description,
            "metrics": {k: v for k, v in analysis.items() if k not in ("value", "confidence", "score")},
        }
