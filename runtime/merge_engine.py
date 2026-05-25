"""
MergeEngine — deterministic merge, no LLM dependency.

Replaces the Gemini-driven merge path. All decision rules that were previously
embedded in _MERGE_PROMPT are now implemented in code:

  - If verification rejects → object is negative (is_positive=false)
  - If any hard negative flag is triggered → object is negative
  - If Pure Negative scene check was positive → all objects are negative
  - If detector and verification both accept AND no negative flags → is_positive=true
  - merge_confidence = weighted combination of upstream scores + adjustments
  - Attribute conflicts resolved across candidates by highest confidence
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from schemas.template_spec import ParsedTaskSpec


def _default_merge_rules() -> dict[str, Any]:
    return {
        "weights": {"detector": 0.3, "verifier": 0.7},
        "attribute_confidence_threshold": 0.3,
        "nms_iou_threshold": 0.5,
    }


def _load_merge_rules(path: str | Path | None = None) -> dict[str, Any]:
    if path:
        p = Path(path)
    else:
        p = Path(__file__).resolve().parent.parent / "config" / "merge_rules.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return _default_merge_rules()


class MergeEngine:
    """Deterministic merge — no LLM dependency.

    Produces the same output format as the legacy Gemini merge path,
    but purely from structured candidate data. Key differences:

    - Zero latency (no API call)
    - Deterministic output (same input → same result)
    - Builds reasoning_trace from candidate data, not execution log text
    """

    def __init__(self, merge_rules: dict[str, Any] | None = None) -> None:
        self._rules = merge_rules or _load_merge_rules()
        self._weights = self._rules.get("weights", {"detector": 0.3, "verifier": 0.7})
        self._attr_threshold = float(self._rules.get("attribute_confidence_threshold", 0.3))

    def merge(
        self,
        *,
        image_path: str,
        parsed: ParsedTaskSpec,
        candidates_data: list[dict[str, Any]],
        scene_pure_negative: bool = False,
    ) -> dict[str, Any]:
        """Produce the final annotation panel from structured pipeline outputs.

        Args:
            image_path: Path to the source image (included in trace metadata).
            parsed: Parsed task specification.
            candidates_data: List of candidate dicts (from Candidate.to_dict()).
            scene_pure_negative: Whether scene-level pure negative was confirmed.

        Returns:
            Dict with keys: adapter, objects, reasoning_trace, resolved_attributes, merge_rules.
        """
        objects: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []

        # YOLO detection trace entry
        trace.append({
            "step": "yolo_detection",
            "input": f"Full image: {image_path}",
            "output": f"{len(candidates_data)} candidate(s) detected",
            "reasoning": f"YOLO-World v2 open-vocabulary detection for '{parsed.object_name}'.",
        })

        # Pure negative → all objects are negative, short-circuit
        if scene_pure_negative:
            trace.append({
                "step": "gemini_negative",
                "input": f"Scene pure negative check on {image_path}",
                "output": "Pure Negative confirmed — no target object in scene",
                "reasoning": "Gemini examined full scene and found no target anywhere.",
            })
            return {
                "adapter": "MergeEngine",
                "objects": [],
                "reasoning_trace": trace,
                "resolved_attributes": {},
                "merge_rules": self._rules,
            }

        for c in candidates_data:
            obj_id = c.get("object_id", "?")
            verif = c.get("verification", {})
            attrs = c.get("attributes", {})
            qual = c.get("quality", {})
            neg = c.get("negative_attributes", {})
            det_score = float(c.get("detector_score", 0))
            verif_score = float(verif.get("score", 0))

            # --- Confidence normalization ---
            merge_conf = self._weighted_confidence(det_score, verif_score)

            # Agreement / conflict adjustments on top of weighted base
            verif_ok = verif.get("ok", False)
            if verif_ok and det_score > 0.5:
                merge_conf += 0.05  # detector + verify agree
            elif verif_ok and det_score <= 0.3:
                merge_conf -= 0.10  # verify rescued low-confidence detection
            elif not verif_ok and det_score > 0.7:
                merge_conf -= 0.10  # verify rejected strong detection → suspicious

            # Attribute confidence signal
            attr_confs = [
                float(a.get("confidence", 0))
                for a in attrs.values()
                if isinstance(a, dict)
            ]
            if attr_confs and all(ac >= 0.7 for ac in attr_confs):
                merge_conf += 0.05

            # Negative flag penalty
            has_negative = any(
                isinstance(nv, dict) and nv.get("value") is True
                for nv in neg.values()
            )
            if has_negative:
                merge_conf -= 0.10

            merge_conf = max(0.0, min(1.0, merge_conf))

            # --- Classification ---
            is_positive = verif_ok
            neg_category: str | None = None
            for nk, nv in neg.items():
                if isinstance(nv, dict) and nv.get("value") is True:
                    is_positive = False
                    neg_category = str(nv.get("attribute_name", nk))
                    break

            objects.append({
                "object_id": obj_id,
                "is_positive": is_positive,
                "negative_category": neg_category,
                "confidence": round(merge_conf, 3),
                "detection_confidence": round(det_score, 3),
                "verification_confidence": round(verif_score, 3),
                "merge_confidence": round(merge_conf, 3),
                "attributes": {
                    k: {"value": v.get("value"), "confidence": v.get("confidence", 0)}
                    for k, v in attrs.items()
                },
                "quality": {
                    k: {"value": v.get("value"), "confidence": v.get("confidence", 0)}
                    for k, v in qual.items()
                },
                "negative_flags": {
                    k: {"value": v.get("value", False), "confidence": v.get("confidence", 0)}
                    for k, v in neg.items()
                },
            })

            # Per-candidate trace entries
            trace.append({
                "step": "gemini_verification",
                "input": f"Crop image for {obj_id}",
                "output": f"ok={verif.get('ok')} score={verif.get('score')}",
                "reasoning": str(verif.get("rationale", "Verification completed.")),
            })
            trace.append({
                "step": "gemini_semantic",
                "input": f"Crop image for {obj_id}",
                "output": str({k: v.get("value") for k, v in attrs.items()}),
                "reasoning": "Semantic attributes extracted from crop.",
            })
            trace.append({
                "step": "opencv_quality",
                "input": f"Crop image for {obj_id}",
                "output": str({k: v.get("value") for k, v in qual.items()}),
                "reasoning": "Quality metrics computed via OpenCV heuristics.",
            })
            trace.append({
                "step": "gemini_negative",
                "input": f"Full scene + bbox for {obj_id}",
                "output": str({k: v.get("value") for k, v in neg.items()}),
                "reasoning": "Negative-sample attributes checked on full scene.",
            })

        # --- Attribute conflict resolution across positive candidates ---
        resolved_attributes: dict[str, dict[str, Any]] = {}
        for obj in objects:
            if not obj["is_positive"]:
                continue
            for attr_key, attr_val in obj.get("attributes", {}).items():
                if not isinstance(attr_val, dict):
                    continue
                current_conf = attr_val.get("confidence", 0)
                best = resolved_attributes.get(attr_key)
                if best is None or current_conf > best.get("confidence", 0):
                    resolved_attributes[attr_key] = {
                        "value": attr_val.get("value"),
                        "confidence": current_conf,
                        "uncertain": current_conf < self._attr_threshold,
                    }

        trace.append({
            "step": "merge",
            "input": f"{len(candidates_data)} candidate(s) with all intermediate results",
            "output": f"{sum(1 for o in objects if o['is_positive'])} positive, "
                      f"{sum(1 for o in objects if not o['is_positive'])} negative",
            "reasoning": (
                f"Weighted voting (detector={self._weights.get('detector')}, "
                f"verifier={self._weights.get('verifier')}) with agreement/conflict signals. "
                f"Attribute conflicts resolved across {len(resolved_attributes)} attribute(s)."
            ),
        })

        return {
            "adapter": "MergeEngine",
            "objects": objects,
            "reasoning_trace": trace,
            "resolved_attributes": resolved_attributes,
            "merge_rules": self._rules,
        }

    def _weighted_confidence(self, det_score: float, verif_score: float) -> float:
        w_det = self._weights.get("detector", 0.3)
        w_ver = self._weights.get("verifier", 0.7)
        return det_score * w_det + verif_score * w_ver
