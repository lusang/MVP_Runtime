"""
Gemini-driven final merge — annotation panel + reasoning trace.

Mock mode (MVP_FORCE_GEMINI_MOCK=1): mechanical average + stub trace.
Real mode: full image + structured prompt sent to Gemini for multi-step review.
"""

from __future__ import annotations

import os
from typing import Any

from schemas.template_spec import ParsedTaskSpec


def _force_mock() -> bool:
    return os.environ.get("MVP_FORCE_GEMINI_MOCK", "1").strip() in ("1", "true", "yes")


class GeminiMerger:
    """Final annotation merge: review all pipeline steps, produce panel + trace."""

    async def merge(
        self,
        *,
        image_path: str,
        parsed: ParsedTaskSpec,
        candidates_data: list[dict[str, Any]],
        scene_pure_negative: bool = False,
        run_id: str = "",
        execution_log_text: str = "",
    ) -> dict[str, Any]:
        if _force_mock():
            return _mock_merge(
                image_path=image_path,
                parsed=parsed,
                candidates_data=candidates_data,
                scene_pure_negative=scene_pure_negative,
            )

        # Edge case: 0 candidates → skip real Gemini call (prompt would be confusing)
        if not candidates_data:
            result = _mock_merge(
                image_path=image_path,
                parsed=parsed,
                candidates_data=candidates_data,
                scene_pure_negative=scene_pure_negative,
            )
            result["adapter"] = "GeminiMergerMock"
            return result

        from models.gemini_client import GeminiClient

        client = GeminiClient()
        result = await client.generate_merge(
            image_path=image_path,
            object_name=parsed.object_name,
            description=parsed.description,
            include=parsed.include,
            exclude=parsed.exclude,
            execution_log=execution_log_text,
            run_id=run_id,
        )
        # If real merge returned empty (error), fall back to mechanical
        if not result.get("objects") and not result.get("reasoning_trace"):
            fallback = _mock_merge(
                image_path=image_path,
                parsed=parsed,
                candidates_data=candidates_data,
                scene_pure_negative=scene_pure_negative,
            )
            fallback["adapter"] = "GeminiMergerFallback"
            fallback["gemini_error"] = result.get("error", "")
            return fallback
        return result


def _mock_merge(
    *,
    image_path: str,
    parsed: ParsedTaskSpec,
    candidates_data: list[dict[str, Any]],
    scene_pure_negative: bool = False,
) -> dict[str, Any]:
    objects: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []

    # YOLO step trace
    trace.append({
        "step": "yolo_detection",
        "input": f"Full image: {image_path}",
        "output": f"{len(candidates_data)} candidate(s) detected",
        "reasoning": f"YOLO-World v2 open-vocabulary detection for '{parsed.object_name}'.",
    })

    if scene_pure_negative:
        trace.append({
            "step": "gemini_negative",
            "input": f"Scene pure negative check on {image_path}",
            "output": "Pure Negative confirmed — no target object in scene",
            "reasoning": "Gemini examined full scene and found no package anywhere.",
        })
        return {
            "adapter": "GeminiMergerMock",
            "objects": [],
            "reasoning_trace": trace,
        }

    for c in candidates_data:
        obj_id = c.get("object_id", "?")
        verif = c.get("verification", {})
        attrs = c.get("attributes", {})
        qual = c.get("quality", {})
        neg = c.get("negative_attributes", {})

        det_score = float(c.get("detector_score", 0))
        verif_score = float(verif.get("score", 0))

        # Independent merge confidence — not just mechanical average
        merge_conf = (det_score + verif_score) / 2.0

        # Agreement/conflict signals
        verif_ok = verif.get("ok", False)
        if verif_ok and det_score > 0.5:
            merge_conf += 0.05  # detector + verify agree
        elif verif_ok and det_score <= 0.3:
            merge_conf -= 0.10  # verify rescued low-confidence detection
        elif not verif_ok and det_score > 0.7:
            merge_conf -= 0.10  # verify rejected strong detection → suspicious

        # Attribute confidence signal
        attr_confidences = [
            float(a.get("confidence", 0))
            for a in attrs.values()
            if isinstance(a, dict)
        ]
        if attr_confidences and all(ac >= 0.7 for ac in attr_confidences):
            merge_conf += 0.05

        # Negative flag penalty
        has_negative = any(
            isinstance(nv, dict) and nv.get("value") is True
            for nv in neg.values()
        )
        if has_negative:
            merge_conf -= 0.10

        merge_conf = max(0.0, min(1.0, merge_conf))

        is_positive = verif.get("ok", False)
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
            "attributes": {k: {"value": v.get("value"), "confidence": v.get("confidence", 0)}
                           for k, v in attrs.items()},
            "quality": {k: {"value": v.get("value"), "confidence": v.get("confidence", 0)}
                        for k, v in qual.items()},
            "negative_flags": {k: {"value": v.get("value", False), "confidence": v.get("confidence", 0)}
                               for k, v in neg.items()},
        })

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

    trace.append({
        "step": "merge",
        "input": f"{len(candidates_data)} candidate(s) with all intermediate results",
        "output": f"{sum(1 for o in objects if o['is_positive'])} positive, {sum(1 for o in objects if not o['is_positive'])} negative",
        "reasoning": "Merge confidence computed independently with agreement/conflict signals: detector+verify agreement, attribute confidence quality, and negative flag penalties applied. Merge confidence is the final authority.",
    })

    return {
        "adapter": "GeminiMergerMock",
        "objects": objects,
        "reasoning_trace": trace,
    }
