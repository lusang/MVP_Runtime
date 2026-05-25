"""
Build canonical `ObjectState` and clean `AnnotationObject` from `CandidateState`.
"""

from __future__ import annotations

from typing import Any

from schemas.api import AnnotationObject
from schemas.candidate_state import CandidateState
from schemas.object_state import ObjectState
from schemas.template_spec import ParsedTaskSpec


class ObjectStateBuilder:
    @staticmethod
    def build(
        *,
        candidate: CandidateState,
        parsed: ParsedTaskSpec,
        scene_pure_negative: bool = False,
        merge_panel: dict[str, Any] | None = None,
    ) -> ObjectState:
        confidence = candidate.confidence
        detection_confidence = candidate.detection_confidence or candidate.detector_score
        verification_confidence = candidate.verification_confidence or candidate.verify_score
        merge_confidence_val: float | None = candidate.merge_confidence

        # Negative: scene-level, verification rejection, or ANY negative attr = True
        negative = scene_pure_negative
        negative_category: str | None = None

        if candidate.verification.get("ok") is False:
            negative = True

        for key, item in candidate.negative_flags.items():
            if isinstance(item, dict) and item.get("value") is True:
                negative = True
                negative_category = str(item.get("attribute_name", key))
                break

        # Merge panel is FINAL authority — overrides pipeline negative/confidence
        is_positive: bool | None = None
        annotation_panel: dict[str, Any] | None = None
        if merge_panel:
            is_positive = merge_panel.get("is_positive")
            annotation_panel = merge_panel
            if is_positive is True:
                negative = False
                negative_category = None
            elif is_positive is False:
                negative = True
                negative_category = merge_panel.get("negative_category") or negative_category
            panel_merge_conf = merge_panel.get("merge_confidence")
            if panel_merge_conf is not None:
                merge_confidence_val = float(panel_merge_conf)
                confidence = merge_confidence_val
            elif merge_panel.get("confidence") is not None:
                confidence = float(merge_panel["confidence"])
            if "detection_confidence" in merge_panel:
                detection_confidence = float(merge_panel["detection_confidence"])
            if "verification_confidence" in merge_panel:
                verification_confidence = float(merge_panel["verification_confidence"])

        return ObjectState(
            object_id=candidate.object_id,
            object_name=parsed.object_name,
            bbox=candidate.bbox,
            preselection_crop_path=candidate.crop_path,
            verification=candidate.verification,
            attributes=candidate.attributes,
            quality=candidate.quality,
            negative_attributes=candidate.negative_flags,
            negative=negative,
            confidence=confidence,
            is_positive=is_positive,
            negative_category=negative_category,
            annotation_panel=annotation_panel,
            visibility=candidate.visibility,
            metrics=candidate.metrics,
            attribute_feasibility=candidate.attribute_feasibility,
            missing_attributes=candidate.missing_attributes,
            analysis_history=candidate.analysis_history,
            detection_confidence=detection_confidence,
            verification_confidence=verification_confidence,
            merge_confidence=merge_confidence_val,
        )

    @staticmethod
    def build_annotation_object(
        *,
        candidate: CandidateState,
        object_name: str,
        merge_panel: dict[str, Any] | None = None,
        scene_pure_negative: bool = False,
    ) -> AnnotationObject:
        """Build clean AnnotationObject for annotation platform consumption.

        merge_panel is the SINGLE canonical source of truth for attributes,
        confidence, and status. candidate state is only used as fallback when
        merge_panel is absent (should not happen in normal flow).
        """

        # --- Status: merge_panel is canonical ---
        status: str = "pending"
        if merge_panel:
            is_positive = merge_panel.get("is_positive")
            if is_positive is True:
                status = "accepted"
            elif is_positive is False:
                status = "rejected"

        # --- Confidence: merge_panel is canonical ---
        confidence = candidate.confidence
        if merge_panel:
            mc = merge_panel.get("merge_confidence") or merge_panel.get("confidence")
            if mc is not None:
                confidence = float(mc)

        # --- Attributes: merge_panel is canonical source ---
        clean_attrs: dict[str, Any] = {}
        if merge_panel:
            # Canonical path: extract attributes from merge panel
            panel_attrs = merge_panel.get("attributes", {})
            for key, val in panel_attrs.items():
                if not isinstance(val, dict):
                    clean_attrs[key] = val
                    continue
                v = val.get("value")
                if v is not None and val.get("infeasible") is not True:
                    clean_attrs[key] = v
        else:
            # Fallback path (should not normally be reached)
            for key, val in candidate.attributes.items():
                if not isinstance(val, dict):
                    continue
                if val.get("infeasible") is True:
                    continue
                v = val.get("value")
                if v is None:
                    continue
                clean_attrs[key] = v

        return AnnotationObject(
            bbox=[candidate.bbox.x1, candidate.bbox.y1, candidate.bbox.x2, candidate.bbox.y2],
            category=object_name,
            attributes=clean_attrs,
            confidence=round(confidence, 4),
            status=status,
        )


def _has_negative_flag(candidate: CandidateState, flag_name: str) -> bool:
    for key, item in candidate.negative_flags.items():
        if not isinstance(item, dict):
            continue
        if item.get("value") is not True:
            continue
        if item.get("attribute_name", key) == flag_name or key == flag_name:
            return True
    return False
