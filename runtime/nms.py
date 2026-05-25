"""
Non-Maximum Suppression — deterministic overlap removal between detection candidates.

Suppressed candidates are marked (exists=False, analysis_history entry) rather than
deleted, preserving the audit trail.
"""

from __future__ import annotations

from schemas.bbox import BBox, compute_iou
from schemas.candidate_state import CandidateState


def apply_nms(
    candidates: list[CandidateState],
    iou_threshold: float = 0.5,
) -> None:
    """
    Sort candidates by detector_score descending and suppress those that
    overlap excessively with a higher-scoring candidate.

    Suppressed candidates (mutated in place):
      - exists = False
      - analysis_history entry: "rejected — NMS suppressed"
    """
    if not candidates:
        return

    sorted_candidates = sorted(candidates, key=lambda c: c.detector_score, reverse=True)

    for i, c in enumerate(sorted_candidates):
        if not c.exists:
            continue
        for j in range(i + 1, len(sorted_candidates)):
            other = sorted_candidates[j]
            if not other.exists:
                continue
            iou = compute_iou(c.bbox, other.bbox)
            if iou >= iou_threshold:
                other.exists = False
                other.record("nms", f"rejected — NMS suppressed (IoU={iou:.3f} >= {iou_threshold})")
