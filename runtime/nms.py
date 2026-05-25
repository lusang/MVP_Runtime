"""
Non-Maximum Suppression — deterministic overlap removal between detection candidates.

Suppressed candidates are marked via ``transition_to(SUPPRESSED)`` rather than
deleted, preserving the audit trail.
"""

from __future__ import annotations

from schemas.bbox import BBox, compute_iou
from schemas.candidate_state import Candidate, CandidateState


def apply_nms(
    candidates: list[Candidate],
    iou_threshold: float = 0.5,
) -> None:
    """
    Sort candidates by detector_score descending and suppress those that
    overlap excessively with a higher-scoring candidate.

    Suppressed candidates (mutated in place):
      - state → CandidateState.SUPPRESSED
      - history entry via transition_to()
    """
    if not candidates:
        return

    sorted_candidates = sorted(candidates, key=lambda c: c.detector_score, reverse=True)

    for i, c in enumerate(sorted_candidates):
        if c.state in (CandidateState.SUPPRESSED, CandidateState.REJECTED):
            continue
        for j in range(i + 1, len(sorted_candidates)):
            other = sorted_candidates[j]
            if other.state in (CandidateState.SUPPRESSED, CandidateState.REJECTED):
                continue
            iou = compute_iou(c.bbox, other.bbox)
            if iou >= iou_threshold:
                other.transition_to(
                    CandidateState.SUPPRESSED,
                    step="nms",
                    reason=f"IoU={iou:.3f} >= {iou_threshold} with {c.object_id}",
                )
