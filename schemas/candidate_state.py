"""
Candidate — per-detection state object with explicit state machine.
Replaces the ad-hoc `exists: bool` with a typed CandidateState enum.

State machine:
  DETECTED → SUPPRESSED  (NMS)
  DETECTED → VERIFIED     (verify ok)
  DETECTED → REJECTED     (verify fail)
  VERIFIED → NEGATIVE     (negative flag triggered)
  VERIFIED → MERGED       (merge consumed)
  REJECTED → NEGATIVE     (negative after verify fail)
  REJECTED → MERGED
  NEGATIVE → MERGED
  SUPPRESSED → (terminal)
  MERGED → (terminal)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from schemas.bbox import BBox


class CandidateState(str, Enum):
    """Explicit state machine for candidate lifecycle.

    This is the single source of truth for what has happened to a candidate.
    Every state transition is recorded in Candidate.history for full audit trail.
    """
    DETECTED = "detected"
    SUPPRESSED = "suppressed"   # NMS — no further processing
    VERIFIED = "verified"       # verify ok → proceed to quality/semantic/negative
    REJECTED = "rejected"       # verify fail → only negative check
    NEGATIVE = "negative"       # negative flag triggered at any stage
    MERGED = "merged"           # consumed by merge step — terminal


# ── Allowed state transitions (immutable mapping) ───────────────────

_ALLOWED_TRANSITIONS: dict[CandidateState, set[CandidateState]] = {
    CandidateState.DETECTED:   {CandidateState.SUPPRESSED, CandidateState.VERIFIED, CandidateState.REJECTED},
    CandidateState.SUPPRESSED: set(),        # terminal
    CandidateState.VERIFIED:   {CandidateState.NEGATIVE, CandidateState.MERGED},
    CandidateState.REJECTED:   {CandidateState.NEGATIVE, CandidateState.MERGED},
    CandidateState.NEGATIVE:   {CandidateState.MERGED},
    CandidateState.MERGED:     set(),        # terminal
}


# ── Terminal states (candidates in these states are excluded from processing) ──

_TERMINAL_STATES = frozenset({CandidateState.SUPPRESSED, CandidateState.MERGED})
_ACTIVE_STATES = frozenset({CandidateState.DETECTED, CandidateState.VERIFIED, CandidateState.REJECTED, CandidateState.NEGATIVE})


# ── Candidate dataclass ─────────────────────────────────────────────


@dataclass
class Candidate:
    """Per-candidate state accumulated across pipeline steps.

    ``state`` is the single source of truth. All previous pipeline steps
    transition it through the state machine. Downstream steps read ``state``
    to decide whether to process this candidate.
    """

    object_id: str
    bbox: BBox
    state: CandidateState = CandidateState.DETECTED
    analysis_path: str = ""
    analysis_bbox: BBox | None = None
    crop_path: str | None = None

    # Detection
    detector_score: float = 0.0

    # Verification
    verify_score: float = 0.0
    verification: dict[str, Any] = field(default_factory=dict)

    # Quality visibility (populated by quality step)
    visibility: dict[str, Any] = field(default_factory=dict)

    # Continuous quality metrics (preserved from OpenCV)
    metrics: dict[str, Any] = field(default_factory=dict)

    # Feasibility (computed after quality step) — None = unknown
    attribute_feasibility: dict[str, bool | None] = field(default_factory=dict)
    missing_attributes: list[str] = field(default_factory=list)

    # Pipeline stage outputs
    attributes: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    negative_flags: dict[str, Any] = field(default_factory=dict)

    # Audit trail — every transition_to() appends an entry
    history: list[dict[str, Any]] = field(default_factory=list)

    # Confidence decomposition — merge is final authority
    detection_confidence: float = 0.0
    verification_confidence: float = 0.0
    merge_confidence: float | None = None
    confidence: float = 0.0

    # ── public API ─────────────────────────────────────────────────

    def transition_to(self, new_state: CandidateState, step: str, reason: str) -> None:
        """Transition candidate to *new_state* and record the move in history.

        Raises ``AssertionError`` if the transition is not allowed.
        """
        allowed = _ALLOWED_TRANSITIONS[self.state]
        assert new_state in allowed, (
            f"Illegal transition: {self.state.value} → {new_state.value} "
            f"(allowed from {self.state.value}: {[s.value for s in allowed]})"
        )
        old_state = self.state
        self.history.append({
            "step": step,
            "from": old_state.value,
            "to": new_state.value,
            "reason": reason,
            "timestamp": time.time(),
        })
        self.state = new_state

    def record(self, step: str, decision: str) -> None:
        """Append a non-transition audit entry (informational only).

        Unlike ``transition_to()`` this does NOT change the candidate's state.
        Use it for logging decisions that are not state transitions
        (e.g. "skipped because X", "quality result recorded").
        """
        self.history.append({"step": step, "decision": decision, "timestamp": time.time()})

    @property
    def is_active(self) -> bool:
        """True if the candidate should be considered for further processing."""
        return self.state in _ACTIVE_STATES

    @property
    def is_suppressed_or_rejected(self) -> bool:
        """True if candidate was NMS-suppressed or verify-rejected."""
        return self.state in (CandidateState.SUPPRESSED, CandidateState.REJECTED)

    # ── backward-compat shim ────────────────────────────────────────
    # Used by eval-based skip conditions in planner.py / step_graph_builder.py.
    # Do NOT use in new code — read ``c.state`` directly instead.

    @property
    def exists(self) -> bool:
        return self.is_active

    # ── confidence ──────────────────────────────────────────────────

    def compute_confidence(self, merge_confidence: float | None = None) -> float:
        """Compute final confidence. merge_confidence overrides mechanical average."""
        self.detection_confidence = self.detector_score
        self.verification_confidence = self.verify_score
        self.merge_confidence = merge_confidence
        if merge_confidence is not None:
            self.confidence = max(0.0, min(1.0, merge_confidence))
        else:
            self.confidence = max(0.0, min(1.0, (self.detector_score + self.verify_score) / 2.0))
        return self.confidence

    # ── serialization ───────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "state": self.state.value,
            "detector_score": self.detector_score,
            "bbox": self.bbox.model_dump(),
            "crop_path": self.crop_path,
            "exists": self.is_active,          # keep for backward compat
            "verify_score": self.verify_score,
            "confidence": self.confidence,
            "detection_confidence": self.detection_confidence,
            "verification_confidence": self.verification_confidence,
            "merge_confidence": self.merge_confidence,
            "visibility": self.visibility,
            "metrics": self.metrics,
            "attribute_feasibility": self.attribute_feasibility,
            "missing_attributes": self.missing_attributes,
            "verification": self.verification,
            "attributes": self.attributes,
            "quality": self.quality,
            "negative_flags": self.negative_flags,
            "history": self.history,
        }
