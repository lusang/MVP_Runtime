"""
SemanticFeatures — the semantic feature vector output by Stage 1 (Semantic Classifier).

These fields are the PRIMARY DRIVER for all downstream runtime decisions.
semantic_type is for observability only — NEVER use it in runtime conditionals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SemanticFeatures:
    """Feature vector describing an attribute's semantic properties.

    All fields default to the most conservative (LLM-dependent) values,
    so a failed classifier degrades to "needs full context + reasoning".
    """

    # ── Primary driver fields ──────────────────────────────────────
    # Runtime decisions are driven by these, never by semantic_type.

    needs_global_context: bool = True
    """True = needs full-image information (cannot be assessed on a crop)."""

    requires_reasoning: bool = True
    """True = requires semantic understanding, not just numeric computation."""

    candidate_level: bool = True
    """False = scene-level attribute, result applies to all candidates."""

    supports_numeric_analysis: bool = False
    """True = can be analyzed numerically (e.g. Laplacian for blur, histogram)."""

    requires_spatial_relation: bool = False
    """True = needs spatial relationships between multiple objects."""

    requires_temporal_context: bool = False
    """True = needs multi-frame temporal context."""

    # ── Observability only — NEVER use in runtime conditionals ─────

    semantic_type: str = "unknown"
    """Human-readable label for logging, debugging, analytics only."""

    reason: str = ""
    """Explanation of why these features were assigned."""


@dataclass
class AttributeCapabilities:
    """Stage 2a output — what capabilities an attribute needs at runtime.

    This is the INTERMEDIATE representation between feature vector and
    concrete handler/model resolution. It describes WHAT is needed,
    not WHO provides it.
    """

    attribute_key: str
    data_flow: Literal["crop", "full_image"]
    required_capabilities: list[str] = field(default_factory=list)
    per_candidate: bool = True


@dataclass
class AttributeRuntimeParams:
    """Stage 2b output — fully resolved runtime parameters for one attribute.

    This is the CONCRETE execution spec consumed by StepGraphBuilder.
    """

    attribute_key: str
    data_flow: Literal["crop", "full_image"]
    handler: str
    per_candidate: bool
    model_id: str
    required_capabilities: list[str] = field(default_factory=list)
    scope: str = "semantic"
    prompt_key: str = ""
