"""
CapabilityMapping (Stage 2a) — maps semantic feature vector to runtime capabilities.

This code is fully deterministic. Given the same SemanticFeatures + scope,
it always produces the same AttributeCapabilities.

The output describes WHAT the attribute needs (data_flow, capabilities),
not WHO provides it. Handler/model resolution happens in Stage 2b (Resolver).
"""

from __future__ import annotations

from schemas.semantic_features import AttributeCapabilities, SemanticFeatures


def map_features(
    features: SemanticFeatures,
    *,
    attribute_key: str,
    scope: str,
    layer: int = 1,
) -> AttributeCapabilities:
    """Map semantic features to data_flow + required capabilities.

    Args:
        features: Semantic feature vector from Stage 1.
        attribute_key: Attribute identifier for traceability.
        scope: "semantic", "quality", or "negative".
        layer: Template layer (1-4). Layer 4 attributes are forced to full_image.

    Returns:
        AttributeCapabilities with data_flow, required_capabilities, per_candidate.
    """
    caps: list[str] = []

    # ── data_flow ───────────────────────────────────────────────────
    if not features.candidate_level:
        data_flow = "full_image"
    elif features.needs_global_context:
        data_flow = "full_image"
    elif features.requires_spatial_relation:
        data_flow = "full_image"
    else:
        data_flow = "crop"

    # ── per_candidate ───────────────────────────────────────────────
    per_candidate = bool(features.candidate_level)

    # ── required_capabilities ───────────────────────────────────────
    # Describe what the attribute needs, not who provides it.
    # Numeric-analyzeable attributes (e.g. blur/occlusion) get BOTH
    # numeric_analysis AND vision_reasoning — the Resolver picks the
    # appropriate handler by priority (numeric_analysis wins).

    if features.supports_numeric_analysis:
        caps.append("numeric_analysis")
    if features.requires_reasoning:
        caps.append("vision_reasoning")
    if not caps:
        caps.append("vision_reasoning")  # conservative default

    if scope == "negative":
        caps.append("negative_classification")

    # Future: requires_spatial_relation → caps.append("spatial_relation")
    # Future: requires_temporal_context  → caps.append("temporal_analysis")

    return AttributeCapabilities(
        attribute_key=attribute_key,
        data_flow=data_flow,
        required_capabilities=caps,
        per_candidate=per_candidate,
        layer=layer,
    )
