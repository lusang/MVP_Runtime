"""
Resolver (Stage 2b) — maps required capabilities to concrete handler + model_id.

This is a hardcoded mapping table with no uncertainty.
required_capabilities describes WHAT is needed; Resolver decides WHO provides it.

Future: Replace hardcoded if/elif with a Registry-based lookup so new handlers
can be registered without editing this file.
"""

from __future__ import annotations

from schemas.semantic_features import AttributeCapabilities, AttributeRuntimeParams


# ── Hardcoded mapping: capability → handler ──────────────────────────────

_CAPABILITY_TO_HANDLER: dict[str, str] = {
    "numeric_analysis": "opencv_quality",
    "vision_reasoning": "gemini",
}

# ── High-stakes attributes that warrant a stronger model ─────────────────

_HIGH_STAKES_NAMES: frozenset[str] = frozenset({
    # Attributes where mis-classification is expensive
    "is_package",
})


def resolve(
    caps: AttributeCapabilities,
    *,
    scope: str,
    attribute_name: str = "",
) -> AttributeRuntimeParams:
    """Resolve capabilities to concrete handler + model_id.

    Args:
        caps: Attribute capabilities from Stage 2a.
        scope: "semantic", "quality", or "negative".
        attribute_name: Attribute name (used for high-stakes model selection).

    Returns:
        AttributeRuntimeParams with handler and model_id resolved.
    """
    handler = _resolve_handler(caps.required_capabilities, scope=scope)
    model_id = _resolve_model_id(handler, attribute_name=attribute_name, scope=scope)
    prompt_key = f"verify_{attribute_name}" if attribute_name else ""

    return AttributeRuntimeParams(
        attribute_key=caps.attribute_key,
        data_flow=caps.data_flow,
        handler=handler,
        per_candidate=caps.per_candidate,
        model_id=model_id,
        required_capabilities=caps.required_capabilities,
        scope=scope,
        prompt_key=prompt_key,
        layer=caps.layer,
    )


def _resolve_handler(caps: list[str], *, scope: str) -> str:
    """Map capabilities → handler name.

    Priority order:
      1. "numeric_analysis" → opencv_quality (fast, deterministic)
      2. "vision_reasoning" + "negative_classification" → gemini_negative
      3. "vision_reasoning" → gemini
      4. fallback → gemini
    """
    if "numeric_analysis" in caps:
        return "opencv_quality"
    if "vision_reasoning" in caps and "negative_classification" in caps:
        return "gemini_negative"
    if "vision_reasoning" in caps:
        return "gemini"
    return "gemini"  # conservative fallback


def _resolve_model_id(handler: str, *, attribute_name: str, scope: str) -> str:
    """Map handler + context → model_id.

    - opencv_quality → rule-engine (no LLM needed)
    - High-stakes attributes → gemini-2.5-pro
    - Default → gemini-2.0-flash
    """
    if handler == "opencv_quality":
        return "rule-engine"
    if _is_high_stakes(attribute_name) and scope in ("semantic", "quality"):
        return "gemini-2.5-pro"
    return "gemini-2.0-flash"


def _is_high_stakes(attribute_name: str) -> bool:
    return attribute_name.lower() in _HIGH_STAKES_NAMES
