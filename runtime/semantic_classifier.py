"""
SemanticClassifier (Stage 1) — understands natural-language attribute descriptions
and extracts a semantic feature vector.

Keyword-rule fallback: does NOT require any LLM. Pure deterministic rules
based on attribute name, type, description, and scope.

When LLM-based classification is added later, it lives here too.
"""

from __future__ import annotations

from schemas.semantic_features import SemanticFeatures
from schemas.template_spec import TemplateAttributeSpec


# ── Known attribute names that indicate numeric-analyzable properties ──
_NUMERIC_ANALYSIS_NAMES = frozenset({
    "blur", "occlusion", "lighting", "brightness",
    "contrast", "sharpness", "noise", "exposure",
})

# ── Description keywords that signal pixel-level numeric analysis ──
_NUMERIC_ANALYSIS_DESC_KEYWORDS = {
    "模糊", "清晰", "边缘", "像素", "遮挡", "曝光",
    "亮度", "对比度", "噪声", "锐度", "光照", "暗",
}

# ── Keywords that indicate full-scene context is needed ──
_GLOBAL_CONTEXT_KEYWORDS = {
    "背景", "环境", "场景", "全局", "全景",
    "context", "scene", "surrounding", "全景",
    "background", "environment", "scenario",
}

# ── Keywords that indicate scene-level (not per-candidate) ──
_SCENE_LEVEL_NAME_PREFIXES = {
    "pure", "scene", "background",
}
_SCENE_LEVEL_DESC_KEYWORDS = {
    "场景中", "场景", "场景级",
    "scene-level", "scene level", "global",
}

# ── Keywords that indicate complex reasoning needed ──
_REASONING_DESC_KEYWORDS = {
    "语义", "判断", "理解", "推理",
    "semantic", "reasoning", "judgment", "understand",
    "视觉", "看起来", "像", "类别",
}


def classify_by_keywords(attr: TemplateAttributeSpec) -> SemanticFeatures:
    """Keyword-rule classifier — no LLM required.

    Falls back to conservative defaults (needs_global_context=True,
    requires_reasoning=True) when no rules match.
    """
    name = attr.name.lower()
    desc = attr.description.lower()
    scope = attr.scope
    attr_type = attr.type

    features = SemanticFeatures(
        semantic_type=_derive_semantic_type(attr),
        reason="",
    )

    # ── needs_global_context ───────────────────────────────────────
    if _has_any_keyword(desc, _GLOBAL_CONTEXT_KEYWORDS):
        features.needs_global_context = True
    elif scope == "quality" and attr_type == "boolean" and _has_any_keyword(desc, {"背景"}):
        # background_clutter-like: scene-level quality boolean → full image
        features.needs_global_context = True
    else:
        features.needs_global_context = False

    # ── requires_reasoning ─────────────────────────────────────────
    # NOTE: NEVER use semantic_type here — only feature vector fields.
    if _has_any_keyword(desc, _REASONING_DESC_KEYWORDS):
        features.requires_reasoning = True
    elif attr_type in ("multi_select", "single_select") and len(attr.options) > 0:
        # Any multi-value classification needs reasoning
        features.requires_reasoning = True
    elif attr_type == "boolean" and not features.supports_numeric_analysis:
        # Boolean with no numeric fallback → requires semantic judgment
        features.requires_reasoning = True
    else:
        features.requires_reasoning = False  # numeric, no complex judgment

    # ── candidate_level ────────────────────────────────────────────
    if name in _SCENE_LEVEL_NAME_PREFIXES or any(
        name.startswith(p) for p in _SCENE_LEVEL_NAME_PREFIXES
    ):
        features.candidate_level = False
    elif _has_any_keyword(desc, _SCENE_LEVEL_DESC_KEYWORDS):
        features.candidate_level = False
    else:
        features.candidate_level = True

    # ── supports_numeric_analysis ──────────────────────────────────
    # Matches by name (hardcoded known set) OR by description keywords
    # (template-agnostic: any quality attribute describing pixel-level
    # properties like blur/occlusion/exposure will be correctly assigned).
    if name in _NUMERIC_ANALYSIS_NAMES or (
        scope == "quality"
        and _has_any_keyword(desc, _NUMERIC_ANALYSIS_DESC_KEYWORDS)
    ):
        features.supports_numeric_analysis = True
    else:
        features.supports_numeric_analysis = False

    # ── spatial / temporal — keyword only for now ──────────────────
    features.requires_spatial_relation = _has_any_keyword(desc, {
        "空间", "相对", "之间", "互相",
        "spatial", "relative", "between", "adjacent",
        "multi-object", "多目标", "多物体",
    })
    features.requires_temporal_context = _has_any_keyword(desc, {
        "时序", "帧", "运动", "变化", "多帧",
        "temporal", "frame", "motion", "video",
        "consecutive", "时间", "sequence",
    })

    # ── reason ─────────────────────────────────────────────────────
    parts = []
    if features.needs_global_context:
        parts.append("needs full-image context")
    if features.requires_reasoning:
        parts.append("requires semantic reasoning")
    if not features.candidate_level:
        parts.append("scene-level attribute")
    if features.supports_numeric_analysis:
        parts.append("supports numeric analysis")
    if features.requires_spatial_relation:
        parts.append("requires spatial relation")
    if features.requires_temporal_context:
        parts.append("requires temporal context")
    features.reason = "; ".join(parts) if parts else "default (crop-level, numeric)"

    return features


# ── Helpers ────────────────────────────────────────────────────────────


def _derive_semantic_type(attr: TemplateAttributeSpec) -> str:
    """Derive a human-readable semantic type label.

    NOT used for runtime decisions — only for logging/debugging/analytics.
    """
    name = attr.name.lower()
    scope = attr.scope

    # Scene-level checks — note: name-based checks before scope-based
    if name in ("pure_negative", "pure negative") or name.startswith("pure"):
        return "scene_pure_negative"
    if name in _NUMERIC_ANALYSIS_NAMES:
        return "crop_numeric"
    if scope == "negative":
        return "crop_negative"
    if scope == "quality" and _has_any_keyword(attr.description.lower(), _GLOBAL_CONTEXT_KEYWORDS):
        return "scene_quality"
    if scope == "quality":
        return "crop_quality"
    if scope == "semantic":
        return "crop_classification"
    return "unknown"


def _has_any_keyword(text: str, keywords: set[str]) -> bool:
    return any(kw in text for kw in keywords)
