"""Tests for keyword-rule Semantic Classifier (Stage 1)."""

from schemas.template_spec import TemplateAttributeSpec
from runtime.semantic_classifier import classify_by_keywords


def _make_attr(
    name: str,
    type: str = "boolean",
    description: str = "",
    scope: str = "semantic",
    options: list | None = None,
) -> TemplateAttributeSpec:
    return TemplateAttributeSpec(
        name=name,
        key=name,
        type=type,
        description=description,
        scope=scope,
        options=options or [],
        handler="gemini",
    )


# ── Scene-level (candidate_level = false) ───────────────────────────────


def test_pure_negative_is_scene_level():
    attr = _make_attr("pure_negative", scope="negative")
    f = classify_by_keywords(attr)
    assert f.candidate_level is False, "pure_negative should be scene-level"
    assert f.semantic_type == "scene_pure_negative"


def test_background_quality_is_scene_level():
    attr = _make_attr(
        "background_clutter", scope="quality",
        description="背景是否存在大量杂波干扰",
    )
    f = classify_by_keywords(attr)
    assert f.candidate_level is False
    assert f.needs_global_context is True
    assert f.requires_reasoning is True
    assert f.semantic_type == "scene_quality"


# ── Per-candidate attributes ───────────────────────────────────────────


def test_object_type_is_candidate_level():
    attr = _make_attr(
        "object_type", type="multi_select",
        options=["瓦楞纸箱", "外卖", "快递软包"],
        description="这个物体客观来说是什么？",
    )
    f = classify_by_keywords(attr)
    assert f.candidate_level is True
    assert f.requires_reasoning is True
    assert f.needs_global_context is False
    assert f.semantic_type == "crop_classification"


def test_is_package_requires_reasoning():
    attr = _make_attr(
        "is_package", type="boolean",
        description="这是不是真正的包裹/快递纸箱？",
    )
    f = classify_by_keywords(attr)
    assert f.requires_reasoning is True
    assert f.candidate_level is True
    assert f.semantic_type == "crop_classification"


# ── Negative attributes ────────────────────────────────────────────────


def test_ambiguous_is_crop_negative():
    attr = _make_attr(
        "ambiguous", scope="negative",
        description="例如：远处一个模糊纸箱、被遮挡的塑料袋",
    )
    f = classify_by_keywords(attr)
    assert f.candidate_level is True
    assert f.requires_reasoning is True
    assert f.semantic_type == "crop_negative"


def test_open_set_negative_is_crop_negative():
    attr = _make_attr(
        "open_set_negative", scope="negative",
        description="完全未知的新物体",
    )
    f = classify_by_keywords(attr)
    assert f.candidate_level is True
    assert f.semantic_type == "crop_negative"


# ── Numeric-quality attributes ─────────────────────────────────────────


def test_blur_supports_numeric_analysis():
    attr = _make_attr("blur", type="single_select",
                      options=["clear", "slight", "heavy"],
                      description="sharp edges or blurry",
                      scope="quality")
    f = classify_by_keywords(attr)
    assert f.supports_numeric_analysis is True
    assert f.semantic_type == "crop_numeric"
    assert f.candidate_level is True


def test_occlusion_supports_numeric_analysis():
    attr = _make_attr("occlusion", scope="quality",
                      description="none/partial/heavy occlusion")
    f = classify_by_keywords(attr)
    assert f.supports_numeric_analysis is True
    assert f.semantic_type == "crop_numeric"


# ── Template.json fixture (resource/Template.json) ─────────────────────


def test_fixture_package_form():
    attr = _make_attr(
        "package_form", type="multi_select",
        options=["box", "envelope", "bag", "irregular", "soft_package"],
        description="regular rectangular box / flat envelope / flexible bag",
    )
    f = classify_by_keywords(attr)
    assert f.candidate_level is True
    assert f.requires_reasoning is True
    assert f.needs_global_context is False


def test_fixture_brand_logo():
    attr = _make_attr(
        "brand_logo", type="single_select",
        options=["amazon", "fedex", "ups", "dhl"],
        description="visible logistics brand logo on package surface",
    )
    f = classify_by_keywords(attr)
    assert f.candidate_level is True
    assert f.requires_reasoning is True
    assert f.needs_global_context is False


def test_fixture_hard_negative():
    attr = _make_attr(
        "Hard Negative", scope="negative",
        description="Visually similar to a package but actually is not",
    )
    f = classify_by_keywords(attr)
    assert f.candidate_level is True
    assert f.requires_reasoning is True
    assert f.semantic_type == "crop_negative"


# ── Edge cases ─────────────────────────────────────────────────────────


def test_empty_description_defaults():
    attr = _make_attr("custom_attr", type="boolean", description="")
    f = classify_by_keywords(attr)
    # Default: conservative — needs reasoning for booleans
    assert f.requires_reasoning is True
    assert f.needs_global_context is False


def test_spatial_relation_keyword():
    attr = _make_attr("multi_object", description="物体之间的空间关系")
    f = classify_by_keywords(attr)
    assert f.requires_spatial_relation is True


def test_temporal_context_keyword():
    attr = _make_attr("action", description="多帧时序动作识别")
    f = classify_by_keywords(attr)
    assert f.requires_temporal_context is True
