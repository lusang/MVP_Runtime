"""Tests for Capability Mapping (Stage 2a)."""

from schemas.semantic_features import SemanticFeatures
from runtime.capability_mapping import map_features


def test_scene_quality_full_image_vision_reasoning():
    """background_clutter: scene-level, quality, needs reasoning."""
    features = SemanticFeatures(
        needs_global_context=True,
        requires_reasoning=True,
        candidate_level=False,
        supports_numeric_analysis=False,
        semantic_type="scene_quality",
    )
    caps = map_features(features, attribute_key="background_clutter", scope="quality")
    assert caps.data_flow == "full_image"
    assert caps.per_candidate is False
    assert "vision_reasoning" in caps.required_capabilities
    assert "numeric_analysis" not in caps.required_capabilities


def test_crop_classification_crop_vision_reasoning():
    """object_type: per-candidate, semantic classification."""
    features = SemanticFeatures(
        needs_global_context=False,
        requires_reasoning=True,
        candidate_level=True,
        supports_numeric_analysis=False,
        semantic_type="crop_classification",
    )
    caps = map_features(features, attribute_key="object_type", scope="semantic")
    assert caps.data_flow == "crop"
    assert caps.per_candidate is True
    assert "vision_reasoning" in caps.required_capabilities


def test_crop_negative_has_negative_classification():
    """ambiguous: per-candidate, negative scope → includes negative_classification."""
    features = SemanticFeatures(
        needs_global_context=False,
        requires_reasoning=True,
        candidate_level=True,
        supports_numeric_analysis=False,
        semantic_type="crop_negative",
    )
    caps = map_features(features, attribute_key="ambiguous", scope="negative")
    assert caps.data_flow == "crop"
    assert caps.per_candidate is True
    assert "vision_reasoning" in caps.required_capabilities
    assert "negative_classification" in caps.required_capabilities


def test_numeric_quality_crop_numeric_analysis():
    """blur/occlusion: per-candidate, numeric → numeric_analysis capability."""
    features = SemanticFeatures(
        needs_global_context=False,
        requires_reasoning=False,
        candidate_level=True,
        supports_numeric_analysis=True,
        semantic_type="crop_numeric",
    )
    caps = map_features(features, attribute_key="blur", scope="quality")
    assert caps.data_flow == "crop"
    assert caps.per_candidate is True
    assert "numeric_analysis" in caps.required_capabilities
    assert "vision_reasoning" not in caps.required_capabilities


def test_scene_pure_negative_full_image():
    """pure_negative: scene-level, negative scope."""
    features = SemanticFeatures(
        needs_global_context=True,
        requires_reasoning=True,
        candidate_level=False,
        semantic_type="scene_pure_negative",
    )
    caps = map_features(features, attribute_key="pure_negative", scope="negative")
    assert caps.data_flow == "full_image"
    assert caps.per_candidate is False
    assert "vision_reasoning" in caps.required_capabilities
    assert "negative_classification" in caps.required_capabilities


def test_spatial_relation_forces_full_image():
    """Multi-object spatial relation → full_image regardless of other flags."""
    features = SemanticFeatures(
        needs_global_context=False,
        requires_reasoning=True,
        candidate_level=True,
        requires_spatial_relation=True,
    )
    caps = map_features(features, attribute_key="spatial_attr", scope="semantic")
    assert caps.data_flow == "full_image"
    assert caps.per_candidate is True


def test_conservative_default_no_reasoning():
    """When requires_reasoning=False and no numeric support → default to vision_reasoning."""
    features = SemanticFeatures(
        needs_global_context=False,
        requires_reasoning=False,
        candidate_level=True,
        supports_numeric_analysis=False,
    )
    caps = map_features(features, attribute_key="unknown", scope="semantic")
    assert "vision_reasoning" in caps.required_capabilities
