"""Tests for Resolver (Stage 2b)."""

from schemas.semantic_features import AttributeCapabilities
from runtime.resolver import resolve


def _caps(
    key: str = "test_attr",
    data_flow: str = "crop",
    caps: list[str] | None = None,
    per_candidate: bool = True,
) -> AttributeCapabilities:
    return AttributeCapabilities(
        attribute_key=key,
        data_flow=data_flow,
        required_capabilities=caps or ["vision_reasoning"],
        per_candidate=per_candidate,
    )


def test_vision_reasoning_maps_to_gemini():
    """vision_reasoning only → handler=gemini, model=gemini-2.0-flash."""
    caps = _caps(caps=["vision_reasoning"])
    result = resolve(caps, scope="semantic", attribute_name="object_type")
    assert result.handler == "gemini"
    assert result.model_id == "gemini-2.0-flash"
    assert result.per_candidate is True


def test_vision_reasoning_negative_maps_to_gemini_negative():
    """vision_reasoning + negative → handler=gemini_negative."""
    caps = _caps(caps=["vision_reasoning", "negative_classification"])
    result = resolve(caps, scope="negative", attribute_name="ambiguous")
    assert result.handler == "gemini_negative"
    assert result.model_id == "gemini-2.0-flash"


def test_numeric_analysis_maps_to_opencv():
    """numeric_analysis → handler=opencv_quality, model=rule-engine."""
    caps = _caps(caps=["numeric_analysis"])
    result = resolve(caps, scope="quality", attribute_name="blur")
    assert result.handler == "opencv_quality"
    assert result.model_id == "rule-engine"


def test_high_stakes_uses_pro_model():
    """is_package is high-stakes → gemini-2.5-pro."""
    caps = _caps(caps=["vision_reasoning"])
    result = resolve(caps, scope="semantic", attribute_name="is_package")
    assert result.handler == "gemini"
    assert result.model_id == "gemini-2.5-pro"


def test_non_high_stakes_uses_flash():
    """Normal attributes → gemini-2.0-flash."""
    caps = _caps(caps=["vision_reasoning"])
    result = resolve(caps, scope="semantic", attribute_name="package_form")
    assert result.model_id == "gemini-2.0-flash"


def test_scene_pure_negative_resolution():
    """pure_negative: scene-level, negative scope."""
    caps = _caps(
        key="pure_negative",
        data_flow="full_image",
        caps=["vision_reasoning", "negative_classification"],
        per_candidate=False,
    )
    result = resolve(caps, scope="negative", attribute_name="pure_negative")
    assert result.handler == "gemini_negative"
    assert result.per_candidate is False
    assert result.data_flow == "full_image"


def test_data_flow_and_per_candidate_preserved():
    """Resolver preserves data_flow and per_candidate from input."""
    caps = _caps(key="bg", data_flow="full_image", per_candidate=False)
    result = resolve(caps, scope="quality", attribute_name="background_clutter")
    assert result.data_flow == "full_image"
    assert result.per_candidate is False


def test_prompt_key_generated():
    """prompt_key is derived from attribute_name."""
    caps = _caps(key="custom_attr")
    result = resolve(caps, scope="semantic", attribute_name="custom_attr")
    assert result.prompt_key == "verify_custom_attr"
