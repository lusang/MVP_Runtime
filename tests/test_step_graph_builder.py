"""Tests for StepGraph Builder (Stage 3)."""

from schemas.pipeline_plan import DataFlow
from schemas.semantic_features import AttributeRuntimeParams
from schemas.template_spec import ParsedTaskSpec, TemplateAttributeSpec
from runtime.step_graph_builder import StepGraphBuilder


def _make_parsed(
    object_name: str = "Package",
    semantic_attrs: list | None = None,
    quality_attrs: list | None = None,
    negative_attrs: list | None = None,
) -> ParsedTaskSpec:
    return ParsedTaskSpec(
        object_name=object_name,
        semantic_attributes=semantic_attrs or [],
        quality_attributes=quality_attrs or [],
        negative_attributes=negative_attrs or [],
    )


def _make_attr(name: str, scope: str = "semantic") -> TemplateAttributeSpec:
    return TemplateAttributeSpec(
        name=name, key=name, type="boolean", description="", scope=scope, handler="gemini",
    )


def _make_param(
    key: str,
    data_flow: str = "crop",
    handler: str = "gemini",
    per_candidate: bool = True,
    model_id: str = "gemini-2.0-flash",
    scope: str = "semantic",
) -> AttributeRuntimeParams:
    return AttributeRuntimeParams(
        attribute_key=key,
        data_flow=data_flow,
        handler=handler,
        per_candidate=per_candidate,
        model_id=model_id,
        required_capabilities=["vision_reasoning"],
        scope=scope,
        prompt_key=f"verify_{key}",
    )


# ── Fixture template: full plan structure ───────────────────────────────


def test_fixture_template_8_steps():
    """Fixture template should produce 8 steps: scene_neg, detect, nms, verify,
    quality, semantic, negative, merge."""
    parsed = _make_parsed(
        object_name="objects",
        quality_attrs=[_make_attr("background_clutter", scope="quality")],
        semantic_attrs=[_make_attr("object_type"), _make_attr("is_package")],
        negative_attrs=[
            _make_attr("pure_negative", scope="negative"),
            _make_attr("ambiguous", scope="negative"),
            _make_attr("open_set_negative", scope="negative"),
        ],
    )
    params = [
        # pure_negative → scene check (special, not in normal params)
        _make_param("background_clutter", data_flow="full_image",
                     handler="gemini", per_candidate=False, scope="quality"),
        _make_param("object_type", scope="semantic"),
        _make_param("is_package", scope="semantic"),
        _make_param("ambiguous", handler="gemini_negative", scope="negative"),
        _make_param("open_set_negative", handler="gemini_negative", scope="negative"),
    ]

    builder = StepGraphBuilder()
    plan = builder.build(parsed, params)

    step_names = [s.step for s in sorted(plan.steps, key=lambda s: s.order)]
    assert step_names == [
        "negative",   # scene-level pure_negative
        "detect",
        "nms",
        "verify",
        "quality",    # background_clutter
        "attribute",  # object_type + is_package (merged)
        "negative",   # ambiguous + open_set_negative (merged)
        "merge",
    ], f"Got steps: {step_names}"

    assert len(plan.early_exit_rules) == 1
    assert plan.early_exit_rules[0].condition == "scene_pure_negative"
    assert len(plan.skip_conditions) >= 2


def test_fixture_template_merges_semantic_attrs():
    """object_type + is_package should be merged into one attribute step."""
    parsed = _make_parsed(
        object_name="objects",
        semantic_attrs=[_make_attr("object_type"), _make_attr("is_package")],
    )
    params = [
        _make_param("object_type", scope="semantic"),
        _make_param("is_package", scope="semantic"),
    ]

    builder = StepGraphBuilder()
    plan = builder.build(parsed, params)

    attr_steps = [s for s in plan.steps if s.step == "attribute"]
    assert len(attr_steps) == 1
    keys = attr_steps[0].params.get("attribute_keys", [])
    assert "object_type" in keys
    assert "is_package" in keys


def test_fixture_template_merges_negative_attrs():
    """ambiguous + open_set_negative should be merged into one negative step."""
    parsed = _make_parsed(
        object_name="objects",
        negative_attrs=[
            _make_attr("ambiguous", scope="negative"),
            _make_attr("open_set_negative", scope="negative"),
        ],
    )
    params = [
        _make_param("ambiguous", handler="gemini_negative", scope="negative"),
        _make_param("open_set_negative", handler="gemini_negative", scope="negative"),
    ]

    builder = StepGraphBuilder()
    plan = builder.build(parsed, params)

    neg_steps = [s for s in plan.steps if s.step == "negative" and not s.params.get("scene_check")]
    assert len(neg_steps) == 1
    keys = neg_steps[0].params.get("attribute_keys", [])
    assert "ambiguous" in keys
    assert "open_set_negative" in keys


# ── No pure_negative → no scene-level step ──────────────────────────────


def test_no_pure_negative_skips_scene_check():
    parsed = _make_parsed(
        object_name="Package",
        semantic_attrs=[_make_attr("package_form")],
    )
    params = [_make_param("package_form", scope="semantic")]

    builder = StepGraphBuilder()
    plan = builder.build(parsed, params)

    step_names = [s.step for s in sorted(plan.steps, key=lambda s: s.order)]
    assert step_names[0] == "detect"  # no scene_neg before detect
    assert len(plan.early_exit_rules) == 0


# ── Semantic split by data_flow ─────────────────────────────────────────


def test_semantic_split_by_data_flow():
    """Semantic attrs with different data_flow should get separate steps."""
    parsed = _make_parsed(
        object_name="Package",
        semantic_attrs=[_make_attr("crop_attr"), _make_attr("full_attr")],
    )
    params = [
        _make_param("crop_attr", data_flow="crop", scope="semantic"),
        _make_param("full_attr", data_flow="full_image", scope="semantic"),
    ]

    builder = StepGraphBuilder()
    plan = builder.build(parsed, params)

    attr_steps = [s for s in plan.steps if s.step == "attribute"]
    assert len(attr_steps) == 2
    # crop should come before full
    assert attr_steps[0].data_flow == DataFlow.CROP
    assert attr_steps[1].data_flow == DataFlow.FULL


# ── Quality before semantic ordering ────────────────────────────────────


def test_quality_before_semantic():
    """Quality step must appear before semantic step."""
    parsed = _make_parsed(
        object_name="Package",
        quality_attrs=[_make_attr("blur", scope="quality")],
        semantic_attrs=[_make_attr("package_form", scope="semantic")],
    )
    params = [
        _make_param("blur", handler="opencv_quality", model_id="rule-engine", scope="quality"),
        _make_param("package_form", scope="semantic"),
    ]

    builder = StepGraphBuilder()
    plan = builder.build(parsed, params)

    step_names = [s.step for s in sorted(plan.steps, key=lambda s: s.order)]
    quality_idx = step_names.index("quality")
    attr_idx = step_names.index("attribute")
    assert quality_idx < attr_idx, "quality must come before semantic"
