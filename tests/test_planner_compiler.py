"""Integration tests for the full Planner compiler pipeline (Stages 1-4)."""

import json
from pathlib import Path

from runtime.planner import compile_plan
from runtime.template_parser import TemplateParser

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "test" / "fixtures" / "request_151049_1tasks.json"


def _load_parsed(fixture_path: Path) -> "ParsedTaskSpec":
    from schemas.template_spec import ParsedTaskSpec
    raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    return TemplateParser().parse(raw["template"])


def _step_names(plan) -> list[str]:
    return [s.step for s in sorted(plan.steps, key=lambda s: s.order)]


# ── Fixture: request_151049_1tasks.json ───────────────────────────────────


def test_fixture_compiler_produces_8_steps():
    """Full fixture template → 8 steps with correct topology."""
    parsed = _load_parsed(FIXTURE)
    plan = compile_plan(parsed)

    names = _step_names(plan)
    assert names == [
        "negative",   # scene-level pure_negative
        "detect",
        "nms",
        "verify",
        "quality",    # background_clutter
        "attribute",  # object_type + is_package
        "negative",   # ambiguous + open_set_negative
        "merge",
    ], f"Got: {names}"


def test_fixture_compiler_early_exit():
    """pure_negative → early_exit_rules present."""
    parsed = _load_parsed(FIXTURE)
    plan = compile_plan(parsed)
    assert len(plan.early_exit_rules) >= 1
    assert any("pure_negative" in r.condition for r in plan.early_exit_rules)


def test_fixture_compiler_skip_conditions():
    """Plan should have skip conditions for empty detections."""
    parsed = _load_parsed(FIXTURE)
    plan = compile_plan(parsed)
    assert len(plan.skip_conditions) >= 1
    assert any("exists" in c.condition for c in plan.skip_conditions)


def test_fixture_compiler_attribute_keys():
    """Semantic step should contain both object_type and is_package."""
    parsed = _load_parsed(FIXTURE)
    plan = compile_plan(parsed)

    attr_step = next(s for s in plan.steps if s.step == "attribute")
    keys = attr_step.params.get("attribute_keys", [])
    assert "object_type" in keys
    assert "is_package" in keys


def test_fixture_compiler_quality_uses_gemini():
    """background_clutter is semantic quality → uses gemini, not opencv."""
    parsed = _load_parsed(FIXTURE)
    plan = compile_plan(parsed)
    quality_step = next(s for s in plan.steps if s.step == "quality")
    assert quality_step.model_id == "gemini-2.0-flash", \
        f"background_clutter needs Gemini, got {quality_step.model_id}"


def test_fixture_compiler_pure_negative_is_scene_level():
    """Scene-level negative check should have per_candidate=false."""
    parsed = _load_parsed(FIXTURE)
    plan = compile_plan(parsed)

    scene_neg = next(
        s for s in plan.steps
        if s.step == "negative" and s.params.get("scene_check")
    )
    assert scene_neg.per_candidate is False
    assert scene_neg.data_flow.value == "full_image"


def test_fixture_compiler_negative_steps_merged():
    """ambiguous + open_set_negative should be in one step."""
    parsed = _load_parsed(FIXTURE)
    plan = compile_plan(parsed)

    neg_steps = [
        s for s in plan.steps
        if s.step == "negative" and not s.params.get("scene_check")
    ]
    assert len(neg_steps) == 1
    keys = neg_steps[0].params.get("attribute_keys", [])
    assert "ambiguous" in keys
    assert "open_set_negative" in keys


# ── Template.json ──────────────────────────────────────────────────────────


def test_template_json_compiler():
    """Template.json should compile successfully."""
    template_path = ROOT / "resource" / "Template.json"
    raw = json.loads(template_path.read_text(encoding="utf-8"))
    parsed = TemplateParser().parse(raw)
    plan = compile_plan(parsed)

    names = _step_names(plan)
    assert "detect" in names
    assert "merge" in names
    assert "nms" in names
    assert "verify" in names
    assert names[-1] == "merge"  # merge should be last
    assert len(plan.steps) >= 4  # at minimum detect, nms, verify, merge
