"""Tests for Plan Validator (Stage 4)."""

import uuid

from schemas.pipeline_plan import DataFlow, EarlyExitRule, PipelinePlan, PlanStep, SkipCondition
from runtime.plan_validator import validate


def _make_plan(steps: list[PlanStep] | None = None) -> PipelinePlan:
    return PipelinePlan(
        plan_id=str(uuid.uuid4()),
        object_name="Package",
        steps=steps or [],
    )


def _step(step: str, order: int, model_id: str = "gemini-2.0-flash") -> PlanStep:
    return PlanStep(
        step=step, model_id=model_id,
        data_flow=DataFlow.CROP, order=order, per_candidate=False,
    )


# ── Valid plans ─────────────────────────────────────────────────────────


def test_valid_plan_passes():
    plan = _make_plan(steps=[
        _step("detect", 0),
        _step("verify", 1),
        _step("merge", 2),
    ])
    result = validate(plan)
    assert result.passed is True
    assert len(result.errors) == 0


def test_full_pipeline_passes():
    plan = _make_plan(steps=[
        _step("negative", 0),   # scene check
        _step("detect", 1),
        _step("nms", 2),
        _step("verify", 3),
        _step("quality", 4),
        _step("attribute", 5),
        _step("negative", 6),
        _step("merge", 7),
    ])
    result = validate(plan)
    assert result.passed is True
    assert len(result.errors) == 0


# ── Required field failures ─────────────────────────────────────────────


def test_missing_plan_id_fails():
    plan = _make_plan(steps=[_step("detect", 0), _step("merge", 1)])
    plan.plan_id = ""
    result = validate(plan)
    assert result.passed is False
    assert any("plan_id" in e for e in result.errors)


def test_missing_object_name_fails():
    plan = _make_plan(steps=[_step("detect", 0), _step("merge", 1)])
    plan.object_name = ""
    result = validate(plan)
    assert result.passed is False
    assert any("object_name" in e for e in result.errors)


# ── Required step failures ──────────────────────────────────────────────


def test_empty_steps_fails():
    # Pydantic schema requires min_length=1, so we test with a single step
    # that's missing the required ones
    plan = _make_plan(steps=[_step("verify", 0)])
    result = validate(plan)
    assert result.passed is False
    assert any("missing required step: detect" in e for e in result.errors)
    assert any("missing required step: merge" in e for e in result.errors)


def test_missing_detect_fails():
    plan = _make_plan(steps=[_step("verify", 0), _step("merge", 1)])
    result = validate(plan)
    assert result.passed is False
    assert any("detect" in e for e in result.errors)


def test_missing_merge_fails():
    plan = _make_plan(steps=[_step("detect", 0), _step("verify", 1)])
    result = validate(plan)
    assert result.passed is False
    assert any("merge" in e for e in result.errors)


# ── Ordering failures ───────────────────────────────────────────────────


def test_detect_after_merge_fails():
    plan = _make_plan(steps=[
        _step("merge", 0),
        _step("detect", 1),
    ])
    result = validate(plan)
    assert result.passed is False
    assert any("before merge" in e for e in result.errors)


# ── Structural warnings ─────────────────────────────────────────────────


def test_merge_not_last_warns():
    plan = _make_plan(steps=[
        _step("detect", 0),
        _step("merge", 1),
        _step("verify", 2),  # after merge
    ])
    result = validate(plan)
    assert result.passed is True  # warnings don't fail
    assert any("last step" in w for w in result.warnings)


def test_quality_after_semantic_warns():
    plan = _make_plan(steps=[
        _step("detect", 0),
        _step("attribute", 1),  # semantic before quality
        _step("quality", 2),
        _step("merge", 3),
    ])
    result = validate(plan)
    assert result.passed is True
    assert any("quality step should come before" in w for w in result.warnings)


def test_no_warnings_for_correct_pipeline():
    plan = _make_plan(steps=[
        _step("negative", 0),
        _step("detect", 1),
        _step("nms", 2),
        _step("verify", 3),
        _step("quality", 4),
        _step("attribute", 5),
        _step("negative", 6),
        _step("merge", 7),
    ])
    result = validate(plan)
    assert result.passed is True
    assert len(result.warnings) == 0
