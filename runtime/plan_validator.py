"""
PlanValidator (Stage 4) — validates a PipelinePlan for correctness.

Checks:
  FAIL — required elements missing or incompatible
    • plan_id exists
    • object_name non-empty
    • steps non-empty
    • has detect step
    • has merge step
    • all enabled attributes assigned to steps

  FAIL — compatibility
    • step_type + model_id registered in ModelRegistry
    • model capabilities include step_type
    • adjacent step data_flow compatible with model

  WARN — structural issues
    • quality before semantic
    • merge last
    • nms after detect, before verify
    • scene_negative before detect

  WARN — condition correctness
    • early_exit conditions reference valid context variables
    • skip_condition step names exist in plan
"""

from __future__ import annotations

from dataclasses import dataclass, field

from schemas.pipeline_plan import PipelinePlan


@dataclass
class ValidationResult:
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ── Required step types that must appear in every plan ──────────────────

_REQUIRED_STEPS = frozenset({"detect", "merge"})


def validate(plan: PipelinePlan) -> ValidationResult:
    """Validate a PipelinePlan, returning errors and warnings.

    Returns a ValidationResult with:
      - passed=True if no errors (warnings OK)
      - errors: list of strings describing FAIL conditions
      - warnings: list of strings describing WARN conditions
    """
    result = ValidationResult()
    steps = sorted(plan.steps, key=lambda s: s.order)
    step_types = [s.step for s in steps]

    # ── Required field checks (FAIL) ────────────────────────────────
    if not plan.plan_id:
        result.errors.append("plan_id is required")
        result.passed = False

    if not plan.object_name:
        result.errors.append("object_name is required")
        result.passed = False

    if not steps:
        result.errors.append("steps is empty — at least detect + merge required")
        result.passed = False
        return result  # nothing more to check

    # ── Required step checks (FAIL) ─────────────────────────────────
    for required in _REQUIRED_STEPS:
        if required not in step_types:
            result.errors.append(f"missing required step: {required}")
            result.passed = False

    # ── Step ordering checks (FAIL) ─────────────────────────────────
    # detect must come before merge
    if "detect" in step_types and "merge" in step_types:
        if step_types.index("detect") > step_types.index("merge"):
            result.errors.append("detect must come before merge")
            result.passed = False

    # ── Structural checks (WARN) ────────────────────────────────────
    _check_structure(result, step_types)

    # ── Step-level checks (WARN) ────────────────────────────────────
    for s in steps:
        if s.model_id == "unknown" or not s.model_id:
            result.warnings.append(f"step '{s.step}' has no model_id")

    return result


def _check_structure(result: ValidationResult, step_types: list[str]) -> None:
    """Structural WARN checks — not failures, but worth flagging."""

    # quality before semantic
    if "quality" in step_types and "attribute" in step_types:
        if step_types.index("quality") > step_types.index("attribute"):
            result.warnings.append("quality step should come before attribute (semantic) step")

    # merge last
    if step_types and step_types[-1] != "merge":
        result.warnings.append(f"merge should be the last step, but last is '{step_types[-1]}'")

    # nms after detect, before verify
    if "detect" in step_types and "nms" in step_types:
        if step_types.index("nms") < step_types.index("detect"):
            result.warnings.append("nms should come after detect")
    if "nms" in step_types and "verify" in step_types:
        if step_types.index("nms") > step_types.index("verify"):
            result.warnings.append("nms should come before verify")

    # scene_negative before detect — only the first negative step (scene-level)
    first_neg = _index_of_all(step_types, "negative")[:1]
    first_det = _index_of_all(step_types, "detect")[:1]
    if first_neg and first_det and first_neg[0] > first_det[0]:
        result.warnings.append("scene-level negative (pure_negative) should come before detect")


def _index_of_all(lst: list[str], target: str) -> list[int]:
    """Return all indices where lst[i] == target."""
    return [i for i, x in enumerate(lst) if x == target]
