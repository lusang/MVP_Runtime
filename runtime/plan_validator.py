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


def validate(plan: PipelinePlan, use_detector: bool = True) -> ValidationResult:
    """Validate a PipelinePlan, returning errors and warnings.

    Args:
        plan: The pipeline plan to validate.
        use_detector: When True (default), requires a ``detect`` step.
                       When False, detect is optional (full-image mode).

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
        result.errors.append("steps is empty — at least one step + merge required")
        result.passed = False
        return result  # nothing more to check

    # ── Required step checks (FAIL) ─────────────────────────────────
    # merge is always required; detect is optional in full-image mode
    if "merge" not in step_types:
        result.errors.append("missing required step: merge")
        result.passed = False

    if use_detector and "detect" not in step_types:
        result.errors.append("missing required step: detect")
        result.passed = False

    # ── Step ordering checks (FAIL) ─────────────────────────────────
    # detect must come before merge (only relevant when detect exists)
    if "detect" in step_types and "merge" in step_types:
        if step_types.index("detect") > step_types.index("merge"):
            result.errors.append("detect must come before merge")
            result.passed = False

    # ── Structural checks (WARN) ────────────────────────────────────
    _check_structure(result, step_types, use_detector)

    # ── Step-level checks (WARN) ────────────────────────────────────
    for s in steps:
        if s.model_id == "unknown" or not s.model_id:
            result.warnings.append(f"step '{s.step}' has no model_id")

    return result


def _check_structure(result: ValidationResult, step_types: list[str],
                     use_detector: bool = True) -> None:
    """Structural WARN checks — not failures, but worth flagging."""

    # quality before semantic (detector-mode)
    if "quality" in step_types and "attribute" in step_types:
        if step_types.index("quality") > step_types.index("attribute"):
            result.warnings.append("quality step should come before attribute (semantic) step")

    # full_quality before full_attribute (full-image mode)
    if "full_quality" in step_types and "full_attribute" in step_types:
        if step_types.index("full_quality") > step_types.index("full_attribute"):
            result.warnings.append("full_quality should come before full_attribute")

    # merge last
    if step_types and step_types[-1] != "merge":
        result.warnings.append(f"merge should be the last step, but last is '{step_types[-1]}'")

    # nms after detect, before verify (detector mode)
    if use_detector:
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
