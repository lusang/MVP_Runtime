"""
StepGraphBuilder (Stage 3) — builds a complete PipelinePlan from per-attribute
runtime parameters.

Responsibilities:
  1. Lay down the fixed topology skeleton (Runtime Invariant)
  2. Group attributes by (scope, data_flow, handler) → merged steps
  3. Assign grouped attributes to topology slots
  4. Set early-exit rules (pure_negative)
  5. Set skip conditions (empty candidates, scene-level gate)
  6. Output a validated PipelinePlan

Runtime Invariant (hardcoded, never changes):
  scene_neg → detect → nms → verify → quality → semantic → negative → merge
"""

from __future__ import annotations

import uuid
from collections import defaultdict

from schemas.pipeline_plan import DataFlow, EarlyExitRule, PipelinePlan, PlanStep, SkipCondition
from schemas.semantic_features import AttributeRuntimeParams
from schemas.template_spec import ParsedTaskSpec


class StepGraphBuilder:
    """Compiles attribute runtime params into an executable PipelinePlan."""

    def build(
        self,
        parsed: ParsedTaskSpec,
        attribute_params: list[AttributeRuntimeParams],
    ) -> PipelinePlan:
        """Build a PipelinePlan from parsed template and resolved attribute params.

        Args:
            parsed: The parsed template spec (used for object_name, pure_negative flag).
            attribute_params: Resolved runtime params for each attribute.

        Returns:
            A complete PipelinePlan ready for execution.
        """
        steps: list[PlanStep] = []
        order = 0

        # Detect pure_negative presence (triggers scene-level step + early_exit)
        # Handles both "Pure Negative" (Template.json) and "pure_negative" (fixture)
        has_pure_negative = any(
            a.enabled and a.name.lower().replace(" ", "_") == "pure_negative"
            for a in parsed.negative_attributes
        )

        # ── 1. Scene-level negative pre-check ───────────────────────
        if has_pure_negative:
            steps.append(self._make_step(
                step="negative",
                model_id="gemini-2.0-flash",
                data_flow=DataFlow.FULL,
                order=order,
                per_candidate=False,
                scope="negative",
                params={"scene_check": True, "attribute_key": "Pure Negative"},
            ))
            order += 1

        # ── 2. Detection ────────────────────────────────────────────
        steps.append(self._make_step(
            step="detect",
            model_id="yolo-world-v2-x",
            data_flow=DataFlow.FULL,
            order=order,
            per_candidate=False,
        ))
        order += 1

        # ── 3. NMS ──────────────────────────────────────────────────
        steps.append(self._make_step(
            step="nms",
            model_id="rule-engine",
            data_flow=DataFlow.FULL,
            order=order,
            per_candidate=False,
            params={"iou_threshold": 0.5},
        ))
        order += 1

        # ── 4. Verification ─────────────────────────────────────────
        steps.append(self._make_step(
            step="verify",
            model_id="gemini-2.0-flash",
            data_flow=DataFlow.CROP,
            order=order,
            per_candidate=True,
        ))
        order += 1

        # ── 5-7. Attribute steps (grouped by scope → data_flow → handler) ──

        # Filter out pure_negative from normal attribute processing —
        # the scene-level step was already added above.
        filtered_params = [
            p for p in attribute_params
            if not (p.scope == "negative"
                    and p.attribute_key.lower().replace(" ", "_") == "pure_negative")
        ]

        # Group params by (scope, data_flow, handler)
        grouped = self._group_attributes(filtered_params)

        for scope in ("quality", "semantic", "negative"):
            if scope not in grouped:
                continue
            for group_key, params_list in grouped[scope].items():
                data_flow_str, handler = group_key
                data_flow = DataFlow(data_flow_str)
                attribute_keys = [p.attribute_key for p in params_list]

                # quality uses step_type="quality"; negative uses "negative"
                if scope == "quality":
                    step_type = "quality"
                elif scope == "negative":
                    step_type = "negative"
                else:
                    step_type = "attribute"

                # Determine model_id (use the highest-stakes one if mixed)
                model_id = self._pick_model_id(params_list)

                steps.append(self._make_step(
                    step=step_type,
                    model_id=model_id,
                    data_flow=data_flow,
                    order=order,
                    per_candidate=params_list[0].per_candidate,
                    scope=scope,
                    params={"attribute_keys": attribute_keys},
                ))
                order += 1

        # ── 8. Merge ────────────────────────────────────────────────
        steps.append(self._make_step(
            step="merge",
            model_id="gemini-2.0-flash",
            data_flow=DataFlow.FULL,
            order=order,
            per_candidate=False,
        ))

        # ── Early exit rules ────────────────────────────────────────
        early_exit_rules: list[EarlyExitRule] = []
        if has_pure_negative:
            early_exit_rules.append(EarlyExitRule(
                condition="scene_pure_negative",
                reason="scene confirmed pure negative — no target object present",
            ))

        # ── Skip conditions ─────────────────────────────────────────
        skip_conditions: list[SkipCondition] = []

        # Skip per-candidate negative steps when scene-level pure_negative confirmed
        if has_pure_negative:
            skip_conditions.append(SkipCondition(
                step="negative",
                condition="scene_pure_negative",
                reason="Pure Negative already confirmed at scene level",
            ))

        # Skip attribute/semantic steps when all candidates rejected
        skip_conditions.append(SkipCondition(
            step="attribute",
            condition="all(not c.exists for c in candidates)",
            reason="all candidates rejected by verification — no targets to assess",
        ))

        return PipelinePlan(
            plan_id=str(uuid.uuid4()),
            object_name=parsed.object_name,
            steps=steps,
            early_exit_rules=early_exit_rules,
            skip_conditions=skip_conditions,
            planner_model="StepGraphBuilder",
            planner_version="2.0",
        )

    # ── Internal helpers ────────────────────────────────────────────────

    def _group_attributes(
        self,
        params: list[AttributeRuntimeParams],
    ) -> dict[str, dict[tuple[str, str], list[AttributeRuntimeParams]]]:
        """Group attribute params by (scope) → (data_flow, handler).

        Returns nested dict: {scope: {(data_flow, handler): [params, ...]}}
        """
        grouped: dict[str, dict[tuple[str, str], list[AttributeRuntimeParams]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        for p in params:
            key = (p.data_flow, p.handler)
            grouped[p.scope][key].append(p)
        return dict(grouped)

    def _pick_model_id(self, params_list: list[AttributeRuntimeParams]) -> str:
        """Pick the best model_id from a group of attributes.

        Uses the most capable model in the group (prefer pro over flash).
        """
        model_ids = {p.model_id for p in params_list}
        if "gemini-2.5-pro" in model_ids:
            return "gemini-2.5-pro"
        return "gemini-2.0-flash"

    def _make_step(
        self,
        *,
        step: str,
        model_id: str,
        data_flow: DataFlow,
        order: int,
        per_candidate: bool,
        scope: str | None = None,
        params: dict | None = None,
    ) -> PlanStep:
        return PlanStep(
            step=step,
            model_id=model_id,
            data_flow=data_flow,
            order=order,
            per_candidate=per_candidate,
            scope=scope,
            params=params or {},
        )
