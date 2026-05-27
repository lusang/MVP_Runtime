"""
StepGraphBuilder (Stage 3) — builds a complete PipelinePlan from per-attribute
runtime parameters.

Two topology modes:
  use_detector=True  (default): scene_neg → detect → nms → verify → per_candidate attributes → merge
  use_detector=False:           scene_neg → full_quality → full_attribute → full_negative → merge
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
        use_detector: bool = True,
    ) -> PipelinePlan:
        """Build a PipelinePlan from parsed template and resolved attribute params.

        Args:
            parsed: The parsed template spec.
            attribute_params: Resolved runtime params for each attribute.
            use_detector: If True, generate detection-driven topology (YOLO + per-candidate).
                          If False, generate full-image analysis topology (Gemini on full image).

        Returns:
            A complete PipelinePlan ready for execution.
        """
        steps: list[PlanStep] = []
        order = 0

        # Detect pure_negative presence (triggers scene-level step + early_exit)
        has_pure_negative = any(
            a.enabled and a.name.lower().replace(" ", "_") == "pure_negative"
            for a in parsed.negative_attributes
        )

        # ── 1. Scene-level negative pre-check (both modes) ───────────
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

        if use_detector:
            self._build_detector_topology(parsed, attribute_params, steps, order,
                                          has_pure_negative)
        else:
            self._build_fullimage_topology(parsed, attribute_params, steps, order,
                                           has_pure_negative)

        return PipelinePlan(
            plan_id=str(uuid.uuid4()),
            object_name=parsed.object_name,
            steps=steps,
            early_exit_rules=self._early_exit_rules(has_pure_negative),
            skip_conditions=self._skip_conditions(use_detector, has_pure_negative,
                                                   attribute_params),
            planner_model="StepGraphBuilder",
            planner_version="2.0",
        )

    # ── Topology: detector-driven (current) ─────────────────────────

    def _build_detector_topology(
        self,
        parsed: ParsedTaskSpec,
        attribute_params: list[AttributeRuntimeParams],
        steps: list[PlanStep],
        order: int,
        has_pure_negative: bool,
    ) -> None:
        # 2. Detection
        steps.append(self._make_step(
            step="detect", model_id="yolo-world-v2-x",
            data_flow=DataFlow.FULL, order=order, per_candidate=False,
        ))
        order += 1

        # 3. NMS
        steps.append(self._make_step(
            step="nms", model_id="rule-engine",
            data_flow=DataFlow.FULL, order=order, per_candidate=False,
            params={"iou_threshold": 0.5},
        ))
        order += 1

        # 4. Verification
        steps.append(self._make_step(
            step="verify", model_id="gemini-2.0-flash",
            data_flow=DataFlow.CROP, order=order, per_candidate=True,
        ))
        order += 1

        # 5-7. Per-candidate attribute steps (Planner-resolved handlers + models)
        filtered_params = [
            p for p in attribute_params
            if not (p.scope == "negative"
                    and p.attribute_key.lower().replace(" ", "_") == "pure_negative")
        ]
        handler_map, model_map = self._build_handler_maps(filtered_params)
        grouped = self._group_attributes(filtered_params)

        for scope in ("quality", "semantic", "negative"):
            if scope not in grouped:
                continue
            for group_key, params_list in grouped[scope].items():
                data_flow_str, handler = group_key
                data_flow = DataFlow(data_flow_str)
                step_type = "quality" if scope == "quality" else ("negative" if scope == "negative" else "attribute")
                model_id = self._pick_model_id(params_list)
                keys = [p.attribute_key for p in params_list]
                steps.append(self._make_step(
                    step=step_type, model_id=model_id,
                    data_flow=data_flow, order=order,
                    per_candidate=params_list[0].per_candidate,
                    scope=scope,
                    params={
                        "attribute_keys": keys,
                        "handler_map": {k: handler_map[k] for k in keys},
                        "model_map": {k: model_map[k] for k in keys},
                    },
                ))
                order += 1

        # 8. Scene-level fallback (0-detection safety net)
        self._add_fallback_steps(attribute_params, steps, order)

    # ── Topology: full-image analysis (no detector) ─────────────────

    def _build_fullimage_topology(
        self,
        parsed: ParsedTaskSpec,
        attribute_params: list[AttributeRuntimeParams],
        steps: list[PlanStep],
        order: int,
        has_pure_negative: bool,
    ) -> None:
        # Filter out pure_negative (handled by scene-level step above)
        filtered_params = [
            p for p in attribute_params
            if not (p.scope == "negative"
                    and p.attribute_key.lower().replace(" ", "_") == "pure_negative")
        ]
        handler_map, model_map = self._build_handler_maps(filtered_params)

        # Collect attributes by scope — force data_flow=full_image for all
        quality_keys = sorted({p.attribute_key for p in filtered_params if p.scope == "quality"})
        semantic_keys = sorted({p.attribute_key for p in filtered_params if p.scope == "semantic"})
        negative_keys = sorted({p.attribute_key for p in filtered_params if p.scope == "negative"})

        if quality_keys:
            steps.append(self._make_step(
                step="full_quality",
                model_id=model_map.get(quality_keys[0], "gemini-2.0-flash"),
                data_flow=DataFlow.FULL,
                order=order,
                per_candidate=False,
                scope="quality",
                params={
                    "attribute_keys": quality_keys,
                    "handler_map": {k: handler_map[k] for k in quality_keys},
                    "model_map": {k: model_map[k] for k in quality_keys},
                },
            ))
            order += 1

        if semantic_keys:
            steps.append(self._make_step(
                step="full_attribute",
                model_id=model_map.get(semantic_keys[0], "gemini-2.5-pro"),
                data_flow=DataFlow.FULL,
                order=order,
                per_candidate=False,
                scope="semantic",
                params={
                    "attribute_keys": semantic_keys,
                    "handler_map": {k: handler_map[k] for k in semantic_keys},
                    "model_map": {k: model_map[k] for k in semantic_keys},
                },
            ))
            order += 1

        if negative_keys:
            steps.append(self._make_step(
                step="full_negative",
                model_id=model_map.get(negative_keys[0], "gemini-2.0-flash"),
                data_flow=DataFlow.FULL,
                order=order,
                per_candidate=False,
                scope="negative",
                params={
                    "attribute_keys": negative_keys,
                    "handler_map": {k: handler_map[k] for k in negative_keys},
                    "model_map": {k: model_map[k] for k in negative_keys},
                },
            ))
            order += 1

        # Merge (last)
        steps.append(self._make_step(
            step="merge", model_id="gemini-2.0-flash",
            data_flow=DataFlow.FULL, order=order, per_candidate=False,
        ))

    # ── Fallback steps (detector mode, 0-detection safety net) ──────

    def _add_fallback_steps(
        self,
        attribute_params: list[AttributeRuntimeParams],
        steps: list[PlanStep],
        order: int,
    ) -> None:
        filtered_params = [
            p for p in attribute_params
            if not (p.scope == "negative"
                    and p.attribute_key.lower().replace(" ", "_") == "pure_negative")
        ]
        handler_map, model_map = self._build_handler_maps(filtered_params)
        fb_quality = [p for p in filtered_params if p.scope == "quality" and p.data_flow == "full_image"]
        fb_negative = [p for p in filtered_params if p.scope == "negative"]

        if fb_quality:
            keys = [p.attribute_key for p in fb_quality]
            steps.append(self._make_step(
                step="quality_fallback",
                model_id=self._pick_model_id(fb_quality),
                data_flow=DataFlow.FULL, order=order,
                per_candidate=False, scope="quality",
                params={
                    "attribute_keys": keys,
                    "handler_map": {k: handler_map[k] for k in keys},
                    "model_map": {k: model_map[k] for k in keys},
                },
            ))
            order += 1

        if fb_negative:
            keys = [p.attribute_key for p in fb_negative]
            steps.append(self._make_step(
                step="negative_fallback",
                model_id=self._pick_model_id(fb_negative),
                data_flow=DataFlow.FULL, order=order,
                per_candidate=False, scope="negative",
                params={
                    "attribute_keys": keys,
                    "handler_map": {k: handler_map[k] for k in keys},
                    "model_map": {k: model_map[k] for k in keys},
                },
            ))
            order += 1

        # Merge (last)
        steps.append(self._make_step(
            step="merge", model_id="gemini-2.0-flash",
            data_flow=DataFlow.FULL, order=order, per_candidate=False,
        ))

    # ── Rules & conditions ──────────────────────────────────────────

    @staticmethod
    def _early_exit_rules(has_pure_negative: bool) -> list[EarlyExitRule]:
        rules: list[EarlyExitRule] = []
        if has_pure_negative:
            rules.append(EarlyExitRule(
                condition="scene_pure_negative",
                reason="scene confirmed pure negative — no target object present",
            ))
        return rules

    def _skip_conditions(
        self,
        use_detector: bool,
        has_pure_negative: bool,
        attribute_params: list[AttributeRuntimeParams],
    ) -> list[SkipCondition]:
        conditions: list[SkipCondition] = []

        if use_detector:
            # Skip per-candidate negative when scene-level pure_negative confirmed
            if has_pure_negative:
                conditions.append(SkipCondition(
                    step="negative",
                    condition="scene_pure_negative",
                    reason="Pure Negative already confirmed at scene level",
                ))

            # Skip attribute/semantic when all candidates rejected
            conditions.append(SkipCondition(
                step="attribute",
                condition="all(not c.exists for c in candidates)",
                reason="all candidates rejected by verification — no targets to assess",
            ))

            # Skip fallback steps when candidates exist
            filtered = [
                p for p in attribute_params
                if not (p.scope == "negative"
                        and p.attribute_key.lower().replace(" ", "_") == "pure_negative")
            ]
            if any(p.scope == "quality" and p.data_flow == "full_image" for p in filtered):
                conditions.append(SkipCondition(
                    step="quality_fallback",
                    condition="len(candidates) > 0",
                    reason="candidates exist — scene-level quality fallback not needed",
                ))
            if any(p.scope == "negative" for p in filtered):
                conditions.append(SkipCondition(
                    step="negative_fallback",
                    condition="len(candidates) > 0",
                    reason="candidates exist — scene-level negative fallback not needed",
                ))

        return conditions

    # ── Internal helpers ────────────────────────────────────────────

    @staticmethod
    def _group_attributes(
        params: list[AttributeRuntimeParams],
    ) -> dict[str, dict[tuple[str, str], list[AttributeRuntimeParams]]]:
        grouped: dict[str, dict[tuple[str, str], list[AttributeRuntimeParams]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        for p in params:
            key = (p.data_flow, p.handler)
            grouped[p.scope][key].append(p)
        return dict(grouped)

    @staticmethod
    def _build_handler_maps(
        attribute_params: list[AttributeRuntimeParams],
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Build per-attribute handler and model maps from Planner-resolved params."""
        handler_map = {p.attribute_key: p.handler for p in attribute_params}
        model_map = {p.attribute_key: p.model_id for p in attribute_params}
        return handler_map, model_map

    @staticmethod
    def _pick_model_id(params_list: list[AttributeRuntimeParams]) -> str:
        model_ids = {p.model_id for p in params_list}
        if "gemini-2.5-pro" in model_ids:
            return "gemini-2.5-pro"
        return "gemini-2.0-flash"

    @staticmethod
    def _make_step(
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
