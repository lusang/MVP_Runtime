"""
Planner — compiles a template into an executable PipelinePlan.

Architecture (v2):
  Stage 1: Semantic Classifier    — attribute → semantic feature vector
  Stage 2a: Capability Mapping    — feature vector → required capabilities
  Stage 2b: Resolver              — capabilities → handler + model_id
  Stage 3: StepGraph Builder      — attribute params → PipelinePlan
  Stage 4: Validator              — validate PipelinePlan

When MVP_DISABLE_PLANNER=1, falls back to _StaticPlanFactory (legacy).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from runtime.capability_mapping import map_features
from runtime.plan_validator import validate
from runtime.prompt_manager import PromptManager
from runtime.resolver import resolve
from runtime.semantic_classifier import classify_by_keywords
from runtime.step_graph_builder import StepGraphBuilder
from schemas.pipeline_plan import DataFlow, EarlyExitRule, PipelinePlan, PlanStep, SkipCondition
from schemas.template_spec import HANDLER_BY_SCOPE, ParsedTaskSpec


def _planner_disabled() -> bool:
    return os.environ.get("MVP_DISABLE_PLANNER", "0").strip() in ("1", "true", "yes")


def _planner_model() -> str:
    return os.environ.get("PLANNER_MODEL", "gemini-2.0-flash").strip()


def _timeout_sec() -> int:
    return int(os.environ.get("GEMINI_TIMEOUT_SEC", "120"))


_PLANNER_PROMPT = """\
You are a pipeline strategy planner for a computer vision annotation system.

Your job: given a template task, a catalog of available models, and historical performance data, produce an optimal execution plan.

=== TEMPLATE TASK ===
Object name: {object_name}
Description: {description}
Include (positive indicators): {include}
Exclude (do NOT count): {exclude}

Semantic attributes:
{semantic_attrs}

Quality attributes:
{quality_attrs}

Negative attributes:
{negative_attrs}

=== AVAILABLE MODELS ===
{model_catalog}

=== HISTORICAL PERFORMANCE (last 30 days) ===
{performance_history}

=== PLANNING RULES ===
1. Every plan MUST include exactly one "detect" step and exactly one "merge" step.
2. Detection models: only models with capability step="detect". Prefer yolo-world-v2-x for accuracy, yolo-world-v2-s for speed.
3. Verification: only models with capability step="verify". gemini-2.0-flash is fast/cheap, gemini-2.5-pro is higher quality.
4. Semantic attributes: models with step="attribute" AND scope includes "semantic".
5. Quality attributes: models with step="quality". opencv-heuristics is free and fast.
6. Negative attributes: models with step="negative" or step="attribute" with scope "negative".
7. Merge: models with step="merge". mechanical-merge is free but basic; gemini models provide reasoning traces.
8. If a template has a "Pure Negative" negative attribute, add a scene-level pre-check (negative step, per_candidate=false, data_flow=full_image) BEFORE detection, and add an early_exit_rule: if scene_pure_negative, skip remaining steps.
8a. QUALITY-FEASIBILITY GATING: The quality step (OpenCV) runs BEFORE semantic attributes. Quality visibility scores (occlusion/blur/lighting) gate which semantic attributes are feasible to assess. Semantic attributes that require clear visibility (e.g. brand_logo) should be skipped when quality is poor.
9. Set skip_conditions when a step is redundant:
   - Skip per-candidate Pure Negative when scene check already ran: condition="scene_pure_negative"
   - Skip semantic step when ALL candidates were rejected by verification: condition="all(not c.exists for c in candidates)"
   - Skip specific attributes when quality makes them infeasible (the executor handles this via missing_attributes per candidate).
10. Order steps by: scene-check (if any) → detect → nms → verify → quality → semantic → negative → merge. Quality MUST come before semantic!
11. Use historical performance to prefer models with higher success_rate and lower latency.
12. If no historical data exists for a model, use its estimated_latency_ms and cost_tier to decide.
13. Do NOT add early_exit_rules for empty detections — the executor handles empty-candidate pipelines gracefully.

=== OUTPUT FORMAT ===
Respond ONLY with a valid JSON object (no markdown, no extra text) matching this schema:

{{
  "plan_id": "<UUID>",
  "object_name": "{object_name}",
  "steps": [
    {{
      "step": "detect|verify|attribute|quality|negative|merge",
      "model_id": "<model_id from catalog>",
      "data_flow": "crop|full_image",
      "order": 0,
      "per_candidate": false,
      "scope": null,
      "params": {{}}
    }}
  ],
  "early_exit_rules": [
    {{
      "condition": "scene_pure_negative",
      "reason": "scene confirmed pure negative — skip remaining steps"
    }}
  ],
  "skip_conditions": [
    {{
      "step": "negative",
      "condition": "scene_pure_negative",
      "reason": "scene already confirmed pure negative"
    }}
  ],
  "planner_model": "{planner_model}",
  "planner_version": "1.0",
  "meta": {{}}
}}"""


def _read_image_bytes(image_path: str) -> bytes:
    with Image.open(image_path) as im:
        if im.mode != "RGB":
            im = im.convert("RGB")
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=85)
        return buf.getvalue()


def _extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from Planner response: {text[:300]}")


# ── New Compiler ────────────────────────────────────────────────────────────


def compile_plan(parsed: ParsedTaskSpec) -> PipelinePlan:
    """Compile a template into an executable PipelinePlan.

    This is the canonical entry point for the v2 compiler pipeline.
    It runs all 4 stages: classifier → mapping → resolver → builder → validator.
    """
    # Collect all enabled attributes across scopes
    all_attrs = (
        [(a, "semantic") for a in parsed.semantic_attributes if a.enabled]
        + [(a, "quality") for a in parsed.quality_attributes if a.enabled]
        + [(a, "negative") for a in parsed.negative_attributes if a.enabled]
    )

    attribute_params = []
    for attr, scope in all_attrs:
        # Stage 1: Semantic Classifier
        features = classify_by_keywords(attr)

        # Stage 2a: Capability Mapping
        caps = map_features(features, attribute_key=attr.key, scope=scope)

        # Stage 2b: Resolver
        params = resolve(caps, scope=scope, attribute_name=attr.name)

        attribute_params.append(params)

    # Stage 3: StepGraph Builder
    builder = StepGraphBuilder()
    plan = builder.build(parsed, attribute_params)

    # Stage 4: Validator
    result = validate(plan)
    if not result.passed:
        raise ValueError(
            f"Plan validation failed: {'; '.join(result.errors)}"
        )

    return plan


# ── Legacy Planner (Gemini-driven, with static fallback) ────────────────────


class Planner:
    """Gemini-driven pipeline planner with static fallback.

    The primary entry point is compile() — a deterministic 4-stage compiler.
    The old plan() method is kept for backward compatibility.
    """

    def __init__(
        self,
        *,
        performance_text_fn: Any = None,
        catalog_text_fn: Any = None,
    ) -> None:
        self._performance_text_fn = performance_text_fn
        self._catalog_text_fn = catalog_text_fn

    async def plan(
        self,
        *,
        parsed: ParsedTaskSpec,
        image_path: str,
        template_name: str = "",
    ) -> PipelinePlan:
        if _planner_disabled():
            return _StaticPlanFactory.build(parsed)

        try:
            return await self._plan_via_gemini(
                parsed=parsed,
                image_path=image_path,
                template_name=template_name,
            )
        except Exception:
            return _StaticPlanFactory.build(parsed)

    async def _plan_via_gemini(
        self,
        *,
        parsed: ParsedTaskSpec,
        image_path: str,
        template_name: str,
    ) -> PipelinePlan:
        from runtime.model_registry import catalog_as_text_for_prompt
        from runtime.performance_tracker import PerformanceTracker

        catalog_text = catalog_as_text_for_prompt()

        perf_text = "No historical performance data available yet."
        if self._performance_text_fn:
            perf_text = self._performance_text_fn(template_name)
        else:
            tracker = PerformanceTracker()
            perf_text = tracker.performance_summary_text(template_name)

        semantic_attrs = _format_attrs(parsed.semantic_attributes)
        quality_attrs = _format_attrs(parsed.quality_attributes)
        negative_attrs = _format_attrs(parsed.negative_attributes)
        model = _planner_model()

        prompt_template = PromptManager.load("planner", default=_PLANNER_PROMPT)
        prompt = prompt_template.format(
            object_name=parsed.object_name,
            description=parsed.description or "N/A",
            include=parsed.include or "N/A",
            exclude=parsed.exclude or "N/A",
            semantic_attrs=semantic_attrs or "(none)",
            quality_attrs=quality_attrs or "(none)",
            negative_attrs=negative_attrs or "(none)",
            model_catalog=catalog_text,
            performance_history=perf_text,
            planner_model=model,
        )

        raw = await asyncio.to_thread(
            _gemini_generate_planner, model, prompt, image_path, _timeout_sec()
        )
        data = _extract_json(raw)
        return PipelinePlan(**data)

    @staticmethod
    def static_plan(parsed: ParsedTaskSpec) -> PipelinePlan:
        return _StaticPlanFactory.build(parsed)

    @staticmethod
    def compile(parsed: ParsedTaskSpec) -> PipelinePlan:
        """Compile a fixed execution plan from template — no Gemini, no replanning.

        This is the canonical entry point for batch execution:
            compiled_plan = Planner.compile(parsed)
            for image in images:
                engine.run_with_plan(image_path=image, plan=compiled_plan, parsed=parsed)
        """
        return compile_plan(parsed)


def _format_attrs(attrs: list) -> str:
    if not attrs:
        return "(none)"
    lines: list[str] = []
    for a in attrs:
        lines.append(f"  - {a.name} (key={a.key}, type={a.type}, scope={a.scope}, handler={a.handler})")
    return "\n".join(lines)


# ── Legacy Static Factory (kept as fallback) ────────────────────────────


class _StaticPlanFactory:
    """Legacy static plan builder — kept as fallback when MVP_DISABLE_PLANNER=1.

    Will be removed in a future version once the v2 compiler is fully validated.
    """

    @staticmethod
    def build(parsed: ParsedTaskSpec) -> PipelinePlan:
        plan_id = str(uuid.uuid4())
        steps: list[PlanStep] = []
        order = 0

        has_pure_negative = _has_pure_negative(parsed)

        # Step 0: Scene-level pure-negative pre-check (if applicable)
        if has_pure_negative:
            steps.append(PlanStep(
                step="negative",
                model_id="gemini-2.0-flash",
                data_flow=DataFlow.FULL,
                order=order,
                per_candidate=False,
                scope="negative",
                params={"scene_check": True, "attribute_key": "Pure Negative"},
            ))
            order += 1

        # Step 1: Detection
        steps.append(PlanStep(
            step="detect",
            model_id="yolo-world-v2-x",
            data_flow=DataFlow.FULL,
            order=order,
            per_candidate=False,
        ))
        order += 1

        # Step 2: Non-Maximum Suppression
        steps.append(PlanStep(
            step="nms",
            model_id="rule-engine",
            data_flow=DataFlow.FULL,
            order=order,
            per_candidate=False,
            params={"iou_threshold": 0.5},
        ))
        order += 1

        # Step 3: Verification (per candidate)
        steps.append(PlanStep(
            step="verify",
            model_id="gemini-2.0-flash",
            data_flow=DataFlow.CROP,
            order=order,
            per_candidate=True,
        ))
        order += 1

        # Steps 3-5: Attributes by scope — quality before semantic
        for scope in ("quality", "semantic", "negative"):
            scope_attrs = getattr(parsed, f"{scope}_attributes", [])
            enabled = [a for a in scope_attrs if a.enabled]
            if not enabled:
                continue

            if scope == "semantic":
                model_id = "gemini-2.0-flash"
                step_type = "attribute"

                crop_keys = frozenset(a.key for a in enabled if a.analysis_scope == "crop")
                full_keys = frozenset(a.key for a in enabled if a.analysis_scope == "full_image")

                if crop_keys:
                    steps.append(PlanStep(
                        step=step_type,
                        model_id=model_id,
                        data_flow=DataFlow.CROP,
                        order=order,
                        per_candidate=True,
                        scope=scope,
                        params={"attribute_keys": list(crop_keys)},
                    ))
                    order += 1
                if full_keys:
                    steps.append(PlanStep(
                        step=step_type,
                        model_id=model_id,
                        data_flow=DataFlow.FULL,
                        order=order,
                        per_candidate=True,
                        scope=scope,
                        params={"attribute_keys": list(full_keys)},
                    ))
                    order += 1
            elif scope == "quality":
                model_id = "opencv-heuristics"
                step_type = "quality"
                data_flow = DataFlow.CROP
                steps.append(PlanStep(
                    step=step_type,
                    model_id=model_id,
                    data_flow=data_flow,
                    order=order,
                    per_candidate=True,
                    scope=scope,
                ))
                order += 1
            else:  # negative
                model_id = "gemini-2.0-flash"
                step_type = "negative"
                data_flow = DataFlow.FULL
                steps.append(PlanStep(
                    step=step_type,
                    model_id=model_id,
                    data_flow=data_flow,
                    order=order,
                    per_candidate=True,
                    scope=scope,
                ))
                order += 1

        # Last step: Merge
        steps.append(PlanStep(
            step="merge",
            model_id="gemini-2.0-flash",
            data_flow=DataFlow.FULL,
            order=order,
            per_candidate=False,
        ))

        early_exit_rules: list[EarlyExitRule] = []
        if has_pure_negative:
            early_exit_rules.append(EarlyExitRule(
                condition="scene_pure_negative",
                reason="scene confirmed pure negative — no target object present",
            ))

        skip_conditions: list[SkipCondition] = []

        if has_pure_negative:
            skip_conditions.append(SkipCondition(
                step="negative",
                condition="scene_pure_negative",
                reason="Pure Negative already confirmed at scene level",
            ))

        skip_conditions.append(SkipCondition(
            step="attribute",
            condition="all(not c.exists for c in candidates)",
            reason="all candidates rejected by verification — no targets to assess",
        ))

        return PipelinePlan(
            plan_id=plan_id,
            object_name=parsed.object_name,
            steps=steps,
            early_exit_rules=early_exit_rules,
            skip_conditions=skip_conditions,
            planner_model="_StaticPlanFactory",
            planner_version="1.0",
        )


def _has_pure_negative(parsed: ParsedTaskSpec) -> bool:
    return any(
        a.enabled and a.name.lower().replace(" ", "_") == "pure_negative"
        for a in parsed.negative_attributes
    )


def _gemini_generate_planner(
    model_id: str,
    prompt: str,
    image_path: str,
    timeout_sec: int,
) -> str:
    from google import genai
    from models.gemini_client import _api_key

    client = genai.Client(api_key=_api_key(), http_options={"timeout": timeout_sec * 1000})
    image_bytes = _read_image_bytes(image_path)
    contents = [
        prompt,
        {"inline_data": {"mime_type": "image/jpeg", "data": image_bytes}},
    ]
    response = client.models.generate_content(model=model_id, contents=contents)
    return response.text or ""
