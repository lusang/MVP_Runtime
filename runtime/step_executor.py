"""
StepExecutor — executes a PipelinePlan dynamically, recording performance per step.
State-driven routing: quality gates semantic attributes, verify gates quality/semantic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from handlers.attribute_handler import AttributeHandler
from handlers.verification_handler import VerificationHandler
from models.gemini_merger import GeminiMerger
from models.gemini_verifier import GeminiVerifier
from models.yolo_detector import YOLODetector
from runtime.performance_tracker import PerformanceTracker
from schemas.candidate_state import CandidateState
from schemas.feasibility import FeasibilityRule, build_feasibility_rules
from schemas.pipeline_plan import EarlyExitRule, PipelinePlan, PlanStep, SkipCondition
from schemas.template_spec import ParsedTaskSpec
from storage.image_crop import bbox_for_full_crop


@dataclass
class _ExecutionContext:
    detections: list[Any] = field(default_factory=list)
    candidates: list[CandidateState] = field(default_factory=list)
    scene_pure_negative: bool = False
    executed_step_ids: list[str] = field(default_factory=list)
    execution_log_lines: list[str] = field(default_factory=list)
    feasibility_rules: dict[str, FeasibilityRule] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    detections: list[Any] = field(default_factory=list)
    candidates: list[CandidateState] = field(default_factory=list)
    scene_pure_negative: bool = False
    merge_result: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    executed_steps: list[str] = field(default_factory=list)


class StepExecutor:
    """Executes a PipelinePlan, dispatching each step to the appropriate model/handler."""

    def __init__(
        self,
        *,
        detector: YOLODetector | None = None,
        verifier: GeminiVerifier | None = None,
        verification_handler: VerificationHandler | None = None,
        attribute_handler: AttributeHandler | None = None,
        merger: GeminiMerger | None = None,
        tracker: PerformanceTracker | None = None,
    ) -> None:
        self._detector = detector or YOLODetector()
        self._verifier = verifier or GeminiVerifier()
        self._verification_handler = verification_handler or VerificationHandler(self._verifier)
        self._attribute_handler = attribute_handler
        self._merger = merger or GeminiMerger()
        self._tracker = tracker

    async def execute(
        self,
        *,
        plan: PipelinePlan,
        image_path: str,
        parsed: ParsedTaskSpec,
        run_id: str = "",
        template_name: str = "",
    ) -> ExecutionResult:
        t0 = time.perf_counter()
        ctx = _ExecutionContext()
        ctx.feasibility_rules = build_feasibility_rules(parsed)

        steps = sorted(plan.steps, key=lambda s: s.order)

        for step in steps:
            if _should_early_exit(plan.early_exit_rules, ctx):
                break
            if _should_skip(plan.skip_conditions, step, ctx):
                continue

            step_t0 = time.perf_counter()
            success = True
            try:
                await self._dispatch(step, ctx, image_path, parsed, run_id)
            except Exception:
                success = False
            step_ms = (time.perf_counter() - step_t0) * 1000.0

            if self._tracker:
                self._tracker.record_step(
                    run_id=run_id,
                    step=step.step,
                    model_id=step.model_id,
                    template_name=template_name,
                    success=success,
                    latency_ms=step_ms,
                    object_count=len(ctx.detections),
                )
            ctx.executed_step_ids.append(f"{step.step}:{step.model_id}")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return ExecutionResult(
            detections=ctx.detections,
            candidates=ctx.candidates,
            scene_pure_negative=ctx.scene_pure_negative,
            merge_result=getattr(ctx, "_merge_result", {}),
            elapsed_ms=elapsed_ms,
            executed_steps=ctx.executed_step_ids,
        )

    async def _dispatch(
        self,
        step: PlanStep,
        ctx: _ExecutionContext,
        image_path: str,
        parsed: ParsedTaskSpec,
        run_id: str,
    ) -> None:
        if step.step == "detect":
            await self._run_detect(step, ctx, image_path, parsed, run_id)
        elif step.step == "verify":
            await self._run_verify(step, ctx, image_path, parsed, run_id)
        elif step.step == "quality":
            await self._run_quality(step, ctx, image_path, parsed, run_id)
        elif step.step in ("attribute", "semantic"):
            await self._run_semantic(step, ctx, image_path, parsed, run_id)
        elif step.step == "negative":
            await self._run_negative(step, ctx, image_path, parsed, run_id)
        elif step.step == "merge":
            await self._run_merge(step, ctx, image_path, parsed, run_id)

    # ── detect ──────────────────────────────────────────────────────

    async def _run_detect(
        self,
        step: PlanStep,
        ctx: _ExecutionContext,
        image_path: str,
        parsed: ParsedTaskSpec,
        run_id: str,
    ) -> None:
        label = f"{step.step}:{step.model_id}"
        detections = await self._detector.detect(
            image_path,
            target_object=parsed.object_name,
            parsed=parsed,
            run_id=run_id,
        )
        ctx.detections = detections

        _log(ctx, label, f"DETECTION")
        if detections:
            _log(ctx, None, f"    {len(detections)} candidate(s) found")
        else:
            _log(ctx, None, f"    0 candidates found")

        for idx, det in enumerate(detections):
            bbox = det.bbox
            analysis_path = det.crop_path or image_path
            analysis_bbox = bbox_for_full_crop(analysis_path) if det.crop_path else bbox
            ctx.candidates.append(CandidateState(
                object_id=f"obj_{idx}",
                detector_score=det.score,
                bbox=bbox,
                crop_path=det.crop_path,
                analysis_path=analysis_path,
                analysis_bbox=analysis_bbox,
            ))
            bbox_str = f"({bbox.x1:.0f},{bbox.y1:.0f})-({bbox.x2:.0f},{bbox.y2:.0f})"
            _log(ctx, None, f"    · obj_{idx}  yolo={det.score:.2f}  bbox={bbox_str}")
        _log(ctx, None, "")

    # ── verify ──────────────────────────────────────────────────────

    async def _run_verify(
        self,
        step: PlanStep,
        ctx: _ExecutionContext,
        image_path: str,
        parsed: ParsedTaskSpec,
        run_id: str,
    ) -> None:
        label = f"{step.step}:{step.model_id}"
        _log(ctx, label, "VERIFICATION")
        for c in ctx.candidates:
            verification = await self._verification_handler.verify_object(
                image_path=c.analysis_path,
                bbox=c.analysis_bbox,
                parsed=parsed,
                object_id=c.object_id,
            )
            c.verification = verification
            c.verify_score = float(verification.get("score", 0.0))
            c.compute_confidence()

            if verification.get("ok") is False:
                c.exists = False
                c.record("verify", "rejected — routing to negative only")
                _log(ctx, None, f"    · {c.object_id}  ✗ rejected → negative")
            else:
                c.exists = True
                c.record("verify", "confirmed, proceeding to quality")
                rationale = str(verification.get("rationale", ""))[:80]
                _log(ctx, None, f"    · {c.object_id}  ✓ ok  score={c.verify_score:.2f}  \"{rationale}\"")
        _log(ctx, None, "")

    # ── quality ─────────────────────────────────────────────────────

    async def _run_quality(
        self,
        step: PlanStep,
        ctx: _ExecutionContext,
        image_path: str,
        parsed: ParsedTaskSpec,
        run_id: str,
    ) -> None:
        label = f"{step.step}:{step.model_id}"
        _log(ctx, label, "QUALITY CHECK")
        for c in ctx.candidates:
            if not c.exists:
                c.record("quality", "skipped — object does not exist")
                _log(ctx, None, f"    · {c.object_id}  skipped (not exists)")
                continue

            if self._attribute_handler:
                new_stage = await self._attribute_handler.analyze_by_scopes(
                    image_path=c.analysis_path,
                    bbox=c.analysis_bbox,
                    parsed=parsed,
                    object_id=c.object_id,
                    scopes={"quality"},
                )
                c.quality = new_stage.quality
                # Extract continuous metrics from quality results
                for kval, qitem in c.quality.items():
                    if isinstance(qitem, dict) and "metrics" in qitem:
                        c.metrics.update(qitem["metrics"])

            # Transfer quality results to visibility
            c.visibility = dict(c.quality)

            # Compute attribute feasibility from visibility
            _compute_feasibility(c, ctx.feasibility_rules)

            anomalies = _quality_anomalies(c.quality)
            if anomalies:
                _log(ctx, None, f"    · {c.object_id}  ⚠ {', '.join(anomalies)}")
                if c.missing_attributes:
                    _log(ctx, None, f"       infeasible: {', '.join(c.missing_attributes)}")
            else:
                if c.missing_attributes:
                    _log(ctx, None, f"    · {c.object_id}  normal (infeasible: {', '.join(c.missing_attributes)})")
                else:
                    _log(ctx, None, f"    · {c.object_id}  normal — all attributes feasible")
        _log(ctx, None, "")

    # ── semantic ────────────────────────────────────────────────────

    async def _run_semantic(
        self,
        step: PlanStep,
        ctx: _ExecutionContext,
        image_path: str,
        parsed: ParsedTaskSpec,
        run_id: str,
    ) -> None:
        label = f"{step.step}:{step.model_id}"
        _log(ctx, label, "SEMANTIC ATTRIBUTES")
        for c in ctx.candidates:
            if not c.exists:
                c.record("semantic", "skipped — object does not exist")
                _log(ctx, None, f"    · {c.object_id}  skipped (not exists)")
                continue

            # Skip attributes that quality deemed infeasible
            skip_keys: frozenset[str] | None = None
            if c.missing_attributes:
                skip_keys = frozenset(c.missing_attributes)
                c.record(
                    "semantic",
                    f"assessing feasible attributes; skipping {', '.join(c.missing_attributes)}",
                )

            if self._attribute_handler:
                new_stage = await self._attribute_handler.analyze_by_scopes(
                    image_path=c.analysis_path,
                    bbox=c.analysis_bbox,
                    parsed=parsed,
                    object_id=c.object_id,
                    scopes={"semantic"},
                    skip_keys=skip_keys,
                )
                c.attributes = new_stage.attributes

            # Mark infeasible attributes as skipped
            for key in c.missing_attributes:
                c.attributes[key] = {
                    "value": None,
                    "confidence": 0.0,
                    "infeasible": True,
                    "reason": "quality insufficient for reliable assessment",
                }

            attrs = c.attributes
            items = [f"{k}={v.get('value','?')}({v.get('confidence',0):.2f})" for k, v in attrs.items()]
            _log(ctx, None, f"    · {c.object_id}  {', '.join(items)}" if items else f"    · {c.object_id}  (none)")
        _log(ctx, None, "")

    # ── negative ────────────────────────────────────────────────────

    async def _run_negative(
        self,
        step: PlanStep,
        ctx: _ExecutionContext,
        image_path: str,
        parsed: ParsedTaskSpec,
        run_id: str,
    ) -> None:
        label = f"{step.step}:{step.model_id}"

        # Scene-level check (before detection)
        if step.per_candidate is False and step.params.get("scene_check"):
            _log(ctx, label, "SCENE CHECK")
            pure_neg_spec = next(
                (s for s in parsed.negative_attributes if s.name == "Pure Negative" and s.enabled),
                None,
            )
            if pure_neg_spec is not None:
                scene_result = await self._verifier.verify_scene_pure_negative(
                    image_path=image_path,
                    parsed=parsed,
                )
                ctx.scene_pure_negative = bool(scene_result.get("value", False))
                if ctx.scene_pure_negative:
                    _log(ctx, None, "    ⚠ Pure Negative confirmed — pipeline will stop")
                else:
                    _log(ctx, None, "    No Pure Negative → continue")
            _log(ctx, None, "")
            return

        # Per-candidate negative check
        _log(ctx, label, "NEGATIVE CHECK")
        skip_keys: frozenset[str] | None = None
        if ctx.scene_pure_negative:
            pure_neg_spec = next(
                (s for s in parsed.negative_attributes if s.name == "Pure Negative" and s.enabled),
                None,
            )
            if pure_neg_spec:
                skip_keys = frozenset({pure_neg_spec.key})

        for c in ctx.candidates:
            if self._attribute_handler:
                new_stage = await self._attribute_handler.analyze_by_scopes(
                    image_path=c.analysis_path,
                    bbox=c.analysis_bbox,
                    parsed=parsed,
                    object_id=c.object_id,
                    full_image_path=image_path,
                    full_bbox=c.bbox,
                    skip_keys=skip_keys,
                    scopes={"negative"},
                )
                c.negative_flags.update(new_stage.negative)

            triggered = _negative_triggered(c.negative_flags)
            if triggered:
                c.record("negative", f"flags triggered: {', '.join(triggered)}")
                _log(ctx, None, f"    · {c.object_id}  ⚠ {', '.join(triggered)}")
            else:
                c.record("negative", "no flags")
                _log(ctx, None, f"    · {c.object_id}  no flags")
        _log(ctx, None, "")

    # ── merge ───────────────────────────────────────────────────────

    async def _run_merge(
        self,
        step: PlanStep,
        ctx: _ExecutionContext,
        image_path: str,
        parsed: ParsedTaskSpec,
        run_id: str,
    ) -> None:
        merge_result = await self._merger.merge(
            image_path=image_path,
            parsed=parsed,
            candidates_data=[c.to_dict() for c in ctx.candidates],
            scene_pure_negative=ctx.scene_pure_negative,
            run_id=run_id,
            execution_log_text="\n".join(ctx.execution_log_lines),
        )
        ctx._merge_result = merge_result


# ── early exit / skip ───────────────────────────────────────────────

def _should_early_exit(rules: list[EarlyExitRule], ctx: _ExecutionContext) -> bool:
    safe_ns = _safe_eval_ns(ctx)
    for rule in rules:
        try:
            if eval(rule.condition, {"__builtins__": {}}, safe_ns):
                return True
        except Exception:
            continue
    return False


def _should_skip(
    conditions: list[SkipCondition],
    step: PlanStep,
    ctx: _ExecutionContext,
) -> bool:
    safe_ns = _safe_eval_ns(ctx)
    for cond in conditions:
        if cond.step != step.step:
            continue
        try:
            if eval(cond.condition, {"__builtins__": {}}, safe_ns):
                return True
        except Exception:
            continue
    return False


def _safe_eval_ns(ctx: _ExecutionContext) -> dict[str, Any]:
    return {
        "len": len,
        "detections": ctx.detections,
        "candidates": ctx.candidates,
        "scene_pure_negative": ctx.scene_pure_negative,
        "detection_count": len(ctx.detections),
    }


# ── helpers ─────────────────────────────────────────────────────────

def _log(ctx: _ExecutionContext, label: str | None, line: str) -> None:
    if label is not None:
        idx = len(ctx.executed_step_ids)
        ctx.execution_log_lines.append(f"[{idx}] {line}")
    else:
        ctx.execution_log_lines.append(line)


def _compute_feasibility(
    candidate: CandidateState,
    rules: dict[str, FeasibilityRule],
) -> None:
    candidate.attribute_feasibility = {}
    candidate.missing_attributes = []
    unknowns: list[str] = []

    for attr_key, rule in rules.items():
        if attr_key == "__default__":
            continue
        feasible = rule.assess(candidate.visibility, candidate.metrics)
        candidate.attribute_feasibility[attr_key] = feasible
        if feasible is False:
            candidate.missing_attributes.append(attr_key)
        elif feasible is None:
            unknowns.append(attr_key)

    if candidate.missing_attributes:
        candidate.record(
            "quality",
            f"infeasible: {', '.join(candidate.missing_attributes)}",
        )
    elif unknowns:
        candidate.record(
            "quality",
            f"feasibility unknown for: {', '.join(unknowns)}",
        )
    else:
        candidate.record("quality", "all attributes feasible")


def _quality_anomalies(quality: dict[str, Any]) -> list[str]:
    defaults = {"occlusion": "none", "blur": "clear", "lighting": "normal"}
    items: list[str] = []
    for k, v in quality.items():
        if isinstance(v, dict) and v.get("value") != defaults.get(k):
            items.append(f"{k}={v.get('value')}")
    return items


def _negative_triggered(neg: dict[str, Any]) -> list[str]:
    items: list[str] = []
    for k, v in neg.items():
        if isinstance(v, dict) and v.get("value") is True:
            items.append(f"{k} ({v.get('confidence',0):.2f})")
    return items
