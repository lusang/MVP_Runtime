"""
StepExecutor — executes a PipelinePlan dynamically with explicit RuntimeState + StepResult.

Key design:
  - RuntimeState wraps candidates + shared state; all steps read/write through it.
  - StepResult is produced by every step; skip/early_exit are explicit, not hidden in if/continue.
  - _execute_one() is the single entry point for step dispatch + decision.
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
from runtime.nms import apply_nms
from runtime.performance_tracker import PerformanceTracker
from schemas.candidate_state import Candidate, CandidateState
from schemas.feasibility import FeasibilityRule, build_feasibility_rules
from schemas.pipeline_plan import EarlyExitRule, PipelinePlan, PlanStep, SkipCondition
from schemas.template_spec import ParsedTaskSpec
from storage.image_crop import bbox_for_full_crop

_HANDLER_FOR_STEP: dict[str, str] = {
    "detect": "YOLODetector",
    "nms": "RuleEngine",
    "verify": "GeminiVerifier",
    "quality": "OpenCVAnalyzer",
    "attribute": "GeminiAttributePlugin",
    "semantic": "GeminiAttributePlugin",
    "negative": "GeminiNegativePlugin",
    "quality_fallback": "GeminiAttributePlugin",
    "negative_fallback": "GeminiNegativePlugin",
    "full_quality": "GeminiAttributePlugin",
    "full_attribute": "GeminiAttributePlugin",
    "full_negative": "GeminiNegativePlugin",
    "merge": "MergeEngine",
}


# ── RuntimeState ────────────────────────────────────────────────────


@dataclass
class RuntimeState:
    """Step-to-step shared state. All steps read/write candidates through this.

    ``active_candidates()`` returns candidates that should be processed
    (not SUPPRESSED or REJECTED). Steps should never directly filter
    ``self.candidates`` — use the accessor instead.
    """

    candidates: list[Candidate] = field(default_factory=list)
    scene_flags: dict[str, Any] = field(default_factory=dict)   # scene_pure_negative, ...
    metrics: dict[str, Any] = field(default_factory=dict)       # cross-step numeric data
    artifacts: dict[str, Any] = field(default_factory=dict)     # debug / intermediate files

    def active_candidates(self) -> list[Candidate]:
        """Candidates that should be processed by the current step."""
        return [c for c in self.candidates
                if c.state not in (CandidateState.SUPPRESSED, CandidateState.REJECTED)]

    def candidate_by_id(self, oid: str) -> Candidate | None:
        return next((c for c in self.candidates if c.object_id == oid), None)


# ── StepResult ──────────────────────────────────────────────────────


@dataclass
class StepResult:
    """Structured result produced by each step execution.

    All execution decisions (skip, early_exit, success, failure) are
    explicit through ``status`` + ``reason``, not hidden in ``if/continue``.
    """

    step: str                  # step_type (detect / nms / verify / ...)
    status: str                # "success" | "skipped" | "failed" | "early_exit"
    reason: str                # human-readable description of the decision
    model_id: str = ""         # which model/handler was targeted
    latency_ms: float = 0.0
    input_count: int = 0
    output_count: int = 0
    error: str | None = None   # set when status == "failed"
    events: list[dict] = field(default_factory=list)


# ── _ExecutionContext (internal, shared across steps) ───────────────


@dataclass
class _ExecutionContext:
    detections: list[Any] = field(default_factory=list)
    executed_step_ids: list[str] = field(default_factory=list)
    execution_log_lines: list[str] = field(default_factory=list)
    feasibility_rules: dict[str, FeasibilityRule] = field(default_factory=dict)


# ── ExecutionResult ─────────────────────────────────────────────────


@dataclass
class ExecutionResult:
    detections: list[Any] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    scene_pure_negative: bool = False
    merge_result: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    executed_steps: list[str] = field(default_factory=list)
    step_results: list[StepResult] = field(default_factory=list)


# ── StepExecutor ────────────────────────────────────────────────────


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
        recorder: Any = None,  # RuntimeRecorder, imported lazily to avoid circular deps
    ) -> None:
        self._detector = detector or YOLODetector()
        self._verifier = verifier or GeminiVerifier()
        self._verification_handler = verification_handler or VerificationHandler(self._verifier)
        self._attribute_handler = attribute_handler
        self._merger = merger or GeminiMerger()
        self._tracker = tracker
        self._recorder = recorder

    # ── public entry point ──────────────────────────────────────────

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
        state = RuntimeState()

        steps = sorted(plan.steps, key=lambda s: s.order)
        step_results: list[StepResult] = []

        for step in steps:
            # Real-time step recording (if a recorder is attached)
            step_id = None
            if self._recorder:
                handler = _HANDLER_FOR_STEP.get(step.step, "")
                step_id = self._recorder.start_step(
                    run_id=run_id,
                    step_name=f"{step.step}:{step.model_id}",
                    step_type=step.step,
                    handler=handler,
                    model_id=step.model_id,
                    input_count=len(state.active_candidates()),
                )

            result = await self._execute_one(step, plan, ctx, state, image_path, parsed, run_id)
            step_results.append(result)

            if self._recorder and step_id:
                self._recorder.finish_step(
                    step_id=step_id,
                    status=result.status,
                    latency_ms=result.latency_ms,
                    output_count=result.output_count,
                    error_message=result.error or "",
                )

            # Record performance after each step
            if self._tracker:
                self._tracker.record_step(
                    run_id=run_id,
                    step=step.step,
                    model_id=step.model_id,
                    template_name=template_name,
                    success=result.status == "success",
                    latency_ms=result.latency_ms,
                    object_count=len(state.candidates),
                )
            ctx.executed_step_ids.append(f"{step.step}:{step.model_id}")

            # early_exit terminates the pipeline
            if result.status == "early_exit":
                break

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        scene_pure_negative = state.scene_flags.get("pure_negative", False)

        return ExecutionResult(
            detections=ctx.detections,
            candidates=state.candidates,
            scene_pure_negative=scene_pure_negative,
            merge_result=getattr(state, "_merge_result", {}),
            elapsed_ms=elapsed_ms,
            executed_steps=ctx.executed_step_ids,
            step_results=step_results,
        )

    # ── single-step execution (wraps skip/early_exit/dispatch) ──────

    async def _execute_one(
        self,
        step: PlanStep,
        plan: PipelinePlan,
        ctx: _ExecutionContext,
        state: RuntimeState,
        image_path: str,
        parsed: ParsedTaskSpec,
        run_id: str,
    ) -> StepResult:
        # 1. Early exit check — terminates entire pipeline
        exit_reason = _check_early_exit(plan.early_exit_rules, state)
        if exit_reason:
            return StepResult(
                step=step.step, model_id=step.model_id,
                status="early_exit", reason=exit_reason,
                input_count=len(state.active_candidates()),
            )

        # 2. Skip check — skip this step, continue to next
        skip_reason = _check_skip(plan.skip_conditions, step, state)
        if skip_reason:
            return StepResult(
                step=step.step, model_id=step.model_id,
                status="skipped", reason=skip_reason,
                input_count=len(state.active_candidates()),
            )

        # 3. Execute
        input_count = len(state.active_candidates())
        step_t0 = time.perf_counter()
        try:
            await self._dispatch(step, ctx, state, image_path, parsed, run_id)
            elapsed = (time.perf_counter() - step_t0) * 1000.0
            output_count = len(state.active_candidates())
            return StepResult(
                step=step.step, model_id=step.model_id,
                status="success", reason="",
                latency_ms=elapsed, input_count=input_count, output_count=output_count,
            )
        except Exception as e:
            elapsed = (time.perf_counter() - step_t0) * 1000.0
            return StepResult(
                step=step.step, model_id=step.model_id,
                status="failed", reason=str(e),
                latency_ms=elapsed, input_count=input_count, output_count=0,
                error=f"{type(e).__name__}: {str(e)[:500]}",
            )

    # ── dispatch ────────────────────────────────────────────────────

    async def _dispatch(
        self,
        step: PlanStep,
        ctx: _ExecutionContext,
        state: RuntimeState,
        image_path: str,
        parsed: ParsedTaskSpec,
        run_id: str,
    ) -> None:
        if step.step == "detect":
            await self._run_detect(step, ctx, state, image_path, parsed, run_id)
        elif step.step == "nms":
            self._run_nms(step, state)
        elif step.step == "verify":
            await self._run_verify(step, ctx, state, image_path, parsed, run_id)
        elif step.step == "quality":
            await self._run_quality(step, ctx, state, image_path, parsed, run_id)
        elif step.step in ("attribute", "semantic"):
            await self._run_semantic(step, ctx, state, image_path, parsed, run_id)
        elif step.step == "negative":
            await self._run_negative(step, ctx, state, image_path, parsed, run_id)
        elif step.step == "quality_fallback":
            await self._run_quality_fallback(step, ctx, state, image_path, parsed, run_id)
        elif step.step == "negative_fallback":
            await self._run_negative_fallback(step, ctx, state, image_path, parsed, run_id)
        elif step.step == "full_quality":
            await self._run_full_quality(step, ctx, state, image_path, parsed, run_id)
        elif step.step == "full_attribute":
            await self._run_full_attribute(step, ctx, state, image_path, parsed, run_id)
        elif step.step == "full_negative":
            await self._run_full_negative(step, ctx, state, image_path, parsed, run_id)
        elif step.step == "merge":
            await self._run_merge(step, ctx, state, image_path, parsed, run_id)

    # ── nms ─────────────────────────────────────────────────────────

    def _run_nms(
        self,
        step: PlanStep,
        state: RuntimeState,
    ) -> None:
        label = f"{step.step}:{step.model_id}"
        iou_threshold = float(step.params.get("iou_threshold", 0.5))
        before = len(state.active_candidates())
        apply_nms(state.candidates, iou_threshold=iou_threshold)
        after = len(state.active_candidates())
        suppressed = before - after
        _log(state, label, f"NMS (IoU ≥ {iou_threshold})")
        if suppressed:
            _log(state, None, f"    {suppressed} candidate(s) suppressed")
        else:
            _log(state, None, f"    0 candidates suppressed")
        _log(state, None, "")

    # ── detect ──────────────────────────────────────────────────────

    async def _run_detect(
        self,
        step: PlanStep,
        ctx: _ExecutionContext,
        state: RuntimeState,
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

        _log(state, label, f"DETECTION")
        if detections:
            _log(state, None, f"    {len(detections)} candidate(s) found")
        else:
            _log(state, None, f"    0 candidates found")

        for idx, det in enumerate(detections):
            bbox = det.bbox
            analysis_path = det.crop_path or image_path
            analysis_bbox = bbox_for_full_crop(analysis_path) if det.crop_path else bbox
            state.candidates.append(Candidate(
                object_id=f"obj_{idx}",
                detector_score=det.score,
                bbox=bbox,
                crop_path=det.crop_path,
                analysis_path=analysis_path,
                analysis_bbox=analysis_bbox,
            ))
            bbox_str = f"({bbox.x1:.0f},{bbox.y1:.0f})-({bbox.x2:.0f},{bbox.y2:.0f})"
            _log(state, None, f"    · obj_{idx}  yolo={det.score:.2f}  bbox={bbox_str}")
        _log(state, None, "")

    # ── verify ──────────────────────────────────────────────────────

    async def _run_verify(
        self,
        step: PlanStep,
        ctx: _ExecutionContext,
        state: RuntimeState,
        image_path: str,
        parsed: ParsedTaskSpec,
        run_id: str,
    ) -> None:
        label = f"{step.step}:{step.model_id}"
        _log(state, label, "VERIFICATION")
        for c in state.candidates:
            if c.is_suppressed_or_rejected:
                c.record("verify", "skipped — NMS suppressed")
                _log(state, None, f"    · {c.object_id}  skipped (NMS suppressed)")
                continue

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
                c.transition_to(CandidateState.REJECTED, "verify", "verification rejected the candidate")
                c.record("verify", "rejected — routing to negative only")
                _log(state, None, f"    · {c.object_id}  ✗ rejected → negative")
            else:
                c.transition_to(CandidateState.VERIFIED, "verify", "verification confirmed the candidate")
                c.record("verify", "confirmed, proceeding to quality")
                rationale = str(verification.get("rationale", ""))[:80]
                _log(state, None, f"    · {c.object_id}  ✓ ok  score={c.verify_score:.2f}  \"{rationale}\"")
        _log(state, None, "")

    # ── quality ─────────────────────────────────────────────────────

    async def _run_quality(
        self,
        step: PlanStep,
        ctx: _ExecutionContext,
        state: RuntimeState,
        image_path: str,
        parsed: ParsedTaskSpec,
        run_id: str,
    ) -> None:
        label = f"{step.step}:{step.model_id}"
        _handler_map = step.params.get("handler_map", {})
        _model_map = step.params.get("model_map", {})
        _log(state, label, "QUALITY CHECK")
        for c in state.candidates:
            if c.is_suppressed_or_rejected:
                c.record("quality", "skipped — object does not exist")
                _log(state, None, f"    · {c.object_id}  skipped (not exists)")
                continue

            if self._attribute_handler:
                new_stage = await self._attribute_handler.analyze_by_scopes(
                    image_path=c.analysis_path,
                    bbox=c.analysis_bbox,
                    parsed=parsed,
                    object_id=c.object_id,
                    scopes={"quality"},
                    handler_map=_handler_map,
                    model_map=_model_map,
                )
                c.quality = new_stage.quality
                for kval, qitem in c.quality.items():
                    if isinstance(qitem, dict) and "metrics" in qitem:
                        c.metrics.update(qitem["metrics"])

            c.visibility = dict(c.quality)
            _compute_feasibility(c, ctx.feasibility_rules)

            anomalies = _quality_anomalies(c.quality)
            if anomalies:
                _log(state, None, f"    · {c.object_id}  ⚠ {', '.join(anomalies)}")
                if c.missing_attributes:
                    _log(state, None, f"       infeasible: {', '.join(c.missing_attributes)}")
            else:
                if c.missing_attributes:
                    _log(state, None, f"    · {c.object_id}  normal (infeasible: {', '.join(c.missing_attributes)})")
                else:
                    _log(state, None, f"    · {c.object_id}  normal — all attributes feasible")
        _log(state, None, "")

    # ── semantic ────────────────────────────────────────────────────

    async def _run_semantic(
        self,
        step: PlanStep,
        ctx: _ExecutionContext,
        state: RuntimeState,
        image_path: str,
        parsed: ParsedTaskSpec,
        run_id: str,
    ) -> None:
        from schemas.pipeline_plan import DataFlow

        label = f"{step.step}:{step.model_id}"
        _handler_map = step.params.get("handler_map", {})
        _model_map = step.params.get("model_map", {})
        is_full = step.data_flow == DataFlow.FULL
        _log(state, label, f"SEMANTIC ATTRIBUTES ({'full_image' if is_full else 'crop'})")

        include_keys: frozenset[str] | None = None
        step_keys = step.params.get("attribute_keys")
        if step_keys:
            include_keys = frozenset(step_keys)

        for c in state.candidates:
            if c.is_suppressed_or_rejected:
                c.record("semantic", "skipped — object does not exist")
                _log(state, None, f"    · {c.object_id}  skipped (not exists)")
                continue

            skip_keys: frozenset[str] | None = None
            if c.missing_attributes:
                skip_keys = frozenset(c.missing_attributes)
                c.record(
                    "semantic",
                    f"assessing feasible attributes; skipping {', '.join(c.missing_attributes)}",
                )

            effective_image = c.analysis_path if not is_full else image_path
            effective_bbox = c.analysis_bbox if not is_full else c.bbox

            if self._attribute_handler:
                new_stage = await self._attribute_handler.analyze_by_scopes(
                    image_path=effective_image,
                    bbox=effective_bbox,
                    parsed=parsed,
                    object_id=c.object_id,
                    scopes={"semantic"},
                    skip_keys=skip_keys,
                    include_keys=include_keys,
                    handler_map=_handler_map,
                    model_map=_model_map,
                )
                c.attributes.update(new_stage.attributes)

            for key in c.missing_attributes:
                c.attributes[key] = {
                    "value": None,
                    "confidence": 0.0,
                    "infeasible": True,
                    "reason": "quality insufficient for reliable assessment",
                }

            attrs_items = [f"{k}={v.get('value','?')}({v.get('confidence',0):.2f})" for k, v in c.attributes.items()]
            _log(state, None, f"    · {c.object_id}  {', '.join(attrs_items)}" if attrs_items else f"    · {c.object_id}  (none)")
        _log(state, None, "")

    # ── negative ────────────────────────────────────────────────────

    async def _run_negative(
        self,
        step: PlanStep,
        ctx: _ExecutionContext,
        state: RuntimeState,
        image_path: str,
        parsed: ParsedTaskSpec,
        run_id: str,
    ) -> None:
        label = f"{step.step}:{step.model_id}"

        # Scene-level check (before detection)
        if step.per_candidate is False and step.params.get("scene_check"):
            _log(state, label, "SCENE CHECK")
            pure_neg_spec = next(
                (s for s in parsed.negative_attributes if s.name == "Pure Negative" and s.enabled),
                None,
            )
            if pure_neg_spec is not None:
                scene_result = await self._verifier.verify_scene_pure_negative(
                    image_path=image_path,
                    parsed=parsed,
                )
                is_pure_negative = bool(scene_result.get("value", False))
                state.scene_flags["pure_negative"] = is_pure_negative
                if is_pure_negative:
                    _log(state, None, "    ⚠ Pure Negative confirmed — pipeline will stop")
                else:
                    _log(state, None, "    No Pure Negative → continue")
            _log(state, None, "")
            return

        # Per-candidate negative check
        _handler_map = step.params.get("handler_map", {})
        _model_map = step.params.get("model_map", {})
        _log(state, label, "NEGATIVE CHECK")
        skip_keys: frozenset[str] | None = None
        if state.scene_flags.get("pure_negative", False):
            pure_neg_spec = next(
                (s for s in parsed.negative_attributes if s.name == "Pure Negative" and s.enabled),
                None,
            )
            if pure_neg_spec:
                skip_keys = frozenset({pure_neg_spec.key})

        for c in state.candidates:
            if c.is_suppressed_or_rejected:
                c.record("negative", "skipped — NMS suppressed")
                _log(state, None, f"    · {c.object_id}  skipped (NMS suppressed)")
                continue
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
                    handler_map=_handler_map,
                    model_map=_model_map,
                )
                c.negative_flags.update(new_stage.negative)

            triggered = _negative_triggered(c.negative_flags)
            if triggered:
                c.record("negative", f"flags triggered: {', '.join(triggered)}")
                _log(state, None, f"    · {c.object_id}  ⚠ {', '.join(triggered)}")
            else:
                c.record("negative", "no flags")
                _log(state, None, f"    · {c.object_id}  no flags")
        _log(state, None, "")

    # ── quality_fallback (scene-level, 0 candidates) ────────────────

    async def _run_quality_fallback(
        self,
        step: PlanStep,
        ctx: _ExecutionContext,
        state: RuntimeState,
        image_path: str,
        parsed: ParsedTaskSpec,
        run_id: str,
    ) -> None:
        """Scene-level quality analysis on full image (fallback for 0 candidates)."""
        if not self._attribute_handler:
            state.scene_flags["fallback_quality"] = {}
            return
        from storage.image_crop import bbox_for_full_crop

        full_bbox = bbox_for_full_crop(image_path)
        result = await self._attribute_handler.analyze_by_scopes(
            image_path=image_path,
            bbox=full_bbox,
            parsed=parsed,
            object_id="scene_fallback",
            scopes={"quality"},
            include_keys=frozenset(step.params.get("attribute_keys", [])),
            handler_map=step.params.get("handler_map", {}),
            model_map=step.params.get("model_map", {}),
        )
        state.scene_flags["fallback_quality"] = result.quality
        _log(state, f"{step.step}:{step.model_id}", "SCENE QUALITY FALLBACK")
        for k, v in result.quality.items():
            val = v.get("value", "?") if isinstance(v, dict) else "?"
            _log(state, None, f"    · {k} = {val}")
        _log(state, None, "")

    # ── negative_fallback (scene-level, 0 candidates) ───────────────

    async def _run_negative_fallback(
        self,
        step: PlanStep,
        ctx: _ExecutionContext,
        state: RuntimeState,
        image_path: str,
        parsed: ParsedTaskSpec,
        run_id: str,
    ) -> None:
        """Scene-level negative analysis on full image (fallback for 0 candidates)."""
        if not self._attribute_handler:
            state.scene_flags["fallback_negative"] = {}
            return
        from storage.image_crop import bbox_for_full_crop

        full_bbox = bbox_for_full_crop(image_path)
        result = await self._attribute_handler.analyze_by_scopes(
            image_path=image_path,
            bbox=full_bbox,
            parsed=parsed,
            object_id="scene_fallback",
            full_image_path=image_path,
            full_bbox=full_bbox,
            scopes={"negative"},
            include_keys=frozenset(step.params.get("attribute_keys", [])),
            handler_map=step.params.get("handler_map", {}),
            model_map=step.params.get("model_map", {}),
        )
        state.scene_flags["fallback_negative"] = result.negative
        _log(state, f"{step.step}:{step.model_id}", "SCENE NEGATIVE FALLBACK")
        for k, v in result.negative.items():
            val = v.get("value", "?") if isinstance(v, dict) else "?"
            _log(state, None, f"    · {k} = {val}")
        _log(state, None, "")

    # ── full_quality (no-detector topology) ─────────────────────────

    async def _run_full_quality(
        self,
        step: PlanStep,
        ctx: _ExecutionContext,
        state: RuntimeState,
        image_path: str,
        parsed: ParsedTaskSpec,
        run_id: str,
    ) -> None:
        """Full-image quality analysis (no-detector topology)."""
        if not self._attribute_handler:
            state.scene_flags["full_quality"] = {}
            return
        from storage.image_crop import bbox_for_full_crop

        full_bbox = bbox_for_full_crop(image_path)
        result = await self._attribute_handler.analyze_by_scopes(
            image_path=image_path,
            bbox=full_bbox,
            parsed=parsed,
            object_id="scene",
            scopes={"quality"},
            include_keys=frozenset(step.params.get("attribute_keys", [])),
            # handler_map/ model_map from Planner — for quality these map to "opencv_quality"
            # unless the template attributes specifically require Gemini.
            handler_map=step.params.get("handler_map", {}),
            model_map=step.params.get("model_map", {}),
        )
        state.scene_flags["full_quality"] = result.quality
        _log(state, f"{step.step}:{step.model_id}", "FULL IMAGE QUALITY")
        for k, v in result.quality.items():
            val = v.get("value", "?") if isinstance(v, dict) else "?"
            _log(state, None, f"    · {k} = {val}")
        _log(state, None, "")

    # ── full_attribute (no-detector topology) ───────────────────────

    async def _run_full_attribute(
        self,
        step: PlanStep,
        ctx: _ExecutionContext,
        state: RuntimeState,
        image_path: str,
        parsed: ParsedTaskSpec,
        run_id: str,
    ) -> None:
        """Full-image semantic attribute analysis (no-detector topology)."""
        if not self._attribute_handler:
            state.scene_flags["full_attributes"] = {}
            return
        from storage.image_crop import bbox_for_full_crop

        full_bbox = bbox_for_full_crop(image_path)
        result = await self._attribute_handler.analyze_by_scopes(
            image_path=image_path,
            bbox=full_bbox,
            parsed=parsed,
            object_id="scene",
            scopes={"semantic"},
            include_keys=frozenset(step.params.get("attribute_keys", [])),
            handler_map=step.params.get("handler_map", {}),
            model_map=step.params.get("model_map", {}),
        )
        state.scene_flags["full_attributes"] = result.attributes
        _log(state, f"{step.step}:{step.model_id}", "FULL IMAGE ATTRIBUTES")
        for k, v in result.attributes.items():
            val = v.get("value", "?") if isinstance(v, dict) else "?"
            _log(state, None, f"    · {k} = {val}")
        _log(state, None, "")

    # ── full_negative (no-detector topology) ────────────────────────

    async def _run_full_negative(
        self,
        step: PlanStep,
        ctx: _ExecutionContext,
        state: RuntimeState,
        image_path: str,
        parsed: ParsedTaskSpec,
        run_id: str,
    ) -> None:
        """Full-image negative analysis (no-detector topology)."""
        if not self._attribute_handler:
            state.scene_flags["full_negative"] = {}
            return
        from storage.image_crop import bbox_for_full_crop

        full_bbox = bbox_for_full_crop(image_path)
        result = await self._attribute_handler.analyze_by_scopes(
            image_path=image_path,
            bbox=full_bbox,
            parsed=parsed,
            object_id="scene",
            full_image_path=image_path,
            full_bbox=full_bbox,
            scopes={"negative"},
            include_keys=frozenset(step.params.get("attribute_keys", [])),
            handler_map=step.params.get("handler_map", {}),
            model_map=step.params.get("model_map", {}),
        )
        state.scene_flags["full_negative"] = result.negative
        _log(state, f"{step.step}:{step.model_id}", "FULL IMAGE NEGATIVE")
        for k, v in result.negative.items():
            val = v.get("value", "?") if isinstance(v, dict) else "?"
            _log(state, None, f"    · {k} = {val}")
        _log(state, None, "")

    # ── merge ───────────────────────────────────────────────────────

    async def _run_merge(
        self,
        step: PlanStep,
        ctx: _ExecutionContext,
        state: RuntimeState,
        image_path: str,
        parsed: ParsedTaskSpec,
        run_id: str,
    ) -> None:
        scene_fallback = {}
        fallback_q = state.scene_flags.get("fallback_quality")
        fallback_n = state.scene_flags.get("fallback_negative")
        if fallback_q or fallback_n:
            scene_fallback = {
                "quality": fallback_q or {},
                "negative": fallback_n or {},
            }
        else:
            full_q = state.scene_flags.get("full_quality")
            full_a = state.scene_flags.get("full_attributes")
            full_n = state.scene_flags.get("full_negative")
            if full_q or full_a or full_n:
                scene_fallback = {
                    "quality": full_q or {},
                    "attributes": full_a or {},
                    "negative": full_n or {},
                }

        merge_result = await self._merger.merge(
            image_path=image_path,
            parsed=parsed,
            candidates_data=[c.to_dict() for c in state.candidates
                                if c.state is not CandidateState.SUPPRESSED],
            scene_pure_negative=state.scene_flags.get("pure_negative", False),
            scene_fallback=scene_fallback,
            run_id=run_id,
            execution_log_text="\n".join(ctx.execution_log_lines),
        )
        state._merge_result = merge_result


# ── early exit / skip (eval-based, migrated to accept RuntimeState) ──


def _check_early_exit(rules: list[EarlyExitRule], state: RuntimeState) -> str | None:
    """Return the reason string if an early-exit rule fires, else None."""
    safe_ns = _safe_eval_ns(state)
    for rule in rules:
        try:
            if eval(rule.condition, {"__builtins__": {}}, safe_ns):
                return rule.reason
        except Exception:
            continue
    return None


def _check_skip(
    conditions: list[SkipCondition],
    step: PlanStep,
    state: RuntimeState,
) -> str | None:
    """Return the reason string if a skip condition fires, else None."""
    safe_ns = _safe_eval_ns(state)
    for cond in conditions:
        if cond.step != step.step:
            continue
        try:
            if eval(cond.condition, {"__builtins__": {}}, safe_ns):
                return cond.reason
        except Exception:
            continue
    return None


def _safe_eval_ns(state: RuntimeState) -> dict[str, Any]:
    return {
        "len": len,
        "detections": [],  # kept for backward-compat eval strings
        "candidates": state.candidates,
        "scene_pure_negative": state.scene_flags.get("pure_negative", False),
        "detection_count": 0,
    }


# ── helpers ─────────────────────────────────────────────────────────


def _log(state: RuntimeState, label: str | None, line: str) -> None:
    """Append a log line to the execution log (stored in state.artifacts)."""
    log_lines: list[str] = state.artifacts.get("_log_lines", [])
    if label is not None:
        idx = len(log_lines)
        log_lines.append(f"[{idx}] {line}")
    else:
        log_lines.append(line)
    state.artifacts["_log_lines"] = log_lines


def _compute_feasibility(
    candidate: Candidate,
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
