"""
Runtime engine — parse → plan → execute → build response.
Uses Planner + StepExecutor when enabled; falls back to static plan otherwise.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from handlers.attribute_handler import AttributeHandler
from handlers.verification_handler import VerificationHandler
from handlers.registry import AttributeHandlerRegistry
from models.gemini_merger import GeminiMerger
from models.gemini_verifier import GeminiVerifier
from models.yolo_detector import YOLODetector
from debug_log import agent_log
from runtime.object_state_builder import ObjectStateBuilder
from runtime.performance_tracker import PerformanceTracker
from runtime.planner import Planner
from runtime.step_executor import ExecutionResult, StepExecutor
from runtime.template_parser import TemplateParser
from schemas.api import AnnotationResult, AnnotationRunResponse, RuntimeTrace
from schemas.pipeline_plan import PipelinePlan
from storage.io import assert_path_exists, read_json_dict


class RuntimeEngine:
    """Orchestrates: parse template → plan pipeline → execute → build response."""

    def __init__(
        self,
        *,
        attribute_registry: AttributeHandlerRegistry,
        detector: YOLODetector | None = None,
        verifier: GeminiVerifier | None = None,
        verification_handler: VerificationHandler | None = None,
        attribute_handler: AttributeHandler | None = None,
        template_parser: TemplateParser | None = None,
        merger: GeminiMerger | None = None,
        planner: Planner | None = None,
        tracker: PerformanceTracker | None = None,
        use_planner: bool = True,
    ) -> None:
        self._template_parser = template_parser or TemplateParser()
        self._detector = detector or YOLODetector()
        self._verifier = verifier or GeminiVerifier()
        self._verification_handler = verification_handler or VerificationHandler(self._verifier)
        self._attribute_handler = attribute_handler or AttributeHandler(attribute_registry)
        self._merger = merger or GeminiMerger()
        self._planner = planner or Planner()
        self._tracker = tracker
        self._use_planner = use_planner

    async def run(self, *, image_path: str, template_path: str) -> AnnotationRunResponse:
        assert_path_exists(image_path, kind="image_path")
        assert_path_exists(template_path, kind="template_path")
        raw_template = await read_json_dict(template_path)
        parsed = self._template_parser.parse(raw_template)

        # Compile plan once from template (no Gemini, no replanning)
        plan = Planner.compile(parsed)
        template_name = Path(template_path).stem

        return await self.run_with_plan(
            image_path=image_path,
            plan=plan,
            parsed=parsed,
            template_name=template_name,
            template_path=template_path,
        )

    async def run_with_plan(
        self,
        *,
        image_path: str,
        plan: PipelinePlan,
        parsed: Any,
        template_name: str = "",
        template_path: str = "",
    ) -> AnnotationRunResponse:
        """Execute a pre-compiled plan against a single image.

        For batch execution, compile the plan once via Planner.compile(template)
        then call this method for each image with the same plan.
        """
        t0 = time.perf_counter()
        run_id = str(uuid.uuid4())

        agent_log(
            hypothesis_id="H1",
            location="runtime/engine.py:run_with_plan",
            message="pipeline_start",
            data={"object_name": parsed.object_name, "image_path": image_path},
            run_id=run_id,
        )

        agent_log(
            hypothesis_id="H10",
            location="runtime/engine.py:run_with_plan",
            message="pipeline_plan",
            data={
                "plan_id": plan.plan_id,
                "planner_model": plan.planner_model,
                "steps": [f"{s.step}:{s.model_id}" for s in plan.steps],
            },
            run_id=run_id,
        )

        # --- Execute ---
        executor = StepExecutor(
            detector=self._detector,
            verifier=self._verifier,
            verification_handler=self._verification_handler,
            attribute_handler=self._attribute_handler,
            merger=self._merger,
            tracker=self._tracker,
        )
        result = await executor.execute(
            plan=plan,
            image_path=image_path,
            parsed=parsed,
            run_id=run_id,
            template_name=template_name,
        )

        # --- Build AnnotationResult (clean, for annotation platforms) ---
        merge_objects = result.merge_result.get("objects", [])
        annotation_objects = []
        for i, candidate in enumerate(result.candidates):
            panel = merge_objects[i] if i < len(merge_objects) else None
            anno_obj = ObjectStateBuilder.build_annotation_object(
                candidate=candidate,
                object_name=parsed.object_name,
                merge_panel=panel,
                scene_pure_negative=result.scene_pure_negative,
            )
            annotation_objects.append(anno_obj)

        annotation_result = AnnotationResult(
            image=image_path,
            objects=annotation_objects,
        )

        # --- Build RuntimeTrace (debug/trace for system analysis) ---
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        quality_scores: list[dict[str, Any]] = []
        for c in result.candidates:
            quality_scores.append({
                "object_id": c.object_id,
                "visibility": c.visibility,
                "metrics": c.metrics,
                "attribute_feasibility": c.attribute_feasibility,
                "missing_attributes": c.missing_attributes,
            })

        annotation_panel: dict[str, Any] | None = None
        if merge_objects:
            annotation_panel = {
                obj.get("object_id", f"obj_{i}"): obj
                for i, obj in enumerate(merge_objects)
            }

        preselection_dir = ""
        if hasattr(self._detector, "_preselection_root"):
            preselection_dir = str(Path(self._detector._preselection_root) / run_id)

        runtime_trace = RuntimeTrace(
            steps=[{"step_id": s} for s in result.executed_steps],
            candidate_history=[c.to_dict() for c in result.candidates],
            planner_decisions={
                "plan_id": plan.plan_id,
                "planner_model": plan.planner_model,
                "planner_version": plan.planner_version,
                "steps": [s.model_dump() for s in plan.steps],
                "early_exit_rules": [r.model_dump() for r in plan.early_exit_rules],
                "skip_conditions": [c.model_dump() for c in plan.skip_conditions],
            },
            quality_scores=quality_scores,
            merge_reasoning=result.merge_result.get("reasoning_trace", []),
            annotation_panel=annotation_panel,
            meta={
                "run_id": run_id,
                "engine": "RuntimeEngine",
                "version": "mvp-2.0",
                "elapsed_ms": round(elapsed_ms, 3),
                "plan_id": plan.plan_id,
                "planner_model": plan.planner_model,
                "object_name": parsed.object_name,
                "template_path": template_path or image_path,
                "scene_pure_negative": result.scene_pure_negative,
                "executed_steps": result.executed_steps,
                "merge_adapter": result.merge_result.get("adapter", "unknown"),
                "preselection_dir": preselection_dir,
            },
        )

        return AnnotationRunResponse(
            annotation_result=annotation_result,
            runtime_trace=runtime_trace,
        )
