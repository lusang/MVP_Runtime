"""
Detailed step-by-step trace of the pipeline execution.
Shows what each step does with the fixture template.
"""
import json, sys, os
sys.path.insert(0, ".")

os.environ["MVP_FORCE_YOLO_MOCK"] = "1"
os.environ["MVP_FORCE_GEMINI_MOCK"] = "1"

from pathlib import Path
from runtime.planner import compile_plan
from runtime.template_parser import TemplateParser
from runtime.step_executor import StepExecutor
from schemas.candidate_state import CandidateState
from schemas.bbox import BBox
from schemas.template_spec import ParsedTaskSpec
from runtime.performance_tracker import PerformanceTracker
from handlers.attribute_handler import AttributeHandler
from handlers.verification_handler import VerificationHandler
from handlers.registry import AttributeHandlerRegistry
from models.gemini_merger import GeminiMerger
from models.gemini_verifier import GeminiVerifier
from models.yolo_detector import YOLODetector
from handlers.plugins.gemini_attribute import GeminiAttributePlugin
from handlers.plugins.gemini_negative import GeminiNegativePlugin
from handlers.plugins.opencv_quality import OpenCVQualityPlugin
from models.opencv_analyzer import OpenCVAnalyzer

fixture = json.loads(open("test/fixtures/request_151049_1tasks.json", encoding="utf-8").read())
parsed = TemplateParser().parse(fixture["template"])
plan = compile_plan(parsed)

# Build executor (same as di/container.py would)
verifier = GeminiVerifier()
analyzer = OpenCVAnalyzer()
registry = AttributeHandlerRegistry()
registry.register("gemini", lambda: GeminiAttributePlugin(verifier))
registry.register("gemini_negative", lambda: GeminiNegativePlugin(verifier))
registry.register("opencv_quality", lambda: OpenCVQualityPlugin(analyzer))

tracker = PerformanceTracker()
tracker.ensure_schema()

executor = StepExecutor(
    detector=YOLODetector(),
    verifier=verifier,
    verification_handler=VerificationHandler(verifier),
    attribute_handler=AttributeHandler(registry),
    merger=GeminiMerger(),
    tracker=tracker,
)

import asyncio

async def trace():
    from runtime.step_executor import _ExecutionContext, _log, _compute_feasibility
    from schemas.feasibility import build_feasibility_rules
    from runtime.nms import apply_nms
    from storage.image_crop import bbox_for_full_crop
    from tests.test_helpers import write_minimal_jpeg

    image_path = "temp/_trace_test_img.jpg"
    Path(image_path).parent.mkdir(parents=True, exist_ok=True)
    write_minimal_jpeg(Path(image_path), width=640, height=480)

    print("=" * 70)
    print("PIPELINE EXECUTION TRACE  (fixture: request_151049_1tasks.json)")
    print("=" * 70)
    print()

    ctx = _ExecutionContext()

    for step in plan.steps:
        print("─" * 70)
        print(f"STEP [{step.order}] {step.step.upper()}  model={step.model_id}  flow={step.data_flow.value}")
        print(f"  params: {step.params}")
        print()

        if step.step == "detect":
            detections = await executor._detector.detect(
                image_path, target_object=parsed.object_name, parsed=parsed, run_id="trace")
            ctx.detections = detections
            for idx, det in enumerate(detections):
                bbox = det.bbox
                analysis_path = det.crop_path or image_path
                analysis_bbox = bbox_for_full_crop(analysis_path) if det.crop_path else bbox
                c = CandidateState(object_id=f"obj_{idx}", detector_score=det.score, bbox=bbox,
                    crop_path=det.crop_path, analysis_path=analysis_path, analysis_bbox=analysis_bbox)
                ctx.candidates.append(c)
            for c in ctx.candidates:
                print(f"  => {c.object_id}: score={c.detector_score:.2f}  bbox=({c.bbox.x1:.0f},{c.bbox.y1:.0f},{c.bbox.x2:.0f},{c.bbox.y2:.0f})  exists={c.exists}")
            print()

        elif step.step == "nms":
            before = sum(1 for c in ctx.candidates if c.exists)
            apply_nms(ctx.candidates, iou_threshold=0.5)
            after = sum(1 for c in ctx.candidates if c.exists)
            for c in ctx.candidates:
                nms_h = [h for h in c.analysis_history if h["step"] == "nms"]
                tag = "SUPPRESSED" if not c.exists else "KEPT"
                info = f"  => {c.object_id}: {tag}"
                if nms_h:
                    info += f"  ({nms_h[0]['decision']})"
                print(info)
            print()

        elif step.step == "verify":
            for c in ctx.candidates:
                if not c.exists:
                    print(f"  => {c.object_id}: SKIPPED (NMS suppressed)")
                    continue
                ver = await executor._verification_handler.verify_object(
                    image_path=c.analysis_path, bbox=c.analysis_bbox, parsed=parsed, object_id=c.object_id)
                c.verification = ver
                c.verify_score = float(ver.get("score", 0.0))
                c.compute_confidence()
                if ver.get("ok") is False:
                    c.exists = False
                    print(f"  => {c.object_id}: REJECTED  score={c.verify_score:.2f}")
                else:
                    c.exists = True
                    print(f"  => {c.object_id}: VERIFIED  ok score={c.verify_score:.2f}  rationale={ver.get('rationale','')[:60]}")
            print()

        elif step.step == "quality":
            for c in ctx.candidates:
                if not c.exists:
                    print(f"  => {c.object_id}: SKIPPED")
                    continue
                result = await executor._attribute_handler.analyze_by_scopes(
                    image_path=c.analysis_path, bbox=c.analysis_bbox, parsed=parsed, object_id=c.object_id, scopes={"quality"})
                c.quality = result.quality
                for kval, qitem in c.quality.items():
                    if isinstance(qitem, dict) and "metrics" in qitem:
                        c.metrics.update(qitem["metrics"])
                c.visibility = dict(c.quality)
                _compute_feasibility(c, ctx.feasibility_rules)
                qv = {k: v.get("value", "?") for k, v in c.quality.items()}
                print(f"  => {c.object_id}: quality={qv}")
                print(f"     visibility={c.visibility}")
                if c.missing_attributes:
                    print(f"     infeasible: {c.missing_attributes}")
                else:
                    print(f"     all feasible")
            print()

        elif step.step == "attribute":
            include_keys = frozenset(step.params.get("attribute_keys", []))
            is_full = step.data_flow.value == "full_image"
            for c in ctx.candidates:
                if not c.exists:
                    print(f"  => {c.object_id}: SKIPPED")
                    continue
                skip_keys = frozenset(c.missing_attributes) if c.missing_attributes else None
                effective_img = c.analysis_path if not is_full else image_path
                effective_bb = c.analysis_bbox if not is_full else c.bbox
                result = await executor._attribute_handler.analyze_by_scopes(
                    image_path=effective_img, bbox=effective_bb, parsed=parsed,
                    object_id=c.object_id, scopes={"semantic"}, skip_keys=skip_keys, include_keys=include_keys)
                c.attributes.update(result.attributes)
                for key in c.missing_attributes:
                    c.attributes[key] = {"value": None, "confidence": 0.0, "infeasible": True}
                ap = {k: {"v": v.get("value"), "c": round(v.get("confidence",0),2)} for k,v in c.attributes.items()}
                print(f"  => {c.object_id}: attrs={ap}")
            print()

        elif step.step == "negative":
            for c in ctx.candidates:
                if not c.exists:
                    print(f"  => {c.object_id}: SKIPPED")
                    continue
                result = await executor._attribute_handler.analyze_by_scopes(
                    image_path=c.analysis_path, bbox=c.analysis_bbox, parsed=parsed,
                    object_id=c.object_id, full_image_path=image_path, full_bbox=c.bbox, scopes={"negative"})
                c.negative_flags.update(result.negative)
                np = {k: v.get("value") for k, v in c.negative_flags.items()}
                print(f"  => {c.object_id}: negative_flags={np}")
            print()

        elif step.step == "merge":
            candidates_data = [c.to_dict() for c in ctx.candidates]
            merge_result = await executor._merger.merge(
                image_path=image_path, parsed=parsed, candidates_data=candidates_data,
                scene_pure_negative=ctx.scene_pure_negative, run_id="trace", execution_log_text="")
            print(f"  => adapter: {merge_result.get('adapter')}")
            print(f"  => merge_rules: {merge_result.get('merge_rules')}")
            print(f"  => objects ({len(merge_result.get('objects',[]))}):")
            for obj in merge_result.get("objects", []):
                print(f"     {obj['object_id']}: {'POS' if obj['is_positive'] else 'NEG'}  conf={obj['merge_confidence']:.3f}  attrs={obj.get('attributes',{})}")
            print(f"  => resolved_attributes:")
            for k, v in merge_result.get("resolved_attributes", {}).items():
                u = " UNCERTAIN" if v.get("uncertain") else ""
                print(f"     {k}: value={v['value']!r}  conf={v['confidence']:.2f}{u}")

    print()
    print("=" * 70)
    print("CANDIDATE STATE (final)")
    print("=" * 70)
    for c in ctx.candidates:
        print(f"  {c.object_id}: exists={c.exists}  det_score={c.detector_score:.2f}  verify_score={c.verify_score:.2f}  confidence={c.confidence:.3f}")
        print(f"    history:")
        for h in c.analysis_history:
            print(f"      [{h['step']}] {h['decision']}")

asyncio.run(trace())
