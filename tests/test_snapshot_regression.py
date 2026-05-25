"""
Snapshot regression test — compares full pipeline output against a golden reference.

This test verifies that the Planner + StepExecutor produce identical results
for a fixed input (fixture JSON + mock models). Any change in the snapshot
indicates a behavioral change that must be reviewed.

Usage:
    # First run — generates snapshot files
    pytest tests/test_snapshot_regression.py --snapshot-update

    # Normal run — compares against existing snapshots
    pytest tests/test_snapshot_regression.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Allow running without conftest.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runtime.planner import compile_plan
from runtime.template_parser import TemplateParser
from schemas.pipeline_plan import DataFlow, PipelinePlan

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "test" / "fixtures" / "request_151049_1tasks.json"
SNAPSHOT_DIR = ROOT / "test" / "snapshots"
PLAN_SNAPSHOT = SNAPSHOT_DIR / "plan_snapshot.json"


# ── Helpers ────────────────────────────────────────────────────────────────


def _load_fixture() -> "ParsedTaskSpec":
    from schemas.template_spec import ParsedTaskSpec
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return TemplateParser().parse(raw["template"])


def _serialize_plan(plan: PipelinePlan) -> dict:
    """Serialize a PipelinePlan into a deterministic JSON-safe dict.
    Filters out non-deterministic fields (plan_id, planner_model, version).
    """
    steps = []
    for s in sorted(plan.steps, key=lambda x: x.order):
        steps.append({
            "order": s.order,
            "step": s.step,
            "model_id": s.model_id,
            "data_flow": s.data_flow.value,
            "per_candidate": s.per_candidate,
            "scope": s.scope,
            "params": s.params,
        })

    return {
        "object_name": plan.object_name,
        "steps": steps,
        "early_exit_rules": [
            {"condition": r.condition, "reason": r.reason}
            for r in plan.early_exit_rules
        ],
        "skip_conditions": [
            {"condition": c.condition, "reason": c.reason}
            for c in plan.skip_conditions
        ],
    }


def _serialize_runtime_result(result) -> dict:
    """Serialize execution result into a deterministic JSON-safe dict.
    Filters out file paths, object_ids, and other non-deterministic fields.
    """
    candidates = []
    for c in result.candidates:
        candidates.append({
            "exists": c.exists,
            "detector_score": round(c.detector_score, 4) if hasattr(c, "detector_score") else None,
            "verify_score": round(c.verify_score, 4) if hasattr(c, "verify_score") else None,
            "confidence": round(c.confidence, 4) if hasattr(c, "confidence") else None,
            "quality": _simplify_quality(c.quality) if hasattr(c, "quality") and c.quality else {},
            "attributes": _simplify_attributes(c.attributes) if hasattr(c, "attributes") and c.attributes else {},
            "negative_flags": _simplify_negative(c.negative_flags) if hasattr(c, "negative_flags") and c.negative_flags else {},
        })

    merge_result = result.merge_result if hasattr(result, "merge_result") else {}
    merge_simple = {
        "objects": [
            {
                "is_positive": o.get("is_positive"),
                "merge_confidence": round(o.get("merge_confidence", 0), 4),
            }
            for o in merge_result.get("objects", [])
        ],
        "resolved_attributes": merge_result.get("resolved_attributes", {}),
    }

    return {
        "scene_pure_negative": result.scene_pure_negative,
        "executed_steps": result.executed_steps,
        "candidates": candidates,
        "merge": merge_simple,
    }


def _simplify_quality(q: dict) -> dict:
    return {
        k: v.get("value") if isinstance(v, dict) else v
        for k, v in q.items()
    }


def _simplify_attributes(a: dict) -> dict:
    return {
        k: {"value": v.get("value"), "confidence": round(v.get("confidence", 0), 4)}
        if isinstance(v, dict) else v
        for k, v in a.items()
    }


def _simplify_negative(n: dict) -> dict:
    return {
        k: v.get("value") if isinstance(v, dict) else v
        for k, v in n.items()
    }


# ─── Snapshot update flag ───────────────────────────────────────────────────


def _update_snapshots() -> bool:
    return os.environ.get("SNAPSHOT_UPDATE", "0").strip() in ("1", "true", "yes")


# ── Plan Snapshot Test ──────────────────────────────────────────────────────


def test_plan_snapshot():
    """Planner.compile() output must match golden reference."""
    parsed = _load_fixture()
    plan = compile_plan(parsed)
    serialized = _serialize_plan(plan)

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    if _update_snapshots() or not PLAN_SNAPSHOT.exists():
        PLAN_SNAPSHOT.write_text(
            json.dumps(serialized, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return

    expected = json.loads(PLAN_SNAPSHOT.read_text(encoding="utf-8"))
    assert serialized == expected, (
        f"Plan snapshot mismatch! "
        f"If the change is intentional, re-generate with: "
        f"SNAPSHOT_UPDATE=1 pytest {__file__}"
    )


# ── Runtime Snapshot Test ───────────────────────────────────────────────────

import asyncio


def test_runtime_snapshot():
    """Full pipeline execution output must match golden reference."""
    asyncio.run(_run_runtime_snapshot())


async def _run_runtime_snapshot():
    """Async body for runtime snapshot test."""
    os.environ["MVP_FORCE_YOLO_MOCK"] = "1"
    os.environ["MVP_FORCE_GEMINI_MOCK"] = "1"

    from runtime.step_executor import StepExecutor
    from handlers.attribute_handler import AttributeHandler
    from handlers.verification_handler import VerificationHandler
    from handlers.registry import AttributeHandlerRegistry
    from handlers.plugins.gemini_attribute import GeminiAttributePlugin
    from handlers.plugins.gemini_negative import GeminiNegativePlugin
    from handlers.plugins.opencv_quality import OpenCVQualityPlugin
    from models.gemini_merger import GeminiMerger
    from models.gemini_verifier import GeminiVerifier
    from models.yolo_detector import YOLODetector
    from models.opencv_analyzer import OpenCVAnalyzer
    from tests.test_helpers import write_minimal_jpeg

    parsed = _load_fixture()
    plan = compile_plan(parsed)

    # Build executor with mocks
    verifier = GeminiVerifier()
    analyzer = OpenCVAnalyzer()
    registry = AttributeHandlerRegistry()
    registry.register("gemini", lambda: GeminiAttributePlugin(verifier))
    registry.register("gemini_negative", lambda: GeminiNegativePlugin(verifier))
    registry.register("opencv_quality", lambda: OpenCVQualityPlugin(analyzer))

    executor = StepExecutor(
        detector=YOLODetector(),
        verifier=verifier,
        verification_handler=VerificationHandler(verifier),
        attribute_handler=AttributeHandler(registry),
        merger=GeminiMerger(),
    )

    image_path = str(ROOT / "temp" / "_snapshot_test_img.jpg")
    Path(image_path).parent.mkdir(parents=True, exist_ok=True)
    write_minimal_jpeg(Path(image_path), width=640, height=480)

    try:
        result = await executor.execute(
            plan=plan,
            image_path=image_path,
            parsed=parsed,
            run_id="snapshot_test",
        )

        serialized = _serialize_runtime_result(result)

        runtime_snapshot = SNAPSHOT_DIR / "runtime_snapshot.json"
        runtime_snapshot.parent.mkdir(parents=True, exist_ok=True)

        if _update_snapshots() or not runtime_snapshot.exists():
            runtime_snapshot.write_text(
                json.dumps(serialized, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return

        expected = json.loads(runtime_snapshot.read_text(encoding="utf-8"))
        assert serialized == expected, (
            f"Runtime snapshot mismatch! "
            f"If the change is intentional, re-generate with: "
            f"SNAPSHOT_UPDATE=1 pytest {__file__}"
        )
    finally:
        img = Path(image_path)
        if img.exists():
            img.unlink()
