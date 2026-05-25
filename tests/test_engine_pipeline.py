"""Integration test for refactored RuntimeEngine pipeline."""

import asyncio
import json
from pathlib import Path

from di.container import build_container
from tests.test_helpers import write_minimal_jpeg

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "resource" / "Template.json"


def test_engine_pipeline_returns_split_response():
    async def _run():
        img = ROOT / "temp" / "_test_engine.jpg"
        img.parent.mkdir(parents=True, exist_ok=True)
        write_minimal_jpeg(img)
        try:
            engine = build_container().runtime_engine
            result = await engine.run(image_path=str(img), template_path=str(TEMPLATE))

            # --- AnnotationResult assertions ---
            ann = result.annotation_result
            assert ann.image == str(img)
            assert len(ann.objects) == 2
            obj = ann.objects[0]
            assert obj.category == "Package"
            assert len(obj.bbox) == 4
            assert obj.status in ("accepted", "rejected", "pending")
            assert isinstance(obj.confidence, float)

            # --- RuntimeTrace assertions ---
            trace = result.runtime_trace
            assert len(trace.steps) > 0
            assert any("detect" in s["step_id"] for s in trace.steps)
            assert any("merge" in s["step_id"] for s in trace.steps)
            assert len(trace.candidate_history) == 2
            assert "plan_id" in trace.planner_decisions
            assert "planner_model" in trace.planner_decisions
            assert len(trace.quality_scores) == 2
            assert "run_id" in trace.meta
            assert trace.meta["object_name"] == "Package"

            # Candidate history should still have full debug data
            ch0 = trace.candidate_history[0]
            assert "attributes" in ch0
            assert "quality" in ch0
            assert "negative_flags" in ch0
        finally:
            if img.is_file():
                img.unlink()

    asyncio.run(_run())
