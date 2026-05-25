"""
Print final merge JSON result only.

  .venv\\Scripts\\python temp\\test_merge_output.py
  .venv\\Scripts\\python temp\\test_merge_output.py --mock
  .venv\\Scripts\\python temp\\test_merge_output.py --image resource/package/neg/006a7b0c5224b8c712160cdcfa460862.jpeg
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.load_env import load_project_env
load_project_env()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run annotation pipeline and print merge JSON")
    p.add_argument("--mock", action="store_true", help="Force mock mode (no API keys needed)")
    p.add_argument("--image", type=str, default=None)
    p.add_argument("--template", type=str, default=None)
    return p.parse_args()


def _apply_mock_env() -> None:
    os.environ["MVP_FORCE_GEMINI_MOCK"] = "1"
    os.environ["MVP_FORCE_YOLO_MOCK"] = "1"
    os.environ["MVP_DISABLE_PLANNER"] = "1"


from handlers.registry import AttributeHandlerRegistry
from handlers.plugins.gemini_attribute import GeminiAttributePlugin
from handlers.plugins.gemini_negative import GeminiNegativePlugin
from handlers.plugins.opencv_quality import OpenCVQualityPlugin
from runtime.engine import RuntimeEngine


def build_engine() -> RuntimeEngine:
    registry = AttributeHandlerRegistry()
    registry.register("gemini", GeminiAttributePlugin)
    registry.register("opencv_quality", OpenCVQualityPlugin)
    registry.register("gemini_negative", GeminiNegativePlugin)
    return RuntimeEngine(attribute_registry=registry)


async def main():
    args = parse_args()

    if args.mock:
        _apply_mock_env()

    engine = build_engine()

    if args.image:
        image_path = args.image
        if not Path(image_path).is_absolute():
            image_path = str(ROOT / image_path)
    else:
        pos_dir = ROOT / "resource" / "package" / "pos"
        images = sorted(pos_dir.glob("*.jpeg"))
        if not images:
            print("No test images found in resource/package/pos/")
            return
        image_path = str(images[0])

    template_path = args.template or str(ROOT / "resource" / "Template.json")

    if not Path(image_path).is_file():
        print(f"Image not found: {image_path}")
        return
    if not Path(template_path).is_file():
        print(f"Template not found: {template_path}")
        return

    response = await engine.run(image_path=image_path, template_path=template_path)

    # Build clean merge result JSON
    trace = response.runtime_trace
    output = {
        "image": response.annotation_result.image,
        "objects": [obj.model_dump() for obj in response.annotation_result.objects],
        "annotation_panel": trace.annotation_panel,
        "merge_reasoning": trace.merge_reasoning,
        "meta": {
            "run_id": trace.meta.get("run_id"),
            "elapsed_ms": trace.meta.get("elapsed_ms"),
            "plan_id": trace.meta.get("plan_id"),
            "planner_model": trace.meta.get("planner_model"),
            "scene_pure_negative": trace.meta.get("scene_pure_negative"),
            "executed_steps": trace.meta.get("executed_steps"),
        },
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
