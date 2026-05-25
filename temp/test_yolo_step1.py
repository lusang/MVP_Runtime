"""
Step 1 only: YOLO detect + pre-selection crops via RuntimeEngine (handler pipeline).
All downstream stages use mocks — no real API, no Gemini.

  .venv\\Scripts\\python temp\\test_yolo_step1.py
"""

from __future__ import annotations

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

from handlers.attribute_handler import AttributeHandler
from handlers.verification_handler import VerificationHandler
from handlers.registry import AttributeHandlerRegistry
from handlers.plugins.gemini_attribute import GeminiAttributePlugin
from handlers.plugins.gemini_negative import GeminiNegativePlugin
from handlers.plugins.opencv_quality import OpenCVQualityPlugin
from models.gemini_verifier import GeminiVerifier
from models.yolo_detector import YOLODetector
from runtime.engine import RuntimeEngine
from runtime.template_parser import TemplateParser

TEMPLATE = ROOT / "resource" / "Template.json"
DEFAULT_IMAGE = ROOT / "temp" / "test.jpg"


def build_engine() -> RuntimeEngine:
    registry = AttributeHandlerRegistry()
    registry.register("gemini", GeminiAttributePlugin)
    registry.register("opencv_quality", OpenCVQualityPlugin)
    registry.register("gemini_negative", GeminiNegativePlugin)
    return RuntimeEngine(attribute_registry=registry)


async def main() -> int:
    img = Path(os.environ.get("MVP_IMAGE_PATH", str(DEFAULT_IMAGE))).resolve()

    raw = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    parsed = TemplateParser().parse(raw)

    print("--- env (YOLO via RuntimeEngine) ---")
    yolo_path = os.environ.get("YOLO_MODEL_PATH") or "(empty → download)"
    print(f"YOLO_MODEL_PATH      = {yolo_path}")
    print(f"YOLO_CONF_THRESHOLD  = {os.environ.get('YOLO_CONF_THRESHOLD', '0.10')}")
    print(f"YOLO_DEVICE          = {os.environ.get('YOLO_DEVICE', '0')}")
    print(f"MVP_FORCE_YOLO_MOCK  = {os.environ.get('MVP_FORCE_YOLO_MOCK', '0')}")
    print(f"object_name          = {parsed.object_name}")
    print(f"include              = {parsed.include}")
    print(f"image                = {img}")
    print(f"template             = {TEMPLATE}")
    print()

    engine = build_engine()
    response = await engine.run(
        image_path=str(img),
        template_path=str(TEMPLATE),
    )

    detections = response.meta.get("detections", [])
    print(f"candidates: {len(detections)}")
    for i, d in enumerate(detections):
        print(f"  [{i}] {json.dumps(d, ensure_ascii=False)}")
    print(f"preselection_dir: {response.meta.get('preselection_dir')}")
    print(f"elapsed_ms:        {response.meta.get('elapsed_ms')}")

    log_path = ROOT / "debug-ea8e9c.log"
    if log_path.is_file():
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        for line in lines[-3:]:
            print(f"debug: {line[:250]}...")

    if not detections:
        print(
            "\nNote: 0 detections on this image is normal for a blank/gray test.jpg. "
            "Set MVP_IMAGE_PATH to a real photo with packages and re-run."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
