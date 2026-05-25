"""YOLO detect + pre-selection crop tests."""

import asyncio
import json
from pathlib import Path

from models.yolo_detector import YOLODetector
from runtime.template_parser import TemplateParser
from tests.test_helpers import write_minimal_jpeg

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "resource" / "Template.json"


def test_yolo_returns_crops_for_candidates():
    async def _run():
        raw = json.loads(TEMPLATE.read_text(encoding="utf-8"))
        parsed = TemplateParser().parse(raw)
        img = ROOT / "temp" / "_yolo_test.jpg"
        img.parent.mkdir(parents=True, exist_ok=True)
        write_minimal_jpeg(img)
        try:
            det = YOLODetector(preselection_root=ROOT / "temp" / "preselection_test")
            cands = await det.detect(
                str(img),
                target_object=parsed.object_name,
                parsed=parsed,
                run_id="test_run",
            )
            assert len(cands) >= 1
            assert cands[0].target_object == "Package"
            assert cands[0].crop_path is not None
            assert Path(cands[0].crop_path).is_file()
        finally:
            if img.is_file():
                img.unlink()

    asyncio.run(_run())
