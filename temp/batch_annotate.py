"""
Batch annotation: compile plan once, process all images in resource/package/, output one JSON.

  .venv\\Scripts\\python temp\\batch_annotate.py --mock          # fast test
  .venv\\Scripts\\python temp\\batch_annotate.py                  # real API
  .venv\\Scripts\\python temp\\batch_annotate.py --output out.json
  .venv\\Scripts\\python temp\\batch_annotate.py --include-trace  # also save runtime_trace
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.load_env import load_project_env

load_project_env()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch annotation over resource/package/")
    p.add_argument("--mock", action="store_true", help="Force mock mode (no API keys)")
    p.add_argument("--output", type=str, default=None, help="Output JSON path (default: temp/batch_result.json)")
    p.add_argument("--include-trace", action="store_true", help="Include runtime_trace per image (large output)")
    p.add_argument("--limit", type=int, default=0, help="Max images to process (0 = all)")
    p.add_argument("--pos-only", action="store_true", help="Only pos images")
    p.add_argument("--neg-only", action="store_true", help="Only neg images")
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
from runtime.planner import Planner
from runtime.template_parser import TemplateParser
from storage.io import read_json_dict


def build_engine() -> RuntimeEngine:
    registry = AttributeHandlerRegistry()
    registry.register("gemini", GeminiAttributePlugin)
    registry.register("opencv_quality", OpenCVQualityPlugin)
    registry.register("gemini_negative", GeminiNegativePlugin)
    return RuntimeEngine(attribute_registry=registry)


def collect_images(root: Path, pos_only: bool, neg_only: bool, limit: int) -> list[Path]:
    """Gather .jpeg images from resource/package/pos and/or neg."""
    images: list[Path] = []

    if not neg_only:
        pos_dir = root / "resource" / "package" / "pos"
        if pos_dir.is_dir():
            imgs = sorted(p for p in pos_dir.glob("*.jpeg") if p.is_file())
            images.extend(imgs)

    if not pos_only:
        neg_dir = root / "resource" / "package" / "neg"
        if neg_dir.is_dir():
            imgs = sorted(p for p in neg_dir.glob("*.jpeg") if p.is_file())
            images.extend(imgs)

    if limit and limit > 0:
        images = images[:limit]

    return images


async def main():
    args = parse_args()

    if args.mock:
        _apply_mock_env()

    template_path = ROOT / "resource" / "Template.json"
    if not template_path.is_file():
        print(f"Template not found: {template_path}")
        return 1

    images = collect_images(ROOT, args.pos_only, args.neg_only, args.limit)
    if not images:
        print("No images found in resource/package/")
        return 1

    output_path = Path(args.output or str(ROOT / "temp" / "batch_result.json"))

    # ── Compile plan once ──────────────────────────────────────────
    print(f"Loading template: {template_path}")
    raw_template = await read_json_dict(str(template_path))
    parser = TemplateParser()
    parsed = parser.parse(raw_template)
    plan = Planner.compile(parsed)

    print(f"Plan compiled: {plan.plan_id}")
    print(f"  planner: {plan.planner_model}")
    print(f"  steps:   {[f'{s.step}:{s.model_id}' for s in plan.steps]}")
    print(f"  object:  {parsed.object_name}")
    print()

    # ── Build engine ───────────────────────────────────────────────
    engine = build_engine()
    total = len(images)
    mock_str = " [MOCK]" if args.mock else ""
    include_trace = args.include_trace

    print(f"Processing {total} images{mock_str}...")
    print(f"Output: {output_path}")
    print(f"Include trace: {include_trace}")
    print()

    # ── Process each image ─────────────────────────────────────────
    results: list[dict] = []
    errors: list[dict] = []
    t0 = time.perf_counter()

    for i, image_path in enumerate(images):
        pct = (i + 1) / total * 100
        label = f"[{i+1}/{total}] {image_path.name}"
        print(f"\r{pct:5.1f}% {label:<80}", end="", flush=True)

        try:
            response = await engine.run_with_plan(
                image_path=str(image_path),
                plan=plan,
                parsed=parsed,
                template_name="Template",
                template_path=str(template_path),
            )

            entry: dict = {
                "image": str(image_path),
                "annotation_result": {
                    "image": response.annotation_result.image,
                    "objects": [obj.model_dump() for obj in response.annotation_result.objects],
                },
            }

            if include_trace:
                entry["runtime_trace"] = {
                    "steps": response.runtime_trace.steps,
                    "merge_reasoning": response.runtime_trace.merge_reasoning,
                    "annotation_panel": response.runtime_trace.annotation_panel,
                    "meta": response.runtime_trace.meta,
                }

            results.append(entry)

        except Exception as exc:
            errors.append({
                "image": str(image_path),
                "error": type(exc).__name__,
                "detail": str(exc)[:500],
            })
            print(f"\n  ERROR: {type(exc).__name__}: {str(exc)[:200]}")

    elapsed = time.perf_counter() - t0
    print(f"\r{'':<90}")
    print(f"Done in {elapsed:.1f}s ({elapsed/total:.2f}s/image)")
    print(f"  success: {len(results)}")
    print(f"  errors:  {len(errors)}")

    # ── Write output ───────────────────────────────────────────────
    output = {
        "template": str(template_path),
        "object_name": parsed.object_name,
        "plan_id": plan.plan_id,
        "planner_model": plan.planner_model,
        "plan_steps": [f"{s.step}:{s.model_id}" for s in plan.steps],
        "total_images": total,
        "success_count": len(results),
        "error_count": len(errors),
        "elapsed_sec": round(elapsed, 1),
        "mock_mode": args.mock,
        "results": results,
    }
    if errors:
        output["errors"] = errors

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    file_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Output: {output_path} ({file_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
