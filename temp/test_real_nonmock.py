"""
Real non-mock pipeline test using fixture request_151049_1tasks.json.
Loads .env, forces GEMINI_MOCK=0, resolves file:// URL, runs engine, prints results.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.load_env import load_project_env
load_project_env()

# Force real API calls (override .env which has MOCK=1)
os.environ["MVP_FORCE_GEMINI_MOCK"] = "0"
os.environ["MVP_FORCE_YOLO_MOCK"] = "0"

from di.container import build_container
from storage.url_resolver import resolve_url


async def main():
    fixture_path = ROOT / "test" / "fixtures" / "request_151049_1tasks.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    # Extract template
    template = fixture["template"]
    callback_url = fixture.get("callback_url", "")
    tasks = fixture.get("tasks", [])

    print(f"Template objects: {[o['name'] for o in template['objects']]}")
    print(f"Callback URL: {callback_url}")
    print(f"Tasks: {len(tasks)}")
    print()

    # Resolve image URL from first task
    task = tasks[0]
    frame = task["frames"][0]
    url = frame["url"]
    print(f"Frame URL: {url}")
    image_path = resolve_url(url)
    img_size = Path(image_path).stat().st_size
    print(f"Resolved to: {image_path} ({img_size / 1024:.0f} KB)")
    print()

    # Save template to temp file (engine expects file path)
    temp_template = ROOT / "temp" / "_nonmock_template.json"
    temp_template.write_text(json.dumps(template, indent=2), encoding="utf-8")

    # Build engine via container (wires all deps)
    container = build_container()
    engine = container.runtime_engine

    # Run
    print("Running pipeline (real API - no mock)...")
    t0 = time.perf_counter()
    response = await engine.run(image_path=str(image_path), template_path=str(temp_template))
    elapsed = time.perf_counter() - t0

    # --- Results ---
    ann = response.annotation_result
    trace = response.runtime_trace
    meta = trace.meta

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Image:       {Path(ann.image).name}")
    print(f"Status:      completed")
    print(f"Elapsed:     {elapsed * 1000:.0f}ms")
    print(f"Object:      {meta.get('object_name', '?')}")
    print(f"Steps:       {meta.get('executed_steps', [])}")
    print(f"Merge:       {meta.get('merge_adapter', '?')}")
    print(f"Pure neg:    {meta.get('scene_pure_negative', '?')}")
    print(f"Run ID:      {meta.get('run_id', '?')}")
    print()

    print(f"Annotation objects ({len(ann.objects)}):")
    for i, obj in enumerate(ann.objects):
        print(f"  [{i}] category={obj.category}  confidence={obj.confidence:.3f}  status={obj.status}")
        print(f"       bbox={[round(v, 1) for v in obj.bbox]}")
        if obj.attributes:
            print(f"       attrs={obj.attributes}")
    print()

    print(f"Candidate history ({len(trace.candidate_history)}):")
    for ch in trace.candidate_history:
        print(f"  {ch.get('object_id', '?'):12s} state={ch.get('state','?')}  "
              f"det={ch.get('detector_score',0):.2f}  ver={ch.get('verify_score',0):.2f}")
    print()

    print(f"Merge reasoning: {len(trace.merge_reasoning)} entries")
    for entry in trace.merge_reasoning:
        print(f"  [{entry.get('step','?')}] {entry.get('reasoning','')[:100]}")
    print()

    if trace.resolved_attributes:
        print(f"Resolved attributes:")
        for k, v in trace.resolved_attributes.items():
            print(f"  {k}: {v!r}")
    print()

    print(f"Merge adapter: {meta.get('merge_adapter', '?')}")
    print(f"Merge rules: {meta.get('merge_rules', {})}")

    # Check DB recording via last_run tool
    print(f"\nDB recorded -> python -m tools.last_run --last 1")

    # Cleanup
    temp_template.unlink(missing_ok=True)
    print()
    print("Done.")

asyncio.run(main())
