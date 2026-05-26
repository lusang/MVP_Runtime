"""
Test the full-image pipeline (no YOLO) against a real image + template.

Usage:
    .venv\Scripts\python.exe temp\test_full_image.py
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

# ── 1. Load env + set full-image mode ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.load_env import load_project_env
load_project_env()

os.environ["MVP_USE_DETECTOR"] = "0"       # full-image mode (no YOLO)
os.environ["MVP_DISABLE_PLANNER"] = "1"    # use compile_plan (deterministic)

# ── 2. Download image from URL ──────────────────────────────────────────
IMAGE_URL = (
    "http://10.8.0.68:8080/api/v1/media/by-file-path"
    "?file_path=C%3A%5CUsers%5CAdministrator%5CDocuments"
    "%5C01_SECURITY_ASSETS%5C_imported"
    "%5C45e46d57f1a7407591a57214ed47ead0%5Cpos"
    "%5C92dc5e7c55efa0944f78f977e958f6d2.jpeg"
)
TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "temp" / "show_temple.json"

print("=" * 70)
print("Downloading image...")
import urllib.request
tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
try:
    urllib.request.urlretrieve(IMAGE_URL, tmp.name)
    img_size = Path(tmp.name).stat().st_size
    print(f"  Downloaded {img_size:,} bytes to {tmp.name}")
except Exception as e:
    print(f"  FAILED to download image: {e}")
    sys.exit(1)

# ── 3. Build engine ─────────────────────────────────────────────────────
print("=" * 70)
print("Initializing RuntimeEngine...")
from di.container import build_container
container = build_container()
engine = container.runtime_engine

# ── 4. Run pipeline ─────────────────────────────────────────────────────
import asyncio

async def main():
    print("=" * 70)
    print("Running pipeline (full-image mode, no YOLO)...")
    t0 = time.perf_counter()
    try:
        response = await engine.run(
            image_path=tmp.name,
            template_path=str(TEMPLATE_PATH),
        )
    except Exception as e:
        import traceback
        print(f"  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)

    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  Done in {elapsed:.1f} ms")
    print()
    return response

response = asyncio.run(main())

# ── 5. Print results ───────────────────────────────────────────────────
print("=" * 70)
print("ANNOTATION RESULT")
print("=" * 70)
ann = response.annotation_result
print(f"  image:  {ann.image}")
print(f"  objects count: {len(ann.objects)}")
for i, obj in enumerate(ann.objects):
    print(f"\n  --- object {i} ---")
    print(f"    bbox:       {obj.bbox}")
    print(f"    category:   {obj.category}")
    print(f"    confidence: {obj.confidence}")
    print(f"    status:     {obj.status}")
    print(f"    attributes ({len(obj.attributes)} keys):")
    for k, v in obj.attributes.items():
        print(f"      {k} = {v}")

print()
print("=" * 70)
print("RUNTIME TRACE")
print("=" * 70)
trace = response.runtime_trace
print(f"  executed_steps: {trace.meta.get('executed_steps', [])}")
print(f"  scene_pure_negative: {trace.meta.get('scene_pure_negative', False)}")
print(f"  planner steps:")
for s in trace.planner_decisions.get("steps", []):
    print(f"    {s['order']}: {s['step']} ({s['model_id']})")
print(f"  merge_adapter: {trace.meta.get('merge_adapter', '')}")
print(f"  merge_rules: {json.dumps(trace.meta.get('merge_rules', {}), indent=4)}")
print(f"  elapsed_ms: {trace.meta.get('elapsed_ms', 0):.1f}")

print()
print("=" * 70)
print("MERGE REASONING TRACE")
print("=" * 70)
for entry in trace.merge_reasoning:
    print(f"  [{entry.get('step', '?')}] {entry.get('reasoning', '')[:200]}")

print()
print("=" * 70)
print("FULL annotation_result (JSON)")
print("=" * 70)
try:
    # Reconstruct full dict for pretty-print
    obj_list = []
    for obj in ann.objects:
        obj_list.append({
            "bbox": obj.bbox,
            "category": obj.category,
            "attributes": dict(obj.attributes),
            "confidence": obj.confidence,
            "status": obj.status,
        })
    full = {"image": ann.image, "objects": obj_list}
    print(json.dumps(full, indent=2, ensure_ascii=False))
except Exception:
    print(ann)

# ── Cleanup ────────────────────────────────────────────────────────────
import time
time.sleep(0.5)  # let file handles release
try:
    os.unlink(tmp.name)
except PermissionError:
    pass
print()
print("Done.")
