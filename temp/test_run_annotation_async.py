r"""
HTTP test script for POST /run_annotation_async.

Usage:
  1. Start the server:
       python main.py
  2. In another terminal, run this script:
       .venv\Scripts\python temp\test_run_annotation_async.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Make project root importable when running from temp/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.load_env import load_project_env

load_project_env()

try:
    import httpx
except ImportError:
    print("Missing httpx. Run: .venv\\Scripts\\pip install httpx")
    sys.exit(1)

ROOT = _PROJECT_ROOT
DEFAULT_IMAGE = ROOT / "temp" / "test.jpg"
DEFAULT_TEMPLATE = ROOT / "resource" / "Template.json"

BASE_URL = os.environ.get("MVP_BASE_URL", "http://127.0.0.1:8001").rstrip("/")


def ensure_test_image(path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        Image.new("RGB", (320, 240), color=(128, 128, 128)).save(path, format="JPEG")
    except ImportError:
        path.write_bytes(bytes([0xFF, 0xD8, 0xFF, 0xDB, 0x00, 0x43, 0x00, 0xFF, 0xD9]))
    print(f"Created placeholder test image: {path}")


def main() -> int:
    ensure_test_image(DEFAULT_IMAGE)

    # Load template for inline embedding
    template_path = DEFAULT_TEMPLATE
    if not template_path.is_file():
        print(f"Template not found: {template_path}")
        return 1

    with open(template_path, encoding="utf-8") as f:
        template = json.load(f)

    payload = {
        "template": template,
        "callback_url": "http://localhost:19999/callback",
        "tasks": [
            {
                "task_id": "batch_test_001",
                "media_type": "image",
                "frames": [
                    {
                        "frame_id": "img_001",
                        "url": str(DEFAULT_IMAGE.as_uri()),
                        "timestamp_ms": 0,
                    }
                ],
            },
        ],
    }

    url = f"{BASE_URL}/run_annotation_async"
    print(f"POST {url}")
    print(f"  template: {template_path.name}")
    print(f"  image   : {DEFAULT_IMAGE}")
    print(f"  tasks   : {len(payload['tasks'])}")
    print()

    try:
        r = httpx.post(url, json=payload, timeout=30.0)
    except httpx.ConnectError:
        print("Cannot connect. Start the server first:")
        print("  .venv\\Scripts\\uvicorn main:app --reload --host 127.0.0.1 --port 8000")
        return 1

    print(f"HTTP {r.status_code}")
    if r.status_code != 202:
        print(r.text)
        return 1

    data = r.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print()
    print("Async request accepted. Background processing started.")
    print("Check server logs for progress. Callbacks will be sent to callback_url.")
    print()
    print("Note: the dead letter queue at storage/dead_letter/ will contain")
    print("results if the callback target is unreachable (expected here).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
