"""
临时 HTTP 测试脚本：调用 POST /run_annotation

使用前：
  1. 在项目根目录启动服务：
       .\\.venv\\Scripts\\uvicorn main:app --reload --host 127.0.0.1 --port 8000
  2. 另开终端运行本脚本：
       .\\.venv\\Scripts\\python temp\\test_run_annotation.py

可选环境变量：
  MVP_BASE_URL   默认 http://127.0.0.1:8000
  MVP_IMAGE_PATH 覆盖测试图片路径
  MVP_TEMPLATE_PATH 覆盖模板路径（默认 resource/Template.json）
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from config.load_env import load_project_env

load_project_env()

try:
    import httpx
except ImportError:
    print("缺少 httpx，请先执行: .venv\\Scripts\\pip install httpx")
    sys.exit(1)

# 项目根目录（本文件在 temp/ 下）
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE = ROOT / "temp" / "test.jpg"
DEFAULT_TEMPLATE = ROOT / "resource" / "Template.json"
OUTPUT_JSON = ROOT / "temp" / "annotation_result.json"

BASE_URL = os.environ.get("MVP_BASE_URL", "http://127.0.0.1:8001").rstrip("/")


def ensure_test_image(path: Path) -> None:
    """若不存在测试图，写入最小合法 JPEG（1x1）。"""
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        Image.new("RGB", (320, 240), color=(128, 128, 128)).save(path, format="JPEG")
    except ImportError:
        path.write_bytes(bytes([0xFF, 0xD8, 0xFF, 0xDB, 0x00, 0x43, 0x00, 0xFF, 0xD9]))
    print(f"已创建占位测试图: {path}")


def main() -> int:
    image_path = Path(os.environ.get("MVP_IMAGE_PATH", str(DEFAULT_IMAGE))).resolve()
    template_path = Path(
        os.environ.get("MVP_TEMPLATE_PATH", str(DEFAULT_TEMPLATE))
    ).resolve()

    ensure_test_image(image_path)

    if not template_path.is_file():
        print(f"模板不存在: {template_path}")
        print("可设置 MVP_TEMPLATE_PATH，或使用 resource/Template.json")
        return 1

    payload = {
        "image_path": str(image_path),
        "template_path": str(template_path),
    }

    url = f"{BASE_URL}/run_annotation"
    print(f"POST {url}")
    print(f"  image_path    = {payload['image_path']}")
    print(f"  template_path = {payload['template_path']}")
    print()

    try:
        response = httpx.post(url, json=payload, timeout=120.0)
    except httpx.ConnectError:
        print("无法连接服务。请先启动:")
        print("  .venv\\Scripts\\uvicorn main:app --reload --host 127.0.0.1 --port 8000")
        return 1

    print(f"HTTP {response.status_code}")
    if response.status_code != 200:
        print(response.text)
        return 1

    data = response.json()
    OUTPUT_JSON.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"完整结果已写入: {OUTPUT_JSON}")

    # --- AnnotationResult ---
    ann = data.get("annotation_result", {})
    objects = ann.get("objects", [])
    print(f"\n=== AnnotationResult ===")
    print(f"image: {ann.get('image')}")
    print(f"objects 数量: {len(objects)}")
    for obj in objects:
        print(f"  - {obj.get('category')}  bbox={obj.get('bbox')}  "
              f"confidence={obj.get('confidence')}  status={obj.get('status')}")
        print(f"    attributes: {obj.get('attributes')}")

    # --- RuntimeTrace ---
    trace = data.get("runtime_trace", {})
    print(f"\n=== RuntimeTrace ===")
    print(f"steps: {len(trace.get('steps', []))} executed")
    print(f"candidate_history: {len(trace.get('candidate_history', []))} candidates")
    planner = trace.get("planner_decisions", {})
    print(f"planner: {planner.get('planner_model')} (plan_id={planner.get('plan_id')})")
    print(f"quality_scores: {len(trace.get('quality_scores', []))} entries")
    print(f"merge_reasoning: {len(trace.get('merge_reasoning', []))} lines")
    meta = trace.get("meta", {})
    print(f"meta: run_id={meta.get('run_id')}  elapsed={meta.get('elapsed_ms')}ms")

    print("\n测试通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
