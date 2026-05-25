"""
从 fixtures 目录加载请求 JSON，发送到 MVP 异步端点。

用法：
  # 使用默认 fixture（request_real.json）
  .venv\Scripts\python test\fixtures\replay_fixture.py

  # 使用指定 fixture
  .venv\Scripts\python test\fixtures\replay_fixture.py --fixture request_placeholder.json

  # 指定 MVP 地址
  .venv\Scripts\python test\fixtures\replay_fixture.py --url http://127.0.0.1:8001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make project root importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

FIXTURES_DIR = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a saved fixture to MVP")
    parser.add_argument(
        "--fixture",
        default="request_real.json",
        help="Fixture filename (under test/fixtures/)",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8001",
        help="MVP base URL",
    )
    args = parser.parse_args()

    fixture_path = FIXTURES_DIR / args.fixture
    if not fixture_path.is_file():
        print(f"Fixture not found: {fixture_path}")
        print(f"Available fixtures: {[p.name for p in FIXTURES_DIR.glob('request_*.json')]}")
        return 1

    with open(fixture_path, encoding="utf-8") as f:
        payload = json.load(f)

    import httpx

    url = f"{args.url.rstrip('/')}/run_annotation_async"
    print(f"POST {url}")
    print(f"  fixture: {fixture_path.name}")
    print(f"  tasks:   {len(payload.get('tasks', []))}")
    print()

    try:
        r = httpx.post(url, json=payload, timeout=30.0)
    except httpx.ConnectError:
        print("Cannot connect to MVP. Is the server running?")
        return 1

    print(f"HTTP {r.status_code}")
    if r.status_code == 202:
        data = r.json()
        print(json.dumps(data, indent=2, ensure_ascii=False))
        print()
        print("Request accepted. Check server logs for processing progress.")
        return 0

    print(r.text)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
