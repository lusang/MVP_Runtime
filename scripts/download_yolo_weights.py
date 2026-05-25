"""
Download YOLO / YOLO-World weights into `weights/`.

YOLO-World also needs CLIP (without replacing a CUDA nightly torch):
  .venv\\Scripts\\pip install --no-deps git+https://github.com/ultralytics/CLIP.git
  .venv\\Scripts\\pip install ftfy

Usage:
  .venv\\Scripts\\python scripts\\download_yolo_weights.py
  .venv\\Scripts\\python scripts\\download_yolo_weights.py yolov8s-worldv2.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.yolo_paths import WEIGHTS_DIR, resolve_yolo_model_path


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "yolov8s-worldv2.pt"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = resolve_yolo_model_path(name)
    if dest.is_file():
        print(f"already exists: {dest} ({dest.stat().st_size} bytes)")
        return

    from ultralytics import YOLO

    print(f"downloading → {dest}")
    YOLO(str(dest))
    if dest.is_file():
        print(f"done: {dest} ({dest.stat().st_size} bytes)")
    else:
        # Ultralytics may download to cwd; move if found
        cwd_file = Path.cwd() / dest.name
        if cwd_file.is_file():
            cwd_file.replace(dest)
            print(f"moved to: {dest}")
        else:
            raise SystemExit(f"download finished but file missing: {dest}")


if __name__ == "__main__":
    main()
