"""Print environment readiness for MVP Runtime (YOLO / CUDA / deps)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.load_env import load_project_env

load_project_env()


def main() -> None:
    import os

    print("=== MVP Runtime environment ===\n")
    loaded = ROOT / "config" / ".env"
    print(f"config/.env exists: {loaded.is_file()}")

    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"gpu: {torch.cuda.get_device_name(0)}")
            cap = torch.cuda.get_device_capability(0)
            print(f"cuda capability: sm_{cap[0]}{cap[1]}")
    except ImportError:
        print("torch: NOT INSTALLED")

    try:
        import ultralytics

        print(f"ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("ultralytics: NOT INSTALLED")

    print("\n--- YOLO (.env) ---")
    for k in ("YOLO_MODEL_PATH", "YOLO_DEVICE", "YOLO_CONF_THRESHOLD", "MVP_FORCE_YOLO_MOCK"):
        print(f"  {k}={os.environ.get(k, '')!r}")

    print("\n--- Gemini (.env) ---")
    key = os.environ.get("GEMINI_API_KEY", "")
    print(f"  GEMINI_API_KEY={'(set)' if key.strip() else '(empty)'}")
    print(f"  MVP_FORCE_GEMINI_MOCK={os.environ.get('MVP_FORCE_GEMINI_MOCK', '')!r}")

    from config.yolo_paths import WEIGHTS_DIR, is_world_weights, resolve_yolo_model_path

    pt = resolve_yolo_model_path()
    print(f"\nweights dir: {WEIGHTS_DIR} exists={WEIGHTS_DIR.is_dir()}")
    print(f"YOLO weights: {pt}")
    print(f"  exists={pt.is_file()}  world={is_world_weights(pt)}")

    print("\nRTX 50-series: if you see sm_120 warning, use YOLO_DEVICE=cpu until PyTorch nightly cu128 is installed.")


if __name__ == "__main__":
    main()
