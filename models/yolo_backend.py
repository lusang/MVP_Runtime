"""
YOLO inference backends: Ultralytics (COCO or YOLO-World) or image-aware mock.
"""

from __future__ import annotations

import os
from pathlib import Path

from config.yolo_paths import is_world_weights, resolve_yolo_model_path
from debug_log import agent_log
from models.object_target import ObjectTarget
from models.world_vocab import world_classes_from_target
from schemas.bbox import BBox
from schemas.detection import DetectionCandidate

_model_cache: dict[str, object] = {}
_world_vocab_key: dict[str, tuple[str, ...]] = {}


def _load_image_size(image_path: str) -> tuple[int, int]:
    from PIL import Image

    try:
        with Image.open(image_path) as im:
            return im.size
    except Exception:
        return (640, 480)


def _mock_candidates(image_path: str, target: ObjectTarget) -> list[DetectionCandidate]:
    """Place two proportional boxes when no Ultralytics model is available."""
    w, h = _load_image_size(image_path)
    boxes = [
        (0.05 * w, 0.08 * h, 0.45 * w, 0.55 * h, 0.91),
        (0.55 * w, 0.12 * h, 0.92 * w, 0.65 * h, 0.72),
    ]
    return [
        DetectionCandidate(
            bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
            label="candidate",
            score=score,
            target_object=target.name,
        )
        for x1, y1, x2, y2, score in boxes
    ]


def _get_ultralytics_model(model_path: Path):
    from ultralytics import YOLO

    key = str(model_path.resolve())
    if key not in _model_cache:
        _model_cache[key] = YOLO(key)
    return _model_cache[key]


def _configure_world_model(model, model_key: str, target: ObjectTarget) -> list[str]:
    classes = world_classes_from_target(target)
    vocab_key = tuple(classes)
    if _world_vocab_key.get(model_key) != vocab_key:
        model.set_classes(classes)
        _world_vocab_key[model_key] = vocab_key
        agent_log(
            hypothesis_id="H6",
            location="models/yolo_backend.py:_configure_world_model",
            message="yolo_world_set_classes",
            data={"classes": classes},
            run_id="yolo",
        )
    return classes


def _boxes_to_candidates(result, target: ObjectTarget) -> list[DetectionCandidate]:
    names = result.names or {}
    out: list[DetectionCandidate] = []
    if result.boxes is None:
        return out

    for box in result.boxes:
        xyxy = box.xyxy[0].tolist()
        cls_id = int(box.cls[0].item()) if box.cls is not None else -1
        label = names.get(cls_id, "candidate")
        score = float(box.conf[0].item()) if box.conf is not None else 0.5
        out.append(
            DetectionCandidate(
                bbox=BBox(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3]),
                label=label,
                score=score,
                target_object=target.name,
            )
        )
    return out


def _ultralytics_candidates(
    image_path: str,
    target: ObjectTarget,
    *,
    model_path: Path,
    conf: float,
) -> tuple[list[DetectionCandidate], str]:
    preferred = os.environ.get("YOLO_DEVICE", "cpu").strip() or "cpu"
    devices_to_try = [preferred]
    if preferred not in ("cpu", "CPU") and "cpu" not in devices_to_try:
        devices_to_try.append("cpu")

    resolved = model_path.resolve()
    world = is_world_weights(resolved)
    model_key = str(resolved)
    model = _get_ultralytics_model(resolved)
    world_classes: list[str] | None = None
    if world:
        world_classes = _configure_world_model(model, model_key, target)

    last_exc: Exception | None = None
    for device in devices_to_try:
        try:
            results = model.predict(source=image_path, conf=conf, device=device, verbose=False)
            if not results:
                tag = "world" if world else "ultralytics"
                return [], f"{tag}_{device}_empty"

            result = results[0]
            out = _boxes_to_candidates(result, target)
            if world:
                return out, f"yolo_world_{device}"
            return out, f"ultralytics_{device}"
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            agent_log(
                hypothesis_id="H2",
                location="models/yolo_backend.py:_ultralytics_candidates",
                message="ultralytics_device_failed",
                data={
                    "device": device,
                    "world": world,
                    "world_classes": world_classes,
                    "error": type(exc).__name__,
                    "detail": str(exc)[:200],
                },
                run_id="yolo",
            )
    if last_exc is not None:
        raise last_exc
    return [], "ultralytics_failed"


def run_yolo_inference(
    image_path: str,
    target: ObjectTarget,
    *,
    run_id: str = "default",
) -> list[DetectionCandidate]:
    """
    Run YOLO or fallback mock. Set env `YOLO_MODEL_PATH` to a `.pt` under `weights/`.

    YOLO-World weights (`*world*.pt`) call `set_classes()` from template `ObjectTarget`.

    `MVP_FORCE_YOLO_MOCK=1` forces mock boxes (proportional to image size).
    """
    force_mock = os.environ.get("MVP_FORCE_YOLO_MOCK", "").strip() in ("1", "true", "yes")
    model_path = resolve_yolo_model_path()
    conf = float(os.environ.get("YOLO_CONF_THRESHOLD", "0.25"))

    agent_log(
        hypothesis_id="H1",
        location="models/yolo_backend.py:run_yolo_inference",
        message="yolo_inference_start",
        data={
            "image_path": image_path,
            "target_name": target.name,
            "force_mock": force_mock,
            "model_path": str(model_path),
            "world": is_world_weights(model_path),
        },
        run_id=run_id,
    )

    if force_mock:
        candidates = _mock_candidates(image_path, target)
        backend = "mock_proportional"
    elif not model_path.is_file() and not os.environ.get("YOLO_MODEL_PATH", "").strip():
        candidates = _mock_candidates(image_path, target)
        backend = "mock_proportional"
    else:
        try:
            candidates, backend = _ultralytics_candidates(
                image_path, target, model_path=model_path, conf=conf
            )
        except Exception as exc:  # noqa: BLE001 — fallback for missing deps/GPU
            agent_log(
                hypothesis_id="H2",
                location="models/yolo_backend.py:run_yolo_inference",
                message="ultralytics_failed_fallback_mock",
                data={"error": type(exc).__name__, "detail": str(exc)[:200]},
                run_id=run_id,
            )
            candidates = _mock_candidates(image_path, target)
            backend = "mock_after_ultralytics_error"

    agent_log(
        hypothesis_id="H3",
        location="models/yolo_backend.py:run_yolo_inference",
        message="yolo_inference_done",
        data={"backend": backend, "count": len(candidates)},
        run_id=run_id,
    )
    return candidates
