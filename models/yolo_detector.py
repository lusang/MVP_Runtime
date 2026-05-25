"""
YOLO detector capability adapter: object-aware detect + pre-selection crops.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from models.object_target import ObjectTarget
from models.yolo_backend import run_yolo_inference
from debug_log import agent_log
from schemas.detection import DetectionCandidate
from schemas.template_spec import ParsedTaskSpec
from storage.image_crop import crop_bbox_to_file


class YOLODetector:
    """
    1. Resolve `ObjectTarget` from template
    2. Run YOLO (or mock) → candidate bboxes
    3. Crop each bbox to `storage/preselection/{run_id}/` as 预选图
    """

    def __init__(self, *, preselection_root: str | Path | None = None) -> None:
        root = preselection_root or Path(__file__).resolve().parent.parent / "storage" / "preselection"
        self._preselection_root = Path(root)

    async def detect(
        self,
        image_path: str,
        *,
        target_object: str,
        parsed: ParsedTaskSpec,
        run_id: str = "default",
    ) -> list[DetectionCandidate]:
        target = ObjectTarget.from_parsed(parsed)
        if target.name != target_object:
            agent_log(
                hypothesis_id="H4",
                location="models/yolo_detector.py:detect",
                message="target_object_name_mismatch",
                data={"arg": target_object, "parsed": target.name},
                run_id=run_id,
            )

        candidates = await asyncio.to_thread(
            run_yolo_inference,
            image_path,
            target,
            run_id=run_id,
        )

        out_dir = self._preselection_root / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        enriched: list[DetectionCandidate] = []
        for idx, det in enumerate(candidates):
            crop_file = out_dir / f"candidate_{idx}.jpg"
            try:
                crop_path = await asyncio.to_thread(
                    crop_bbox_to_file,
                    image_path=image_path,
                    bbox=det.bbox,
                    output_path=crop_file,
                )
                crop_str = str(crop_path)
            except Exception as exc:  # noqa: BLE001
                agent_log(
                    hypothesis_id="H5",
                    location="models/yolo_detector.py:detect",
                    message="crop_failed",
                    data={"index": idx, "error": type(exc).__name__, "detail": str(exc)[:200]},
                    run_id=run_id,
                )
                crop_str = None

            enriched.append(
                det.model_copy(
                    update={
                        "crop_path": crop_str,
                        "target_object": target.name,
                    }
                )
            )

        agent_log(
            hypothesis_id="H3",
            location="models/yolo_detector.py:detect",
            message="detect_with_crops_done",
            data={
                "count": len(enriched),
                "crops": [d.crop_path for d in enriched],
            },
            run_id=run_id,
        )
        return enriched
