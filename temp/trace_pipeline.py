"""
Diagnostic: trace every Gemini I/O and verify negative-sample pipeline.

  .venv\\Scripts\\python temp\\trace_pipeline.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.load_env import load_project_env
load_project_env()

# Inject a tracing wrapper around GeminiClient to capture I/O
import models.gemini_client as _gmod

_orig_verify_object = None
_orig_verify_attribute = None
_orig_verify_scene = None
_orig_generate_merge = None
_TRACE: list[dict] = []


def _install_trace():
    global _orig_verify_object, _orig_verify_attribute, _orig_verify_scene, _orig_generate_merge
    from models.gemini_client import GeminiClient

    _orig_verify_object = GeminiClient.verify_object
    _orig_verify_attribute = GeminiClient.verify_attribute
    _orig_verify_scene = GeminiClient.verify_scene_pure_negative
    _orig_generate_merge = GeminiClient.generate_merge

    async def traced_verify_object(self, **kwargs):
        call = {
            "method": "verify_object",
            "input": {
                "image_path": kwargs.get("image_path"),
                "object_name": kwargs.get("object_name"),
                "description": kwargs.get("description", "")[:120],
                "include": kwargs.get("include", "")[:120],
                "exclude": kwargs.get("exclude", "")[:120],
                "object_id": kwargs.get("object_id"),
            },
        }
        result = await _orig_verify_object(self, **kwargs)
        call["output"] = {
            "ok": result.get("ok"),
            "score": result.get("score"),
            "rationale": result.get("rationale", "")[:200],
            "adapter": result.get("adapter"),
        }
        _TRACE.append(call)
        return result

    async def traced_verify_attribute(self, **kwargs):
        call = {
            "method": "verify_attribute",
            "input": {
                "image_path": kwargs.get("image_path"),
                "object_name": kwargs.get("object_name"),
                "attribute_name": kwargs.get("attribute_name"),
                "attribute_type": kwargs.get("attribute_type"),
                "options": kwargs.get("options", []),
                "description": kwargs.get("description", "")[:100],
                "scope": kwargs.get("scope"),
                "object_id": kwargs.get("object_id"),
            },
        }
        result = await _orig_verify_attribute(self, **kwargs)
        call["output"] = {
            "value": result.get("value"),
            "confidence": result.get("confidence"),
            "verified": result.get("verified"),
            "adapter": result.get("adapter"),
        }
        _TRACE.append(call)
        return result

    async def traced_verify_scene(self, **kwargs):
        call = {
            "method": "verify_scene_pure_negative",
            "input": {
                "image_path": kwargs.get("image_path"),
                "object_name": kwargs.get("object_name"),
                "description": kwargs.get("description", "")[:120],
                "include": kwargs.get("include", "")[:120],
                "exclude": kwargs.get("exclude", "")[:120],
            },
        }
        result = await _orig_verify_scene(self, **kwargs)
        call["output"] = {
            "value": result.get("value"),
            "has_object": result.get("has_object"),
            "confidence": result.get("confidence"),
            "rationale": str(result.get("rationale", ""))[:200],
            "adapter": result.get("adapter"),
        }
        _TRACE.append(call)
        return result

    async def traced_generate_merge(self, **kwargs):
        call = {
            "method": "generate_merge",
            "input": {
                "image_path": kwargs.get("image_path"),
                "object_name": kwargs.get("object_name"),
                "candidate_count": kwargs.get("candidate_count"),
                "candidates_summary": kwargs.get("candidates_summary", "")[:300],
            },
        }
        result = await _orig_generate_merge(self, **kwargs)
        call["output"] = {
            "adapter": result.get("adapter"),
            "object_count": len(result.get("objects", [])),
            "trace_steps": len(result.get("reasoning_trace", [])),
        }
        _TRACE.append(call)
        return result

    GeminiClient.verify_object = traced_verify_object
    GeminiClient.verify_attribute = traced_verify_attribute
    GeminiClient.verify_scene_pure_negative = traced_verify_scene
    GeminiClient.generate_merge = traced_generate_merge


_install_trace()


from handlers.attribute_handler import AttributeHandler
from handlers.verification_handler import VerificationHandler
from handlers.registry import AttributeHandlerRegistry
from handlers.plugins.gemini_attribute import GeminiAttributePlugin
from handlers.plugins.gemini_negative import GeminiNegativePlugin
from handlers.plugins.opencv_quality import OpenCVQualityPlugin
from runtime.engine import RuntimeEngine


def build_engine() -> RuntimeEngine:
    registry = AttributeHandlerRegistry()
    registry.register("gemini", GeminiAttributePlugin)
    registry.register("opencv_quality", OpenCVQualityPlugin)
    registry.register("gemini_negative", GeminiNegativePlugin)
    return RuntimeEngine(attribute_registry=registry)


async def run_one(label: str, image_path: str) -> None:
    _TRACE.clear()

    engine = build_engine()
    response = await engine.run(
        image_path=image_path,
        template_path=str(ROOT / "resource" / "Template.json"),
    )

    dashes = "=" * 70
    print(f"\n{dashes}")
    print(f"  {label}")
    print(f"  Image: {Path(image_path).name}")
    print(f"{dashes}")

    print(f"\n[Step 1] YOLO detections: {len(response.objects)}")
    for i, det in enumerate(response.meta.get("detections", [])):
        print(f"  [{i}] label={det.get('label')}  conf={det.get('confidence'):.3f}  bbox={det.get('bbox')}")

    print(f"\n--- Gemini API Calls ({len(_TRACE)} total) ---")
    for i, call in enumerate(_TRACE):
        print(f"\n  Call #{i+1}: {call['method']}")
        print(f"    ── INPUT ──")
        for k, v in call["input"].items():
            print(f"      {k}: {v}")
        print(f"    ── OUTPUT ──")
        for k, v in call["output"].items():
            print(f"      {k}: {v}")

    print(f"\n[Final] ObjectState:")
    for obj in response.objects:
        print(f"  object_id:      {obj.object_id}")
        print(f"  object_name:    {obj.object_name}")
        print(f"  confidence:     {obj.confidence:.3f}")
        print(f"  negative:       {obj.negative}")
        print(f"  negative_category: {obj.negative_category}")
        print(f"  is_positive:    {obj.is_positive}")
        print(f"  verify ok:      {obj.verification.get('ok')}")
        print(f"  verify score:   {obj.verification.get('score')}")
        print(f"  semantic:       {json.dumps({k: v.get('value') for k, v in obj.attributes.items()}, ensure_ascii=False)}")
        print(f"  quality:        {json.dumps({k: v.get('value') for k, v in obj.quality.items()}, ensure_ascii=False)}")
        print(f"  negative_flags: {json.dumps({k: v.get('value') for k, v in obj.negative_attributes.items()}, ensure_ascii=False)}")
        if obj.annotation_panel:
            print(f"  annotation_panel: {json.dumps({k: v for k, v in obj.annotation_panel.items() if k != 'attributes' and k != 'quality' and k != 'negative_flags'}, ensure_ascii=False)}")

    print(f"\n[Merge] reasoning_trace: {len(response.reasoning_trace)} steps")
    for step in response.reasoning_trace:
        print(f"  [{step.get('step')}] {step.get('reasoning', '')[:120]}")
    print(f"[Merge] annotation_panel keys: {list(response.annotation_panel.keys()) if response.annotation_panel else 'None'}")

    # Negative path analysis
    if not response.objects:
        print(f"\n  ⚠ No objects detected. If this is a neg image, YOLO correctly found nothing.")
    else:
        for obj in response.objects:
            if obj.negative:
                print(f"\n  ✓ Negative flag IS set on {obj.object_id} — downstream should skip annotation.")
            else:
                print(f"\n  → {obj.object_id} is NOT flagged as negative. Checking why...")
                verif_ok = obj.verification.get("ok")
                print(f"    verification.ok = {verif_ok}")
                for k, v in obj.negative_attributes.items():
                    print(f"    negative.{k} = value={v.get('value')}, verified={v.get('verified')}")

    print(f"\n  elapsed_ms: {response.meta.get('elapsed_ms'):.0f}")


async def main():
    # POS image
    pos_img = str(ROOT / "resource" / "package" / "pos" / "0300446c1b9800831e54ddcbbfe08511.jpeg")
    await run_one("POSITIVE SAMPLE (should be Package)", pos_img)

    # NEG image
    import glob
    neg_files = sorted(glob.glob(str(ROOT / "resource" / "package" / "neg" / "*.jpeg")))
    neg_img = neg_files[0]
    await run_one("NEGATIVE SAMPLE (should be rejected)", neg_img)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
