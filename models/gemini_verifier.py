"""
Gemini / VLM verifier adapter — real API via `GeminiClient` or mock fallback.

Set MVP_FORCE_GEMINI_MOCK=1 to skip real API calls regardless of GEMINI_API_KEY.
"""

from __future__ import annotations

import os
from typing import Any

from schemas.bbox import BBox
from schemas.template_spec import ParsedTaskSpec


def _gemini_env_status() -> dict[str, Any]:
    return {
        "api_key_configured": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
        "model": os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
        "api_base_set": bool(os.environ.get("GEMINI_API_BASE", "").strip()),
        "force_mock": os.environ.get("MVP_FORCE_GEMINI_MOCK", "1").strip()
        in ("1", "true", "yes"),
    }


def _force_mock() -> bool:
    return os.environ.get("MVP_FORCE_GEMINI_MOCK", "1").strip() in ("1", "true", "yes")


class GeminiVerifier:
    """Routes to real GeminiClient or mock based on MVP_FORCE_GEMINI_MOCK."""

    async def verify_object(
        self,
        *,
        image_path: str,
        bbox: BBox,
        parsed: ParsedTaskSpec,
        object_id: str,
    ) -> dict[str, Any]:
        if _force_mock():
            return {
                "adapter": "GeminiVerifierMock",
                "ok": True,
                "score": 0.88,
                "rationale": f"Mock: bbox matches target object '{parsed.object_name}'.",
                "object_name": parsed.object_name,
                "image_path": image_path,
                "bbox": bbox.model_dump(),
                "object_id": object_id,
                "description": parsed.description,
                "gemini_env": _gemini_env_status(),
            }

        from models.gemini_client import GeminiClient

        client = GeminiClient()
        result = await client.verify_object(
            image_path=image_path,
            object_name=parsed.object_name,
            description=parsed.description,
            include=parsed.include,
            exclude=parsed.exclude,
            object_id=object_id,
        )
        result["bbox"] = bbox.model_dump()
        result["gemini_env"] = _gemini_env_status()
        return result

    async def verify_scene_pure_negative(
        self,
        *,
        image_path: str,
        parsed: ParsedTaskSpec,
    ) -> dict[str, Any]:
        if _force_mock():
            return {
                "adapter": "GeminiVerifierMock",
                "value": False,
                "confidence": 0.95,
                "rationale": "Mock: assuming scene may contain package objects.",
                "has_object": True,
                "object_name": parsed.object_name,
                "image_path": image_path,
            }

        from models.gemini_client import GeminiClient

        client = GeminiClient()
        return await client.verify_scene_pure_negative(
            image_path=image_path,
            object_name=parsed.object_name,
            description=parsed.description,
            include=parsed.include,
            exclude=parsed.exclude,
        )

    async def verify_attribute(
        self,
        *,
        image_path: str,
        bbox: BBox,
        parsed: ParsedTaskSpec,
        object_id: str,
        attribute_name: str,
        attribute_type: str,
        options: list[Any],
        description: str,
        scope: str,
    ) -> dict[str, Any]:
        if _force_mock():
            default_value: Any
            if attribute_type == "boolean":
                default_value = False
            elif attribute_type == "multi_select":
                default_value = [options[0]] if options else []
            else:
                default_value = options[0] if options else None

            return {
                "adapter": "GeminiVerifierMock",
                "scope": scope,
                "attribute_name": attribute_name,
                "attribute_type": attribute_type,
                "value": default_value,
                "confidence": 0.85,
                "verified": True,
                "object_name": parsed.object_name,
                "object_id": object_id,
                "image_path": image_path,
                "bbox": bbox.model_dump(),
                "description": description,
            }

        from models.gemini_client import GeminiClient

        client = GeminiClient()
        result = await client.verify_attribute(
            image_path=image_path,
            object_name=parsed.object_name,
            attribute_name=attribute_name,
            attribute_type=attribute_type,
            options=options,
            description=description,
            scope=scope,
            object_id=object_id,
        )
        result["bbox"] = bbox.model_dump()
        return result
