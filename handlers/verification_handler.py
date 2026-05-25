"""
Verification handler — Gemini object-level verification per detection.
"""

from __future__ import annotations

from typing import Any

from models.gemini_verifier import GeminiVerifier
from schemas.bbox import BBox
from schemas.template_spec import ParsedTaskSpec


class VerificationHandler:
    """Async façade over `GeminiVerifier.verify_object`."""

    def __init__(self, verifier: GeminiVerifier | None = None) -> None:
        self._verifier = verifier or GeminiVerifier()

    async def verify_object(
        self,
        *,
        image_path: str,
        bbox: BBox,
        parsed: ParsedTaskSpec,
        object_id: str,
    ) -> dict[str, Any]:
        return await self._verifier.verify_object(
            image_path=image_path,
            bbox=bbox,
            parsed=parsed,
            object_id=object_id,
        )
