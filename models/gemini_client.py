"""
Real Gemini multimodal client via google-genai SDK.

Respects MVP_FORCE_GEMINI_MOCK env var — when set, delegates to mock responses.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

import backoff

from debug_log import agent_log
from runtime.prompt_manager import PromptManager


def _api_key() -> str:
    key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set in config/.env")
    return key


def _model_id() -> str:
    return (os.environ.get("GEMINI_MODEL") or "gemini-2.0-flash").strip()


def _timeout_sec() -> int:
    return int(os.environ.get("GEMINI_TIMEOUT_SEC", "120"))


def _read_image_bytes(image_path: str) -> bytes:
    from PIL import Image
    import io

    with Image.open(image_path) as im:
        if im.mode != "RGB":
            im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        return buf.getvalue()


_VERIFY_OBJECT_PROMPT = """\
You are a precision vision annotation agent.
Task: determine if the provided image crop contains a "{object_name}".

Target definition: {description}
Include (positive indicators): {include}
Exclude (do NOT count these): {exclude}

Respond ONLY with a valid JSON object (no markdown, no extra text):
{{"ok": true or false, "score": 0.0 to 1.0, "rationale": "one sentence explaining your decision"}}"""


_VERIFY_ATTRIBUTE_PROMPT = """\
You are a precision vision annotation agent.
For this image crop containing a "{object_name}", determine the attribute: {attribute_name}

Attribute description: {description}
Type: {attribute_type}
ALLOWED OPTIONS (this is a CLOSED set — NO other values are permitted): {options}

CRITICAL ENUM CONSTRAINT:
- You MUST choose ONLY from the ALLOWED OPTIONS list above.
- If NONE of the allowed options convincingly match → you MUST return null.
- Under NO circumstances may you return a value not in the allowed options list.
- Do NOT invent, approximate, paraphrase, or translate any option.
- Do NOT return a "best guess" that is outside the allowed set.

Respond ONLY with a valid JSON object (no markdown, no extra text):
{{"value": <exact option from the list, or null>, "confidence": 0.0 to 1.0}}"""


_VERIFY_NEGATIVE_ATTRIBUTE_PROMPT = """\
You are a precision vision annotation agent.
Task: a detector proposed that this FULL SCENE IMAGE contains a "{object_name}" within a specific region (bbox provided for reference). But the detector might be wrong. Examine the ENTIRE scene for context and determine the negative-sample attribute: {attribute_name}

Attribute description: {description}
Type: {attribute_type}
ALLOWED OPTIONS (this is a CLOSED set — NO other values are permitted): {options}

CRITICAL ENUM CONSTRAINT:
- You MUST choose ONLY from the ALLOWED OPTIONS list above.
- If NONE of the allowed options convincingly match → you MUST return null.
- Under NO circumstances may you return a value not in the allowed options list.
- Do NOT invent, approximate, paraphrase, or translate any option.

IMPORTANT — consider the FULL scene context:
- If the highlighted object is worn/held by a person, it may be a Hard Negative
- If the region is too blurry/far to tell, it may be Ambiguous
- If no "{object_name}" exists anywhere in the scene, this is a Pure Negative
- If the object is a completely novel/unknown form, it is Open-set Negative

Respond ONLY with a valid JSON object (no markdown, no extra text):
{{"value": <exact option from the list, or null>, "confidence": 0.0 to 1.0, "rationale": "one sentence why"}}"""


_VERIFY_SCENE_NEGATIVE_PROMPT = """\
You are a precision vision annotation agent.
Task: examine this FULL SCENE IMAGE and determine if there is ANY "{object_name}" visible anywhere.

Target definition: {description}
Positive indicators (these count as target): {include}
Exclude (these are NOT targets, even if they look similar): {exclude}

Answer ONLY one question: is there at least one "{object_name}" visible anywhere in this scene?

Respond ONLY with a valid JSON object (no markdown, no extra text):
{{"has_object": true or false, "confidence": 0.0 to 1.0, "rationale": "one sentence explaining your assessment of the full scene"}}"""


_MERGE_PROMPT = """\
Merge output schema reference — merge is now handled by runtime.merge_engine.MergeEngine (no LLM call).

The deterministic MergeEngine implements the decision rules and produces this
output format directly from structured candidate data:

{{
  "objects": [
    {{
      "is_positive": true or false,
      "negative_category": null or "pure_negative" or "hard_negative" or "ambiguous" or "open_set_negative",
      "confidence": 0.0 to 1.0,
      "detection_confidence": <float>,
      "verification_confidence": <float>,
      "merge_confidence": <float>,
      "attributes": {{ "name": {{"value": ..., "confidence": 0.0-1.0}}, ... }},
      "quality": {{ "name": {{"value": ..., "confidence": 0.0-1.0}}, ... }},
      "negative_flags": {{ "name": {{"value": true/false, "confidence": 0.0-1.0}}, ... }}
    }}
  ],
  "reasoning_trace": [
    {{
      "step": "yolo_detection" | "gemini_verification" | "gemini_semantic" | "opencv_quality" | "gemini_negative" | "merge",
      "input": "...",
      "output": "...",
      "reasoning": "1-2 sentences"
    }}
  ],
  "resolved_attributes": {{
    "name": {{"value": ..., "confidence": 0.0-1.0, "uncertain": true/false}}
  }},
  "merge_rules": {{"weights": {{"detector": 0.3, "verifier": 0.7}}}}
}}"""


def _extract_json(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction from model response."""
    # Try direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # Try to extract from markdown code fence
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Try to find first { ... } block
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from Gemini response: {text[:300]}")


def _validate_attribute_value(
    value: Any,
    attribute_type: str,
    options: list[Any],
    attribute_name: str = "",
    run_id: str = "",
) -> Any:
    """Post-parse validation: ensures value is a member of the allowed options set.

    Returns corrected value, or a safe fallback if invalid.
    """
    if attribute_type == "boolean":
        if not isinstance(value, bool):
            agent_log(
                hypothesis_id="H9",
                location="models/gemini_client.py:_validate_attribute_value",
                message="attribute_value_invalid_bool",
                data={"attribute_name": attribute_name, "value": value, "fallback": False},
                run_id=run_id,
            )
            return False
        return value

    if attribute_type == "single_select":
        if value is None:
            return None
        if options and value not in options:
            lower_map = {str(o).lower(): o for o in options}
            corrected = lower_map.get(str(value).lower())
            if corrected is not None:
                agent_log(
                    hypothesis_id="H9",
                    location="models/gemini_client.py:_validate_attribute_value",
                    message="attribute_value_case_corrected",
                    data={"attribute_name": attribute_name, "raw": value, "corrected": corrected},
                    run_id=run_id,
                )
                return corrected
            agent_log(
                hypothesis_id="H9",
                location="models/gemini_client.py:_validate_attribute_value",
                message="attribute_value_not_in_options",
                data={"attribute_name": attribute_name, "value": value, "options": options, "fallback": None},
                run_id=run_id,
            )
            return None
        return value

    if attribute_type == "multi_select":
        if not isinstance(value, list):
            agent_log(
                hypothesis_id="H9",
                location="models/gemini_client.py:_validate_attribute_value",
                message="attribute_value_invalid_multiselect",
                data={"attribute_name": attribute_name, "value": value, "fallback": [options[0]] if options else []},
                run_id=run_id,
            )
            return [options[0]] if options else []
        invalid = [v for v in value if v not in options]
        if invalid:
            agent_log(
                hypothesis_id="H9",
                location="models/gemini_client.py:_validate_attribute_value",
                message="attribute_value_filtered_multi",
                data={"attribute_name": attribute_name, "removed": invalid},
                run_id=run_id,
            )
        return [v for v in value if v in options]

    return value


class GeminiClient:
    """Real Gemini multimodal client. Raises RuntimeError if API key is missing."""

    def __init__(self, tracer: Any | None = None) -> None:
        self._api_key = _api_key()
        self._model_id = _model_id()
        self._timeout = _timeout_sec()
        self._tracer = tracer

    def _trace(self, run_id: str, step_name: str, prompt: str, response: str) -> None:
        """Send observation to tracer if available."""
        if self._tracer is not None and hasattr(self._tracer, "observe"):
            try:
                self._tracer.observe(
                    run_id=run_id,
                    step_name=step_name,
                    model=self._model_id,
                    prompt=prompt,
                    response=response,
                )
            except Exception:
                pass  # tracer must never break the pipeline

    async def verify_object(
        self,
        *,
        image_path: str,
        object_name: str,
        description: str,
        include: str,
        exclude: str,
        object_id: str,
        run_id: str = "",
    ) -> dict[str, Any]:
        prompt = PromptManager.load("verify_object", default=_VERIFY_OBJECT_PROMPT)
        prompt = prompt.format(
            object_name=object_name,
            description=description,
            include=include or "N/A",
            exclude=exclude or "N/A",
        )

        try:
            raw = await asyncio.to_thread(
                _gemini_generate, self._model_id, prompt, image_path, self._timeout
            )
            parsed = _extract_json(raw)
            self._trace(run_id, "verify_object", prompt, raw)
            agent_log(
                hypothesis_id="H8",
                location="models/gemini_client.py:verify_object",
                message="gemini_verify_object_done",
                data={
                    "object_id": object_id,
                    "ok": parsed.get("ok"),
                    "score": parsed.get("score"),
                },
                run_id=run_id,
            )
            return {
                "adapter": "GeminiClient",
                "ok": bool(parsed.get("ok", False)),
                "score": float(parsed.get("score", 0.0)),
                "rationale": str(parsed.get("rationale", "")),
                "object_name": object_name,
                "image_path": image_path,
                "object_id": object_id,
                "description": description,
            }
        except Exception as exc:
            agent_log(
                hypothesis_id="H9",
                location="models/gemini_client.py:verify_object",
                message="gemini_verify_object_failed",
                data={
                    "object_id": object_id,
                    "error": type(exc).__name__,
                    "detail": str(exc)[:300],
                },
                run_id=run_id,
            )
            return {
                "adapter": "GeminiClientError",
                "ok": False,
                "score": 0.0,
                "rationale": f"Gemini API error: {type(exc).__name__}: {str(exc)[:200]}",
                "object_name": object_name,
                "image_path": image_path,
                "object_id": object_id,
                "description": description,
                "error": str(exc)[:200],
            }

    async def verify_attribute(
        self,
        *,
        image_path: str,
        object_name: str,
        attribute_name: str,
        attribute_type: str,
        options: list[Any],
        description: str,
        scope: str,
        object_id: str,
        run_id: str = "",
    ) -> dict[str, Any]:
        template_name = "verify_negative_attribute" if scope == "negative" else "verify_attribute"
        default_prompt = _VERIFY_NEGATIVE_ATTRIBUTE_PROMPT if scope == "negative" else _VERIFY_ATTRIBUTE_PROMPT
        prompt = PromptManager.load(template_name, default=default_prompt)
        prompt = prompt.format(
            object_name=object_name,
            attribute_name=attribute_name,
            description=description or "N/A",
            attribute_type=attribute_type,
            options=json.dumps(options, ensure_ascii=False) if options else "N/A",
        )

        try:
            raw = await asyncio.to_thread(
                _gemini_generate, self._model_id, prompt, image_path, self._timeout
            )
            parsed = _extract_json(raw)
            raw_value = parsed.get("value")
            value = _validate_attribute_value(
                raw_value,
                attribute_type=attribute_type,
                options=options,
                attribute_name=attribute_name,
                run_id=run_id,
            )
            self._trace(run_id, f"verify_attribute:{attribute_name}", prompt, raw)
            agent_log(
                hypothesis_id="H8",
                location="models/gemini_client.py:verify_attribute",
                message="gemini_verify_attribute_done",
                data={
                    "object_id": object_id,
                    "attribute_name": attribute_name,
                    "raw_value": raw_value,
                    "validated_value": value,
                },
                run_id=run_id,
            )
            return {
                "adapter": "GeminiClient",
                "scope": scope,
                "attribute_name": attribute_name,
                "attribute_type": attribute_type,
                "value": value,
                "confidence": float(parsed.get("confidence", 0.0)),
                "verified": True,
                "object_name": object_name,
                "object_id": object_id,
                "image_path": image_path,
                "description": description,
            }
        except Exception as exc:
            agent_log(
                hypothesis_id="H9",
                location="models/gemini_client.py:verify_attribute",
                message="gemini_verify_attribute_failed",
                data={
                    "object_id": object_id,
                    "attribute_name": attribute_name,
                    "error": type(exc).__name__,
                    "detail": str(exc)[:300],
                },
                run_id=run_id,
            )
            default_value: Any
            if attribute_type == "boolean":
                default_value = False
            elif attribute_type == "multi_select":
                default_value = [options[0]] if options else []
            else:
                default_value = options[0] if options else None
            return {
                "adapter": "GeminiClientError",
                "scope": scope,
                "attribute_name": attribute_name,
                "attribute_type": attribute_type,
                "value": default_value,
                "confidence": 0.0,
                "verified": False,
                "object_name": object_name,
                "object_id": object_id,
                "image_path": image_path,
                "description": description,
                "error": str(exc)[:200],
            }


    async def verify_scene_pure_negative(
        self,
        *,
        image_path: str,
        object_name: str,
        description: str,
        include: str,
        exclude: str,
        run_id: str = "",
    ) -> dict[str, Any]:
        prompt = PromptManager.load("verify_scene_negative", default=_VERIFY_SCENE_NEGATIVE_PROMPT)
        prompt = prompt.format(
            object_name=object_name,
            description=description,
            include=include or "N/A",
            exclude=exclude or "N/A",
        )
        try:
            raw = await asyncio.to_thread(
                _gemini_generate, self._model_id, prompt, image_path, self._timeout
            )
            parsed = _extract_json(raw)
            has_obj = bool(parsed.get("has_object", True))
            self._trace(run_id, "verify_scene_pure_negative", prompt, raw)
            agent_log(
                hypothesis_id="H8",
                location="models/gemini_client.py:verify_scene_pure_negative",
                message="gemini_scene_pure_negative_done",
                data={"has_object": has_obj},
                run_id=run_id,
            )
            return {
                "adapter": "GeminiClient",
                "value": not has_obj,
                "confidence": float(parsed.get("confidence", 0.0)),
                "rationale": str(parsed.get("rationale", "")),
                "has_object": has_obj,
                "object_name": object_name,
                "image_path": image_path,
            }
        except Exception as exc:
            agent_log(
                hypothesis_id="H9",
                location="models/gemini_client.py:verify_scene_pure_negative",
                message="gemini_scene_pure_negative_failed",
                data={"error": type(exc).__name__, "detail": str(exc)[:300]},
                run_id=run_id,
            )
            return {
                "adapter": "GeminiClientError",
                "value": False,
                "confidence": 0.0,
                "rationale": f"Gemini API error: {type(exc).__name__}: {str(exc)[:200]}",
                "has_object": True,
                "object_name": object_name,
                "image_path": image_path,
                "error": str(exc)[:200],
            }

    async def generate_merge(
        self,
        *,
        image_path: str,
        object_name: str,
        description: str,
        include: str,
        exclude: str,
        execution_log: str,
        run_id: str = "",
    ) -> dict[str, Any]:
        prompt = PromptManager.load("merge", default=_MERGE_PROMPT)
        prompt = prompt.format(
            object_name=object_name,
            description=description,
            include=include or "N/A",
            exclude=exclude or "N/A",
            execution_log=execution_log,
        )
        try:
            raw = await asyncio.to_thread(
                _gemini_generate, self._model_id, prompt, image_path, self._timeout
            )
            parsed = _extract_json(raw)
            self._trace(run_id, "generate_merge", prompt, raw)
            agent_log(
                hypothesis_id="H8",
                location="models/gemini_client.py:generate_merge",
                message="gemini_merge_done",
                data={"object_count": len(parsed.get("objects", [])),
                       "trace_steps": len(parsed.get("reasoning_trace", []))},
                run_id=run_id,
            )
            return {
                "adapter": "GeminiClient",
                "objects": parsed.get("objects", []),
                "reasoning_trace": parsed.get("reasoning_trace", []),
            }
        except Exception as exc:
            agent_log(
                hypothesis_id="H9",
                location="models/gemini_client.py:generate_merge",
                message="gemini_merge_failed",
                data={"error": type(exc).__name__, "detail": str(exc)[:300]},
                run_id=run_id,
            )
            return {
                "adapter": "GeminiClientError",
                "objects": [],
                "reasoning_trace": [],
                "error": str(exc)[:200],
            }


# ── cached client + retry ────────────────────────────────────────────

_client_cache: dict[str, Any] = {}


def _get_genai_client(api_key: str, timeout_sec: int) -> Any:
    """Return a cached genai.Client, creating one if necessary."""
    cache_key = f"{api_key[:8]}@{timeout_sec}"
    if cache_key not in _client_cache:
        from google import genai
        _client_cache[cache_key] = genai.Client(
            api_key=api_key,
            http_options={"timeout": timeout_sec * 1000},
        )
    return _client_cache[cache_key]


def _is_retryable(exc: Exception) -> bool:
    """True for transient errors: 5xx, 429 (rate limit), or network errors."""
    from google.genai import errors as gemini_errors

    if isinstance(exc, gemini_errors.ServerError):
        return True
    if isinstance(exc, gemini_errors.APIError):
        return getattr(exc, "code", 0) in (429, 500, 502, 503)
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    return False


@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=3,
    max_time=60.0,
    giveup=lambda exc: not _is_retryable(exc),
)
def _gemini_generate(
    model_id: str,
    prompt: str,
    image_path: str,
    timeout_sec: int,
) -> str:
    """Synchronous call to google-genai SDK with retry support."""
    api_key = _api_key()
    client = _get_genai_client(api_key, timeout_sec)

    image_bytes = _read_image_bytes(image_path)
    contents = [
        prompt,
        {"inline_data": {"mime_type": "image/jpeg", "data": image_bytes}},
    ]

    response = client.models.generate_content(
        model=model_id,
        contents=contents,
    )
    return response.text or ""
