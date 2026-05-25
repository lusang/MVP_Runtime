"""
PromptManager — loads prompt templates from prompts/ directory with env var override.

Priority (highest to lowest):
  1. Environment variable: ``MVP_PROMPT_{NAME}`` (uppercased, dashes → underscores)
  2. File: ``prompts/{name}.txt``
  3. Hardcoded default (fallback string passed to load())

Usage:
    prompt = PromptManager.format("verify_object", object_name="Package", ...)
    raw   = PromptManager.load("merge")
"""

from __future__ import annotations

import os
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class PromptManager:
    """Simple file-based prompt template loader with env-var override."""

    @staticmethod
    def load(name: str, *, default: str = "") -> str:
        """Load a prompt template.

        Resolution order:
          1. Environment variable ``MVP_PROMPT_<NAME>``
          2. File ``prompts/<name>.txt``
          3. ``default`` parameter

        ``name`` is slugified: lowercase, dashes become underscores,
        then uppercased for the env-var lookup.
        """
        env_key = f"MVP_PROMPT_{name.upper().replace('-', '_')}"
        env_val = os.environ.get(env_key)
        if env_val is not None:
            return env_val

        file_path = _PROMPTS_DIR / f"{name}.txt"
        if file_path.is_file():
            return file_path.read_text(encoding="utf-8")

        return default

    @staticmethod
    def format(name: str, **kwargs: object) -> str:
        """Load a prompt template and apply ``str.format(**kwargs)``.

        Raises ``KeyError`` if a placeholder in the template has no
        corresponding keyword argument.
        """
        prompt = PromptManager.load(name)
        return prompt.format(**kwargs)
