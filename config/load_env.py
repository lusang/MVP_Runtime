"""
Load environment variables from `config/.env` (then optional project-root `.env`).

Import this once at application startup (see `main.py`).
"""

from __future__ import annotations

from pathlib import Path

_CONFIG_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CONFIG_DIR.parent


def load_project_env() -> Path | None:
    """
    Load dotenv files in order (later files do not override earlier by default).

    Priority: variables already in OS environ are kept (dotenv typically does not override).
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return None

    loaded: Path | None = None
    for candidate in (_CONFIG_DIR / ".env", _PROJECT_ROOT / ".env"):
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            loaded = candidate
    return loaded
