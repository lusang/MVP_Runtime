"""
ASGI entrypoint — application factory lives here; domain logic stays in `runtime/`.
"""

import logging
import logging.handlers
import os
from pathlib import Path

from config.load_env import load_project_env

load_project_env()

# ── Logging configuration ────────────────────────────────────────────────
_log_level = (os.environ.get("MVP_LOG_LEVEL") or "INFO").upper()
_log_file = Path(__file__).resolve().parent / "storage" / "runtime.log"

_log_file.parent.mkdir(parents=True, exist_ok=True)

_log_formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

# Clear existing handlers to prevent duplicates on uvicorn reload
_root_logger = logging.getLogger()
for h in list(_root_logger.handlers):
    _root_logger.removeHandler(h)

# File handler (rotates at midnight, keep 7 days)
_file_handler = logging.handlers.TimedRotatingFileHandler(
    _log_file, when="midnight", interval=1, backupCount=7, encoding="utf-8",
)
_file_handler.setFormatter(_log_formatter)
_file_handler.setLevel(_log_level)

_root_logger.setLevel(_log_level)
_root_logger.addHandler(_file_handler)

# Silence noisy third-party loggers
for _noisy in ("httpx", "httpcore", "urllib3", "PIL", "matplotlib", "fsspec", "watchfiles"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logging.info("=== MVP Runtime starting (log_level=%s) ===", _log_level)
logging.info("Log file: %s", _log_file)

from fastapi import FastAPI

from api.routes import router as annotation_router

app = FastAPI(
    title="Vision Agent Pre-annotation Runtime (MVP)",
    version="0.1.0",
    description="Mocked adapters + modular handlers. Swap `models/*` for production inference.",
)

app.include_router(annotation_router)

if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("MVP_HOST", "0.0.0.0")
    port = int(os.environ.get("MVP_PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)
