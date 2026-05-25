"""
ASGI entrypoint — application factory lives here; domain logic stays in `runtime/`.
"""

import os

from config.load_env import load_project_env

load_project_env()

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
