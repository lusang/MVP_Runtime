"""
FastAPI dependencies — resolve singletons from `AppContainer`.

Routes stay thin; wiring stays centralized in `di/container.py`.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from di.container import AppContainer, build_container
from handlers.registry import AttributeHandlerRegistry
from runtime.engine import RuntimeEngine

_container: AppContainer | None = None


def _get_container() -> AppContainer:
    global _container
    if _container is None:
        _container = build_container()
    return _container


def get_container() -> AppContainer:
    """Mutable process-global container (MVP). Replace with lifespan hooks if needed."""
    return _get_container()


def get_runtime_engine(
    container: Annotated[AppContainer, Depends(get_container)],
) -> RuntimeEngine:
    """Inject the shared `RuntimeEngine` instance."""
    return container.runtime_engine


def get_attribute_registry(
    container: Annotated[AppContainer, Depends(get_container)],
) -> AttributeHandlerRegistry:
    """Optional: expose registry for admin/debug routes."""
    return container.attribute_registry
