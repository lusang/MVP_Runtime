"""
Attribute handler registry — maps template `handler` ids to plugin factories.

Factories enable dependency injection (shared analyzers, config) without
hardcoding constructors inside `RuntimeEngine`.
"""

from __future__ import annotations

from collections.abc import Callable

from handlers.plugins.protocol import AttributeHandlerPlugin

PluginFactory = Callable[[], AttributeHandlerPlugin]


class AttributeHandlerRegistry:
    """
    Thread-safe enough for MVP (register at startup, read during requests).

    Unknown handler ids raise `KeyError` so templates fail loudly.
    """

    def __init__(self) -> None:
        self._factories: dict[str, PluginFactory] = {}

    def register(self, handler_id: str, factory: PluginFactory) -> None:
        hid = handler_id.strip()
        if not hid:
            raise ValueError("handler_id must be non-empty")
        if hid in self._factories:
            raise ValueError(f"duplicate attribute handler id: {hid}")
        self._factories[hid] = factory

    def resolve(self, handler_id: str) -> AttributeHandlerPlugin:
        hid = handler_id.strip()
        factory = self._factories.get(hid)
        if factory is None:
            raise KeyError(f"unknown attribute handler: {hid!r}")
        return factory()

    def registered_ids(self) -> frozenset[str]:
        return frozenset(self._factories.keys())
