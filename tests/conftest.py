"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def force_yolo_mock_in_tests(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Unit tests avoid loading multi-GB YOLO-World weights from the developer `.env`."""
    if "world_vocab" in request.node.nodeid:
        return
    monkeypatch.setenv("MVP_FORCE_YOLO_MOCK", "1")
