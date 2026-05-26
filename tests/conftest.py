"""Shared pytest fixtures.

Sets MVP_FORCE_YOLO_MOCK=1 at module level to prevent YOLO weight loading.
Also sets MVP_FORCE_GEMINI_MOCK=1 to avoid real API calls in tests.
"""

from __future__ import annotations

import os

import pytest

# Set mock env vars at module level, before ANY test runs.
# This is more reliable than scope="function" monkeypatch which can race
# with module-level fixtures (e.g. module-scoped engine fixtures) in
# certain test ordering scenarios.
os.environ.setdefault("MVP_FORCE_YOLO_MOCK", "1")
os.environ.setdefault("MVP_FORCE_GEMINI_MOCK", "1")


@pytest.fixture(autouse=True)
def force_yolo_mock_in_tests(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Per-test guard: ensure mock env vars stay set."""
    if "world_vocab" in request.node.nodeid:
        return
    monkeypatch.setenv("MVP_FORCE_YOLO_MOCK", "1")
