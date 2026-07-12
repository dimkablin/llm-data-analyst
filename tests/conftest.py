"""Shared pytest configuration and fixtures.

Marker conventions
------------------
- ``@pytest.mark.live``  – test requires a running LLM endpoint or external service.
  Skip in CI with:  pytest -m "not live"

- ``@pytest.mark.e2e``   – full-stack end-to-end test; requires backend + Ollama.
  Skip in CI with:  pytest -m "not e2e"

Offline tests (no markers) must be importable and runnable without any network
access, running containers, or API keys.
"""
from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Auto-mark files that are obviously live/e2e by name convention so
# test authors don't have to add markers manually to every function.
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        path = str(item.fspath)
        if "live" in path or "e2e" in path:
            # Add both markers so authors can filter by either.
            item.add_marker(pytest.mark.live)
            item.add_marker(pytest.mark.e2e)
