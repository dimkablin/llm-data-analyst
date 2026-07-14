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
