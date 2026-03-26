"""LLM transport errors should be classified without relying on full tracebacks."""

from __future__ import annotations

import errno

import httpx

from backend.agent import _is_llm_transport_failure


def test_connect_error_is_transport() -> None:
    inner = OSError(errno.ENETUNREACH, "Network is unreachable")
    exc = httpx.ConnectError("failed", request=None)
    exc.__cause__ = inner
    assert _is_llm_transport_failure(exc) is True


def test_connection_refused_oserror_is_transport() -> None:
    assert _is_llm_transport_failure(OSError(errno.ECONNREFUSED, "refused")) is True


def test_value_error_not_transport() -> None:
    assert _is_llm_transport_failure(ValueError("bad json")) is False
