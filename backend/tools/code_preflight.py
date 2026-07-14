"""Lightweight code preflight before sandbox execution."""

from __future__ import annotations

from typing import Any

_FORBIDDEN_NAMES = frozenset({"df", "pd", "np", "px", "go", "chart", "tool_result"})

# Injected into sandbox when include_plotly=True (must match sandbox.execute).
_PLOTLY_SCOPE_NAMES = frozenset({"px", "go", "make_subplots"})


def list_sandbox_user_var_names(scope: dict[str, Any]) -> list[str]:
    """Return user-visible sandbox variable names for LLM repair context."""
    return sorted(
        key
        for key in scope
        if not key.startswith("_") and key not in _FORBIDDEN_NAMES
    )


def preflight_sandbox_code(
    code: str,
    scope: dict[str, Any],
    *,
    extra_allowed: set[str] | frozenset[str] | None = None,
) -> tuple[str, str | None]:
    """Return code and optional blocking error before sandbox execution.

    This layer deliberately avoids dataframe schema inference, column-name
    repair, unknown-variable repair, import stripping, and domain-specific
    recovery. Those are runtime facts: the ReAct loop should execute code,
    observe the actual result/error, then choose the next tool call.
    """
    _ = scope, extra_allowed
    return code, None
