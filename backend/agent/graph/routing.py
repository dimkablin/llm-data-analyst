from __future__ import annotations

from typing import Any


def route_after_prepare_context(state: dict[str, Any]) -> str:
    if state.get("done") or state.get("response"):
        return "finalize"
    return "agent"
