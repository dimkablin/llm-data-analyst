from __future__ import annotations

from typing import Any

from backend.agent.context_manager import AgentContextRequest
from backend.agent.dependencies import AgentRuntimeDependencies
from backend.agent.state import AgentGraphState


def prepare_context_node(
    state: AgentGraphState,
    deps: AgentRuntimeDependencies,
) -> dict[str, Any]:
    """Prepare generic agent context for the tool-calling runtime."""
    if state.get("registry_snapshot") is not None:
        return {}
    if deps.context_builder is None:
        msg = "AgentRuntimeDependencies.context_builder is not configured"
        raise RuntimeError(msg)
    prepared = deps.context_builder.build(AgentContextRequest(state=dict(state)))
    return prepared.state_update
