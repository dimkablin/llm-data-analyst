"""LangGraph runtime for the analysis agent.

This package is the target runtime for the backend agent.  During migration the
legacy ``backend.agent`` code can be used as a behavior reference, but the final
production path should route through this LangGraph implementation only.
"""

from backend.agent_graph.graph import build_agent_graph
from backend.agent_graph.models import AgentGraphRequest, AgentGraphResult
from backend.agent_graph.routing import is_chat_query
from backend.agent_graph.runner import AgentGraphRunner
from backend.agent_graph.runtime import GraphRuntimeContext, RuntimeContextStore
from backend.agent_graph.services import AgentRuntimeServices
from backend.agent_graph.state import AgentGraphState

__all__ = [
    "AgentGraphRequest",
    "AgentGraphResult",
    "AgentGraphRunner",
    "AgentGraphState",
    "AgentRuntimeServices",
    "GraphRuntimeContext",
    "RuntimeContextStore",
    "build_agent_graph",
    "is_chat_query",
]
