from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from backend.agent.dependencies import AgentRuntimeDependencies
from backend.agent.graph.nodes.agent import agent_node
from backend.agent.graph.nodes.finalize import finalize_node
from backend.agent.graph.nodes.prepare_context import prepare_context_node
from backend.agent.graph.routing import route_after_prepare_context
from backend.agent.state import AgentGraphState


def build_query_graph(deps: AgentRuntimeDependencies) -> StateGraph:
    graph = StateGraph(AgentGraphState)

    graph.add_node("prepare_context", lambda state: prepare_context_node(state, deps))
    graph.add_node("agent", lambda state: agent_node(state, deps))
    graph.add_node("finalize", lambda state: finalize_node(state, deps))

    graph.add_edge(START, "prepare_context")
    graph.add_conditional_edges(
        "prepare_context",
        route_after_prepare_context,
        {"agent": "agent", "finalize": "finalize"},
    )
    graph.add_edge("agent", "finalize")
    graph.add_edge("finalize", END)

    return graph
