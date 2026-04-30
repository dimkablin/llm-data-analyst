from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

from backend.agent_graph.nodes import AgentGraphNodes
from backend.agent_graph.runtime import RuntimeContextStore
from backend.agent_graph.nodes import (
    should_continue_tools,
    should_skip_to_finalize,
)
from backend.agent_graph.state import AgentGraphState


@dataclass(slots=True)
class AgentGraphBuilder:
    """Builds the LangGraph runtime from injected node dependencies."""

    runtime_context_store: RuntimeContextStore

    def build(self) -> Any:
        """Build the LangGraph runtime skeleton.

        The graph mirrors the target decomposition of the current manual
        ``_direct_tool_loop``.  The migration goal is to make this graph the
        only backend agent runtime.
        """

        nodes = AgentGraphNodes(runtime_context_store=self.runtime_context_store)
        graph = StateGraph(AgentGraphState)

        graph.add_node("prepare", nodes.prepare)
        graph.add_node("route", nodes.route)
        graph.add_node("chat", nodes.chat)
        graph.add_node("summary", nodes.summary)
        graph.add_node("prepare_analysis", nodes.prepare_analysis)
        graph.add_node("planner", nodes.plan)
        graph.add_node("llm", nodes.call_llm)
        graph.add_node("tools", nodes.execute_tools)
        graph.add_node("finalize", nodes.finalize)

        graph.add_edge(START, "prepare")
        graph.add_edge("prepare", "route")
        graph.add_conditional_edges(
            "route",
            should_skip_to_finalize,
            {
                "chat": "chat",
                "summary": "summary",
                "prepare_analysis": "prepare_analysis",
            },
        )
        graph.add_edge("chat", "finalize")
        graph.add_edge("summary", "finalize")
        graph.add_edge("prepare_analysis", "planner")
        graph.add_edge("planner", "llm")
        graph.add_conditional_edges(
            "llm",
            should_continue_tools,
            {"tools": "tools", "finalize": "finalize"},
        )
        graph.add_edge("tools", "llm")
        graph.add_edge("finalize", END)

        return graph.compile()


def build_agent_graph(
    runtime_context_store: RuntimeContextStore | None = None,
) -> Any:
    """Compatibility helper for tests and lightweight callers."""

    store = runtime_context_store or RuntimeContextStore()
    return AgentGraphBuilder(runtime_context_store=store).build()
