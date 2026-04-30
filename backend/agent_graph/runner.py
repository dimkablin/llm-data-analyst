from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.agent_graph.graph import AgentGraphBuilder
from backend.agent_graph.memory import WorkingMemoryFlusher
from backend.agent_graph.models import AgentGraphRequest, AgentGraphResult
from backend.agent_graph.runtime import GraphRuntimeContext, RuntimeContextStore
from backend.sessions.session_memory import StructuredSessionMemory


@dataclass(slots=True)
class AgentGraphRunner:
    """OO entrypoint for the LangGraph agent runtime.

    The runner owns graph invocation and runtime-context lifecycle.  Node
    classes own behavior.  This separation keeps API integration code from
    knowing how the graph is assembled and keeps graph state free of live
    objects.
    """

    runtime_context_store: RuntimeContextStore = field(default_factory=RuntimeContextStore)
    recursion_limit: int = 20
    _graph: Any | None = field(default=None, init=False, repr=False)

    @property
    def graph(self) -> Any:
        if self._graph is None:
            self._graph = AgentGraphBuilder(
                runtime_context_store=self.runtime_context_store,
            ).build()
        return self._graph

    def run(
        self,
        request: AgentGraphRequest,
        *,
        runtime_context: GraphRuntimeContext | None = None,
    ) -> AgentGraphResult:
        context_key = ""
        if runtime_context is not None:
            context_key = self.runtime_context_store.put(runtime_context)

        try:
            final_state = self.graph.invoke(
                request.to_state(runtime_context_key=context_key),
                config={"recursion_limit": self.recursion_limit},
            )
            if runtime_context is not None:
                self._flush_working_memory(final_state, runtime_context)
            return AgentGraphResult.from_state(final_state)
        finally:
            if context_key:
                self.runtime_context_store.discard(context_key)

    @staticmethod
    def _flush_working_memory(
        final_state: dict[str, Any],
        runtime_context: GraphRuntimeContext,
    ) -> None:
        if final_state.get("route") != "analysis":
            return
        services = runtime_context.services
        if services is None or not isinstance(services.session_memory, StructuredSessionMemory):
            return
        working_memory = final_state.get("working_memory")
        if not isinstance(working_memory, dict):
            return
        WorkingMemoryFlusher(services.session_memory).flush(working_memory)
