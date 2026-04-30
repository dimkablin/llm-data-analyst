from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from backend.agent_graph.services import AgentRuntimeServices


@dataclass(slots=True)
class GraphRuntimeContext:
    """Non-serializable dependencies for one graph invocation.

    The graph state stores only ``runtime_context_key``.  Keeping these objects
    outside state is the first step toward real LangGraph checkpointing: a
    durable checkpoint should contain facts about the run, not live Python
    objects that cannot survive process restart.
    """

    services: AgentRuntimeServices | None = None
    df: Any | None = None
    callbacks: list[Any] = field(default_factory=list)
    tools: list[Any] = field(default_factory=list)
    sandbox: Any | None = None
    tool_db_runtime: Any | None = None
    llm_factory: Any | None = None
    execution_system_prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class RuntimeContextStore:
    """Small in-process registry for graph runtime dependencies.

    This is intentionally narrow.  The next migration step can replace the
    in-memory registry with a rehydration layer based on session id, user id and
    request metadata without changing the graph state's shape.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, GraphRuntimeContext] = {}

    def put(self, context: GraphRuntimeContext) -> str:
        key = uuid.uuid4().hex
        with self._lock:
            self._items[key] = context
        return key

    def get(self, key: str) -> GraphRuntimeContext:
        with self._lock:
            context = self._items.get(key)
        if context is None:
            raise KeyError(f"Graph runtime context not found: {key}")
        return context

    def discard(self, key: str) -> None:
        with self._lock:
            self._items.pop(key, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
