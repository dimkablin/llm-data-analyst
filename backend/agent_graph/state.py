from __future__ import annotations

from typing import Any, Literal, TypedDict


Route = Literal["chat", "summary", "analysis"]
NodeStatus = Literal["pending", "running", "done", "error"]


class ArtifactRefState(TypedDict, total=False):
    """Serializable projection of an artifact produced by a tool."""

    id: str
    name: str
    type: str
    tool_name: str
    step_index: int
    schema: dict[str, str] | None
    row_count: int | None
    summary: str | None


class WorkingMemoryState(TypedDict, total=False):
    """Per-run working memory kept in the graph state.

    Only JSON-like values belong here.  Runtime objects such as DataFrames,
    callbacks, tools, sandboxes, DB clients and runner instances must stay in
    ``GraphRuntimeContext``.
    """

    goal: str
    step_index: int
    tool_call_count: int
    artifact_refs: list[ArtifactRefState]
    sandbox_var_names: list[str]
    current_plan: list[str]
    completed_actions: list[str]
    last_tool_result_summary: str


class MessageState(TypedDict, total=False):
    """Checkpoint-friendly message representation.

    LangChain messages can be reconstructed at node boundaries.  Keeping state
    plain here avoids coupling durable checkpoints to provider-specific message
    classes.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None
    tool_call_id: str | None
    tool_calls: list[dict[str, Any]]
    additional_kwargs: dict[str, Any]


class ToolCallState(TypedDict, total=False):
    id: str
    name: str
    args: dict[str, Any]
    type: str


class AgentGraphState(TypedDict, total=False):
    """Serializable state for the future LangGraph-first agent runtime."""

    # Request
    prompt: str
    history: list[dict[str, Any]]
    use_history: bool
    include_reasoning: bool
    trace_context: dict[str, Any]
    session_source: dict[str, Any]
    selected_skill_ids: list[str]

    # Runtime lookup key.  The value is serializable; the actual runtime object
    # lives outside state and can later be rehydrated from services/session ids.
    runtime_context_key: str

    # Routing and budgets
    route: Route
    done: bool
    stop_reason: str
    step_index: int
    max_steps: int
    status: NodeStatus
    error: str

    # Conversation/tool loop
    available_tool_keys: list[str]
    capability_context: dict[str, Any]
    messages: list[MessageState]
    pending_tool_calls: list[ToolCallState]
    tool_call_count: int
    tool_names: list[str]
    working_memory: WorkingMemoryState

    # Output
    final_text: str
    reasoning: str | None
    reasoning_steps: list[str]
    artifact_refs: list[ArtifactRefState]
