from __future__ import annotations

from typing import Annotated, Any, TypedDict

import pandas as pd
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from backend.agent.contracts import AnalysisTaskContract
from backend.agent.models import AgentResponse
from backend.agent.working_memory import AnalysisWorkingMemory
from backend.skills.contracts import SkillExecutionRequirement


class AgentGraphState(TypedDict, total=False):
    # Input
    df: pd.DataFrame | None
    prompt: str
    messages: Annotated[list[BaseMessage], add_messages]
    history: list[dict[str, Any]]
    use_history: bool
    include_reasoning: bool
    callbacks: list
    trace_context: dict[str, Any]
    session_source: dict[str, Any]
    selected_skill_ids: list[str]
    cancel_event: Any

    # Prepare -> Agent handoff
    done: bool
    stop_reason: str
    step_index: int
    max_steps: int
    tools: list
    capability_context: dict[str, Any]
    task_contract: AnalysisTaskContract
    skill_execution_requirements: list[SkillExecutionRequirement]
    llm_unreachable: bool
    sandbox: Any
    tool_db_runtime: Any  # RuntimeDBConnectionConfig | None - resolved once in context preparation
    context_budget: Any
    retrieved_context: Any

    working_memory: AnalysisWorkingMemory | None  # per-query ephemeral state

    # Output
    response: AgentResponse
