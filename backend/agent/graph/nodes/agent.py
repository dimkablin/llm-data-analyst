from __future__ import annotations

import logging
import time
from typing import Any

from backend.agent.callbacks import ToolCollector
from backend.agent.contracts import AnalysisTaskContract
from backend.agent.context_compaction import compact_context_if_needed
from backend.agent.dependencies import AgentRuntimeDependencies
from backend.agent.models import AgentResponse
from backend.agent.services.events import emit_phase_event, emit_progress_event
from backend.agent.services.message_builder import (
    ExecutionSystemPromptRequest,
    MessageBuildRequest,
    build_execution_context_messages,
    build_execution_system_prompt,
    build_messages,
)
from backend.agent.state import AgentGraphState
from backend.agent.tool_loop import (
    ToolLoopRequest,
    artifact_recovery_text,
    collect_tool_stats,
    direct_tool_loop,
)
from backend.agent.working_memory import AnalysisWorkingMemory
from backend.artifacts.execution import artifact_type_label

logger = logging.getLogger(__name__)


def _build_agent_messages(
    *,
    state: AgentGraphState,
    deps: AgentRuntimeDependencies,
    execution_system_prompt: str,
    context_messages: list,
) -> list:
    messages = build_messages(
        MessageBuildRequest(
            prompt=state.get("prompt", ""),
            history=state.get("history", []),
            use_history=state.get("use_history", True),
            settings=deps.settings,
            user_memory=deps.user_memory,
            session_memory=deps.session_memory,
            system_prompt=execution_system_prompt,
            enable_thinking=state.get("include_reasoning", False),
        )
    )
    if context_messages and messages:
        messages = [*messages[:-1], *context_messages, messages[-1]]
    return messages


def agent_node(
    state: AgentGraphState,
    deps: AgentRuntimeDependencies,
) -> dict[str, Any]:
    """Execute the generic agent tool loop for prepared analysis state."""
    df = state.get("df")
    tools = state.get("tools", [])
    callbacks = state.get("callbacks", [])

    step_index = 1
    max_steps = int(state.get("max_steps", deps.settings.agent_inner_recursion_limit))

    tool_db_runtime = state.get("tool_db_runtime")
    sandbox = state.get("sandbox")
    working_memory: AnalysisWorkingMemory | None = state.get("working_memory")

    execution_prompt_request = ExecutionSystemPromptRequest(
        settings=deps.settings,
        skill_registry=deps.skill_registry,
        enabled_analytical_skill_ids=deps.enabled_analytical_skill_ids,
        capability_context=state.get("capability_context"),
        sandbox=sandbox,
        selected_skill_ids=state.get("selected_skill_ids") or [],
        df=df,
        session_source=state.get("session_source"),
        tool_db_runtime=tool_db_runtime,
    )
    execution_system_prompt = build_execution_system_prompt(execution_prompt_request)
    task_contract = state.get("task_contract") or AnalysisTaskContract.from_prompt(
        state.get("prompt", "")
    )
    contract_prompt_block = deps.prompt_context_builder.task_contract_prompt_block(
        task_contract
    )
    if contract_prompt_block:
        execution_system_prompt += f"\n\n{contract_prompt_block}"

    context_messages = build_execution_context_messages(execution_prompt_request)
    messages = _build_agent_messages(
        state=state,
        deps=deps,
        execution_system_prompt=execution_system_prompt,
        context_messages=context_messages,
    )
    if state.get("use_history", True):
        compaction = compact_context_if_needed(
            messages=messages,
            history=list(state.get("history", []) or []),
            settings=deps.settings,
            session_memory=deps.session_memory,
            callbacks=callbacks,
            include_reasoning=state.get("include_reasoning", False),
        )
        if compaction.status == "done":
            messages = _build_agent_messages(
                state=state,
                deps=deps,
                execution_system_prompt=execution_system_prompt,
                context_messages=context_messages,
            )

    emit_phase_event(
        callbacks,
        phase="act",
        title="Выполнение анализа",
        content="",
        step_index=step_index,
        max_steps=max_steps,
        status="streaming",
    )
    emit_progress_event(
        callbacks,
        phase="act",
        title="Выполняю анализ",
        details="Подбираю инструмент и формирую вызов tool.",
        step_index=step_index,
        max_steps=max_steps,
    )

    tool_collector = next((cb for cb in callbacks if isinstance(cb, ToolCollector)), None)
    tool_events_offset = len(tool_collector.events) if tool_collector else 0

    started_at = time.perf_counter()
    try:
        loop_request = ToolLoopRequest(
            settings=deps.settings,
            include_reasoning=state.get("include_reasoning", False),
            tools=tools,
            callbacks=callbacks,
            max_iterations=max_steps,
            trace_context=state.get("trace_context"),
            working_memory=working_memory,
            messages=messages,
            cancel_event=state.get("cancel_event"),
        )
        response = direct_tool_loop(loop_request)
        messages = list(loop_request.messages or messages)
    except Exception as exc:
        artifacts, tool_calls, tool_names = collect_tool_stats(callbacks)
        response = AgentResponse(
            final_text=artifact_recovery_text(artifacts),
            reasoning=f"Agent step failed: {exc}",
            artifacts=artifacts,
            route="analysis",
            tool_calls=tool_calls,
            tool_names=tool_names,
        )

    elapsed_sec = time.perf_counter() - started_at
    if elapsed_sec > max(1, deps.settings.agent_step_timeout_sec):
        response.reasoning = (
            (response.reasoning or "")
            + f"\n\nStep timeout guard triggered ({int(elapsed_sec * 1000)} ms)."
        ).strip()

    tool_summary_lines: list[str] = []
    if tool_collector is not None:
        for ev in tool_collector.events[tool_events_offset:]:
            if ev.get("phase") == "start":
                name = ev.get("tool_name", "")
                inp = (ev.get("input_preview") or "").strip()
                if inp:
                    tool_summary_lines.append(f"**{name}**: {inp[:400]}")
            elif ev.get("phase") == "end" and ev.get("code_preview"):
                code = ev["code_preview"].strip()
                tool_summary_lines.append(f"```sql\n{code[:800]}\n```")
    if not tool_summary_lines and response.tool_names:
        tool_summary_lines.append(f"Инструменты: {', '.join(response.tool_names)}")
    if response.artifacts:
        types = [artifact_type_label(getattr(a, "artifact_type", "")) for a in response.artifacts]
        tool_summary_lines.append(f"Артефакты: {', '.join(t for t in types if t)}")

    emit_phase_event(
        callbacks,
        phase="act",
        title="Анализ завершён",
        content="\n".join(tool_summary_lines) if tool_summary_lines else "Шаг выполнен.",
        step_index=step_index,
        max_steps=max_steps,
        status="done",
    )

    return {
        "response": response,
        "step_index": step_index,
        "working_memory": working_memory,
        "messages": messages,
    }
