from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.messages import SystemMessage
from langchain_core.tools import ToolException

from backend.agent.callbacks import ToolCollector
from backend.agent.context_compaction import compact_context_if_needed
from backend.agent.dependencies import AgentRuntimeDependencies
from backend.agent.models import AgentOutcome, AgentResponse, ErrorCategory
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
    if context_messages:
        insert_at = 0
        while insert_at < len(messages) and isinstance(messages[insert_at], SystemMessage):
            insert_at += 1
        messages = [*messages[:insert_at], *context_messages, *messages[insert_at:]]
    return messages


def _with_semantic_metric_footer(
    text: str,
    artifacts: list[Any],
) -> str:
    metrics = _semantic_metric_contracts(artifacts)

    footer: list[str] = []
    seen: set[tuple[str, str]] = set()
    for metric in metrics:
        name = str(metric.get("name") or metric.get("key") or "").strip()
        formula = str(metric.get("formula") or "").strip()
        signature = (name.casefold(), formula.casefold())
        if not name or not formula or signature in seen:
            continue
        seen.add(signature)
        if name.casefold() in text.casefold() and formula.casefold() in text.casefold():
            continue
        footer.append(f"Semantic metric: {name}; Formula: {formula}")

    return "\n\n".join([text.strip(), *footer]).strip()


def _semantic_metric_contracts(artifacts: list[Any]) -> list[dict[str, Any]]:
    roots = artifacts[-1:]
    artifacts_by_id = {
        str(artifact_id): artifact for artifact in artifacts if (artifact_id := getattr(artifact, "id", None))
    }
    lineage: list[Any] = []
    pending = list(roots)
    seen: set[str] = set()
    while pending:
        artifact = pending.pop()
        artifact_id = str(getattr(artifact, "id", None) or f"object:{id(artifact)}")
        if artifact_id in seen:
            continue
        seen.add(artifact_id)
        lineage.append(artifact)
        pending.extend(
            artifacts_by_id[parent_id]
            for parent_id in getattr(artifact, "parent_ids", [])
            if parent_id in artifacts_by_id
        )

    metrics: list[dict[str, Any]] = []
    for artifact in lineage:
        meta = getattr(artifact, "meta", {})
        semantic = meta.get("semantic_metric") if isinstance(meta, dict) else None
        if not isinstance(semantic, dict):
            continue
        metrics.extend(metric for metric in semantic.get("metrics") or [] if isinstance(metric, dict))
    return metrics


def agent_node(
    state: AgentGraphState,
    deps: AgentRuntimeDependencies,
) -> dict[str, Any]:
    """Execute the generic agent tool loop for prepared analysis state."""
    df = state.get("df")
    requested_tool_key = state.get("requested_tool_key")
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
        requested_tool_key=requested_tool_key,
        df=df,
        session_source=state.get("session_source"),
        tool_db_runtime=tool_db_runtime,
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
    messages = []
    try:
        execution_system_prompt = build_execution_system_prompt(execution_prompt_request)

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
        response.final_text = _with_semantic_metric_footer(
            response.final_text,
            response.artifacts,
        )
        messages = list(loop_request.messages or messages)
    except Exception as exc:
        artifacts, tool_calls, tool_names = collect_tool_stats(callbacks)
        error_category = ErrorCategory.TOOL if isinstance(exc, ToolException) else ErrorCategory.INTERNAL
        response = AgentResponse(
            final_text=artifact_recovery_text(artifacts),
            reasoning=f"Agent step failed: {exc}",
            artifacts=artifacts,
            route="analysis",
            tool_calls=tool_calls,
            tool_names=tool_names,
            outcome=(
                AgentOutcome.partial(error_category) if artifacts else AgentOutcome.failed(error_category)
            ),
        )

    elapsed_sec = time.perf_counter() - started_at
    if elapsed_sec > max(1, deps.settings.agent_step_timeout_sec):
        response.reasoning = (
            (response.reasoning or "") + f"\n\nStep timeout guard triggered ({int(elapsed_sec * 1000)} ms)."
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
