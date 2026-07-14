from __future__ import annotations

from backend.agent.constants import LLM_UNAVAILABLE_USER_TEXT
from backend.agent.dependencies import AgentRuntimeDependencies
from backend.agent.models import AgentResponse
from backend.agent.services.events import emit_phase_event, emit_progress_event
from backend.agent.services.finalization import (
    fallback_text as build_fallback_text,
)
from backend.agent.state import AgentGraphState
from backend.agent.tool_loop import artifact_summary_text, collect_tool_stats


def finalize_node(
    state: AgentGraphState,
    _deps: AgentRuntimeDependencies,
) -> dict[str, AgentResponse]:
    """Return the prepared final response without analytical repair passes."""
    callbacks = state.get("callbacks", [])
    step_index = int(state.get("step_index", 0))
    max_steps = int(state.get("max_steps", 1))

    emit_phase_event(
        callbacks,
        phase="finalize",
        title="Финализация",
        content="",
        step_index=step_index,
        max_steps=max_steps,
        status="streaming",
    )
    emit_progress_event(
        callbacks,
        phase="finalize",
        title="Формирую финальный ответ",
        details="Собираю финальный ответ из готового состояния выполнения.",
        step_index=step_index,
        max_steps=max_steps,
    )

    response = state.get("response")
    prompt = state.get("prompt", "")
    df = state.get("df")
    stop_reason = state.get("stop_reason")
    collected_artifacts, tool_calls, tool_names = collect_tool_stats(callbacks)

    if response is None:
        artifacts = list(collected_artifacts)
        text = (
            LLM_UNAVAILABLE_USER_TEXT
            if state.get("llm_unreachable")
            else artifact_summary_text(artifacts)
            or build_fallback_text(prompt, df, stop_reason=stop_reason)
        )
        response = AgentResponse(
            final_text=text,
            reasoning=(
                "LLM invoke failed"
                if state.get("llm_unreachable")
                else "No response produced by graph."
            ),
            artifacts=artifacts,
            route="analysis",
            tool_calls=tool_calls,
            tool_names=tool_names,
            llm_unreachable=bool(state.get("llm_unreachable")),
        )
    else:
        if collected_artifacts and not response.artifacts:
            response.artifacts = list(collected_artifacts)
        if getattr(response, "llm_unreachable", False):
            response.final_text = (response.final_text or "").strip() or LLM_UNAVAILABLE_USER_TEXT
        elif not response.final_text.strip():
            response.final_text = (
                artifact_summary_text(list(response.artifacts or []))
                or build_fallback_text(prompt, df, stop_reason=stop_reason)
            )

    emit_phase_event(
        callbacks,
        phase="finalize",
        title="Финализация",
        content="Ответ сформирован.",
        step_index=step_index,
        max_steps=max_steps,
        status="done",
    )

    return {"response": response}
