from __future__ import annotations

from backend.agent.callbacks import (
    AgentProgressCollector,
    ContextUsageCollector,
    PhaseCollector,
)
from backend.agent.context_window import ContextUsageSnapshot


def collect_progress_collectors(callbacks: list) -> list[AgentProgressCollector]:
    return [callback for callback in callbacks if isinstance(callback, AgentProgressCollector)]


def emit_progress_event(
    callbacks: list,
    *,
    phase: str,
    title: str,
    details: str = "",
    step_index: int | None = None,
    max_steps: int | None = None,
) -> None:
    for collector in collect_progress_collectors(callbacks):
        collector.add_event(
            phase=phase,
            title=title,
            details=details,
            step_index=step_index,
            max_steps=max_steps,
        )


def collect_phase_collectors(callbacks: list) -> list[PhaseCollector]:
    return [callback for callback in callbacks if isinstance(callback, PhaseCollector)]


def collect_context_usage_collectors(callbacks: list) -> list[ContextUsageCollector]:
    return [
        callback for callback in callbacks if isinstance(callback, ContextUsageCollector)
    ]


def emit_context_usage_event(
    callbacks: list,
    snapshot: ContextUsageSnapshot,
) -> None:
    for collector in collect_context_usage_collectors(callbacks):
        collector.add_snapshot(snapshot)


def emit_phase_event(
    callbacks: list,
    *,
    phase: str,
    title: str,
    content: str = "",
    step_index: int | None = None,
    max_steps: int | None = None,
    status: str | None = None,
) -> None:
    for collector in collect_phase_collectors(callbacks):
        collector.add_phase(
            phase=phase,
            title=title,
            content=content,
            step_index=step_index,
            max_steps=max_steps,
            status=status,
        )
        graph_tracker = getattr(collector, "graph_tracker", None)
        if graph_tracker is None:
            continue
        safe_step_index = step_index if isinstance(step_index, int) else 0
        if status == "streaming":
            graph_tracker.phase_start(phase, safe_step_index)
        elif status in ("done", "pass", "fail", "error"):
            graph_tracker.phase_end(
                phase,
                safe_step_index,
                status="done" if status in ("done", "pass") else "error",
            )
        collector._graph_version += 1  # noqa: SLF001
