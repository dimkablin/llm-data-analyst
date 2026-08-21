from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import replace
from types import ModuleType

from opentelemetry import trace
from opentelemetry.trace import StatusCode

import backend.observability.phoenix as phoenix
from backend.agent.models import AgentOutcome, AgentResponse, ErrorCategory
from backend.observability.phoenix import (
    query_trace_context,
    record_agent_outcome_on_active_span,
)


class _RecordingSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}
        self.status = None

    def is_recording(self) -> bool:
        return True

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def set_status(self, status) -> None:
        self.status = status


def test_invalid_agent_outcome_marks_active_span_error(monkeypatch) -> None:
    span = _RecordingSpan()
    monkeypatch.setattr(trace, "get_current_span", lambda: span)
    response = AgentResponse(
        final_text="Provider unavailable.",
        reasoning="",
        artifacts=[],
        route="analysis",
        contract_valid=False,
        terminal_status="unavailable",
    )

    record_agent_outcome_on_active_span(response)

    assert span.attributes["agent.contract_valid"] is False
    assert span.attributes["agent.task_contract_satisfied"] is False
    assert span.attributes["agent.response_envelope_valid"] is True
    assert span.attributes["agent.error_category"] == "tool"
    assert span.attributes["agent.terminal_status"] == "unavailable"
    assert span.status.status_code is StatusCode.ERROR


def test_model_connection_failure_is_distinct_from_tool_errors(monkeypatch) -> None:
    span = _RecordingSpan()
    monkeypatch.setattr(trace, "get_current_span", lambda: span)
    response = AgentResponse(
        final_text="Model unavailable.",
        reasoning="connection refused",
        artifacts=[],
        llm_unreachable=True,
        outcome=AgentOutcome.unavailable(ErrorCategory.MODEL),
    )

    record_agent_outcome_on_active_span(response)

    assert span.attributes["agent.error_category"] == "model"
    assert span.attributes["agent.tool_error_count"] == 0
    assert span.status.status_code is StatusCode.ERROR


def test_query_trace_context_exposes_recording_outcome_span(monkeypatch) -> None:
    span = _RecordingSpan()

    class _Tracer:
        @contextmanager
        def start_as_current_span(self, _name: str):
            monkeypatch.setattr(trace, "get_current_span", lambda: span)
            yield span

    @contextmanager
    def _using_attributes(**_kwargs):
        yield

    monkeypatch.setattr(phoenix, "_initialized", True)
    monkeypatch.setattr(phoenix, "initialize_phoenix", lambda: None)
    monkeypatch.setattr(
        phoenix,
        "settings",
        replace(phoenix.settings, phoenix_enabled=True),
    )
    monkeypatch.setattr(trace, "get_tracer", lambda _name: _Tracer())
    openinference = ModuleType("openinference")
    instrumentation = ModuleType("openinference.instrumentation")
    instrumentation.using_attributes = _using_attributes
    openinference.instrumentation = instrumentation
    monkeypatch.setitem(sys.modules, "openinference", openinference)
    monkeypatch.setitem(sys.modules, "openinference.instrumentation", instrumentation)

    with query_trace_context(
        session_id="session-1",
        user_id=1,
        username="tester",
        request_kind="query",
        use_history=False,
        include_reasoning=False,
        query="question",
    ):
        record_agent_outcome_on_active_span(
            AgentResponse(
                final_text="Provider unavailable.",
                reasoning="",
                artifacts=[],
                route="analysis",
                contract_valid=False,
                terminal_status="unavailable",
            )
        )

    assert span.attributes["session.id"] == "session-1"
    assert span.attributes["metadata.request_kind"] == "query"
    assert span.status.status_code is StatusCode.ERROR


def test_partial_outcome_is_degraded_not_error(monkeypatch) -> None:
    span = _RecordingSpan()
    monkeypatch.setattr(trace, "get_current_span", lambda: span)
    record_agent_outcome_on_active_span(
        AgentResponse(
            final_text="Useful interim result with warning.",
            reasoning="Recovered after retries.",
            artifacts=[],
            route="analysis",
            outcome=AgentOutcome.partial(ErrorCategory.TOOL),
        )
    )

    assert span.attributes["agent.degraded"] is True
    assert span.status is None or span.status.status_code is not StatusCode.ERROR
