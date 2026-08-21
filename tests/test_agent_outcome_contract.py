from __future__ import annotations

from types import SimpleNamespace

from backend.agent.models import AgentOutcome, AgentResponse, ErrorCategory, TerminalStatus
from backend.agent.runner import AgentRunner
from backend.agent.runtime_contracts import AgentRunRequest
from backend.agent.services.message_builder import (
    ExecutionSystemPromptRequest,
    build_execution_system_prompt,
)
from backend.core.config import Settings


def test_agent_response_defaults_are_safe_failure() -> None:
    response = AgentResponse(
        final_text="fallback",
        reasoning=None,
        artifacts=[],
    )

    assert response.response_envelope_valid is True
    assert response.task_contract_satisfied is False
    assert response.terminal_status is TerminalStatus.FAILED
    assert response.error_category is ErrorCategory.INTERNAL
    assert response.task_contract_satisfied is False


def test_graph_exception_never_becomes_success(monkeypatch, tmp_path) -> None:
    runner = AgentRunner(settings=Settings(storage_dir=str(tmp_path)))
    cache_writes: list[object] = []
    monkeypatch.setattr(
        runner,
        "_cache_set",
        lambda *_args, **_kwargs: cache_writes.append((_args, _kwargs)),
    )
    monkeypatch.setattr(
        runner,
        "_prepare_request",
        lambda _request: SimpleNamespace(
            state_update={
                "registry_snapshot": SimpleNamespace(fingerprint="snapshot"),
                "tools": [],
            }
        ),
    )
    monkeypatch.setattr(runner._graph, "invoke", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    response = runner.run(
        AgentRunRequest(
            prompt="question",
            trace_context={"request_kind": "query"},
        )
    ).response

    assert response.response_envelope_valid is True
    assert response.task_contract_satisfied is False
    assert response.terminal_status is TerminalStatus.FAILED
    assert response.error_category is ErrorCategory.GRAPH
    assert response.task_contract_satisfied is False
    assert cache_writes == []


def test_legacy_contract_valid_alias_maps_to_task_satisfaction() -> None:
    response = AgentResponse(
        final_text="done",
        reasoning=None,
        artifacts=[],
        outcome=AgentOutcome.success(),
    )

    assert response.contract_valid is True


def test_cache_key_includes_registry_snapshot(tmp_path) -> None:
    runner = AgentRunner(settings=Settings(storage_dir=str(tmp_path)))
    common = {
        "df": None,
        "prompt": "same request",
        "history": [],
        "use_history": False,
        "include_reasoning": False,
    }

    first = runner._query_cache_key(**common, registry_snapshot_fingerprint="one")
    second = runner._query_cache_key(**common, registry_snapshot_fingerprint="two")

    assert first != second


def test_cache_key_separates_forced_plan_preference(tmp_path) -> None:
    adaptive = AgentRunner(settings=Settings(storage_dir=str(tmp_path)))
    always = AgentRunner(
        settings=Settings(
            storage_dir=str(tmp_path),
            always_use_analysis_plan=True,
        )
    )
    common = {
        "df": None,
        "prompt": "same request",
        "history": [],
        "use_history": False,
        "include_reasoning": False,
    }

    assert adaptive._query_cache_key(**common) != always._query_cache_key(**common)


def test_shared_cache_does_not_cross_registry_snapshots(monkeypatch, tmp_path) -> None:
    shared: dict[str, object] = {}
    monkeypatch.setattr(
        "backend.agent.runner.redis_cache.get_pickle",
        lambda key: shared.get(key),
    )
    monkeypatch.setattr(
        "backend.agent.runner.redis_cache.set_pickle",
        lambda key, value, ttl_sec: shared.__setitem__(key, value),
    )
    settings = Settings(storage_dir=str(tmp_path), agent_cache_enabled=True)
    first = AgentRunner(settings=settings)
    second = AgentRunner(settings=settings)
    monkeypatch.setattr(
        first,
        "_prepare_request",
        lambda _request: SimpleNamespace(
            state_update={"registry_snapshot": SimpleNamespace(fingerprint="one")}
        ),
    )
    monkeypatch.setattr(
        second,
        "_prepare_request",
        lambda _request: SimpleNamespace(
            state_update={"registry_snapshot": SimpleNamespace(fingerprint="two")}
        ),
    )
    monkeypatch.setattr(
        first._graph,
        "invoke",
        lambda *_args, **_kwargs: {
            "response": AgentResponse(
                final_text="provider one",
                reasoning=None,
                artifacts=[],
                outcome=AgentOutcome.success(),
            )
        },
    )
    monkeypatch.setattr(
        second._graph,
        "invoke",
        lambda *_args, **_kwargs: {
            "response": AgentResponse(
                final_text="provider two",
                reasoning=None,
                artifacts=[],
                outcome=AgentOutcome.success(),
            )
        },
    )
    request = AgentRunRequest(
        prompt="same",
        trace_context={"request_kind": "query"},
    )

    assert first.run(request).response.final_text == "provider one"
    assert second.run(request).response.final_text == "provider two"
    assert len(shared) == 2


def test_disabled_tools_are_absent_from_effective_system_prompt(tmp_path) -> None:
    runner = AgentRunner(
        settings=Settings(storage_dir=str(tmp_path)),
        allowed_tool_keys={"memory", "session_note"},
    )
    prepared = runner._prepare_request(AgentRunRequest(prompt="hello"))
    prompt = build_execution_system_prompt(
        ExecutionSystemPromptRequest(
            settings=runner.settings,
            skill_registry=runner.skill_registry,
            capability_context=prepared.state_update["capability_context"],
        )
    )

    assert "`memory`" in prompt
    assert "`forecast_tool`" not in prompt
    assert "`sql_tool`" not in prompt
    assert "`pandas_tool`" not in prompt


def test_public_runner_does_not_create_runtime_contract_from_words(tmp_path) -> None:
    runner = AgentRunner(
        settings=Settings(storage_dir=str(tmp_path)),
        allowed_tool_keys={"memory", "session_note"},
    )

    prepared = runner._prepare_request(
        AgentRunRequest(prompt="Сделай прогноз на следующий месяц")
    )

    assert "task_contract" not in prepared.state_update
    snapshot = prepared.state_update["registry_snapshot"]
    assert snapshot.resolution_for("forecast").status == "unavailable"


def test_resumed_legacy_history_does_not_restore_inactive_tool(tmp_path) -> None:
    runner = AgentRunner(
        settings=Settings(storage_dir=str(tmp_path)),
        allowed_tool_keys={"memory", "session_note"},
    )

    prepared = runner._prepare_request(
        AgentRunRequest(
            prompt="continue",
            use_history=True,
            history=[
                {
                    "role": "assistant",
                    "content": "Previously I called `forecast_tool`.",
                }
            ],
        )
    )

    snapshot = prepared.state_update["registry_snapshot"]
    assert "forecast_tool" not in snapshot.catalog.tool_keys
    assert "forecast_tool" not in prepared.state_update["capability_context"][
        "available_tool_keys"
    ]


def test_prompt_model_and_executor_share_snapshot_surface(tmp_path) -> None:
    runner = AgentRunner(
        settings=Settings(storage_dir=str(tmp_path)),
        allowed_tool_keys={"memory", "session_note"},
    )

    prepared = runner._prepare_request(AgentRunRequest(prompt="hello"))
    snapshot = prepared.state_update["registry_snapshot"]
    tool_names = [tool.name for tool in prepared.state_update["tools"]]

    assert tool_names == snapshot.catalog.tool_keys
    assert tool_names == prepared.state_update["capability_context"][
        "available_tool_keys"
    ]
    assert snapshot.fingerprint == prepared.state_update["capability_context"][
        "registry_snapshot_fingerprint"
    ]
