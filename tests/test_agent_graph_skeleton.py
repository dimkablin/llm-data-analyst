from __future__ import annotations

import json

import pandas as pd
from langchain_core.messages import AIMessage

from backend.agent_graph import (
    AgentGraphRequest,
    AgentGraphResult,
    AgentGraphRunner,
    AgentGraphState,
    AgentRuntimeServices,
    GraphRuntimeContext,
    RuntimeContextStore,
    build_agent_graph,
)
from backend.agent_graph.adapter import AgentGraphQueryRunner
from backend.agent_graph.routing import is_chat_query
from backend.core.config import Settings
from backend.sessions.session_memory import StructuredSessionMemory


class _FakeToolBoundLlm:
    def __init__(self, responses):
        self._responses = list(responses)
        self.bound_tools = []

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages, config=None):
        _ = messages, config
        return self._responses.pop(0)


class _FakeLlmFactory:
    def __init__(self, responses):
        self._llm = _FakeToolBoundLlm(responses)

    def build(self, **kwargs):
        _ = kwargs
        return self._llm


class _FakeGraphRunner:
    def __init__(self) -> None:
        self.runtime_context = None

    def run(self, request, *, runtime_context=None):
        self.runtime_context = runtime_context
        return AgentGraphResult(
            final_text=f"handled: {request.prompt}",
            route="chat",
            tool_calls=0,
            tool_names=[],
        )


class _FakeTitleLlm:
    def invoke(self, messages, config=None):
        _ = messages, config
        return AIMessage(content='"Выручка по регионам продаж."')


class _FakeUserMemoryService:
    def __init__(self) -> None:
        self.scheduled = []

    def schedule_consolidation(self, user_id, notes, llm_invoke):
        self.scheduled.append((user_id, notes, llm_invoke))


class _FakeSessionStore:
    def __init__(self) -> None:
        self.appended_notes = []
        self.persisted_memory = None

    def append_session_memory(self, session_id, note):
        self.appended_notes.append((session_id, note))

    def set_structured_memory(self, session_id, memory):
        self.persisted_memory = (session_id, memory)


class _FakePlannerTool:
    name = "planner_tool"

    def invoke(self, tool_call, config=None):
        _ = tool_call, config
        return "1. Inspect the metric with value_tool\n2. Summarize the result"


class _FakeArtifactTool:
    name = "value_tool"

    def invoke(self, tool_call, config=None):
        _ = tool_call, config
        return (
            "created metric",
            {
                "artifact_type": "value",
                "items": {"total_revenue": 42},
            },
        )


def test_agent_graph_skeleton_compiles_and_runs_empty_prompt() -> None:
    graph = build_agent_graph()

    result = graph.invoke(
        {
            "prompt": "",
            "history": [],
            "use_history": False,
            "include_reasoning": False,
            "trace_context": {},
            "session_source": {},
            "selected_skill_ids": [],
        },
        config={"recursion_limit": 10},
    )

    assert result["route"] == "chat"
    assert result["stop_reason"] == "chat_route"
    assert result["status"] == "done"


def test_agent_graph_state_contract_stays_json_like() -> None:
    state: AgentGraphState = {
        "prompt": "analyze revenue",
        "history": [{"role": "user", "content": "hello"}],
        "use_history": True,
        "include_reasoning": False,
        "trace_context": {"session_id": "s1"},
        "session_source": {"source_type": "csv"},
        "selected_skill_ids": [],
        "runtime_context_key": "ctx",
        "working_memory": {
            "goal": "analyze revenue",
            "step_index": 0,
            "tool_call_count": 0,
            "artifact_refs": [],
            "sandbox_var_names": [],
            "current_plan": [],
            "completed_actions": [],
            "last_tool_result_summary": "",
        },
    }

    json.dumps(state)


def test_runtime_context_store_keeps_live_objects_outside_state() -> None:
    store = RuntimeContextStore()
    callbacks = [object()]
    key = store.put(GraphRuntimeContext(callbacks=callbacks))

    context = store.get(key)

    assert context.callbacks is callbacks
    store.discard(key)


def test_agent_graph_runner_owns_context_lifecycle() -> None:
    runner = AgentGraphRunner()
    context = GraphRuntimeContext(callbacks=[object()])

    result = runner.run(
        AgentGraphRequest(prompt="", use_history=False),
        runtime_context=context,
    )

    assert result.route == "chat"
    assert result.stop_reason == "chat_route"
    assert len(runner.runtime_context_store) == 0


def test_agent_graph_routes_analysis_prompt_to_analysis_path() -> None:
    result = AgentGraphRunner().run(
        AgentGraphRequest(prompt="analyze revenue by region", use_history=False),
    )

    assert result.route == "analysis"
    assert result.stop_reason == ""


def test_agent_graph_routes_summary_prompt_to_summary_path() -> None:
    result = AgentGraphRunner().run(
        AgentGraphRequest(prompt="executive summary", use_history=False),
    )

    assert result.route == "summary"
    assert result.stop_reason == "summary_route"


def test_agent_graph_chat_intent_helper_replaces_old_runner_import() -> None:
    assert is_chat_query("")
    assert is_chat_query("привет")
    assert is_chat_query("как дела")
    assert not is_chat_query("analyze revenue")
    assert not is_chat_query("покажи таблицу")


def test_agent_graph_builds_analysis_context_with_dataframe() -> None:
    services = AgentRuntimeServices.create(
        settings=Settings(
            llm_base_url="http://localhost:11434/v1",
            llm_model="test-model",
            llm_api_key="test-key",
            skills_dir="./skills",
            storage_dir=".test_storage_agent_graph",
        ),
    )
    context = GraphRuntimeContext(
        services=services,
        df=pd.DataFrame({"revenue": [10, 20], "region": ["A", "B"]}),
        llm_factory=_FakeLlmFactory([AIMessage(content="analysis done")]),
    )

    result = AgentGraphRunner().run(
        AgentGraphRequest(
            prompt="analyze revenue by region",
            use_history=False,
            trace_context={"session_id": "graph-test"},
        ),
        runtime_context=context,
    )

    assert "pandas_tool" in result.raw_state["available_tool_keys"]
    assert "value_tool" in result.raw_state["available_tool_keys"]
    assert result.raw_state["capability_context"]["source_mode"] == "dataset"
    assert "table_analysis" in result.raw_state["capability_context"]["available_capability_keys"]
    assert "revenue" in context.execution_system_prompt
    assert "pandas_tool" in context.execution_system_prompt


def test_agent_graph_executes_tool_call_and_continues_to_final_answer() -> None:
    notes: list[str] = []
    services = AgentRuntimeServices.create(
        settings=Settings(
            llm_base_url="http://localhost:11434/v1",
            llm_model="test-model",
            llm_api_key="test-key",
            skills_dir="./skills",
            storage_dir=".test_storage_agent_graph",
        ),
        memory_note_callback=notes.append,
    )
    context = GraphRuntimeContext(
        services=services,
        llm_factory=_FakeLlmFactory(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call-1",
                            "name": "memory",
                            "args": {"text": "User prefers concise analysis."},
                        },
                    ],
                ),
                AIMessage(content="done"),
            ],
        ),
    )

    result = AgentGraphRunner().run(
        AgentGraphRequest(prompt="analyze and remember preference", use_history=False),
        runtime_context=context,
    )

    assert result.final_text == "done"
    assert result.tool_calls == 1
    assert result.tool_names == ["memory"]
    assert notes == ["User prefers concise analysis."]


def test_agent_graph_runs_planner_node_before_llm() -> None:
    services = AgentRuntimeServices.create(
        settings=Settings(
            llm_base_url="http://localhost:11434/v1",
            llm_model="test-model",
            llm_api_key="test-key",
            skills_dir="./skills",
            storage_dir=".test_storage_agent_graph",
        ),
    )
    context = GraphRuntimeContext(
        services=services,
        tools=[_FakePlannerTool()],
        llm_factory=_FakeLlmFactory([AIMessage(content="planned answer")]),
    )

    result = AgentGraphRunner().run(
        AgentGraphRequest(prompt="analyze revenue", use_history=False),
        runtime_context=context,
    )

    assert result.final_text == "planned answer"
    assert "Preliminary Analysis Plan" in context.execution_system_prompt
    assert result.raw_state["working_memory"]["current_plan"][0].startswith("1. Inspect")


def test_agent_graph_maps_artifact_refs_and_flushes_session_memory() -> None:
    session_memory = StructuredSessionMemory()
    services = AgentRuntimeServices.create(
        settings=Settings(
            llm_base_url="http://localhost:11434/v1",
            llm_model="test-model",
            llm_api_key="test-key",
            skills_dir="./skills",
            storage_dir=".test_storage_agent_graph",
        ),
        session_memory=session_memory,
    )
    context = GraphRuntimeContext(
        services=services,
        tools=[_FakeArtifactTool()],
        llm_factory=_FakeLlmFactory(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call-value",
                            "name": "value_tool",
                            "args": {"code": "tool_result = {'total_revenue': 42}"},
                        },
                    ],
                ),
                AIMessage(content="answer with artifact"),
            ],
        ),
    )

    result = AgentGraphRunner().run(
        AgentGraphRequest(prompt="analyze metric", use_history=False),
        runtime_context=context,
    )

    assert result.final_text == "answer with artifact"
    assert result.artifact_refs[0]["name"] == "total_revenue"
    assert session_memory.turn_count == 1
    assert session_memory.artifact_index[0].name == "total_revenue"
    assert any("value_tool" in finding for finding in session_memory.key_findings)


def test_query_runner_adapter_exposes_legacy_run_query_contract() -> None:
    query_runner = AgentGraphQueryRunner(
        Settings(
            llm_base_url="http://localhost:11434/v1",
            llm_model="test-model",
            llm_api_key="test-key",
            skills_dir="./skills",
        ),
    )
    fake_graph_runner = _FakeGraphRunner()
    query_runner._graph_runner = fake_graph_runner

    response = query_runner.run_query(
        None,
        "hello",
        [],
        False,
        False,
        [],
        {"request_kind": "evaluate"},
        {},
        [],
    )

    assert response.final_text == "handled: hello"
    assert response.route == "chat"
    assert fake_graph_runner.runtime_context.services is query_runner.services


def test_query_runner_adapter_generates_chat_title() -> None:
    query_runner = AgentGraphQueryRunner(
        Settings(
            llm_base_url="http://localhost:11434/v1",
            llm_model="test-model",
            llm_api_key="test-key",
            skills_dir="./skills",
        ),
    )
    query_runner._build_llm = lambda **kwargs: _FakeTitleLlm()  # noqa: SLF001

    title = query_runner.generate_chat_title(
        dataset_name="sales.csv",
        user_queries=["покажи выручку по регионам"],
        trace_context={"session_id": "s1"},
    )

    assert title == "Выручка по регионам продаж"


def test_query_runner_adapter_processes_post_run_effects() -> None:
    session_memory = StructuredSessionMemory(notes="Existing note")
    query_runner = AgentGraphQueryRunner(
        Settings(
            llm_base_url="http://localhost:11434/v1",
            llm_model="test-model",
            llm_api_key="test-key",
            skills_dir="./skills",
        ),
        session_memory=session_memory,
    )
    query_runner._build_llm = lambda **kwargs: _FakeTitleLlm()  # noqa: SLF001
    query_runner.services.memory_note_callback("Remember concise analysis")
    query_runner.services.session_note_callback("Session revenue insight")
    user_memory_service = _FakeUserMemoryService()
    session_store = _FakeSessionStore()

    report = query_runner.process_post_run_effects(
        user_id=42,
        session_id="session-1",
        user_memory_service=user_memory_service,
        session_store=session_store,
    )

    assert not report.has_failures
    assert report.user_memory_notes == 1
    assert report.session_memory_notes == 1
    assert report.structured_session_memory_persisted
    assert user_memory_service.scheduled[0][0] == 42
    assert user_memory_service.scheduled[0][1] == ["Remember concise analysis"]
    assert session_store.appended_notes == [("session-1", "Session revenue insight")]
    assert session_store.persisted_memory == ("session-1", session_memory)
    assert session_memory.notes == "Existing note\nSession revenue insight"

    second_report = query_runner.process_post_run_effects(
        user_id=42,
        session_id="session-1",
        user_memory_service=user_memory_service,
        session_store=session_store,
    )

    assert second_report.user_memory_notes == 0
    assert second_report.session_memory_notes == 0
    assert len(user_memory_service.scheduled) == 1
    assert session_store.appended_notes == [("session-1", "Session revenue insight")]
