"""
Behavior and contract tests — full-stack tier.

Requires: langchain_core, langgraph, duckdb (installed in the full runtime env).
Tests are skipped gracefully when dependencies are not available.

Covers:
  1. Unified graph flow: prepare_context → agent → finalize (no legacy nodes)
  2. Single tool-calling engine (_direct_tool_loop, native bind_tools)
  3. Skills injected into execution system prompt
  4. AgentResponse/finalize contract
"""

from __future__ import annotations

import textwrap
import threading
from dataclasses import replace
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ─── dependency guards ────────────────────────────────────────────────────────
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from backend.agent.graph.nodes.finalize import finalize_node
from backend.agent.graph.nodes.prepare_context import prepare_context_node
from backend.agent.models import AgentResponse
from backend.agent.prompts import execution_agent_prompt
from backend.agent.runner import AgentRunner
from backend.agent.services.message_builder import (
    ExecutionSystemPromptRequest,
    build_execution_system_prompt,
)
from backend.agent.tool_loop import ToolLoopRequest, direct_tool_loop
from backend.artifacts.execution import ExecArtifactType, ExecutionArtifact
from backend.core.config import Settings
from backend.notebook.manifest_store import ManifestStore
from backend.notebook.session_source import SessionManifest, SessionSource
from backend.skills.registry import SkillRegistry

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def settings():
    return Settings(
        llm_base_url="http://localhost:11434/v1",
        llm_model="test-model",
        llm_api_key="test-key",
        agent_analysis_depth="light",
        skills_dir="./skills",
    )


def _write_skill(tmp_path, folder: str, content: str) -> None:
    d = tmp_path / folder
    d.mkdir(exist_ok=True)
    (d / "SKILL.md").write_text(textwrap.dedent(content), encoding="utf-8")


@pytest.fixture()
def skill_registry_with_skills(tmp_path):
    _write_skill(
        tmp_path,
        "sql_tool",
        """\
        ---
        name: SQL Tool
        description: Execute SQL queries against the database
        kind: tool
        tool_key: sql_tool
        triggers: sql, query, database
        ---
        ## SQL Instructions
        ### API
        Use sql_tool to run SQL queries.
        ### Final result protocol
        Return a table artifact.
    """,
    )
    _write_skill(
        tmp_path,
        "cohort_analysis",
        """\
        ---
        name: Cohort Analysis
        description: Run cohort retention analysis
        kind: analytical
        triggers: cohort, retention, удержание
        ---
        ## Cohort Analysis Method
        ### Algorithm
        Steps: 1. Group users by date. 2. Compute retention.
        ### Rules
        Use loaded data only.
    """,
    )
    reg = SkillRegistry.from_path(tmp_path)
    reg.load()
    return reg


@pytest.fixture()
def runner(settings, tmp_path):
    reg = SkillRegistry.from_path(tmp_path)
    reg.load()
    return AgentRunner(settings=settings, skill_registry=reg)


def _build_execution_prompt_for_runner(runner: AgentRunner, **overrides: Any) -> str:
    return build_execution_system_prompt(
        ExecutionSystemPromptRequest(
            settings=runner.settings,
            skill_registry=runner.skill_registry,
            enabled_analytical_skill_ids=runner.enabled_analytical_skill_ids,
            **overrides,
        )
    )


def _build_execution_context_text_for_runner(runner: AgentRunner, **overrides: Any) -> str:
    from backend.agent.services.message_builder import build_execution_context_messages

    messages = build_execution_context_messages(
        ExecutionSystemPromptRequest(
            settings=runner.settings,
            skill_registry=runner.skill_registry,
            enabled_analytical_skill_ids=runner.enabled_analytical_skill_ids,
            **overrides,
        )
    )
    return "\n\n".join(str(message.content) for message in messages)


def _run_direct_tool_loop(runner: AgentRunner, **kwargs: Any) -> AgentResponse:
    system_prompt = kwargs.pop("execution_system_prompt", "system")
    prompt = kwargs.pop("prompt", "")
    history = kwargs.pop("history", [])
    use_history = kwargs.pop("use_history", False)
    if "messages" not in kwargs:
        messages = [SystemMessage(content=system_prompt)]
        if use_history:
            for item in history:
                content = str(item.get("content", ""))
                if item.get("role") == "assistant":
                    messages.append(AIMessage(content=content))
                else:
                    messages.append(HumanMessage(content=content))
        messages.append(HumanMessage(content=prompt))
        kwargs["messages"] = messages
    return direct_tool_loop(
        ToolLoopRequest(
            settings=runner.settings,
            **kwargs,
        )
    )


def test_base_exec_tools_are_not_parallel_safe():
    from backend.tools.impl.anomaly_planfact_tool import AnomalyPlanfactTool
    from backend.tools.impl.base_tool import BaseExecTool
    from backend.tools.impl.forecast_tool import ForecastTool
    from backend.tools.impl.pandas_tool import PandasTool
    from backend.tools.impl.plotly_tool import PlotlyTool

    assert BaseExecTool.parallel_safe is False
    assert PandasTool.parallel_safe is False
    assert PlotlyTool.parallel_safe is False
    assert ForecastTool.parallel_safe is False
    assert AnomalyPlanfactTool.parallel_safe is False


def test_read_only_structured_tools_opt_into_parallelism():
    from backend.tools.impl.data_catalog_tool import DataCatalogTool
    from backend.tools.impl.generation_tools import GenerateSummaryTool
    from backend.tools.impl.get_tool_instructions_tool import GetToolInstructionsTool
    from backend.tools.impl.planner_tool import PlannerTool
    from backend.tools.impl.rag_tool import RagTool

    assert DataCatalogTool.parallel_safe is True
    assert RagTool.parallel_safe is True
    assert PlannerTool.parallel_safe is True
    assert GetToolInstructionsTool.parallel_safe is True
    assert GenerateSummaryTool.parallel_safe is True


def test_stateful_or_external_write_structured_tools_stay_sequential():
    from backend.tools.impl.database_tool import DatabaseTool
    from backend.tools.impl.generation_tools import GenerateReportTool
    from backend.tools.impl.memory_tool import MemoryTool, SessionNoteTool
    from backend.tools.impl.mcp_tool import MCPTool
    from backend.tools.impl.sql_tool import SQLTool

    assert DatabaseTool.parallel_safe is False
    assert SQLTool.parallel_safe is False
    assert GenerateReportTool.parallel_safe is False
    assert MemoryTool.parallel_safe is False
    assert SessionNoteTool.parallel_safe is False
    assert MCPTool.parallel_safe is False


def test_mixed_parallel_safe_batch_stays_sequential():
    from backend.agent.tool_loop import _tool_batch_concurrency

    class SafeTool:
        parallel_safe: ClassVar[bool] = True

        def __init__(self, name: str) -> None:
            self.name = name

    class UnsafeTool:
        parallel_safe: ClassVar[bool] = False

        def __init__(self, name: str) -> None:
            self.name = name

    tools = {
        "safe_a": SafeTool("safe_a"),
        "safe_b": SafeTool("safe_b"),
        "unsafe": UnsafeTool("unsafe"),
    }

    assert _tool_batch_concurrency(
        [
            {"name": "safe_a"},
            {"name": "safe_b"},
            {"name": "unsafe"},
        ],
        tools,
        limit=3,
    ) == 1


def test_tool_collector_counts_parallel_callbacks_without_lost_updates():
    from concurrent.futures import ThreadPoolExecutor

    from backend.agent.callbacks import ToolCollector

    collector = ToolCollector()

    def run_one(i: int) -> None:
        tool_name = "search_tool" if i % 2 else "data_catalog_tool"
        run_id = f"run-{i}"
        collector.on_tool_start(
            {"name": tool_name},
            f'{{"tool_call_id": "call-{i}", "x": 1}}',
            run_id=run_id,
        )
        collector.on_tool_end('{"text": "ok"}', run_id=run_id)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(run_one, range(50)))

    assert collector.tool_calls == 50
    assert len([event for event in collector.events if event["phase"] == "start"]) == 50
    assert len([event for event in collector.events if event["phase"] == "end"]) == 50


def test_tool_collector_parallel_callbacks_resolve_end_by_run_id_not_last_tool_name():
    from backend.agent.callbacks import ToolCollector

    collector = ToolCollector()

    collector.on_tool_start(
        {"name": "search_tool"},
        '{"tool_call_id": "search-call", "query": "q"}',
        run_id="run-search",
    )
    collector.on_tool_start(
        {"name": "data_catalog_tool"},
        '{"tool_call_id": "catalog-call", "action": "list_tables"}',
        run_id="run-catalog",
    )
    collector.on_tool_end('{"text": "search done"}', run_id="run-search")

    activities = collector.to_persisted_activities(
        unfinished_status="error",
        unfinished_error="unfinished",
    )

    by_id = {item.get("tool_call_id"): item for item in activities}
    assert by_id["search-call"]["tool_name"] == "search_tool"
    assert by_id["search-call"]["status"] == "done"
    assert by_id["catalog-call"]["tool_name"] == "data_catalog_tool"
    assert by_id["catalog-call"]["status"] == "error"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Unified execution graph: prepare_context → agent → finalize
# ─────────────────────────────────────────────────────────────────────────────


class TestGraphStructure:
    """Compiled LanGraph must have exactly the three expected nodes, no ReAct nodes."""

    def test_graph_has_no_legacy_react_nodes(self, runner):
        node_names = set(getattr(runner._graph, "nodes", {}).keys())
        forbidden = {"react_node", "thought_node", "action_node", "observation_node"}
        overlap = node_names & forbidden
        assert not overlap, f"Legacy ReAct nodes in compiled graph: {overlap}"

    def test_graph_contains_required_nodes(self, runner):
        node_names = set(getattr(runner._graph, "nodes", {}).keys())
        if not node_names:
            pytest.skip("Graph node introspection not available in this LanGraph version")
        required = {"prepare_context", "agent", "finalize"}
        missing = required - node_names
        assert not missing, f"Required graph nodes missing: {missing}"

    def test_prepare_context_does_not_short_circuit_greeting(self, runner):
        """A greeting should enter the generic agent runtime instead of keyword routing."""
        state: dict[str, Any] = {
            "_runner": runner,
            "prompt": "привет",
            "df": None,
            "history": [],
            "use_history": False,
            "include_reasoning": False,
            "callbacks": [],
            "trace_context": {},
            "session_source": {},
            "selected_skill_ids": [],
        }
        result = prepare_context_node(state, runner.dependencies)

        assert result.get("done") is False
        assert result.get("stop_reason") == ""

    def test_prepare_context_keeps_summary_request_in_agent_runtime(self, runner):
        """Summary requests should be handled by tools/skills, not context bypass."""
        state: dict[str, Any] = {
            "_runner": runner,
            "prompt": "подведи итог анализа",
            "df": None,
            "history": [{"role": "user", "content": "что-то"}],
            "use_history": True,
            "include_reasoning": False,
            "callbacks": [],
            "trace_context": {},
            "session_source": {},
            "selected_skill_ids": [],
        }
        result = prepare_context_node(state, runner.dependencies)

        assert result.get("done") is False
        tool_names = {getattr(tool, "name", "") for tool in result.get("tools", [])}
        assert "generate_summary_tool" in tool_names

    def test_prepare_context_sets_done_false_for_analysis(self, runner):
        """An analysis request must fall through to the agent node (done=False)."""
        df = pd.DataFrame({"sales": [100, 200, 300], "region": ["A", "B", "C"]})
        state: dict[str, Any] = {
            "_runner": runner,
            "prompt": "покажи топ продажи по регионам",
            "df": df,
            "history": [],
            "use_history": False,
            "include_reasoning": False,
            "callbacks": [],
            "trace_context": {"session_id": "test-session"},
            "session_source": {},
            "selected_skill_ids": [],
        }

        mock_sandbox = MagicMock()
        mock_sandbox.describe_for_prompt.return_value = ""
        mock_sandbox.ensure_storage_dir.return_value = None

        with patch("backend.agent.context_manager.SandboxManager") as mock_sm:
            mock_sm.get_instance.return_value.get_or_create.return_value = mock_sandbox
            result = prepare_context_node(state, runner.dependencies)

        assert result.get("done") is False, (
            "Analysis prompt must not be short-circuited — agent node must run"
        )

    def test_run_query_denies_skill_when_required_tool_is_not_allowed(self, settings, tmp_path):
        """A selected skill must fail cleanly when current permissions block required tools."""
        _write_skill(
            tmp_path,
            "cohort_analysis",
            """\
            ---
            name: Cohort Analysis
            description: Run cohort retention analysis
            kind: analytical
            triggers: cohort, retention
            ---
            ## Cohort Analysis Method
            ### Algorithm
            Group users by signup period and compute retention.
            ### Rules
            Use loaded data only.
            ### Required tools
            - pandas_tool
        """,
        )
        registry = SkillRegistry.from_path(tmp_path).load()
        runner = AgentRunner(
            settings=settings,
            skill_registry=registry,
            allowed_tool_keys={"get_tool_instructions"},
        )

        response = runner.run_query(
            None,
            "run cohort retention analysis",
            history=[],
            use_history=False,
            include_reasoning=False,
            callbacks=[],
            trace_context={"session_id": "skill-denied"},
            session_source={},
            selected_skill_ids=["cohort_analysis"],
        )

        assert "pandas_tool" in response.final_text
        assert "необходимый tool" in response.final_text
        assert "выключен" in response.final_text or "недоступен" in response.final_text
        assert "Не удалось завершить анализ" not in response.final_text

    def test_prepare_context_adds_uploaded_table_descriptors_to_capability_context(self, runner, tmp_path):
        session_id = "csv-descriptor-session"
        runner = AgentRunner(
            settings=replace(runner.settings, storage_dir=str(tmp_path)),
            skill_registry=runner.skill_registry,
        )
        ManifestStore(tmp_path).save(
            session_id,
            SessionManifest(
                session_id=session_id,
                sources=[
                    SessionSource(
                        alias="orders_csv",
                        source_type="csv",
                        display_name="Orders upload",
                        file_name="orders.csv",
                        csv_session_id=session_id,
                        csv_table_names=["orders"],
                        schema_hint={"customer_id": "object", "amount": "int64"},
                        preprocessing_summary={"header_row_index": 1},
                        row_count=42,
                        column_count=2,
                    )
                ],
            ),
        )
        state: dict[str, Any] = {
            "_runner": runner,
            "prompt": "analyze uploaded tables",
            "df": None,
            "history": [],
            "use_history": False,
            "include_reasoning": False,
            "callbacks": [],
            "trace_context": {"session_id": session_id},
            "session_source": {
                "source_type": "csv",
                "csv_loaded": True,
                "csv_session_id": session_id,
                "csv_table_names": ["orders"],
            },
            "selected_skill_ids": [],
        }

        mock_sandbox = MagicMock()
        mock_sandbox.ensure_storage_dir.return_value = None

        with patch("backend.agent.context_manager.SandboxManager") as mock_sm:
            mock_sm.get_instance.return_value.get_or_create.return_value = mock_sandbox
            result = prepare_context_node(state, runner.dependencies)

        prompt_block = result["capability_context"]["prompt_block"]
        assert "orders.csv" in prompt_block
        assert "orders_csv" in prompt_block
        assert "customer_id" in prompt_block
        assert "amount" in prompt_block
        assert "42" in prompt_block

    def test_prepare_context_adds_source_inventory_from_manifest_without_live_duckdb(self, runner, tmp_path):
        session_id = "csv-inventory-session"
        runner = AgentRunner(
            settings=replace(runner.settings, storage_dir=str(tmp_path)),
            skill_registry=runner.skill_registry,
        )
        ManifestStore(tmp_path).save(
            session_id,
            SessionManifest(
                session_id=session_id,
                sources=[
                    SessionSource(
                        alias="orders_csv",
                        source_type="csv",
                        display_name="orders.csv",
                        file_name="orders.csv",
                        csv_session_id=session_id,
                        csv_table_names=["orders"],
                        schema_hint={"order_id": "int64", "amount": "float64"},
                    )
                ],
            ),
        )
        state: dict[str, Any] = {
            "_runner": runner,
            "prompt": "analyze uploaded tables",
            "df": None,
            "history": [],
            "use_history": False,
            "include_reasoning": False,
            "callbacks": [],
            "trace_context": {"session_id": session_id},
            "session_source": {
                "source_type": "csv",
                "csv_loaded": True,
                "csv_session_id": session_id,
                "csv_table_names": ["orders"],
            },
            "selected_skill_ids": [],
        }
        mock_sandbox = MagicMock()
        mock_sandbox.ensure_storage_dir.return_value = None

        with patch("backend.agent.context_manager.SandboxManager") as mock_sm:
            mock_sm.get_instance.return_value.get_or_create.return_value = mock_sandbox
            result = prepare_context_node(state, runner.dependencies)

        inventory = result["capability_context"]["source_inventory"]
        assert inventory["tables"][0]["qualified_name"] == "orders"
        assert inventory["tables"][0]["columns"] == ["order_id", "amount"]
        assert "[SOURCE INVENTORY]" in result["capability_context"]["prompt_block"]

    def test_prepare_context_prompts_catalog_first_for_multiple_source_tables(self, runner, tmp_path):
        session_id = "multi-source-catalog-session"
        runner = AgentRunner(
            settings=replace(runner.settings, storage_dir=str(tmp_path)),
            skill_registry=runner.skill_registry,
        )
        ManifestStore(tmp_path).save(
            session_id,
            SessionManifest(
                session_id=session_id,
                sources=[
                    SessionSource(
                        alias="orders_csv",
                        source_type="csv",
                        display_name="orders.csv",
                        file_name="orders.csv",
                        csv_session_id=session_id,
                        csv_table_names=["orders"],
                        schema_hint={"order_id": "int64", "customer_id": "int64"},
                    ),
                    SessionSource(
                        alias="customers_csv",
                        source_type="csv",
                        display_name="customers.csv",
                        file_name="customers.csv",
                        csv_session_id=session_id,
                        csv_table_names=["customers"],
                        schema_hint={"customer_id": "int64", "segment": "object"},
                    ),
                ],
            ),
        )
        state: dict[str, Any] = {
            "_runner": runner,
            "prompt": "join uploaded tables",
            "df": None,
            "history": [],
            "use_history": False,
            "include_reasoning": False,
            "callbacks": [],
            "trace_context": {"session_id": session_id},
            "session_source": {
                "source_type": "csv",
                "csv_loaded": True,
                "csv_session_id": session_id,
                "csv_table_names": ["orders", "customers"],
            },
            "selected_skill_ids": [],
        }
        mock_sandbox = MagicMock()
        mock_sandbox.ensure_storage_dir.return_value = None

        with patch("backend.agent.context_manager.SandboxManager") as mock_sm:
            mock_sm.get_instance.return_value.get_or_create.return_value = mock_sandbox
            result = prepare_context_node(state, runner.dependencies)

        capability_context = result["capability_context"]
        assert "data_catalog_tool" in capability_context["available_tool_keys"]
        assert "CATALOG-FIRST" in capability_context["prompt_block"]
        assert "qualified_name" in capability_context["prompt_block"]

    def test_prepare_context_chat_context_uses_uploaded_table_descriptors_from_manifest(self, runner, tmp_path):
        session_id = "csv-chat-descriptor-session"
        runner = AgentRunner(
            settings=replace(runner.settings, storage_dir=str(tmp_path)),
            skill_registry=runner.skill_registry,
        )
        ManifestStore(tmp_path).save(
            session_id,
            SessionManifest(
                session_id=session_id,
                sources=[
                    SessionSource(
                        alias="orders_csv",
                        source_type="csv",
                        display_name="Orders upload",
                        file_name="orders.csv",
                        csv_session_id=session_id,
                        csv_table_names=["orders"],
                        schema_hint={"order_id": "object", "amount": "int64"},
                        row_count=11,
                        column_count=2,
                    ),
                    SessionSource(
                        alias="customers_csv",
                        source_type="csv",
                        display_name="Customers upload",
                        file_name="customers.csv",
                        csv_session_id=session_id,
                        csv_table_names=["customers"],
                        schema_hint={"customer_id": "object", "customer_name": "object"},
                        row_count=22,
                        column_count=2,
                    ),
                ],
            ),
        )
        state: dict[str, Any] = {
            "_runner": runner,
            "prompt": "привет какие данные есть?",
            "df": pd.DataFrame({"first_file_only": [1]}),
            "history": [],
            "use_history": False,
            "include_reasoning": False,
            "callbacks": [],
            "trace_context": {"session_id": session_id},
            "session_source": {
                "source_type": "csv",
                "csv_loaded": True,
                "csv_session_id": session_id,
                "csv_table_names": ["orders", "customers"],
            },
            "selected_skill_ids": [],
        }

        result = prepare_context_node(state, runner.dependencies)

        assert result.get("done") is False
        prompt_block = result["capability_context"]["prompt_block"]
        assert "`orders`" in prompt_block
        assert "`customers`" in prompt_block
        assert "orders.csv" in prompt_block
        assert "customers.csv" in prompt_block
        assert "order_id" in prompt_block
        assert "customer_name" in prompt_block
        assert "first_file_only" not in prompt_block


# ─────────────────────────────────────────────────────────────────────────────
# 2. Single tool-calling engine (native bind_tools, no ReAct)
# ─────────────────────────────────────────────────────────────────────────────


class TestExecutionEngine:
    """_direct_tool_loop is the one and only execution engine."""

    def test_agent_runner_has_no_legacy_react_methods(self, runner):
        """AgentRunner must not expose any legacy ReAct execution methods."""
        legacy = [
            "run_react",
            "_run_react",
            "react_loop",
            "_react_loop",
            "parse_thought",
            "_parse_thought",
            "parse_action",
            "_parse_action",
            "run_agent_with_react",
        ]
        present = [name for name in legacy if hasattr(runner, name)]
        assert not present, f"Legacy ReAct methods still present: {present}"

    def test_direct_tool_loop_returns_analysis_route(self, runner):
        """_direct_tool_loop must always return AgentResponse with route='analysis'."""
        mock_response = AIMessage(content="Анализ выполнен.")

        with patch("backend.agent.tool_loop.build_runtime_llm") as mock_build:
            mock_bound = MagicMock()
            mock_bound.invoke.return_value = mock_response
            mock_build.return_value.bind_tools.return_value = mock_bound

            result = _run_direct_tool_loop(
                runner,
                prompt="test",
                history=[],
                use_history=False,
                include_reasoning=False,
                tools=[],
                execution_system_prompt="sys",
                callbacks=[],
                max_iterations=1,
            )

        assert isinstance(result, AgentResponse)
        assert result.route == "analysis"

    def test_tool_result_fed_back_as_tool_message(self, runner):
        """After a tool call the result must be appended as ToolMessage for the next LLM call."""
        call_responses = [
            AIMessage(
                content="",
                tool_calls=[{"name": "fake_tool", "args": {}, "id": "call-1", "type": "tool_call"}],
            ),
            AIMessage(content="Done"),
        ]

        @tool
        def fake_tool() -> str:
            """Return fake output for direct loop tests."""
            return "tool output"

        captured_batches: list[list] = []

        def capturing_invoke(messages, config=None):
            captured_batches.append(list(messages))
            return call_responses.pop(0)

        with patch("backend.agent.tool_loop.build_runtime_llm") as mock_build:
            mock_bound = MagicMock()
            mock_bound.invoke.side_effect = capturing_invoke
            mock_build.return_value.bind_tools.return_value = mock_bound

            _run_direct_tool_loop(
                runner,
                prompt="run tool",
                history=[],
                use_history=False,
                include_reasoning=False,
                tools=[fake_tool],
                execution_system_prompt="sys",
                callbacks=[],
                max_iterations=3,
            )

        assert len(captured_batches) == 2
        second_batch = captured_batches[1]
        tool_messages = [m for m in second_batch if isinstance(m, ToolMessage)]
        assert len(tool_messages) >= 1, "Tool result not fed back as ToolMessage"
        assert tool_messages[0].tool_call_id == "call-1"

    def test_direct_tool_loop_executes_multiple_tool_calls_in_one_cycle(self, runner):
        seen: list[str] = []

        @tool
        def fake_tool(value: str) -> str:
            """Return the value for direct loop tests."""
            seen.append(value)
            return f"out:{value}"

        responses = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "fake_tool",
                        "args": {"value": "one"},
                        "id": "call-1",
                        "type": "tool_call",
                    },
                    {
                        "name": "fake_tool",
                        "args": {"value": "two"},
                        "id": "call-2",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content="Done"),
        ]
        captured_batches: list[list] = []

        def invoke(messages, config=None):
            captured_batches.append(list(messages))
            return responses.pop(0)

        with patch("backend.agent.tool_loop.build_runtime_llm") as mock_build:
            mock_bound = MagicMock()
            mock_bound.invoke.side_effect = invoke
            mock_build.return_value.bind_tools.return_value = mock_bound

            result = _run_direct_tool_loop(
                runner,
                prompt="run two tools",
                history=[],
                use_history=False,
                include_reasoning=False,
                tools=[fake_tool],
                execution_system_prompt="sys",
                callbacks=[],
                max_iterations=2,
            )

        assert seen == ["one", "two"]
        assert result.tool_calls == 2
        tool_messages = [
            message for message in captured_batches[1] if isinstance(message, ToolMessage)
        ]
        assert {message.tool_call_id for message in tool_messages} == {"call-1", "call-2"}

    def test_direct_tool_loop_limits_tool_calls_per_cycle(self, runner):
        seen: list[str] = []

        @tool
        def fake_tool(value: str) -> str:
            """Return the value for direct loop tests."""
            seen.append(value)
            return f"out:{value}"

        responses = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "fake_tool",
                        "args": {"value": "one"},
                        "id": "call-1",
                        "type": "tool_call",
                    },
                    {
                        "name": "fake_tool",
                        "args": {"value": "two"},
                        "id": "call-2",
                        "type": "tool_call",
                    },
                    {
                        "name": "fake_tool",
                        "args": {"value": "three"},
                        "id": "call-3",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content="Done"),
        ]
        captured_batches: list[list] = []

        def invoke(messages, config=None):
            captured_batches.append(list(messages))
            return responses.pop(0)

        capped_runner = AgentRunner(
            settings=replace(runner.settings, max_tools_per_cycle=2),
            skill_registry=runner.skill_registry,
        )

        with patch("backend.agent.tool_loop.build_runtime_llm") as mock_build:
            mock_bound = MagicMock()
            mock_bound.invoke.side_effect = invoke
            mock_build.return_value.bind_tools.return_value = mock_bound

            result = _run_direct_tool_loop(
                capped_runner,
                prompt="run three tools",
                history=[],
                use_history=False,
                include_reasoning=False,
                tools=[fake_tool],
                execution_system_prompt="sys",
                callbacks=[],
                max_iterations=2,
            )

        assert seen == ["one", "two"]
        assert result.tool_calls == 2
        tool_messages = [
            message for message in captured_batches[1] if isinstance(message, ToolMessage)
        ]
        assert {message.tool_call_id for message in tool_messages} == {
            "call-1",
            "call-2",
            "call-3",
        }
        skipped = next(message for message in tool_messages if message.tool_call_id == "call-3")
        assert "MAX_TOOLS_PER_CYCLE=2" in str(skipped.content)

    def test_parallel_policy_uses_tool_class_flag_not_name_allowlist(self, runner):
        from langchain_core.tools import BaseTool
        from pydantic import BaseModel

        class Args(BaseModel):
            value: str

        class ArbitrarySafeTool(BaseTool):
            name: str = "not_in_any_allowlist"
            description: str = "safe arbitrary tool"
            args_schema: type[BaseModel] = Args
            parallel_safe: ClassVar[bool] = True

            def _run(self, value: str) -> str:
                return value

        captured_configs: list[dict[str, Any]] = []

        def fake_tool_node_invoke(self, input, config=None, runtime=None):
            del self, runtime
            captured_configs.append(dict(config or {}))
            return [
                ToolMessage(content="ok", name=call["name"], tool_call_id=call["id"])
                for call in input
            ]

        responses = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "not_in_any_allowlist",
                        "args": {"value": "a"},
                        "id": "call-1",
                        "type": "tool_call",
                    },
                    {
                        "name": "not_in_any_allowlist",
                        "args": {"value": "b"},
                        "id": "call-2",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content="Done"),
        ]

        def invoke(messages, config=None):
            del messages, config
            return responses.pop(0)

        capped_runner = AgentRunner(
            settings=replace(runner.settings, max_tools_per_cycle=2),
            skill_registry=runner.skill_registry,
        )

        with patch("backend.agent.tool_loop.ToolNode.invoke", fake_tool_node_invoke):
            with patch("backend.agent.tool_loop.build_runtime_llm") as mock_build:
                mock_bound = MagicMock()
                mock_bound.invoke.side_effect = invoke
                mock_build.return_value.bind_tools.return_value = mock_bound
                _run_direct_tool_loop(
                    capped_runner,
                    prompt="run arbitrary safe tools",
                    history=[],
                    use_history=False,
                    include_reasoning=False,
                    tools=[ArbitrarySafeTool()],
                    execution_system_prompt="sys",
                    callbacks=[],
                    max_iterations=2,
                )

        assert captured_configs[0]["max_concurrency"] == 2

    def test_base_exec_tool_batch_keeps_single_concurrency_even_if_flag_is_overridden(self, runner):
        from backend.tools.impl.base_tool import BaseExecTool

        class FakeExecTool(BaseExecTool):
            name: str = "fake_exec_tool"
            description: str = "exec tool"
            parallel_safe: ClassVar[bool] = True

        captured_configs: list[dict[str, Any]] = []

        def fake_tool_node_invoke(self, input, config=None, runtime=None):
            del self, runtime
            captured_configs.append(dict(config or {}))
            return [
                ToolMessage(content="ok", name=call["name"], tool_call_id=call["id"])
                for call in input
            ]

        responses = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "fake_exec_tool",
                        "args": {"code": "1"},
                        "id": "call-1",
                        "type": "tool_call",
                    },
                    {
                        "name": "fake_exec_tool",
                        "args": {"code": "2"},
                        "id": "call-2",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content="Done"),
        ]

        def invoke(messages, config=None):
            del messages, config
            return responses.pop(0)

        capped_runner = AgentRunner(
            settings=replace(runner.settings, max_tools_per_cycle=2),
            skill_registry=runner.skill_registry,
        )

        with patch("backend.agent.tool_loop.ToolNode.invoke", fake_tool_node_invoke):
            with patch("backend.agent.tool_loop.build_runtime_llm") as mock_build:
                mock_bound = MagicMock()
                mock_bound.invoke.side_effect = invoke
                mock_build.return_value.bind_tools.return_value = mock_bound
                _run_direct_tool_loop(
                    capped_runner,
                    prompt="run exec tools",
                    history=[],
                    use_history=False,
                    include_reasoning=False,
                    tools=[FakeExecTool(pd.DataFrame())],
                    execution_system_prompt="sys",
                    callbacks=[],
                    max_iterations=2,
                )

        assert captured_configs[0]["max_concurrency"] == 1

    def test_parallel_safe_tools_execute_concurrently_through_langgraph(self, runner):
        import time

        from langchain_core.tools import BaseTool
        from pydantic import BaseModel

        class Args(BaseModel):
            value: str

        class SlowSafeTool(BaseTool):
            name: str = "slow_safe_tool"
            description: str = "slow safe tool"
            args_schema: type[BaseModel] = Args
            parallel_safe: ClassVar[bool] = True

            def _run(self, value: str) -> str:
                time.sleep(0.2)
                return value

        responses = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "slow_safe_tool",
                        "args": {"value": "a"},
                        "id": "call-1",
                        "type": "tool_call",
                    },
                    {
                        "name": "slow_safe_tool",
                        "args": {"value": "b"},
                        "id": "call-2",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content="Done"),
        ]

        def invoke(messages, config=None):
            del messages, config
            return responses.pop(0)

        capped_runner = AgentRunner(
            settings=replace(runner.settings, max_tools_per_cycle=2),
            skill_registry=runner.skill_registry,
        )

        with patch("backend.agent.tool_loop.build_runtime_llm") as mock_build:
            mock_bound = MagicMock()
            mock_bound.invoke.side_effect = invoke
            mock_build.return_value.bind_tools.return_value = mock_bound
            started_at = time.perf_counter()
            _run_direct_tool_loop(
                capped_runner,
                prompt="parallel safe tools",
                history=[],
                use_history=False,
                include_reasoning=False,
                tools=[SlowSafeTool()],
                execution_system_prompt="sys",
                callbacks=[],
                max_iterations=2,
            )
            elapsed = time.perf_counter() - started_at

        assert elapsed < 0.35

    def test_direct_tool_loop_stops_before_tool_when_cancelled_after_llm(self, runner):
        cancel_event = threading.Event()
        seen: list[str] = []

        @tool
        def fake_tool() -> str:
            """Record unexpected execution."""
            seen.append("called")
            return "out"

        def cancelling_invoke(_messages, config=None):
            cancel_event.set()
            return AIMessage(
                content="",
                tool_calls=[{"name": "fake_tool", "args": {}, "id": "call-1", "type": "tool_call"}],
            )

        with patch("backend.agent.tool_loop.build_runtime_llm") as mock_build:
            mock_bound = MagicMock()
            mock_bound.invoke.side_effect = cancelling_invoke
            mock_build.return_value.bind_tools.return_value = mock_bound

            _run_direct_tool_loop(
                runner,
                prompt="run tool",
                history=[],
                use_history=False,
                include_reasoning=False,
                tools=[fake_tool],
                execution_system_prompt="sys",
                callbacks=[],
                max_iterations=3,
                cancel_event=cancel_event,
            )

        assert seen == []

    def test_max_iterations_guard_terminates_loop(self, runner):
        """Loop must terminate after max_iterations even if LLM keeps calling tools."""
        always_tool = AIMessage(
            content="",
            tool_calls=[{"name": "fake_tool", "args": {}, "id": "c", "type": "tool_call"}],
        )
        @tool
        def fake_tool() -> str:
            """Return fake output for direct loop tests."""
            return "out"

        count = 0

        def counting(messages, config=None):
            nonlocal count
            count += 1
            return always_tool

        max_iter = 3
        with patch("backend.agent.tool_loop.build_runtime_llm") as mock_build:
            mock_bound = MagicMock()
            mock_bound.invoke.side_effect = counting
            mock_build.return_value.bind_tools.return_value = mock_bound

            result = _run_direct_tool_loop(
                runner,
                prompt="loop test",
                history=[],
                use_history=False,
                include_reasoning=False,
                tools=[fake_tool],
                execution_system_prompt="sys",
                callbacks=[],
                max_iterations=max_iter,
            )

        assert count <= max_iter
        assert isinstance(result, AgentResponse)

    def test_unknown_tool_does_not_crash_loop(self, runner):
        """Unknown tool name must produce an error message, not crash."""
        call_responses = [
            AIMessage(
                content="",
                tool_calls=[{"name": "ghost_tool", "args": {}, "id": "c1", "type": "tool_call"}],
            ),
            AIMessage(content="Done"),
        ]

        with patch("backend.agent.tool_loop.build_runtime_llm") as mock_build:
            mock_bound = MagicMock()
            mock_bound.invoke.side_effect = lambda msgs, config=None: call_responses.pop(0)
            mock_build.return_value.bind_tools.return_value = mock_bound

            result = _run_direct_tool_loop(
                runner,
                prompt="test",
                history=[],
                use_history=False,
                include_reasoning=False,
                tools=[],  # empty — ghost_tool not registered
                execution_system_prompt="sys",
                callbacks=[],
                max_iterations=3,
            )

        assert isinstance(result, AgentResponse)

    def test_repeated_identical_tool_errors_stay_visible_to_next_llm_turn(self, runner):
        """The generic ReAct loop should show repeated errors to the LLM, not stop."""
        call_responses = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "fake_tool",
                        "args": {"code": "bad()"},
                        "id": "c1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "fake_tool",
                        "args": {"code": "bad()"},
                        "id": "c2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="should not be reached"),
        ]
        seen: list[str] = []

        @tool
        def fake_tool(code: str) -> str:
            """Return a repeated fake tool error."""
            seen.append(code)
            return "❌ Ошибка при создании таблиц: same sandbox error"

        with patch("backend.agent.tool_loop.build_runtime_llm") as mock_build:
            mock_bound = MagicMock()
            seen_messages = []

            def invoke(messages, config=None):
                seen_messages.append(list(messages))
                return call_responses.pop(0)

            mock_bound.invoke.side_effect = invoke
            mock_build.return_value.bind_tools.return_value = mock_bound

            result = _run_direct_tool_loop(
                runner,
                prompt="test",
                history=[],
                use_history=False,
                include_reasoning=False,
                tools=[fake_tool],
                execution_system_prompt="sys",
                callbacks=[],
                max_iterations=5,
            )

        assert len(seen) == 2
        assert mock_bound.invoke.call_count == 3
        assert result.final_text == "should not be reached"
        third_turn_messages = seen_messages[-1]
        tool_messages = [
            message for message in third_turn_messages if isinstance(message, ToolMessage)
        ]
        assert len(tool_messages) == 2
        assert all("same sandbox error" in str(message.content) for message in tool_messages)
        ai_tool_calls = [
            message
            for message in third_turn_messages
            if isinstance(message, AIMessage) and getattr(message, "tool_calls", None)
        ]
        assert ai_tool_calls[-1].tool_calls[0]["args"] == {"code": "bad()"}

    def test_llm_transport_failure_returns_unreachable_flag(self, runner):
        """ConnectionError from LLM must produce llm_unreachable=True, not raise."""
        with patch("backend.agent.tool_loop.build_runtime_llm") as mock_build:
            mock_bound = MagicMock()
            mock_bound.invoke.side_effect = ConnectionError("refused")
            mock_build.return_value.bind_tools.return_value = mock_bound

            result = _run_direct_tool_loop(
                runner,
                prompt="fail",
                history=[],
                use_history=False,
                include_reasoning=False,
                tools=[],
                execution_system_prompt="sys",
                callbacks=[],
                max_iterations=1,
            )

        assert result.llm_unreachable is True
        assert isinstance(result.final_text, str)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Skills injected into execution system prompt
# ─────────────────────────────────────────────────────────────────────────────


class TestSkillsInPrompt:
    """Skills must appear in the assembled execution prompt when relevant."""

    def test_execution_agent_prompt_is_prompt_base(self, runner):
        prompt = _build_execution_prompt_for_runner(
            runner,
            capability_context={
                "source_mode": "dataset",
                "tool_descriptions": "",
                "available_tool_keys": [],
            },
        )
        first_line = execution_agent_prompt.strip().split("\n")[0].strip()
        assert first_line in prompt

    def test_selected_skill_appears_in_execution_context_messages(self, settings, skill_registry_with_skills):
        runner = AgentRunner(settings=settings, skill_registry=skill_registry_with_skills)
        skills = skill_registry_with_skills.list_skills()
        analytical = next(s for s in skills if s.kind == "analytical")

        context = _build_execution_context_text_for_runner(
            runner,
            capability_context={
                "source_mode": "dataset",
                "tool_descriptions": "",
                "available_tool_keys": [],
            },
            selected_skill_ids=[analytical.skill_id],
        )
        assert analytical.name in context

    def test_tool_skill_brief_appears_for_available_tool(self, settings, skill_registry_with_skills):
        runner = AgentRunner(settings=settings, skill_registry=skill_registry_with_skills)
        prompt = _build_execution_prompt_for_runner(
            runner,
            capability_context={
                "source_mode": "dataset",
                "tool_descriptions": "",
                "available_tool_keys": ["sql_tool"],
            },
        )
        assert "sql_tool" in prompt

    def test_analytical_skills_brief_in_context_messages_when_skills_exist(self, settings, skill_registry_with_skills):
        runner = AgentRunner(settings=settings, skill_registry=skill_registry_with_skills)
        context = _build_execution_context_text_for_runner(
            runner,
            capability_context={
                "source_mode": "dataset",
                "tool_descriptions": "",
                "available_tool_keys": [],
            },
        )
        assert "cohort" in context.lower()

    def test_no_react_markers_in_assembled_prompt(self, runner):
        prompt = _build_execution_prompt_for_runner(
            runner,
            capability_context={
                "source_mode": "dataset",
                "tool_descriptions": "",
                "available_tool_keys": [],
            },
        )
        for marker in ("Thought:", "Action Input:", "Observation:"):
            assert marker not in prompt, f"ReAct marker '{marker}' in assembled prompt"

    def test_df_schema_included_when_df_present(self, runner):
        df = pd.DataFrame({"revenue": [100, 200], "region": ["A", "B"]})
        context = _build_execution_context_text_for_runner(
            runner,
            capability_context={
                "source_mode": "dataset",
                "tool_descriptions": "",
                "available_tool_keys": [],
            },
            df=df,
        )
        assert "revenue" in context or "region" in context


# ─────────────────────────────────────────────────────────────────────────────
class TestAgentResponseContract:
    """AgentResponse must always carry a valid route value."""

    VALID_ROUTES: ClassVar[set[str]] = {"analysis", "summary"}

    def test_default_route_is_valid(self):
        r = AgentResponse(final_text="", reasoning=None, artifacts=[])
        assert r.route in self.VALID_ROUTES

    @pytest.mark.parametrize("route", ["analysis", "summary"])
    def test_all_canonical_routes_accepted(self, route: str):
        r = AgentResponse(final_text="", reasoning=None, artifacts=[], route=route)
        assert r.route == route

    def test_runner_does_not_expose_keyword_quick_route(self):
        assert not hasattr(AgentRunner, "_quick_route")

    def test_finalize_keeps_existing_answer_without_review_gate(self, runner):
        artifact = ExecutionArtifact(
            artifact_type=ExecArtifactType.DATAFRAME,
            producer_tool="sql_tool",
            name="variance_table",
            data=pd.DataFrame({"metric": ["a"], "value": [10]}),
            meta={"aggregation": True},
        )
        response = AgentResponse(
            final_text="Суть: построена таблица. Ключевые цифры: 10. Инсайты: требуется график.",
            reasoning=None,
            artifacts=[artifact],
            route="analysis",
            tool_calls=1,
            tool_names=["sql_tool"],
        )

        result = finalize_node(
            {
                "_runner": runner,
                "response": response,
                "prompt": "plot top deviations",
                "callbacks": [],
                "trace_context": {},
                "step_index": 1,
                "max_steps": 1,
            },
            runner.dependencies,
        )

        finalized = result["response"]
        assert finalized.final_text == response.final_text
        assert finalized.reasoning is None

    def test_finalize_does_not_enforce_artifact_contract_requirements(self, runner):
        artifact = ExecutionArtifact(
            artifact_type=ExecArtifactType.DATAFRAME,
            producer_tool="sql_tool",
            name="variance_table",
            data=pd.DataFrame({"metric": ["a"], "value": [10]}),
            meta={"aggregation": True},
        )
        response = AgentResponse(
            final_text="Суть: построена таблица. Ключевые цифры: 10.",
            reasoning=None,
            artifacts=[artifact],
            route="analysis",
            tool_calls=1,
            tool_names=["sql_tool"],
        )

        result = finalize_node(
            {
                "_runner": runner,
                "response": response,
                "prompt": "plot top deviations",
                "callbacks": [],
                "trace_context": {},
                "step_index": 1,
                "max_steps": 1,
            },
            runner.dependencies,
        )

        finalized = result["response"]
        assert finalized.final_text == response.final_text
        assert finalized.reasoning is None

    def test_finalize_allows_final_answer_without_artifacts(self, runner):
        response = AgentResponse(
            final_text="Короткий вывод по вопросу без приложенных артефактов.",
            reasoning=None,
            artifacts=[],
            route="analysis",
            tool_calls=1,
            tool_names=["sql_tool"],
        )

        result = finalize_node(
            {
                "_runner": runner,
                "response": response,
                "prompt": "plot top deviations",
                "callbacks": [],
                "trace_context": {},
                "step_index": 1,
                "max_steps": 1,
            },
            runner.dependencies,
        )

        finalized = result["response"]
        assert finalized.artifacts == []
        assert finalized.final_text == response.final_text
        assert finalized.reasoning is None
