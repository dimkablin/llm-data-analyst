"""
Behavior and contract tests — full-stack tier.

Requires: langchain_core, langgraph, duckdb (installed in the full runtime env).
Tests are skipped gracefully when dependencies are not available.

Covers:
  1. Unified graph flow: dispatch → agent → finalize (no legacy nodes)
  2. Single tool-calling engine (_direct_tool_loop, native bind_tools)
  3. Skills injected into execution system prompt
  4. ToolBuildContext invariants
  5. ToolRegistry factory guards
  6. AgentResponse route contract
"""
from __future__ import annotations

import textwrap
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ─── dependency guards ────────────────────────────────────────────────────────

langchain_core = pytest.importorskip(
    "langchain_core",
    reason="langchain_core not installed — full-stack tests skipped",
)
langgraph = pytest.importorskip(
    "langgraph",
    reason="langgraph not installed — full-stack tests skipped",
)
duckdb = pytest.importorskip(
    "duckdb",
    reason="duckdb not installed — full-stack tests skipped",
)

# These imports are only reached when all deps are available.
from langchain_core.messages import ToolMessage  # noqa: E402

from backend.agent.prompts import execution_agent_prompt  # noqa: E402
from backend.agent.runner import DEPTH_PROFILES, AgentResponse, AgentRunner  # noqa: E402
from backend.core.config import Settings  # noqa: E402
from backend.skills.registry import SkillRegistry  # noqa: E402
from backend.tools.context import ToolBuildContext  # noqa: E402
from backend.tools.impl.factory import (  # noqa: E402
    MemoryToolFactory,
    PandasToolFactory,
    PlannerToolFactory,
    SQLToolFactory,
    SessionNoteToolFactory,
    ValueToolFactory,
)
from backend.tools.policy import is_tool_allowed  # noqa: E402
from backend.tools.registry import ToolRegistry  # noqa: E402


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


@pytest.fixture()
def skill_registry_with_skills(tmp_path):
    (tmp_path / "sql_skill.md").write_text(
        textwrap.dedent("""\
            ---
            name: SQL Tool
            description: Execute SQL queries against the database
            kind: tool
            tool_key: sql_tool
            triggers: sql, query, database
            ---
            ## SQL Instructions
            Use sql_tool to run SQL queries.
        """),
        encoding="utf-8",
    )
    (tmp_path / "cohort.md").write_text(
        textwrap.dedent("""\
            ---
            name: Cohort Analysis
            description: Run cohort retention analysis
            kind: analytical
            triggers: cohort, retention, удержание
            ---
            ## Cohort Analysis Method
            Steps: 1. Group users by date. 2. Compute retention.
        """),
        encoding="utf-8",
    )
    reg = SkillRegistry.from_path(tmp_path)
    reg.load()
    return reg


@pytest.fixture()
def runner(settings, tmp_path):
    reg = SkillRegistry.from_path(tmp_path)
    reg.load()
    return AgentRunner(settings=settings, skill_registry=reg)


@pytest.fixture()
def minimal_ctx(settings):
    return ToolBuildContext(settings=settings)


@pytest.fixture()
def ctx_with_df(settings):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    return ToolBuildContext(settings=settings, df=df)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Unified execution graph: dispatch → agent → finalize
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
        required = {"dispatch", "agent", "finalize"}
        missing = required - node_names
        assert not missing, f"Required graph nodes missing: {missing}"

    def test_dispatch_sets_done_true_for_greeting(self, runner):
        """A greeting must short-circuit to finalize (done=True, stop_reason=chat_route)."""
        state: dict[str, Any] = {
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
        with patch.object(
            runner,
            "chat",
            return_value=AgentResponse(
                final_text="Привет!", reasoning=None, artifacts=[], route="chat"
            ),
        ):
            result = runner._dispatch_node(state)

        assert result.get("done") is True
        assert result.get("stop_reason") == "chat_route"

    def test_dispatch_sets_done_true_for_summary_route(self, runner):
        """A management-note prompt must bypass the agent node (done=True)."""
        state: dict[str, Any] = {
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
        with patch.object(
            runner,
            "_build_management_note",
            return_value=AgentResponse(
                final_text="Итог: ...", reasoning=None, artifacts=[], route="summary"
            ),
        ):
            result = runner._dispatch_node(state)

        assert result.get("done") is True

    def test_dispatch_sets_done_false_for_analysis(self, runner):
        """An analysis request must fall through to the agent node (done=False)."""
        df = pd.DataFrame({"sales": [100, 200, 300], "region": ["A", "B", "C"]})
        state: dict[str, Any] = {
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

        with patch("backend.agent.runner.SandboxManager") as mock_sm:
            mock_sm.get_instance.return_value.get_or_create.return_value = mock_sandbox
            result = runner._dispatch_node(state)

        assert result.get("done") is False, (
            "Analysis prompt must not be short-circuited — agent node must run"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Single tool-calling engine (native bind_tools, no ReAct)
# ─────────────────────────────────────────────────────────────────────────────


class TestExecutionEngine:
    """_direct_tool_loop is the one and only execution engine."""

    def test_agent_runner_has_no_legacy_react_methods(self, runner):
        """AgentRunner must not expose any legacy ReAct execution methods."""
        legacy = [
            "run_react", "_run_react", "react_loop", "_react_loop",
            "parse_thought", "_parse_thought", "parse_action", "_parse_action",
            "run_agent_with_react",
        ]
        present = [name for name in legacy if hasattr(runner, name)]
        assert not present, f"Legacy ReAct methods still present: {present}"

    def test_direct_tool_loop_returns_analysis_route(self, runner):
        """_direct_tool_loop must always return AgentResponse with route='analysis'."""
        mock_response = MagicMock()
        mock_response.tool_calls = []
        mock_response.content = "Анализ выполнен."
        mock_response.additional_kwargs = {}

        with patch.object(runner, "_build_llm") as mock_build:
            mock_bound = MagicMock()
            mock_bound.invoke.return_value = mock_response
            mock_build.return_value.bind_tools.return_value = mock_bound

            result = runner._direct_tool_loop(
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
            MagicMock(
                tool_calls=[
                    {"name": "fake_tool", "args": {}, "id": "call-1", "type": "tool_call"}
                ],
                content="",
                additional_kwargs={},
            ),
            MagicMock(tool_calls=[], content="Done", additional_kwargs={}),
        ]

        fake_tool = MagicMock()
        fake_tool.name = "fake_tool"
        fake_tool.invoke.return_value = "tool output"

        captured_batches: list[list] = []

        def capturing_invoke(messages, config=None):
            captured_batches.append(list(messages))
            return call_responses.pop(0)

        with patch.object(runner, "_build_llm") as mock_build:
            mock_bound = MagicMock()
            mock_bound.invoke.side_effect = capturing_invoke
            mock_build.return_value.bind_tools.return_value = mock_bound

            runner._direct_tool_loop(
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

    def test_max_iterations_guard_terminates_loop(self, runner):
        """Loop must terminate after max_iterations even if LLM keeps calling tools."""
        always_tool = MagicMock(
            tool_calls=[
                {"name": "fake_tool", "args": {}, "id": "c", "type": "tool_call"}
            ],
            content="",
            additional_kwargs={},
        )
        fake_tool = MagicMock()
        fake_tool.name = "fake_tool"
        fake_tool.invoke.return_value = "out"

        count = 0

        def counting(messages, config=None):
            nonlocal count
            count += 1
            return always_tool

        max_iter = 3
        with patch.object(runner, "_build_llm") as mock_build:
            mock_bound = MagicMock()
            mock_bound.invoke.side_effect = counting
            mock_build.return_value.bind_tools.return_value = mock_bound

            result = runner._direct_tool_loop(
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
            MagicMock(
                tool_calls=[
                    {"name": "ghost_tool", "args": {}, "id": "c1", "type": "tool_call"}
                ],
                content="",
                additional_kwargs={},
            ),
            MagicMock(tool_calls=[], content="Done", additional_kwargs={}),
        ]

        with patch.object(runner, "_build_llm") as mock_build:
            mock_bound = MagicMock()
            mock_bound.invoke.side_effect = lambda msgs, config=None: call_responses.pop(0)
            mock_build.return_value.bind_tools.return_value = mock_bound

            result = runner._direct_tool_loop(
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

    def test_llm_transport_failure_returns_unreachable_flag(self, runner):
        """ConnectionError from LLM must produce llm_unreachable=True, not raise."""
        with patch.object(runner, "_build_llm") as mock_build:
            mock_bound = MagicMock()
            mock_bound.invoke.side_effect = ConnectionError("refused")
            mock_build.return_value.bind_tools.return_value = mock_bound

            result = runner._direct_tool_loop(
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
        prompt = runner._build_execution_system_prompt(
            capability_context={
                "source_mode": "dataset",
                "tool_descriptions": "",
                "available_tool_keys": [],
            },
        )
        first_line = execution_agent_prompt.strip().split("\n")[0].strip()
        assert first_line in prompt

    def test_selected_skill_appears_in_execution_prompt(
        self, settings, skill_registry_with_skills
    ):
        runner = AgentRunner(settings=settings, skill_registry=skill_registry_with_skills)
        skills = skill_registry_with_skills.list_skills()
        analytical = next(s for s in skills if s.kind == "analytical")

        prompt = runner._build_execution_system_prompt(
            capability_context={
                "source_mode": "dataset",
                "tool_descriptions": "",
                "available_tool_keys": [],
            },
            selected_skill_ids=[analytical.skill_id],
        )
        assert analytical.name in prompt

    def test_tool_skill_brief_appears_for_available_tool(
        self, settings, skill_registry_with_skills
    ):
        runner = AgentRunner(settings=settings, skill_registry=skill_registry_with_skills)
        prompt = runner._build_execution_system_prompt(
            capability_context={
                "source_mode": "dataset",
                "tool_descriptions": "",
                "available_tool_keys": ["sql_tool"],
            },
        )
        assert "sql_tool" in prompt

    def test_analytical_skills_brief_in_prompt_when_skills_exist(
        self, settings, skill_registry_with_skills
    ):
        runner = AgentRunner(settings=settings, skill_registry=skill_registry_with_skills)
        prompt = runner._build_execution_system_prompt(
            capability_context={
                "source_mode": "dataset",
                "tool_descriptions": "",
                "available_tool_keys": [],
            },
        )
        assert "cohort" in prompt.lower()

    def test_no_react_markers_in_assembled_prompt(self, runner):
        prompt = runner._build_execution_system_prompt(
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
        prompt = runner._build_execution_system_prompt(
            capability_context={
                "source_mode": "dataset",
                "tool_descriptions": "",
                "available_tool_keys": [],
            },
            df=df,
        )
        assert "revenue" in prompt or "region" in prompt


# ─────────────────────────────────────────────────────────────────────────────
# 4. ToolBuildContext invariants
# ─────────────────────────────────────────────────────────────────────────────


class TestToolBuildContextContract:
    def test_has_data_false_without_any_source(self, settings):
        ctx = ToolBuildContext(settings=settings)
        assert ctx.has_data is False

    def test_has_data_true_with_dataframe(self, settings):
        ctx = ToolBuildContext(settings=settings, df=pd.DataFrame({"x": [1]}))
        assert ctx.has_data is True

    def test_has_data_true_with_csv_session(self, settings):
        ctx = ToolBuildContext(
            settings=settings, csv_loaded=True, csv_session_id="sess-abc"
        )
        assert ctx.has_data is True

    def test_tool_df_returns_empty_df_when_no_data(self, settings):
        ctx = ToolBuildContext(settings=settings)
        result = ctx.tool_df
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_tool_df_returns_actual_df_when_present(self, settings):
        df = pd.DataFrame({"x": [1, 2, 3]})
        ctx = ToolBuildContext(settings=settings, df=df)
        assert len(ctx.tool_df) == 3


# ─────────────────────────────────────────────────────────────────────────────
# 5. ToolRegistry factory guards
# ─────────────────────────────────────────────────────────────────────────────


class TestToolRegistryContract:
    def test_planner_tool_available_without_data(self, minimal_ctx):
        assert PlannerToolFactory().is_available(minimal_ctx) is True

    def test_pandas_tool_requires_dataframe(self, minimal_ctx, ctx_with_df):
        factory = PandasToolFactory()
        assert factory.is_available(minimal_ctx) is False
        assert factory.is_available(ctx_with_df) is True

    def test_value_tool_requires_dataframe(self, minimal_ctx, ctx_with_df):
        factory = ValueToolFactory()
        assert factory.is_available(minimal_ctx) is False
        assert factory.is_available(ctx_with_df) is True

    def test_sql_tool_requires_db_or_csv(self, settings):
        factory = SQLToolFactory()
        assert factory.is_available(ToolBuildContext(settings=settings)) is False
        ctx_csv = ToolBuildContext(
            settings=settings, csv_loaded=True, csv_session_id="sess-1"
        )
        assert factory.is_available(ctx_csv) is True

    def test_memory_tool_always_available(self, minimal_ctx):
        assert MemoryToolFactory(on_note=lambda _: None).is_available(minimal_ctx) is True

    def test_session_note_tool_always_available(self, minimal_ctx):
        assert SessionNoteToolFactory(on_note=lambda _: None).is_available(minimal_ctx) is True

    def test_build_tools_excludes_data_tools_without_df(self, settings):
        """pandas_tool and value_tool must not appear when no DataFrame is present."""
        registry = ToolRegistry.from_services()
        tools = registry.build_tools(ToolBuildContext(settings=settings))
        names = {getattr(t, "name", "") for t in tools}
        assert "pandas_tool" not in names
        assert "value_tool" not in names

    def test_build_tools_includes_planner_without_restrictions(self, settings):
        registry = ToolRegistry.from_services()
        tools = registry.build_tools(ToolBuildContext(settings=settings))
        names = {getattr(t, "name", "") for t in tools}
        assert "planner_tool" in names

    def test_allowed_tool_keys_restricts_build_tools(self, settings):
        """allowed_tool_keys must block disallowed tools even when data is present."""
        registry = ToolRegistry.from_services()
        df = pd.DataFrame({"x": [1, 2, 3]})
        ctx = ToolBuildContext(
            settings=settings, df=df, allowed_tool_keys={"planner_tool"}
        )
        tools = registry.build_tools(ctx)
        names = {getattr(t, "name", "") for t in tools}
        assert "pandas_tool" not in names
        assert "planner_tool" in names

    def test_is_tool_allowed_none_permits_any(self):
        assert is_tool_allowed("anything", None) is True


# ─────────────────────────────────────────────────────────────────────────────
# 6. AgentResponse route contract
# ─────────────────────────────────────────────────────────────────────────────


class TestDepthProfileContract:
    """Depth profiles define the tool-calling iteration budget per analysis depth.

    Protected contract: three levels must exist with monotonically increasing limits.
    """

    def test_all_required_levels_defined(self):
        assert set(DEPTH_PROFILES.keys()) >= {"light", "medium", "deep"}

    def test_limits_are_in_sensible_range(self):
        for name, profile in DEPTH_PROFILES.items():
            limit = profile.get("inner_recursion_limit", 0)
            assert 0 < limit <= 20, f"Profile '{name}' has out-of-range limit: {limit}"

    def test_limits_are_monotonically_increasing(self):
        """light < medium < deep — more depth = more tool-calling iterations."""
        assert (
            DEPTH_PROFILES["light"]["inner_recursion_limit"]
            < DEPTH_PROFILES["medium"]["inner_recursion_limit"]
            < DEPTH_PROFILES["deep"]["inner_recursion_limit"]
        )


class TestAgentResponseContract:
    """AgentResponse must always carry a valid route value."""

    VALID_ROUTES = {"chat", "analysis", "rag", "summary"}

    def test_default_route_is_valid(self):
        r = AgentResponse(final_text="", reasoning=None, artifacts=[])
        assert r.route in self.VALID_ROUTES

    @pytest.mark.parametrize("route", ["chat", "analysis", "rag", "summary"])
    def test_all_canonical_routes_accepted(self, route: str):
        r = AgentResponse(final_text="", reasoning=None, artifacts=[], route=route)
        assert r.route == route

    def test_quick_route_returns_none_for_analysis_prompt(self):
        """_quick_route must return None for analysis prompts (fall-through to agent)."""
        result = AgentRunner._quick_route(
            "покажи топ-10 продаж по регионам", has_rag=False, has_data=True
        )
        assert result is None

    def test_quick_route_returns_chat_for_greeting(self):
        assert AgentRunner._quick_route("привет", has_rag=False, has_data=False) == "chat"

    def test_quick_route_returns_summary_for_management_note(self):
        assert (
            AgentRunner._quick_route(
                "подведи итог анализа", has_rag=False, has_data=False
            )
            == "summary"
        )

    def test_quick_route_returns_rag_when_rag_available(self):
        assert (
            AgentRunner._quick_route(
                "что написано в документации", has_rag=True, has_data=False
            )
            == "rag"
        )

    def test_quick_route_never_returns_rag_without_service(self):
        result = AgentRunner._quick_route(
            "что написано в документации", has_rag=False, has_data=False
        )
        assert result != "rag"
