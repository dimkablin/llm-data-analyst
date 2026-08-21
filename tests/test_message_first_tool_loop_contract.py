from __future__ import annotations

from dataclasses import replace
from typing import get_args, get_type_hints
from unittest.mock import MagicMock, patch

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages

from backend.agent.callbacks import ToolCollector
from backend.agent.services.message_builder import (
    ExecutionSystemPromptRequest,
    build_execution_context_messages,
    build_execution_system_prompt,
)
from backend.agent.state import AgentGraphState
from backend.agent.tool_loop import ToolLoopRequest, direct_tool_loop
from backend.core.config import Settings
from backend.skills.registry import SkillRegistry
from backend.tools.context import ToolBuildContext
from backend.tools.impl.factory import AnomalyPlanfactToolFactory, PlotlyToolFactory
from backend.tools.impl.pandas_tool import PandasTool
from backend.tools.sandbox import SessionSandbox


def test_agent_graph_state_messages_use_langgraph_add_messages() -> None:
    hints = get_type_hints(AgentGraphState, include_extras=True)

    assert "messages" in hints
    assert add_messages in get_args(hints["messages"])


def test_generic_code_tool_timeout_defaults_to_20_seconds() -> None:
    settings = Settings()
    context = ToolBuildContext(settings=settings, df=pd.DataFrame({"value": [1]}))

    assert settings.tool_exec_timeout_sec == 20
    assert PlotlyToolFactory().build(context).execution_timeout_sec == 20
    assert AnomalyPlanfactToolFactory(MagicMock()).build(context).execution_timeout_sec == 20


def test_skill_context_is_system_and_data_context_is_separate_message() -> None:
    registry = SkillRegistry.from_path("__missing_skills__")
    registry.build_analytical_skills_brief_block = (  # type: ignore[method-assign]
        lambda enabled_skill_ids=None: "available skill summary"
    )
    registry.build_prompt_block = lambda skill_ids: "full selected skill"  # type: ignore[method-assign]
    settings = Settings()
    request = ExecutionSystemPromptRequest(
        settings=settings,
        skill_registry=registry,
        enabled_analytical_skill_ids={"investment_risk"},
        selected_skill_ids=["investment_risk"],
        requested_tool_key="pandas_tool",
        df=pd.DataFrame({"revenue": [1, 2]}),
        session_source={"source_label": "demo"},
    )

    system_prompt = build_execution_system_prompt(request)
    context_messages = build_execution_context_messages(request)

    assert "SKILL_CATALOG_CONTEXT:\navailable skill summary" in system_prompt
    assert "SKILL_CONTEXT:\nfull selected skill" in system_prompt
    assert not any("SKILL_CONTEXT:" in str(message.content) for message in context_messages)
    assert any(str(message.content).startswith("DATA_CONTEXT:") for message in context_messages)
    assert any(
        str(message.content).startswith("REQUESTED_TOOL_CONTEXT:")
        and "must call this tool" in str(message.content)
        and "pandas_tool" in str(message.content)
        for message in context_messages
    )


def test_application_context_precedes_history_and_follow_up() -> None:
    from backend.agent.graph.nodes.agent import _build_agent_messages
    from backend.agent.runner import AgentRunner

    runner = AgentRunner()
    messages = _build_agent_messages(
        state={
            "prompt": "а теперь только по Москве",
            "history": [
                {"role": "user", "content": "покажи пассажиропоток по филиалам"},
                {
                    "role": "assistant",
                    "content": "Вот пассажиропоток по филиалам.",
                    "artifacts": [
                        {
                            "id": "artifact-passengers",
                            "execution_artifact_id": "artifact-passengers",
                            "type": "table",
                            "text": "passenger_by_branch",
                            "execution": {
                                "data_complete": True,
                                "schema": {
                                    "columns": ["branch", "passengers"],
                                    "dtypes": {"branch": "object", "passengers": "int64"},
                                    "row_count": 1,
                                },
                            },
                            "data": {
                                "format": "split",
                                "data": {
                                    "columns": ["branch", "passengers"],
                                    "index": [0],
                                    "data": [["A", 10]],
                                },
                            },
                        }
                    ],
                },
            ],
            "use_history": True,
        },
        deps=runner.dependencies,
        execution_system_prompt="system policy",
        context_messages=[HumanMessage(content="DATA_CONTEXT:\nschema")],
    )

    assert isinstance(messages[0], SystemMessage)
    assert "[INTERNAL_ARTIFACT_CONTEXT]" in str(messages[0].content)
    assert [str(message.content) for message in messages[1:]] == [
        "DATA_CONTEXT:\nschema",
        "покажи пассажиропоток по филиалам",
        "Вот пассажиропоток по филиалам.",
        "а теперь только по Москве",
    ]


def test_requested_tool_context_uses_generic_normal_loop_contract() -> None:
    registry = SkillRegistry.from_path("__missing_skills__")
    request = ExecutionSystemPromptRequest(
        settings=Settings(),
        skill_registry=registry,
        enabled_analytical_skill_ids=set(),
        requested_tool_key="forecast_tool",
        df=pd.DataFrame({"month": ["2026-01"], "attrition": [10]}),
    )

    context_messages = build_execution_context_messages(request)
    requested_context = "\n".join(
        str(message.content)
        for message in context_messages
        if str(message.content).startswith("REQUESTED_TOOL_CONTEXT:")
    )

    assert "forecast_tool" in requested_context
    assert "normal tool loop" in requested_context
    assert "manually computing" not in requested_context


def test_pandas_print_output_becomes_table_without_hidden_repair() -> None:
    df = pd.DataFrame(
        {
            "month": ["2026-01", "2026-02"],
            "channel": ["online", "offline"],
            "revenue": [10, 20],
        }
    )
    sandbox = SessionSandbox()
    sandbox.put("sales_by_month_channel", df)
    sandbox.put("unused_dataframe", pd.DataFrame({"secret": [1]}))
    tool = PandasTool(df, sandbox=sandbox, tool_cache_size=0)
    before_count = sandbox.execution_count

    with patch.object(PandasTool, "_fix_with_llm", side_effect=AssertionError("hidden repair"), create=True):
        text, payload = tool._run("print(sales_by_month_channel.columns.tolist())")

    assert sandbox.execution_count == before_count + 1
    assert "output" in text
    assert payload["table"]["output"]["output"].tolist() == ["['month', 'channel', 'revenue']"]


def test_fake_react_flow_sees_print_output_as_tool_result() -> None:
    df = pd.DataFrame(
        {
            "month": ["2026-01", "2026-02"],
            "channel": ["online", "offline"],
            "revenue": [10, 20],
        }
    )
    sandbox = SessionSandbox()
    sandbox.put("sales_by_month_channel", df)
    tool = PandasTool(df, sandbox=sandbox, tool_cache_size=0)
    collector = ToolCollector()
    captured_messages: list[list[object]] = []

    class FakeLLM:
        def bind_tools(self, _tools):
            return self

        def get_num_tokens_from_messages(self, messages):
            return sum(len(str(getattr(message, "content", ""))) for message in messages)

        def invoke(self, messages, config=None):
            del config
            captured_messages.append(list(messages))
            if len(captured_messages) == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "pandas_tool",
                            "args": {
                                "code": (
                                    "print('rows=2')\n"
                                    "tool_result = {'schema_version': '1.0', "
                                    "'artifact_type': 'table', "
                                    "'items': {'sales_summary': sales_by_month_channel.copy()}}"
                                )
                            },
                            "id": "call-bad",
                            "type": "tool_call",
                        }
                    ],
                )
            if len(captured_messages) == 2:
                assert any(
                    isinstance(message, ToolMessage)
                    and message.tool_call_id == "call-bad"
                    and "rows=2" in str(message.content)
                    and "sales_summary" in str(message.content)
                    for message in messages
                )
                return AIMessage(content="Done")
            return AIMessage(content="Done")

    settings = replace(
        Settings(),
        llm_provider="vllm",
        llm_num_ctx=100_000,
        llm_warmup_enabled=False,
    )

    with patch("backend.agent.tool_loop.build_runtime_llm", return_value=FakeLLM()):
        response = direct_tool_loop(
            ToolLoopRequest(
                settings=settings,
                include_reasoning=False,
                tools=[tool],
                callbacks=[collector],
                max_iterations=3,
                messages=[HumanMessage(content="show columns")],
            )
        )

    assert response.final_text == "Done"
    assert len(captured_messages) == 2
    second_batch_tool_messages = [
        message for message in captured_messages[1] if isinstance(message, ToolMessage)
    ]
    assert second_batch_tool_messages[0].tool_call_id == "call-bad"


def test_recovered_tool_error_does_not_poison_final_success() -> None:
    @tool
    def flaky_tool(action: str) -> str:
        """Return one simulated timeout and then a successful observation."""
        if action == "fail":
            raise TimeoutError("timed out")
        return "recovered"

    class FakeLLM:
        calls = 0

        def bind_tools(self, _tools):
            return self

        def get_num_tokens_from_messages(self, messages):
            return sum(len(str(getattr(message, "content", ""))) for message in messages)

        def invoke(self, _messages, config=None):
            del config
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "flaky_tool",
                            "args": {"action": "fail"},
                            "id": "call-fail",
                            "type": "tool_call",
                        }
                    ],
                )
            if self.calls == 2:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "flaky_tool",
                            "args": {"action": "recover"},
                            "id": "call-recover",
                            "type": "tool_call",
                        }
                    ],
                )
            return AIMessage(content="Recovered answer")

    settings = replace(
        Settings(),
        llm_provider="vllm",
        llm_num_ctx=100_000,
        llm_warmup_enabled=False,
    )
    with patch("backend.agent.tool_loop.build_runtime_llm", return_value=FakeLLM()):
        response = direct_tool_loop(
            ToolLoopRequest(
                settings=settings,
                include_reasoning=False,
                tools=[flaky_tool],
                max_iterations=4,
                messages=[HumanMessage(content="recover after a timeout")],
            )
        )

    assert response.final_text == "Recovered answer"
    assert response.task_contract_satisfied is True
    assert response.tool_error_count == 1


def test_tool_result_returns_to_model_for_final_synthesis() -> None:
    source = pd.DataFrame({"branch": ["A", "B"], "gap_pct": [5.0, -3.0]})
    sandbox = SessionSandbox()
    sandbox.put("source", source)
    tool = PandasTool(source, sandbox=sandbox, tool_cache_size=0)
    collector = ToolCollector()

    class FakeLLM:
        def __init__(self):
            self.calls = 0
            self.synthesis_messages = []

        def bind_tools(self, _tools):
            return self

        def get_num_tokens_from_messages(self, messages):
            return sum(len(str(getattr(message, "content", ""))) for message in messages)

        def invoke(self, messages, config=None):
            del config
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "pandas_tool",
                            "args": {
                                "code": (
                                    "result = source.copy()\n"
                                    "tool_result = {'schema_version': '1.0', "
                                    "'artifact_type': 'table', "
                                    "'items': {'result': result}}"
                                )
                            },
                            "id": "call-final",
                            "type": "tool_call",
                        }
                    ],
                )
            self.synthesis_messages = list(messages)
            return AIMessage(content="Branch A leads by 8 percentage points.")

    llm = FakeLLM()
    settings = replace(
        Settings(),
        llm_provider="vllm",
        llm_num_ctx=100_000,
        llm_warmup_enabled=False,
    )

    with patch("backend.agent.tool_loop.build_runtime_llm", return_value=llm):
        response = direct_tool_loop(
            ToolLoopRequest(
                settings=settings,
                include_reasoning=False,
                tools=[tool],
                callbacks=[collector],
                max_iterations=3,
                messages=[HumanMessage(content="show final rows")],
            )
        )

    assert llm.calls == 2
    assert response.contract_valid is True
    assert response.final_text == "Branch A leads by 8 percentage points."
    assert len(response.artifacts) == 1
    assert response.artifacts[0].name == "result"
    assert any(isinstance(message, ToolMessage) for message in llm.synthesis_messages)
