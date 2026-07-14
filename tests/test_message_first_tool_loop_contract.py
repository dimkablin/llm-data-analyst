from __future__ import annotations

from dataclasses import replace
from typing import get_args, get_type_hints
from unittest.mock import patch

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
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
from backend.tools.impl.pandas_tool import PandasTool
from backend.tools.sandbox import SessionSandbox


def test_agent_graph_state_messages_use_langgraph_add_messages() -> None:
    hints = get_type_hints(AgentGraphState, include_extras=True)

    assert "messages" in hints
    assert add_messages in get_args(hints["messages"])


def test_skill_and_data_context_are_separate_messages() -> None:
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
        df=pd.DataFrame({"revenue": [1, 2]}),
        session_source={"source_label": "demo"},
    )

    system_prompt = build_execution_system_prompt(request)
    context_messages = build_execution_context_messages(request)

    assert "full selected skill" not in system_prompt
    assert any(
        isinstance(message, HumanMessage)
        and str(message.content).startswith("SKILL_CONTEXT:")
        and "full selected skill" in str(message.content)
        for message in context_messages
    )
    assert any(str(message.content).startswith("DATA_CONTEXT:") for message in context_messages)


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
    assert payload["table"]["output"]["output"].tolist() == [
        "['month', 'channel', 'revenue']"
    ]


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
                            "args": {"code": "print(sales_by_month_channel.columns.tolist())"},
                            "id": "call-bad",
                            "type": "tool_call",
                        }
                    ],
                )
            if len(captured_messages) == 2:
                assert any(
                    isinstance(message, ToolMessage)
                    and message.tool_call_id == "call-bad"
                    and "output" in str(message.content)
                    and "month" in str(message.content)
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
