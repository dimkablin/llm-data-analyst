from __future__ import annotations

import json
from dataclasses import replace
from typing import ClassVar
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ValidationError

from backend.agent.tool_loop import ToolLoopRequest, direct_tool_loop
from backend.agent.working_memory import AnalysisWorkingMemory
from backend.core.config import Settings
from backend.tools.context import ToolBuildContext
from backend.tools.impl.update_plan_tool import UpdatePlanInput, UpdatePlanTool
from backend.tools.registry import ToolRegistry


def test_update_plan_replaces_the_complete_working_checklist() -> None:
    memory = AnalysisWorkingMemory(goal="analyze data")
    tool = UpdatePlanTool(memory)

    first_result = tool.invoke(
        {
            "plan": [
                {"step": "Inspect the schema", "status": "in_progress"},
                {"step": "Calculate the result", "status": "pending"},
            ]
        }
    )
    second_result = tool.invoke(
        {
            "plan": [
                {"step": "Inspect the schema", "status": "completed"},
                {"step": "Use the discovered relationship", "status": "in_progress"},
            ]
        }
    )

    assert json.loads(first_result) == {
        "plan": [
            {"step": "Inspect the schema", "status": "in_progress"},
            {"step": "Calculate the result", "status": "pending"},
        ],
        "completed": 0,
        "total": 2,
    }
    assert json.loads(second_result) == {
        "plan": [
            {"step": "Inspect the schema", "status": "completed"},
            {"step": "Use the discovered relationship", "status": "in_progress"},
        ],
        "completed": 1,
        "total": 2,
    }
    assert [item.step for item in memory.current_plan] == [
        "Inspect the schema",
        "Use the discovered relationship",
    ]


@pytest.mark.parametrize(
    "plan",
    [
        [
            {"step": "Inspect data", "status": "in_progress"},
            {"step": "inspect DATA", "status": "pending"},
        ],
        [
            {"step": "Inspect data", "status": "in_progress"},
            {"step": "Calculate result", "status": "in_progress"},
        ],
    ],
)
def test_update_plan_rejects_ambiguous_checklist_state(plan: list[dict[str, str]]) -> None:
    with pytest.raises(ValidationError):
        UpdatePlanInput.model_validate({"plan": plan})


def test_update_plan_rejects_more_than_eight_steps() -> None:
    with pytest.raises(ValidationError):
        UpdatePlanInput.model_validate(
            {"plan": [{"step": f"Step {index}", "status": "pending"} for index in range(9)]}
        )


def test_update_plan_returns_text_without_creating_an_artifact() -> None:
    memory = AnalysisWorkingMemory(goal="answer")

    result = UpdatePlanTool(memory).invoke({"plan": [{"step": "Answer the question", "status": "completed"}]})

    assert isinstance(result, str)


def test_registry_binds_update_plan_to_the_request_working_memory() -> None:
    memory = AnalysisWorkingMemory(goal="analyze")
    tools = ToolRegistry.from_services().build_tools(
        ToolBuildContext(settings=Settings(), working_memory=memory)
    )
    tool = next(tool for tool in tools if tool.name == "update_plan")

    tool.invoke({"plan": [{"step": "Inspect data", "status": "in_progress"}]})

    assert memory.current_plan[0].step == "Inspect data"


def test_tool_loop_passes_latest_plan_as_tool_result_without_blocking_final_answer() -> None:
    memory = AnalysisWorkingMemory(goal="analyze")
    tool = UpdatePlanTool(memory)
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
                            "name": "update_plan",
                            "args": {
                                "plan": [
                                    {"step": "Inspect data", "status": "completed"},
                                    {"step": "Calculate result", "status": "in_progress"},
                                ]
                            },
                            "id": "call-plan",
                            "type": "tool_call",
                        }
                    ],
                )
            return AIMessage(content="Partial but honest answer")

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
                max_iterations=3,
                messages=[SystemMessage(content="base instructions"), HumanMessage(content="analyze")],
                working_memory=memory,
            )
        )

    assert all(
        sum(isinstance(message, SystemMessage) for message in invocation) == 1
        for invocation in captured_messages
    )
    assert all(
        "CURRENT ANALYSIS PLAN" not in str(message.content)
        for invocation in captured_messages
        for message in invocation
        if isinstance(message, SystemMessage)
    )
    plan_result = next(
        message
        for message in captured_messages[1]
        if isinstance(message, ToolMessage) and message.name == "update_plan"
    )
    assert json.loads(str(plan_result.content)) == {
        "plan": [
            {"step": "Inspect data", "status": "completed"},
            {"step": "Calculate result", "status": "in_progress"},
        ],
        "completed": 1,
        "total": 2,
    }
    assert response.final_text == "Partial but honest answer"
    assert response.task_contract_satisfied is True


def test_always_plan_uses_native_tool_choice_only_for_the_first_llm_call() -> None:
    memory = AnalysisWorkingMemory(goal="list tables")
    plan_tool = UpdatePlanTool(memory)
    bindings: list[dict[str, object]] = []

    class BoundLLM:
        def __init__(self, forced: bool) -> None:
            self.forced = forced

        def invoke(self, _messages, config=None):
            del config
            if self.forced:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "update_plan",
                            "args": {
                                "plan": [
                                    {"step": "List tables", "status": "in_progress"},
                                ]
                            },
                            "id": "forced-plan",
                            "type": "tool_call",
                        }
                    ],
                )
            return AIMessage(content="Done")

    class FakeLLM:
        def bind_tools(self, _tools, **kwargs):
            bindings.append(kwargs)
            return BoundLLM(forced=kwargs.get("tool_choice") == "update_plan")

        def get_num_tokens_from_messages(self, messages):
            return sum(len(str(getattr(message, "content", ""))) for message in messages)

    settings = replace(
        Settings(),
        always_use_analysis_plan=True,
        llm_num_ctx=100_000,
        llm_warmup_enabled=False,
    )
    with patch("backend.agent.tool_loop.build_runtime_llm", return_value=FakeLLM()):
        response = direct_tool_loop(
            ToolLoopRequest(
                settings=settings,
                include_reasoning=False,
                tools=[plan_tool],
                max_iterations=3,
                messages=[SystemMessage(content="base"), HumanMessage(content="list tables")],
                working_memory=memory,
            )
        )

    assert bindings == [{}, {"tool_choice": "update_plan"}]
    assert response.tool_names == ["update_plan"]
    assert response.final_text == "Done"


def test_always_plan_can_update_plan_alongside_later_tool_work() -> None:
    memory = AnalysisWorkingMemory(goal="analyze")
    plan_tool = UpdatePlanTool(memory)
    normal_invocations = 0
    captured_messages: list[list[object]] = []

    class WorkInput(BaseModel):
        query: str

    class WorkTool(BaseTool):
        name: str = "work_tool"
        description: str = "Return requested evidence"
        args_schema: type[BaseModel] = WorkInput
        parallel_safe: ClassVar[bool] = True

        def _run(self, query: str) -> str:
            return f"evidence:{query}"

    class BoundLLM:
        def __init__(self, forced: bool) -> None:
            self.forced = forced

        def invoke(self, messages, config=None):
            nonlocal normal_invocations
            del config
            captured_messages.append(list(messages))
            if self.forced:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "update_plan",
                            "args": {
                                "plan": [
                                    {"step": "Inspect evidence", "status": "in_progress"},
                                ]
                            },
                            "id": "initial-plan",
                            "type": "tool_call",
                        }
                    ],
                )
            normal_invocations += 1
            if normal_invocations == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "update_plan",
                            "args": {
                                "plan": [
                                    {"step": "Inspect evidence", "status": "completed"},
                                    {"step": "Verify result", "status": "in_progress"},
                                ]
                            },
                            "id": "revised-plan",
                            "type": "tool_call",
                        },
                        {
                            "name": "work_tool",
                            "args": {"query": "verification"},
                            "id": "work",
                            "type": "tool_call",
                        },
                    ],
                )
            return AIMessage(content="Done")

    class FakeLLM:
        def bind_tools(self, _tools, **kwargs):
            return BoundLLM(forced=kwargs.get("tool_choice") == "update_plan")

        def get_num_tokens_from_messages(self, messages):
            return sum(len(str(getattr(message, "content", ""))) for message in messages)

    settings = replace(
        Settings(),
        always_use_analysis_plan=True,
        llm_num_ctx=100_000,
        llm_warmup_enabled=False,
    )
    with patch("backend.agent.tool_loop.build_runtime_llm", return_value=FakeLLM()):
        response = direct_tool_loop(
            ToolLoopRequest(
                settings=settings,
                include_reasoning=False,
                tools=[plan_tool, WorkTool()],
                max_iterations=4,
                messages=[SystemMessage(content="base"), HumanMessage(content="analyze")],
                working_memory=memory,
            )
        )

    final_tool_results = {
        message.name: str(message.content)
        for message in captured_messages[-1]
        if isinstance(message, ToolMessage)
    }
    assert response.tool_names == ["update_plan", "work_tool"]
    assert response.tool_calls == 3
    assert set(final_tool_results) == {"update_plan", "work_tool"}
    assert final_tool_results["work_tool"] == "evidence:verification"
    assert response.final_text == "Done"
