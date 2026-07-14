from pydantic import BaseModel

from backend.agent.runner import AgentRunner
from backend.auth import UserMemory
from backend.core.config import Settings
from backend.sessions.session_memory import SessionMemory


def test_message_build_request_is_pydantic_contract() -> None:
    from backend.agent.services.message_builder import MessageBuildRequest

    request = MessageBuildRequest(
        prompt="hello",
        history=[],
        use_history=False,
        settings=Settings(),
        user_memory=UserMemory(profile="Data lead", notes=""),
        session_memory=SessionMemory(),
    )

    assert isinstance(request, BaseModel)
    assert request.prompt == "hello"


def test_build_messages_injects_memory_without_runner() -> None:
    from langchain_core.messages import SystemMessage

    from backend.agent.services.message_builder import MessageBuildRequest, build_messages

    messages = build_messages(
        MessageBuildRequest(
            prompt="hello",
            history=[],
            use_history=False,
            settings=Settings(),
            user_memory=UserMemory(profile="Data lead", notes=""),
            session_memory=SessionMemory(),
        )
    )

    system_messages = [message for message in messages if isinstance(message, SystemMessage)]

    assert any("Data lead" in str(message.content) for message in system_messages)


def test_build_messages_skips_history_already_in_context_summary() -> None:
    from backend.agent.services.message_builder import MessageBuildRequest, build_messages

    memory = SessionMemory(
        context_summary="old-marker already summarized",
        compacted_message_count=2,
    )
    messages = build_messages(
        MessageBuildRequest(
            prompt="current prompt",
            history=[
                {"role": "user", "content": "old-marker-0"},
                {"role": "assistant", "content": "old-marker-1"},
                {"role": "user", "content": "fresh-marker"},
            ],
            use_history=True,
            settings=Settings(),
            user_memory=UserMemory(profile="", notes=""),
            session_memory=memory,
        )
    )

    text = "\n".join(str(message.content) for message in messages)

    assert "old-marker already summarized" in text
    assert "old-marker-0" not in text
    assert "old-marker-1" not in text
    assert "fresh-marker" in text


def test_build_messages_omits_context_summary_when_history_disabled() -> None:
    from backend.agent.services.message_builder import MessageBuildRequest, build_messages

    memory = SessionMemory(
        notes="persistent session note",
        context_summary="old-marker already summarized",
        compacted_message_count=2,
    )
    messages = build_messages(
        MessageBuildRequest(
            prompt="current prompt",
            history=[{"role": "user", "content": "fresh-marker"}],
            use_history=False,
            settings=Settings(),
            user_memory=UserMemory(profile="", notes=""),
            session_memory=memory,
        )
    )

    text = "\n".join(str(message.content) for message in messages)

    assert "persistent session note" in text
    assert "old-marker already summarized" not in text
    assert "fresh-marker" not in text
    assert "current prompt" in text


def test_execution_prompt_builder_matches_runner_contract() -> None:
    from backend.agent.services.message_builder import (
        ExecutionSystemPromptRequest,
        build_execution_system_prompt,
    )

    runner = AgentRunner()
    capability_context = {
        "source_mode": "dataset",
        "tool_descriptions": "",
        "available_tool_keys": ["sql_tool"],
    }

    request = ExecutionSystemPromptRequest(
        settings=runner.settings,
        skill_registry=runner.skill_registry,
        enabled_analytical_skill_ids=runner.enabled_analytical_skill_ids,
        capability_context=capability_context,
    )
    prompt = build_execution_system_prompt(request)

    assert "sql_tool" in prompt
    assert "Thought:" not in prompt


def test_execution_prompt_contains_prompt_only_general_analytics_rule() -> None:
    from backend.agent.services.message_builder import (
        ExecutionSystemPromptRequest,
        build_execution_system_prompt,
    )

    runner = AgentRunner()
    capability_context = {
        "source_mode": "dataset",
        "tool_descriptions": "",
        "available_tool_keys": ["get_tool_instructions", "sql_tool"],
    }

    request = ExecutionSystemPromptRequest(
        settings=runner.settings,
        skill_registry=runner.skill_registry,
        enabled_analytical_skill_ids=runner.enabled_analytical_skill_ids,
        capability_context=capability_context,
    )
    prompt = build_execution_system_prompt(request)

    assert "active_workflow" not in prompt
    assert "CSV/XLSX" in prompt
    assert 'get_tool_instructions("general_analytics")' in prompt


def test_execution_prompt_explains_core_tool_pipeline_contract() -> None:
    from backend.agent.services.message_builder import (
        ExecutionSystemPromptRequest,
        build_execution_system_prompt,
    )

    runner = AgentRunner()
    capability_context = {
        "source_mode": "dataset",
        "tool_descriptions": "",
        "available_tool_keys": ["sql_tool", "pandas_tool", "plotly_tool"],
    }

    prompt = build_execution_system_prompt(
        ExecutionSystemPromptRequest(
            settings=runner.settings,
            skill_registry=runner.skill_registry,
            enabled_analytical_skill_ids=runner.enabled_analytical_skill_ids,
            capability_context=capability_context,
        )
    )

    assert "Основной технический pipeline" in prompt
    assert "`sql_tool`" in prompt
    assert "`pandas_tool`" in prompt
    assert "`plotly_tool`" in prompt
    assert "одном session sandbox" in prompt
    assert "Failed tool calls do not create sandbox variables" in prompt


def test_execution_prompt_requires_planner_for_new_analytical_tasks_only() -> None:
    from backend.agent.services.message_builder import (
        ExecutionSystemPromptRequest,
        build_execution_context_messages,
        build_execution_system_prompt,
    )

    runner = AgentRunner()
    capability_context = {
        "source_mode": "dataset",
        "tool_descriptions": "",
        "available_tool_keys": ["planner_tool", "get_tool_instructions", "sql_tool"],
    }

    request = ExecutionSystemPromptRequest(
        settings=runner.settings,
        skill_registry=runner.skill_registry,
        enabled_analytical_skill_ids=runner.enabled_analytical_skill_ids,
        capability_context=capability_context,
    )
    prompt = build_execution_system_prompt(request)
    context_text = "\n\n".join(
        str(message.content) for message in build_execution_context_messages(request)
    )

    assert "Новая аналитическая задача" in prompt
    assert "`planner_tool` → `get_tool_instructions(\"general_analytics\")`" in prompt
    assert "выбросы" in prompt
    assert "аномалии" in prompt
    assert "переделай график" in prompt
    assert "SKILL_CATALOG_CONTEXT:" in context_text
