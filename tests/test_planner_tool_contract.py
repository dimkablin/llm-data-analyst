from __future__ import annotations

from typing import Any

from backend.tools.impl.planner_tool import PlannerTool


def test_planner_tool_uses_compact_execution_sketch_prompt(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class _FakeResponse:
        def __init__(self) -> None:
            self.content = (
                "- source: active_context\n"
                "- steps:\n"
                "  - [ ] direct: answer from context\n"
                "- answer: concise"
            )
            self.additional_kwargs: dict[str, Any] = {}

    class _FakeLLM:
        def invoke(self, messages):
            captured["messages"] = messages
            return _FakeResponse()

    def _fake_make_reasoning_llm(**kwargs):
        captured["llm_kwargs"] = kwargs
        return _FakeLLM()

    monkeypatch.setattr(
        "backend.tools.impl.planner_tool.make_reasoning_llm",
        _fake_make_reasoning_llm,
    )

    tool = PlannerTool(
        llm_model="test-model",
        llm_base_url="http://test",
        tool_descriptions="- `sql_tool`: SQL",
    )

    result = tool._run("посмотри документ который я загрузил")

    system_prompt = captured["messages"][0].content
    llm_kwargs = captured["llm_kwargs"]

    assert result.startswith("- source:")
    assert "compact execution planner" in system_prompt
    assert "plan-checkpoint assistant" in system_prompt
    assert "Maximum 3 bullets" in system_prompt
    assert "Maximum 450 characters" in system_prompt
    assert "- source:" in system_prompt
    assert "- steps:" in system_prompt
    assert "- [ ] <tool_or_direct>: <what to obtain next>" in system_prompt
    assert "- [x] <tool_or_direct>: <completed step" in system_prompt
    assert "- answer:" in system_prompt
    assert "markdown headings" in system_prompt
    assert 'get_tool_instructions("general_analytics")' in system_prompt
    assert llm_kwargs["temperature"] == 0.0
    assert llm_kwargs["max_tokens"] == 1024
