from __future__ import annotations

from pathlib import Path

from backend.skills import SkillRegistry
from backend.tools.impl.get_tool_instructions_tool import GetToolInstructionsTool
from backend.tools.instructions import ToolInstructionRegistry

_TOOL_SKILL = (
    "---\nname: My Tool\ndescription: Does things.\nkind: tool\ntool_key: my_tool\n---\n\n"
    "### API\nfoo(x: int) -> None\n\n"
    "### Final result protocol\nLast expression must be tool_result.\n\n"
    "### Rules\n- Always set x > 0\n"
)
_DETAILS_CONTENT = "## Examples\n```python\nfoo(1)\n```\n"


def _make_registry(tmp_path: Path, with_details: bool = False) -> SkillRegistry:
    skill_dir = tmp_path / "my_tool"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(_TOOL_SKILL, encoding="utf-8")
    if with_details:
        (skill_dir / "DETAILS.md").write_text(_DETAILS_CONTENT, encoding="utf-8")
    return SkillRegistry.from_path(tmp_path).load()


def _make_tool_instruction_registry(tmp_path: Path, with_details: bool = False) -> ToolInstructionRegistry:
    tool_dir = tmp_path / "tools" / "my_tool"
    tool_dir.mkdir(parents=True)
    (tool_dir / "TOOL.md").write_text(
        "---\n"
        "id: my_tool\n"
        "name: My Tool\n"
        "kind: tool\n"
        "tool_key: my_tool\n"
        "description: Does things.\n"
        "enabled_by_default: true\n"
        "---\n\n"
        "### API\nfrom TOOL.md\n\n"
        "### Final result protocol\nReturn content.\n",
        encoding="utf-8",
    )
    if with_details:
        (tool_dir / "DETAILS.md").write_text("## Tool Details\nExample", encoding="utf-8")
    return ToolInstructionRegistry.from_path(tool_dir.parent).load()


def test_default_returns_core(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path, with_details=True)
    tool = GetToolInstructionsTool(registry)
    result = tool._run(skill_id="my_tool")
    assert "API" in result
    assert "Examples" not in result  # details not included by default


def test_default_includes_hint_when_details_exist(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path, with_details=True)
    tool = GetToolInstructionsTool(registry)
    result = tool._run(skill_id="my_tool")
    assert "details=True" in result
    assert "my_tool" in result


def test_no_hint_when_no_details(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path, with_details=False)
    tool = GetToolInstructionsTool(registry)
    result = tool._run(skill_id="my_tool")
    assert "details=True" not in result


def test_details_true_returns_details(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path, with_details=True)
    tool = GetToolInstructionsTool(registry)
    result = tool._run(skill_id="my_tool", details=True)
    assert "Examples" in result
    assert "API" not in result  # core not repeated


def test_details_true_missing_returns_graceful_fallback(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path, with_details=False)
    tool = GetToolInstructionsTool(registry)
    result = tool._run(skill_id="my_tool", details=True)
    assert "not available" in result.lower()
    assert "API" not in result  # core NOT repeated


def test_unknown_skill_returns_available_list(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path)
    tool = GetToolInstructionsTool(registry)
    result = tool._run(skill_id="nonexistent_tool")
    assert "sql_tool" in result


def test_tool_docs_take_precedence_over_tool_skills(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path)
    tool_registry = _make_tool_instruction_registry(tmp_path)
    tool = GetToolInstructionsTool(registry, tool_instruction_registry=tool_registry)

    result = tool._run(skill_id="my_tool")

    assert "from TOOL.md" in result
    assert "Always set x > 0" not in result


def test_tool_docs_details_true_returns_tool_details(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path)
    tool_registry = _make_tool_instruction_registry(tmp_path, with_details=True)
    tool = GetToolInstructionsTool(registry, tool_instruction_registry=tool_registry)

    result = tool._run(skill_id="my_tool", details=True)

    assert result == "## Tool Details\nExample"
