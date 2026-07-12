from __future__ import annotations

from pathlib import Path

from backend.skills import SkillRegistry
from backend.tools.impl.get_tool_instructions_tool import GetToolInstructionsTool

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
    assert "my_tool" in result
