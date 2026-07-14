from __future__ import annotations

from pathlib import Path

import pytest

from backend.agent.prompts import execution_agent_prompt
from backend.tools.instructions import (
    ToolInstructionError,
    ToolInstructionRegistry,
    extract_markdown_section,
    get_default_tool_instruction_registry,
    tool_description,
    tool_section_text,
)


def _write_tool(
    tmp_path: Path,
    tool_key: str,
    content: str,
    details: str | None = None,
) -> None:
    tool_dir = tmp_path / tool_key
    tool_dir.mkdir()
    (tool_dir / "TOOL.md").write_text(content, encoding="utf-8")
    if details is not None:
        (tool_dir / "DETAILS.md").write_text(details, encoding="utf-8")


def _tool_md(tool_key: str = "sql_tool") -> str:
    return (
        "---\n"
        f"id: {tool_key}\n"
        "name: SQL Tool\n"
        "kind: tool\n"
        f"tool_key: {tool_key}\n"
        "description: Run read-only SQL over session tables.\n"
        "enabled_by_default: true\n"
        "triggers:\n"
        "  - sql\n"
        "  - table\n"
        "---\n\n"
        "## Purpose\nUse for SQL.\n\n"
        "### API\nUse structured arguments.\n\n"
        "### Final result protocol\nReturns a table artifact.\n"
    )


def test_tool_instruction_registry_loads_tool_docs(tmp_path: Path) -> None:
    _write_tool(tmp_path, "sql_tool", _tool_md("sql_tool"))

    registry = ToolInstructionRegistry.from_path(tmp_path).load()
    document = registry.get("sql_tool")

    assert document.metadata.tool_key == "sql_tool"
    assert document.metadata.enabled_by_default is True
    assert document.metadata.triggers == ("sql", "table")
    assert "structured arguments" in document.body


def test_tool_instruction_registry_loads_details(tmp_path: Path) -> None:
    _write_tool(
        tmp_path,
        "sql_tool",
        _tool_md("sql_tool"),
        details="## Examples\nSELECT 1",
    )

    document = ToolInstructionRegistry.from_path(tmp_path).load().get("sql_tool")

    assert document.has_details is True
    assert document.details_markdown == "## Examples\nSELECT 1"


def test_tool_instruction_registry_rejects_missing_tools_dir(tmp_path: Path) -> None:
    with pytest.raises(ToolInstructionError, match="does not exist"):
        ToolInstructionRegistry.from_path(tmp_path / "missing").load()


def test_tool_instruction_registry_rejects_non_tool_docs(tmp_path: Path) -> None:
    _write_tool(
        tmp_path,
        "broken",
        "---\n"
        "id: broken\n"
        "name: Broken\n"
        "kind: analytical\n"
        "description: Not a tool.\n"
        "---\n\n"
        "### API\nNope.",
    )

    with pytest.raises(ToolInstructionError, match="kind='tool'"):
        ToolInstructionRegistry.from_path(tmp_path).load()


def test_extract_markdown_section_returns_requested_section() -> None:
    markdown = (
        "## Purpose\n"
        "Overview.\n\n"
        "### API\n"
        "Call with JSON.\n\n"
        "### Final result protocol\n"
        "Return JSON."
    )

    assert extract_markdown_section(markdown, "API") == "Call with JSON."


def test_extract_markdown_section_rejects_missing_section() -> None:
    with pytest.raises(ToolInstructionError, match="not found"):
        extract_markdown_section("## Purpose\nOverview.", "API")


def test_default_tool_instruction_registry_loads_project_tools() -> None:
    registry = get_default_tool_instruction_registry()
    tool_keys = {document.metadata.tool_key for document in registry.list_tools()}

    assert "sql_tool" in tool_keys
    assert "pandas_tool" in tool_keys
    assert "get_tool_instructions" in tool_keys


def test_default_tool_instruction_registry_loads_extended_project_tool_docs() -> None:
    registry = get_default_tool_instruction_registry()

    for tool_key in ("sql_tool", "pandas_tool", "plotly_tool", "database_tool"):
        document = registry.get(tool_key)
        assert document.has_details is True
        assert document.details_markdown


def test_tool_description_is_loaded_from_project_tool_markdown() -> None:
    assert tool_description("sql_tool").startswith("Run read-only SQL")


def test_tool_section_text_unwraps_markdown_code_fence() -> None:
    section = tool_section_text("planner_tool", "Internal system prompt")

    assert "compact execution planner" in section
    assert not section.startswith("```")
    assert not section.endswith("```")


def test_pandas_tool_doc_is_local_execution_contract_not_workflow() -> None:
    document = get_default_tool_instruction_registry().get("pandas_tool")
    text = document.body.lower()

    assert "### data flow" not in text
    assert "do not call `sql_tool`" in text
    assert "from inside this code" in text


def test_pandas_inspection_contract_is_prompted_as_table_artifact() -> None:
    tool_text = get_default_tool_instruction_registry().get("pandas_tool").body.lower()
    prompt_text = execution_agent_prompt.lower()

    for text in (tool_text, prompt_text):
        assert "inspection" in text
        assert "diagnostics" in text
        assert "compact table artifact" in text
