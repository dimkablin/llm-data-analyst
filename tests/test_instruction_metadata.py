from __future__ import annotations

from pathlib import Path

import pytest

from backend.instructions import (
    InstructionKind,
    InstructionMarkdownError,
    InstructionMetadata,
    parse_instruction_markdown,
    read_instruction_document,
)


def test_instruction_metadata_normalizes_string_triggers() -> None:
    metadata = InstructionMetadata.model_validate(
        {
            "id": "My_Tool",
            "name": "My Tool",
            "description": "A useful tool.",
            "kind": "tool",
            "tool_key": "my_tool",
            "triggers": "SQL, sql, Table",
            "enabled_by_default": False,
        }
    )

    assert metadata.id == "my_tool"
    assert metadata.kind == InstructionKind.TOOL
    assert metadata.triggers == ("sql", "table")
    assert metadata.enabled_by_default is False


def test_instruction_metadata_requires_tool_key_for_tools() -> None:
    with pytest.raises(ValueError, match="tool_key"):
        InstructionMetadata.model_validate(
            {
                "id": "my_tool",
                "name": "My Tool",
                "description": "A useful tool.",
                "kind": "tool",
            }
        )


def test_parse_instruction_markdown_uses_defaults() -> None:
    document = parse_instruction_markdown(
        "---\nname: Cohort\ndescription: Cohort analysis.\ntriggers: cohort, retention\n---\n\n"
        "### Algorithm\nRun analysis.\n\n### Rules\nUse evidence.",
        source_path="skills/cohort/SKILL.md",
        default_id="cohort",
    )

    assert document.instruction_id == "cohort"
    assert document.metadata.kind == InstructionKind.ANALYTICAL
    assert document.metadata.triggers == ("cohort", "retention")
    assert "Algorithm" in document.body


def test_read_instruction_document_loads_details(tmp_path: Path) -> None:
    instruction_path = tmp_path / "TOOL.md"
    details_path = tmp_path / "DETAILS.md"
    instruction_path.write_text(
        "---\n"
        "name: SQL Tool\n"
        "description: Run read-only SQL.\n"
        "kind: tool\n"
        "tool_key: sql_tool\n"
        "---\n\n"
        "### API\nUse structured args.",
        encoding="utf-8",
    )
    details_path.write_text("## Examples\nSELECT 1", encoding="utf-8")

    document = read_instruction_document(
        instruction_path,
        default_id="sql_tool",
        default_kind="tool",
        details_path=details_path,
    )

    assert document.metadata.tool_key == "sql_tool"
    assert document.details_markdown == "## Examples\nSELECT 1"


def test_parse_instruction_markdown_rejects_missing_frontmatter() -> None:
    with pytest.raises(InstructionMarkdownError, match="frontmatter"):
        parse_instruction_markdown(
            "### API\nMissing metadata.",
            source_path="tools/broken/TOOL.md",
            default_id="broken",
            default_kind="tool",
        )
