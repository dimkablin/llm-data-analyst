from __future__ import annotations

from backend.agent.dataset_profiles import build_sql_generation_hints
from backend.tools.instructions import tool_description


def test_sql_hints_show_exact_names_as_code_identifiers() -> None:
    hints = build_sql_generation_hints(["Промо_активность", "market value"])

    assert "`Промо_активность`" in hints
    assert "`market value`" in hints


def test_sql_tool_description_prompts_identifier_contract() -> None:
    description = tool_description("sql_tool")

    assert "double-quote identifiers" in description
    assert "preserve their spelling" in description
