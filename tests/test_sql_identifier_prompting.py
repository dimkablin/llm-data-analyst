from __future__ import annotations

import backend.tools.impl  # noqa: F401 - side-effect import avoids sql_table_service circular import

from backend.agent.dataset_profiles import build_sql_generation_hints
from backend.data_access.sql_table_service import SQLTableService
from backend.tools.instructions import tool_description


def test_duckdb_prompt_quotes_cyrillic_and_spaced_columns() -> None:
    text = SQLTableService._quoted_columns_str(
        ["Промо_активность", "market value", "plain_col"],
        "duckdb",
    )

    assert '"Промо_активность"' in text
    assert '"market value"' in text
    assert '"plain_col"' in text


def test_sql_hints_show_exact_names_as_code_identifiers() -> None:
    hints = build_sql_generation_hints(["Промо_активность", "market value"])

    assert "`Промо_активность`" in hints
    assert "`market value`" in hints


def test_sql_tool_description_prompts_identifier_contract() -> None:
    description = tool_description("sql_tool")

    assert "double-quote identifiers" in description
    assert "never replace spaces" in description
