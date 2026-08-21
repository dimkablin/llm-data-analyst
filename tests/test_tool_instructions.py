from __future__ import annotations

from pathlib import Path

import pytest

from backend.agent.prompts import execution_agent_prompt
from backend.tools.impl.sql_tool import SQLToolArgs
from backend.tools.instructions import (
    ToolInstructionError,
    ToolInstructionRegistry,
    extract_markdown_section,
    get_default_tool_instruction_registry,
    tool_description,
)


def test_analytics_prompts_forbid_dataframe_links_in_answers() -> None:
    general = Path("skills/general_analytics/SKILL.md").read_text(encoding="utf-8")
    investment = Path("skills/investment_market_analysis/SKILL.md").read_text(encoding="utf-8")

    assert "Never format them as Markdown links" in execution_agent_prompt
    assert "Never turn tool-derived values into Markdown links" in general
    assert "expose artifact expressions" in investment


def test_general_analytics_separates_total_rows_from_components() -> None:
    general = Path("skills/general_analytics/SKILL.md").read_text(encoding="utf-8")

    assert "roll-up" in general
    assert "components" in general
    assert "either total or members" in general
    assert "verify key uniqueness" in general
    assert "date alone duplicates repeated entities" in general


def test_general_analytics_requires_explicit_coarser_time_grain() -> None:
    general = Path("skills/general_analytics/SKILL.md").read_text(encoding="utf-8")

    assert "requested period is coarser" in general
    assert "date_trunc" in general
    assert "grouping by the raw date remains raw-date grain" in general.lower()


def test_planfact_skill_requires_its_own_source_tables() -> None:
    text = Path("skills/planfact_variance_analysis/SKILL.md").read_text(encoding="utf-8")

    assert "not for generic plan-versus-fact metrics" in text
    assert "required `planfact_*` tables must exist" in text


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
        "---\nid: broken\nname: Broken\nkind: analytical\ndescription: Not a tool.\n---\n\n### API\nNope.",
    )

    with pytest.raises(ToolInstructionError, match="kind='tool'"):
        ToolInstructionRegistry.from_path(tmp_path).load()


def test_extract_markdown_section_returns_requested_section() -> None:
    markdown = "## Purpose\nOverview.\n\n### API\nCall with JSON.\n\n### Final result protocol\nReturn JSON."

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
    description = tool_description("sql_tool")
    assert description.startswith("Run read-only SQL")
    assert "conditionally aggregate in a CTE" in description
    semantic_description = tool_description("semantic_catalog_read_tool")
    assert "missing, incomplete, or ambiguous" in semantic_description
    semantic_details = get_default_tool_instruction_registry().get("semantic_catalog_read_tool").body
    assert "already present" in semantic_details
    assert "without retrieving them again" in semantic_details
    plan_description = tool_description("update_plan")
    assert "before multi-step analytical work" in plan_description
    assert "progress or new evidence changes the remaining route" in plan_description


def test_runtime_tool_descriptions_keep_evidence_and_chart_grounded() -> None:
    pandas_description = tool_description("pandas_tool")
    assert "exact variable and output columns" in pandas_description
    assert "convert object/string dates with pd.to_datetime before .dt access" in pandas_description
    assert "df.groupby(keys)[numeric_measure_columns]" in pandas_description
    assert "never retype artifact rows as Python literals" in pandas_description
    assert "separate scenario columns already exist" in pandas_description
    assert "assign the final dataframe to tool_result" in pandas_description
    plotly_description = tool_description("plotly_tool")
    assert "materially clarifies a comparison or trend" in plotly_description
    assert "explicit chart bans win" in plotly_description
    assert "`chart` is already bound, so never import plotly_tool" in plotly_description


def test_sql_tool_schema_prompts_portable_numeric_aggregates() -> None:
    mode_field = SQLToolArgs.model_json_schema()["properties"]["mode"]
    sql_field = SQLToolArgs.model_json_schema()["properties"]["sql"]
    metrics_field = SQLToolArgs.model_json_schema()["properties"]["metrics"]
    dimensions_field = SQLToolArgs.model_json_schema()["properties"]["dimensions"]
    filters_field = SQLToolArgs.model_json_schema()["properties"]["filters"]
    time_dimension_field = SQLToolArgs.model_json_schema()["properties"]["time_dimension"]
    time_grain_field = SQLToolArgs.model_json_schema()["properties"]["time_grain"]
    assert "answer_ready" not in SQLToolArgs.model_json_schema()["properties"]

    assert "raw numeric aggregates" in tool_description("sql_tool")
    assert "metric_resolution=resolved" in mode_field["description"]
    assert "do not inspect coverage or rebuild the formula" in mode_field["description"]
    assert "AVG(value) AS avg_value" in sql_field["description"]
    assert "CROSS JOIN LATERAL (VALUES ...)" in sql_field["description"]
    assert "project the value-table alias columns" in sql_field["description"]
    assert "switch to explicit UNION ALL branches" in sql_field["description"]
    assert "UNION ALL their final aggregates" in sql_field["description"]
    assert "one final UNION ALL query" in sql_field["description"]
    assert "never resend the same SQL" in sql_field["description"]
    assert "joint member combinations" in sql_field["description"]
    assert "corresponding value column" in sql_field["description"]
    assert "OR-ed nonzero measures" in sql_field["description"]
    assert "source MAX(date)" in sql_field["description"]
    assert "averaging all history beside MAX(date)" in sql_field["description"]
    assert "alias of the table that declares it" in sql_field["description"]
    assert "Sandbox artifact names are not database relations" in sql_field["description"]
    assert "balanced CTEs" in sql_field["description"]
    assert "AVG(value) AS avg_value" in sql_field["examples"][0]
    sql_doc = get_default_tool_instruction_registry().get("sql_tool").body
    assert "synthesized convenience totals are not source fields" in sql_doc
    assert "PostgreSQL has no `UNPIVOT` syntax" in sql_doc
    assert "project the value-table alias columns" in sql_doc
    assert "switch to explicit `UNION ALL` branches" in sql_doc
    assert "latest complete year containing every compared period" in sql_doc
    assert "inside one final query" in sql_doc
    assert "Never resend failed SQL unchanged" in sql_doc
    assert "never a join condition" in sql_doc
    assert "MAX(source_date)" in sql_doc
    assert "latest snapshot filters rows" in sql_doc
    assert "non-unique date alone" in sql_doc
    normalized_sql_doc = " ".join(sql_doc.split())
    assert "dimension type, member, period, value, baseline" in normalized_sql_doc
    assert "For absolute slices" in normalized_sql_doc
    assert "without inventing a comparison" in normalized_sql_doc
    assert "Python dataframes, not database relations" in sql_doc
    assert "group to the final period/slice/item grain" in sql_doc
    assert "Metric-defined filters are compiled automatically" in sql_doc
    assert "do not repeat them" in filters_field["description"]
    assert "complete requested time window" in filters_field["description"]
    assert "Do not put time grain labels" in dimensions_field["description"]
    assert "together with time_grain" in time_dimension_field["description"]
    assert "does not infer a grain" in time_grain_field["description"]
    assert "let the agent synthesize the answer" in sql_doc
    assert "Top-k candidates are context" in metrics_field["description"]
    assert "not executable keys" in metrics_field["description"]
    assert "compatible base table" in metrics_field["description"]
    assert "metric_resolution=resolved" in sql_doc
    assert "allowed_dimensions" in sql_doc
    assert "do not inspect coverage or rebuild a resolved" in sql_doc.lower()


def test_pandas_tool_brief_description_exposes_its_execution_boundary() -> None:
    registry = get_default_tool_instruction_registry()
    description = registry.build_brief_block({"pandas_tool"}).lower()

    assert "one dataframe transformation" in description
    assert "dependent filtering, reshaping, derived columns, and ranking" in description
    assert "existing dataframe artifact" in description
    assert "omit input_artifacts" in description
    assert "never an artifact name" in description
    assert "separate top-level actions" in description
    assert "unique descriptive item key" in description
    assert "never generic result or table" in description
    pandas_doc = registry.get("pandas_tool").body
    assert "Do not use Pandas for grouping, deltas, shares, ranking" in pandas_doc


def test_sql_and_plotly_briefs_expose_artifact_and_axis_contracts() -> None:
    registry = get_default_tool_instruction_registry()
    plotly_description = registry.build_brief_block({"plotly_tool"}).lower()

    assert "distinct name" in SQLToolArgs.model_json_schema()["properties"]["artifact_name"]["description"]
    assert "omit input_artifacts" in plotly_description
    assert "never an artifact name" in plotly_description
    assert "categoryorder" in plotly_description
    assert "never category_order" in plotly_description
    assert "do not duplicate a sufficient final table" in registry.get("plotly_tool").body


def test_forecast_contract_requires_a_structured_tool_call() -> None:
    text = get_default_tool_instruction_registry().get("forecast_tool").body

    assert "Call the registered `forecast_tool` directly with structured arguments" in text
    assert '"question"' in text
    assert '"horizon"' in text
    assert "forecast.forecast_result" not in text
    assert "follow each bound JSON Schema" in execution_agent_prompt
    assert "active capability catalog" in execution_agent_prompt
    assert "Chronos MCP" not in execution_agent_prompt


def test_pandas_tool_doc_is_local_execution_contract_not_workflow() -> None:
    document = get_default_tool_instruction_registry().get("pandas_tool")
    text = document.body.lower()
    details = document.details_markdown.lower()

    assert "### data flow" not in text
    assert "one dataframe transformation" in text
    assert "preceding successful tool observation" in text
    assert "top-level action" in text
    assert "already available as `pd`" in text
    assert "isolated staging namespace" in details
    assert "validated `tool_result.items`" in details
    assert "variables persist between tool calls" not in details
    assert "summary_rows" not in details


def test_pandas_inspection_contract_is_prompted_as_table_artifact() -> None:
    tool_text = get_default_tool_instruction_registry().get("pandas_tool").body.lower()
    assert "inspection" in tool_text
    assert "diagnostics" in tool_text
    assert "compact table artifact" in tool_text
    assert "required artifact" in execution_agent_prompt.lower()
