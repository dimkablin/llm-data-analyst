"""Tests for session data catalog and code preflight."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from backend.data_access.data_catalog import (
    CatalogProfileOptions,
    build_snapshot_from_dataframe,
    build_snapshot_from_db_helper,
    format_catalog_prompt_block,
    fuzzy_match_column,
    fuzzy_match_identifier,
    snapshot_from_json,
    snapshot_to_json,
)
from backend.data_access.db_connectors import CatalogColumn as DBColumn
from backend.data_access.db_connectors import CatalogTable as DBTable
from backend.tools.code_preflight import (
    preflight_sandbox_code,
)
from backend.tools.impl.db_helpers import DBAnalyticsHelper
from backend.tools.schema_registry import infer_sql_alias_map


def test_build_snapshot_from_dataframe_includes_columns() -> None:
    df = pd.DataFrame(
        {
            "Месяц": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "Выручка": [100.0, 200.0],
            "Канал": ["online", "offline"],
        }
    )
    snap = build_snapshot_from_dataframe(df)
    assert len(snap.tables) == 1
    names = [c.name for c in snap.tables[0].columns]
    assert "Месяц" in names
    assert "Выручка" in names


def test_profile_is_sampled_limited_and_masks_sensitive_values() -> None:
    snap = build_snapshot_from_dataframe(
        pd.DataFrame(
            {
                "amount": [10.0, None, 999.0],
                "email": ["one@example.com", "two@example.com", "three@example.com"],
                "ignored": [1, 2, 3],
            }
        ),
        options=CatalogProfileOptions(
            max_tables=1,
            max_columns_per_table=2,
            sample_rows=2,
            top_values=1,
        ),
    )

    amount, email = snap.tables[0].columns
    assert [amount.name, email.name] == ["amount", "email"]
    assert amount.null_ratio == 0.5
    assert amount.distinct_count == 1
    assert amount.min_value == "10.0"
    assert email.examples == ["<redacted>", "<redacted>"]
    assert email.top_values == ["<redacted>"]


def test_format_catalog_prompt_block_lists_tables() -> None:
    snap = build_snapshot_from_dataframe(
        pd.DataFrame({"a": [1], "b": [2]}),
        qualified_name="sales",
    )
    snap.tables[0].qualified_name = "sales"
    text = format_catalog_prompt_block(snap)
    assert "Каталог данных" in text
    assert "`a`" in text
    assert "sales" in text
    assert "первым 1000 строкам" in text


def test_profile_sample_provenance_survives_json_round_trip() -> None:
    snap = build_snapshot_from_dataframe(
        pd.DataFrame({"amount": [1, 2, 3]}),
        options=CatalogProfileOptions(sample_rows=2),
    )

    restored = snapshot_from_json(snapshot_to_json(snap))

    assert restored is not None
    assert restored.profile_sample_strategy == "first_rows"
    assert restored.profile_sample_limit == 2


def test_format_catalog_prompt_block_is_schema_only() -> None:
    snap = build_snapshot_from_dataframe(
        pd.DataFrame(
            {
                "direct_info": [
                    '{"route":[{"airport":"DME"},{"airport":"AER"}]}',
                    '{"route":[{"airport":"LED"},{"airport":"DME"}]}',
                ],
                "created_at": pd.to_datetime(["2026-03-01", "2026-03-03"]),
                "email": ["one@example.com", "two@example.com"],
            }
        ),
        qualified_name="session.searches",
        options=CatalogProfileOptions(sample_rows=2, top_values=2),
    )

    text = format_catalog_prompt_block(snap)

    assert "direct_info" in text
    assert "DME" not in text
    assert '"route"' not in text
    assert "range=" not in text
    assert "one@example.com" not in text
    assert "<redacted>" not in text
    assert "Эвристика ролей" not in text


def test_fuzzy_match_column_cyrillic() -> None:
    cols = ["Выручка", "Канал"]
    assert fuzzy_match_column("выручка", cols) == "Выручка"


def test_fuzzy_match_identifier_typo_suffix() -> None:
    candidates = ["top_3_with_share", "top_3", "top_categories_by_revenue"]
    assert fuzzy_match_identifier("top_3_with_rounded", candidates) == "top_3_with_share"


def test_fuzzy_match_column_visits_to_traffic() -> None:
    cols = ["total_revenue", "total_traffic", "avg_conversion", "avg_discount"]
    assert fuzzy_match_column("total_visits", cols) == "total_traffic"


def test_fuzzy_match_unrealized_pnl_abs_to_pct() -> None:
    cols = [
        "lot_id",
        "ticker",
        "market_value_mln_rub",
        "unrealized_pnl_pct",
    ]
    assert fuzzy_match_column("unrealized_pnl_abs", cols) == "unrealized_pnl_pct"


def test_preflight_does_not_rewrite_aggregate_column_alias() -> None:
    scope = {
        "traffic_by_channel": pd.DataFrame(
            {
                "channel": ["online"],
                "total_revenue": [10.0],
                "total_traffic": [100],
            }
        )
    }
    code = (
        "summary = traffic_by_channel.groupby('channel').agg("
        "{'total_revenue': 'sum', 'total_visits': 'sum'})\n"
        "tool_result = {'schema_version': '1.0', 'artifact_type': 'table', "
        "'items': {'out': summary.reset_index()}}"
    )
    fixed, err = preflight_sandbox_code(code, scope)
    assert err is None
    assert fixed == code
    assert "total_visits" in fixed


def test_preflight_does_not_rewrite_dataframe_column_alias() -> None:
    scope = {
        "shoe_monthly_metrics": pd.DataFrame(
            {
                "Месяц": ["2024-02", "2024-03"],
                "total_traffic": [100, 80],
                "avg_conversion": [0.1, 0.11],
            }
        )
    }
    code = (
        "delta = shoe_monthly_metrics['total_visits'].diff()\n"
        "tool_result = {'schema_version': '1.0', 'artifact_type': 'value', "
        "'items': {'traffic_delta': float(delta.iloc[-1])}}"
    )
    fixed, err = preflight_sandbox_code(code, scope)
    assert err is None
    assert fixed == code
    assert "total_visits" in fixed


def test_preflight_does_not_replace_df_with_single_artifact_var() -> None:
    artifact = pd.DataFrame({"x": [1, 2]})
    scope = {"monthly_sales": artifact}
    code = "summary = df['x'].sum()\ntool_result = {'schema_version': '1.0', 'artifact_type': 'value', 'items': {'total': summary}}"
    fixed, err = preflight_sandbox_code(code, scope)
    assert err is None
    assert fixed == code
    assert "df[" in fixed


def test_preflight_allows_summary_assigned_in_same_block() -> None:
    scope = {
        "portfolio_structure": pd.DataFrame(
            {
                "security_type": ["bond"],
                "account_type": ["broker"],
                "ticker": ["X"],
                "risk_profile": ["high"],
                "lot_count": [1],
            }
        )
    }
    code = (
        "summary = portfolio_structure.copy()\n"
        "tool_result = {'schema_version': '1.0', 'artifact_type': 'table', "
        "'items': {'structure': summary}}"
    )
    _, err = preflight_sandbox_code(code, scope)
    assert err is None


def test_preflight_allows_make_subplots_for_plotly() -> None:
    from backend.tools.code_preflight import _PLOTLY_SCOPE_NAMES

    scope = {"portfolio_structure": pd.DataFrame({"a": [1, 2]})}
    code = (
        "fig = make_subplots(rows=1, cols=2)\n"
        "tool_result = chart.result(fig, artifact_name='portfolio_charts')"
    )
    _, err = preflight_sandbox_code(
        code,
        scope,
        extra_allowed=_PLOTLY_SCOPE_NAMES,
    )
    assert err is None


def test_preflight_allows_lambda_row_and_loop_vars() -> None:
    scope = {"portfolio_structure": pd.DataFrame({"ticker": ["A"], "w": [1.0]})}
    code = (
        "risk = portfolio_structure.groupby('ticker')['w'].max()\n"
        "rows = portfolio_structure.apply(lambda row: row['w'], axis=1)\n"
        "flags = []\n"
        "for flag, val in risk.items():\n"
        "    flags.append((flag, val))\n"
        "tool_result = {'schema_version': '1.0', 'artifact_type': 'table', "
        "'items': {'out': rows}}"
    )
    _, err = preflight_sandbox_code(code, scope)
    assert err is None


def test_preflight_allows_local_helper_function_used_by_apply() -> None:
    scope = {
        "fact_march_raw": pd.DataFrame(
            {
                "dds_article": ["CF02080000 rent", "CF03010000 support"],
                "amount": [100.0, 50.0],
            }
        )
    }
    code = (
        "def extract_cf_key(value):\n"
        "    text = str(value)\n"
        "    return text.split()[0] if text else None\n"
        "\n"
        "fact = fact_march_raw.copy()\n"
        "fact['cf_key'] = fact['dds_article'].apply(extract_cf_key)\n"
        "tool_result = {'schema_version': '1.0', 'artifact_type': 'table', "
        "'items': {'fact_with_cf_key': fact}}\n"
        "tool_result\n"
    )

    _, err = preflight_sandbox_code(code, scope)

    assert err is None


def test_preflight_does_not_apply_schema_registry_alias_before_pandas_execution() -> None:
    scope = {
        "actuals_agg": pd.DataFrame(
            {
                "article_code": ["A-100"],
                "department": ["Sales"],
                "actual_amount": [100.0],
            }
        )
    }
    code = (
        "summary = actuals_agg.groupby('Article Code')['Amount'].sum().reset_index()\n"
        "tool_result = {'schema_version': '1.0', 'artifact_type': 'table', "
        "'items': {'summary': summary}}"
    )

    fixed, err = preflight_sandbox_code(code, scope)

    assert err is None
    assert fixed == code
    assert "Article Code" in fixed
    assert "Amount" in fixed


def test_infer_sql_alias_map_tracks_source_columns_to_sql_output_aliases() -> None:
    sql = """
        SELECT
            "Article Code" AS article_code,
            SUM("Amount") AS actual_amount,
            department
        FROM actuals
        GROUP BY "Article Code", department
    """

    aliases = infer_sql_alias_map(sql, ["article_code", "actual_amount", "department"])

    assert aliases == {
        "Article Code": "article_code",
        "Amount": "actual_amount",
    }


def test_preflight_does_not_block_missing_dataframe_column() -> None:
    scope = {
        "actuals_agg": pd.DataFrame(
            {
                "article_code": ["A-100"],
                "department": ["Sales"],
                "actual_amount": [100.0],
            }
        )
    }
    code = (
        "summary = actuals_agg.groupby('Plan Article')['actual_amount'].sum().reset_index()\n"
        "tool_result = {'schema_version': '1.0', 'artifact_type': 'table', "
        "'items': {'summary': summary}}"
    )

    fixed, err = preflight_sandbox_code(code, scope)

    assert err is None
    assert fixed == code
    assert "Plan Article" in fixed


def test_preflight_does_not_validate_columns_on_local_dataframe_copy() -> None:
    scope = {
        "actuals_agg": pd.DataFrame(
            {
                "article_code": ["A-100"],
                "department": ["Sales"],
                "actual_amount": [100.0],
            }
        )
    }
    code = (
        "work = actuals_agg.copy()\n"
        "work['Article Code'] = work['Article Code'].astype(str)\n"
        "tool_result = {'schema_version': '1.0', 'artifact_type': 'table', "
        "'items': {'work': work}}"
    )

    fixed, err = preflight_sandbox_code(code, scope)

    assert err is None
    assert fixed == code
    assert "Article Code" in fixed


def test_preflight_allows_local_groupby_columns_created_in_code() -> None:
    scope = {
        "df_time": pd.DataFrame(
            {
                "Дата документа": ["2026-01-01", "2026-01-01", "2026-01-02"],
                "Сумма": [100.0, 150.0, 90.0],
            }
        )
    }
    code = """
daily_agg = df_time.groupby(df_time["Дата документа"]).agg({"Сумма": ["sum", "count", "mean", "std"]}).reset_index()
daily_agg.columns = ["Дата", "Daily_Sum", "Transaction_Count", "Avg_Transaction", "Std_Transaction"]
daily_agg["Z_Score_Sum"] = (daily_agg["Daily_Sum"] - daily_agg["Daily_Sum"].mean()) / daily_agg["Daily_Sum"].std()
tool_result = {"schema_version": "1.0", "artifact_type": "table", "items": {"daily": daily_agg}}
tool_result
"""
    fixed, err = preflight_sandbox_code(code, scope)

    assert err is None
    assert fixed == code


def test_preflight_allows_list_comp_x() -> None:
    scope = {"portfolio_structure": pd.DataFrame({"a": [1, 2, 3]})}
    code = (
        "top = [x for x in portfolio_structure['a'].tolist() if x > 1]\n"
        "tool_result = {'schema_version': '1.0', 'artifact_type': 'value', "
        "'items': {'n': len(top)}}"
    )
    _, err = preflight_sandbox_code(code, scope)
    assert err is None


def test_preflight_does_not_autofix_near_miss_variable_name() -> None:
    scope = {
        "top_categories_by_revenue": pd.DataFrame({"Категория": ["Обувь"], "total_revenue": [100.0]}),
        "top_3_with_share": pd.DataFrame({"Категория": ["Обувь"], "share": [0.5]}),
    }
    code = (
        "summary = top_3_with_rounded.copy()\n"
        "tool_result = {'schema_version': '1.0', 'artifact_type': 'table', "
        "'items': {'top_3_categories_summary': summary}}"
    )
    fixed, err = preflight_sandbox_code(code, scope)
    assert err is None
    assert fixed == code
    assert "top_3_with_rounded" in fixed


def test_preflight_does_not_block_unknown_names_before_runtime() -> None:
    scope = {"portfolio_structure": pd.DataFrame({"a": [1]})}
    code = (
        "tool_result = {'schema_version': '1.0', 'artifact_type': 'table', 'items': {'structure': summary}}"
    )
    fixed, err = preflight_sandbox_code(code, scope)
    assert err is None
    assert fixed == code


def test_preflight_allows_hasattr_call_not_as_variable() -> None:
    scope = {"monthly_sales": pd.DataFrame({"volume": [1, 2], "channel": ["a", "b"]})}
    code = (
        "x = monthly_sales.copy()\n"
        "flag = hasattr(x, 'volume')\n"
        "tool_result = {'schema_version': '1.0', 'artifact_type': 'table', "
        "'items': {'out': x}}\n"
        "tool_result\n"
    )
    fixed, err = preflight_sandbox_code(code, scope)
    assert err is None
    assert "hasattr" in fixed


def test_preflight_allows_hasattr_control_flow() -> None:
    scope = {"monthly_sales": pd.DataFrame({"volume": [10, 20]})}
    code = (
        "work = monthly_sales.copy()\n"
        "if hasattr(work, 'volume'):\n"
        "    total = work['volume'].sum()\n"
        "tool_result = {'schema_version': '1.0', 'artifact_type': 'value', "
        "'items': {'total': total}}\n"
        "tool_result\n"
    )
    _, err = preflight_sandbox_code(code, scope)
    assert err is None


def test_preflight_preserves_imports_for_runtime_validation() -> None:
    code = "import pandas as pd\nfrom plotly import express as px\nx = 1\n"
    fixed, err = preflight_sandbox_code(code, {})
    assert err is None
    assert fixed == code


def test_pandas_tool_returns_runtime_keyerror_without_schema_hint() -> None:
    from backend.tools.impl.pandas_tool import PandasTool
    from backend.tools.sandbox import SessionSandbox

    df = pd.DataFrame({"actual": [1, 2]})
    sandbox = SessionSandbox()
    sandbox.bind_dataframe(df)
    tool = PandasTool(
        df,
        sandbox=sandbox,
        tool_cache_size=0,
    )

    text, payload = tool._run(
        """
missing = df["missing"]
tool_result = {"schema_version": "1.0", "artifact_type": "table", "items": {"missing": missing.to_frame()}}
tool_result
"""
    )

    assert "KeyError: missing" in text
    assert payload["error_type"] == "KeyError"
    assert payload["missing_symbol"] == "missing"
    assert "Подсказка" not in text
    assert "actual" not in text


def test_pandas_tool_uses_canonical_artifact_envelope_and_actionable_validation() -> None:
    from backend.tools.impl.pandas_tool import PandasTool
    from backend.tools.sandbox import SessionSandbox

    df = pd.DataFrame({"actual": [1, 2]})
    sandbox = SessionSandbox()
    sandbox.bind_dataframe(df)
    tool = PandasTool(df, sandbox=sandbox, tool_cache_size=0)

    text, payload = tool._run(
        """
result = df.copy()
tool_result = {"schema_version": "1.0", "artifact_type": "table", "items": {"result": result}}
tool_result
"""
    )
    assert "failed" not in text.lower()
    assert payload["schema_version"] == "1.0"
    assert payload["artifact_type"] == "table"
    assert list(payload["items"]) == ["result"]

    error_text, error_payload = tool._run(
        """
result = df.copy()
tool_result = {"schema_version": "2.0", "artifact_type": "table", "items": {"result": result}}
tool_result
"""
    )
    assert error_payload["status"] == "error"
    assert "2.0" in error_text
    assert "1.0" in error_payload["contract_hint"]


def test_db_snapshot_preserves_authoritative_column_type() -> None:
    helper = DBAnalyticsHelper(runtime=SimpleNamespace(options={"schema": "public"}, db_type="postgresql"))
    table = DBTable(
        schema="public",
        name="events",
        table_type="table",
        qualified_name="public.events",
    )
    column = DBColumn(
        schema="public",
        table="events",
        name="event_date",
        data_type="date",
        is_nullable=False,
        ordinal_position=1,
    )
    adapter = SimpleNamespace(list_tables_with_columns=lambda _schema: {"events": (table, [column])})

    with (
        patch.object(helper, "_catalog_adapter", return_value=adapter),
        patch.object(
            helper,
            "preview_table",
            return_value=pd.DataFrame({"event_date": [date(2025, 1, 1)]}),
        ),
    ):
        snapshot = build_snapshot_from_db_helper(helper)

    event_date = snapshot.tables[0].columns[0]
    assert event_date.dtype == "date"
    assert "`event_date` (date)" in format_catalog_prompt_block(snapshot)
