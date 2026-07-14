"""Tests for session data catalog and code preflight."""

from __future__ import annotations

import pandas as pd

from backend.data_access.data_catalog import (
    CatalogProfileOptions,
    build_snapshot_from_dataframe,
    format_catalog_prompt_block,
    fuzzy_match_column,
    fuzzy_match_identifier,
)
from backend.tools.code_preflight import (
    preflight_sandbox_code,
)
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
        "top_categories_by_revenue": pd.DataFrame(
            {"Категория": ["Обувь"], "total_revenue": [100.0]}
        ),
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
        "tool_result = {'schema_version': '1.0', 'artifact_type': 'table', "
        "'items': {'structure': summary}}"
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

    text, _payload = tool._run(
        """
missing = df["missing"]
tool_result = {"schema_version": "1.0", "artifact_type": "table", "items": {"missing": missing.to_frame()}}
tool_result
"""
    )

    assert "KeyError: missing" in text
    assert "Подсказка" not in text
    assert "actual" not in text
