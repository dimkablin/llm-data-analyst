import pandas as pd

from backend.agent.dataset_profiles import (
    build_dataset_profile_block,
    build_sql_generation_hints,
    infer_column_roles,
)


def test_universal_playbook_for_generic_csv() -> None:
    df = pd.DataFrame(
        {
            "order_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "region": ["North", "South"],
            "revenue": [100.0, 200.0],
        }
    )
    block = build_dataset_profile_block(
        df,
        dataset_name="orders.csv",
        session_source={
            "source_type": "csv",
            "csv_loaded": True,
            "csv_table_names": ["orders"],
        },
    )

    assert "general_analytics" in block
    assert "order_date" in block
    assert "GROUP BY" in block


def test_universal_playbook_for_db_session() -> None:
    block = build_dataset_profile_block(
        None,
        session_source={"source_type": "db_connection"},
        db_name="analytics_pg",
        db_type="postgresql",
    )

    assert "analytics_pg" in block
    assert "database_tool" in block


def test_dataset_profile_does_not_add_portfolio_or_risk_playbook() -> None:
    df = pd.DataFrame(
        {
            "ticker": ["BRIZ"],
            "portfolio_weight_pct": [12.0],
            "market_value_mln_rub": [5.0],
            "risk_profile": ["high"],
            "security_type": ["stock"],
            "account_type": ["brokerage"],
            "unrealized_pnl_pct": [3.5],
        }
    )
    block = build_dataset_profile_block(
        df,
        dataset_name="client_energy_portfolio_lots.csv",
    )

    forbidden_hardcode = {
        "structure_by_*",
        "concentration_*",
        "risk_classification_chart",
        "top_positions_by_*",
        "pnl_distribution_chart",
    }
    assert forbidden_hardcode.isdisjoint(block)


def test_dataset_profile_does_not_add_retail_sales_playbook() -> None:
    df = pd.DataFrame(
        columns=[
            "month",
            "channel",
            "category",
            "brand",
            "revenue",
            "sales_volume",
            "promo_activity",
        ]
    )
    block = build_dataset_profile_block(df, dataset_name="sportmaster.csv")

    assert "promo_effectiveness" not in block


def test_dataset_profile_does_not_add_demo_invest_db_playbook() -> None:
    block = build_dataset_profile_block(
        None,
        session_source={"source_type": "db_connection"},
        db_schema="demo_invest",
    )
    hints = build_sql_generation_hints(
        ["instrument_snapshot_demo", "price_history_demo", "news_impact_demo"],
        db_schema="demo_invest",
    )

    forbidden_hardcode = {
        "instrument_snapshot_demo",
        "price_history_demo",
        "news_impact_demo",
        "NREH",
        "company_name ILIKE",
    }
    assert forbidden_hardcode.isdisjoint(block)
    assert forbidden_hardcode.isdisjoint(hints)


def test_infer_column_roles() -> None:
    df = pd.DataFrame(
        {
            "month": ["2024-01", "2024-02"],
            "channel": ["online", "offline"],
            "revenue": [10.0, 20.0],
            "sku_id": [1, 2],
        }
    )
    roles = infer_column_roles(df)

    assert "month" in roles.time
    assert "revenue" in roles.metrics
    assert "channel" in roles.dimensions


def test_sql_hints_always_present() -> None:
    hints = build_sql_generation_hints(["product", "revenue"])

    assert "GROUP BY" in hints
