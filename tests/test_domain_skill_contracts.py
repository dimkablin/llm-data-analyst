from __future__ import annotations

from pathlib import Path

from backend.agent.prompts import execution_agent_prompt
from backend.data_access.sql_table_service import SQLTableService, TableCandidate


def _table(name: str, columns: list[str]) -> TableCandidate:
    return TableCandidate(
        source_kind="db",
        dialect="sqlite",
        table_name=name,
        qualified_name=name,
        schema="main",
        columns=columns,
        source_label="test",
        source_ref_id="test-db",
    )


def test_domain_skills_declare_tool_artifact_and_evidence_sections() -> None:
    for skill_id in (
        "portfolio_risk_analysis",
        "investment_market_analysis",
        "retail_sales_analysis",
    ):
        text = Path("skills", skill_id, "SKILL.md").read_text(encoding="utf-8").lower()

        assert "### required tools" in text
        assert "### required artifacts" in text
        assert "### evidence rules" in text


def test_general_analytics_no_longer_owns_investment_or_portfolio_workflows() -> None:
    text = Path("skills/general_analytics/SKILL.md").read_text(encoding="utf-8").lower()

    moved_domain_tokens = {
        "demo_invest",
        "instrument_snapshot",
        "price_history",
        "news_impact",
        "risk_profile",
        "portfolio_weight_pct",
        "unrealized_pnl_pct",
    }
    assert moved_domain_tokens.isdisjoint(text)


def test_sql_table_service_ranks_by_generic_table_and_column_matches() -> None:
    service = SQLTableService.__new__(SQLTableService)
    candidates = [
        _table("orders", ["customer", "amount", "month"]),
        _table("instrument_snapshot", ["ticker", "sector", "market_value"]),
        _table("price_history", ["ticker", "close_price", "trade_date"]),
        _table("news_impact", ["ticker", "headline", "impact_score"]),
    ]

    ranked = service._rank_candidates(
        "show portfolio risk and finance trend by customer amount",
        candidates,
    )

    assert ranked[0].table_name == "orders"
    assert SQLTableService._score_table_candidate(
        "portfolio risk finance market news",
        _table("instrument_snapshot", ["ticker", "sector"]),
    ) == 0


def test_general_analytics_declares_default_tabular_workflow() -> None:
    text = Path("skills/general_analytics/SKILL.md").read_text(encoding="utf-8")

    assert "enabled_by_default: true" in text
    assert "default workflow" in text.lower()
    assert "CSV/XLSX" in text
    assert "DuckDB" in text


def test_data_quality_audit_materializes_source_before_pandas() -> None:
    text = Path("skills/data_quality_audit/SKILL.md").read_text(encoding="utf-8")

    assert 'get_tool_instructions("general_analytics")' in text
    assert "Source preparation" in text
    assert text.index("Source preparation") < text.index("DQ report per column")
    assert "`sql_tool`" in text
    assert "artifact variable" in text


def test_execution_prompt_keeps_domain_routing_out_of_generic_table() -> None:
    prompt = execution_agent_prompt.lower()

    assert 'get_tool_instructions("<skill_id>")' in prompt

    moved_domain_tokens = {
        "demo_invest",
        "instrument_snapshot_demo",
        "price_history_demo",
        "news_impact_demo",
        "portfolio_risk_analysis",
        "investment_market_analysis",
        "`risk_profile`",
        "`portfolio_weight_pct`",
    }
    assert moved_domain_tokens.isdisjoint(prompt)
