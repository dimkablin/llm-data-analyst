from __future__ import annotations

from backend.agent.services.runtime_context import build_chat_data_context


def test_planfact_prompt_context_includes_tables_fields_and_period_rule() -> None:
    context = build_chat_data_context(
        None,
        {
            "source_type": "planfact",
            "source_mode": "duckdb",
            "source_label": "Plan-Fact",
        },
    )

    assert "planfact_by_cfo_period" in context
    assert "planfact_by_cfo_article_period" in context
    assert "service_content" in context
    assert "plan_counterparty" in context
    assert "fact_counterparty" in context
    assert "fact_contract" in context
    assert "SELECT DISTINCT period FROM planfact_fact_monthly" in context
    assert "MAX(period)" not in context
