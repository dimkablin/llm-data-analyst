from __future__ import annotations

import re
from pathlib import Path

from backend.agent.prompts import execution_agent_prompt


def test_domain_skills_declare_capability_artifact_and_evidence_sections() -> None:
    for skill_id in (
        "portfolio_risk_analysis",
        "investment_market_analysis",
        "retail_sales_analysis",
    ):
        text = Path("skills", skill_id, "SKILL.md").read_text(encoding="utf-8").lower()

        assert "### required capabilities" in text
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


def test_general_analytics_declares_default_tabular_workflow() -> None:
    text = Path("skills/general_analytics/SKILL.md").read_text(encoding="utf-8")

    assert "enabled_by_default: true" in text
    assert "default workflow" in text.lower()
    assert "CSV/XLSX" in text
    assert "DuckDB" in text
    assert "The plan is not an answer" in text
    assert "main agent reasoning" in text


def test_general_analytics_recovers_without_reusing_bad_intermediate_results() -> None:
    text = Path("skills/general_analytics/SKILL.md").read_text(encoding="utf-8")

    assert "Never use preview, head, sample, or limited rows as the analysis dataset" in text
    assert "complete non-overlapping partitions" in text
    assert "add a chart" in text
    assert "materially improves" in text
    assert "Explicit chart bans win" in text
    assert "answer immediately" in text
    assert "Do not repeat an equivalent successful call that returned an empty candidate" in text
    assert "Never combine mutually exclusive scenario rows" in text
    assert "roll-up" in text
    assert "components" in text
    assert "Keep the intervention distinct from the problem and KPI" in text


def test_data_quality_audit_materializes_source_before_pandas() -> None:
    text = Path("skills/data_quality_audit/SKILL.md").read_text(encoding="utf-8")

    assert 'get_tool_instructions("general_analytics")' in text
    assert "Source preparation" in text
    assert text.index("Source preparation") < text.index("DQ report per column")
    assert "`sql_tool`" in text
    assert "artifact variable" in text


def test_execution_prompt_keeps_domain_routing_out_of_generic_table() -> None:
    prompt = execution_agent_prompt.lower()

    assert "semantic capabilities and task outcomes" in prompt
    assert "active capability catalog" in prompt

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


def test_fpk_skill_is_domain_reference_not_demo_script() -> None:
    skill = Path("skills/fpk-management-analysis/SKILL.md").read_text(encoding="utf-8")
    details = Path("skills/fpk-management-analysis/DETAILS.md").read_text(encoding="utf-8")
    combined = f"{skill}\n{details}".casefold()
    metadata = skill.split("---", maxsplit=2)[1].casefold()
    normalized_skill = " ".join(skill.split()).casefold()
    normalized_details = " ".join(details.split())

    assert "enabled_by_default: false" in metadata
    assert "triggers:" not in metadata
    assert "terminal_artifacts" not in combined
    assert re.search(r"(?im)^```(?:sql|python)\b", combined) is None
    assert re.search(r"(?im)^#{2,}\s+\d+[.)]\s+", details) is None

    required_skill_contracts = (
        "рабочий план в основном цикле",
        "`rag_tool`",
        "capability `forecast`",
        "связанный с ней инструмент",
        "Период, зерно, разрезы",
        "семантического слоя",
        "явного сообщения пользователя",
    )
    assert all(contract in skill for contract in required_skill_contracts)
    assert "не смешивать" in normalized_skill
    assert all(token in skill for token in ("`plan`", "`fact`", "`forecast`"))
    assert all(
        f"`demo_fpk.{table}`" in skill for table in ("stat_stats", "stat_csi", "stat_isoo", "stat_manual")
    )
    assert "`шифр` — код темы" in skill
    assert "не соединять `шифр`" in skill

    for table in ("stat_stats", "stat_csi", "stat_isoo", "stat_manual"):
        assert f"## `{table}`" in details
    assert "## Сопоставление филиалов" in details
    assert "Дата без филиала не является достаточным ключом соединения." in normalized_details
    assert "Формулы именованных бизнес-метрик" in details
