from __future__ import annotations

from backend.skills import SkillRegistry
from backend.tools.catalog import KNOWN_TOOL_KEYS

GENERAL_SKILL_ID = "business_domain_analysis"

REMOVED_PROMPT_SPECIFIC_SKILL_IDS = (
    "investment_opportunity_screening",
    "retail_root_cause_analysis",
    "retail_forecast_anomaly_analysis",
    "demo_report_synthesis",
)

GENERIC_ANALYSIS_TERMS = (
    "schema discovery",
    "metric role",
    "dimension role",
    "time role",
    "entity role",
    "evidence table",
    "forecast_tool",
    "anomaly_planfact_tool",
    "generate_report_tool",
)

FORBIDDEN_PROMPT_HARDCODE_TERMS = (
    "sportmaster",
    "спортмастер",
    "нордресурс",
    "nordresource",
    "nike",
    "adidas",
    "регион центр",
    "обувь",
    "март 2024",
    "нет акции",
    "client energy portfolio",
    "account_type",
    "portfolio_weight_pct",
    "unrealized_pnl_pct",
)


def _registry() -> SkillRegistry:
    return SkillRegistry.from_path("skills").load()


def _skill_text(skill_id: str) -> str:
    skill = _registry().get(skill_id)
    return " ".join(
        (
            skill.name,
            skill.description,
            " ".join(skill.triggers),
            skill.core_markdown,
        )
    ).lower()


def test_single_general_business_domain_skill_loads() -> None:
    skill = _registry().get(GENERAL_SKILL_ID)

    assert skill.name == "Business Domain Analysis"
    assert skill.enabled_by_default is True
    assert skill.kind == "analytical"


def test_prompt_specific_demo_skills_are_not_registered() -> None:
    loaded_ids = {skill.skill_id for skill in _registry().list_skills()}

    assert set(REMOVED_PROMPT_SPECIFIC_SKILL_IDS).isdisjoint(loaded_ids)


def test_general_business_domain_skill_has_generic_tool_contract() -> None:
    contract = _registry().get(GENERAL_SKILL_ID).execution_contract

    assert contract.required_tools == (
        "data_catalog_tool",
        "sql_tool",
        "pandas_tool",
        "plotly_tool",
    )
    assert contract.required_artifacts
    assert contract.evidence_rules
    assert set(contract.required_tools).issubset(KNOWN_TOOL_KEYS)


def test_general_business_domain_skill_covers_methods_not_demo_prompts() -> None:
    text = _skill_text(GENERAL_SKILL_ID)

    for term in GENERIC_ANALYSIS_TERMS:
        assert term.lower() in text


def test_business_domain_skills_do_not_contain_prompt_hardcode() -> None:
    skill_ids = (
        GENERAL_SKILL_ID,
        "retail_sales_analysis",
        "portfolio_risk_analysis",
        "promo_effectiveness",
    )

    for skill_id in skill_ids:
        text = _skill_text(skill_id)
        for term in FORBIDDEN_PROMPT_HARDCODE_TERMS:
            assert term not in text, f"{skill_id} contains prompt hardcode {term!r}"
