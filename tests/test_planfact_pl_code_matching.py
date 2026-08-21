from __future__ import annotations

import pandas as pd

from backend.data_access.planfact_source_service import PlanfactSourceService


def _service() -> PlanfactSourceService:
    return object.__new__(PlanfactSourceService)


def test_pl_mapping_prefers_cost_article_and_falls_back_to_nomenclature() -> None:
    service = _service()
    fact = pd.DataFrame(
        {
            "Дата": ["2026-03-01", "2026-03-02"],
            "ЦФО": ["A", "A"],
            "Статьи затрат": ["22030600 | Связь", ""],
            "Номенклатурная группа": ["11030700 | Другое", "Связь (22030600)"],
            "Сумма": [10, 20],
        }
    )
    plan_cfg = {"monthly_metric": "PL"}
    fact_cfg = {"date_column": "Дата", "cfo_column": "ЦФО", "article_column": "Статья ДДС", "amount_column": "Сумма"}

    service._apply_pl_article_key_defaults(plan_cfg, fact_cfg, fact)
    _, by_article = service._build_fact_monthly(fact, fact_cfg)

    assert fact_cfg["article_key_columns"] == ["Статьи затрат", "Номенклатурная группа"]
    assert by_article[["article_key", "fact_amount"]].to_dict("records") == [
        {"article_key": "22030600", "fact_amount": 30},
    ]


def test_pl_plan_uses_code_from_article() -> None:
    service = _service()
    plan = pd.DataFrame({"ЦФО": ["A"], "Статья PL": ["Связь (22030600)"], "PL Mar": [100]})

    result = service._build_plan_long(
        plan,
        {
            "cfo_column": "ЦФО",
            "article_column": "Статья PL",
            "monthly_metric": "PL",
            "article_key_mode": "pl_code",
            "monthly_columns": {"2026-03": "PL Mar"},
        },
    )

    assert result.loc[0, "article_key"] == "22030600"


def test_pl_mapping_keeps_existing_article_when_code_columns_are_absent() -> None:
    service = _service()
    fact_cfg = {"article_column": "Статья начисления"}

    service._apply_pl_article_key_defaults(
        {"monthly_metric": "PL"},
        fact_cfg,
        pd.DataFrame({"Статья начисления": ["Аренда"]}),
    )

    assert fact_cfg == {"article_column": "Статья начисления"}


def test_article_code_extraction_rejects_dates_and_kpp() -> None:
    assert PlanfactSourceService.extract_article_code("22030600 | Услуги связи") == "22030600"
    assert PlanfactSourceService.extract_article_code("Услуги связи (22030600)") == "22030600"
    assert PlanfactSourceService.extract_article_code("PL22030600") == "22030600"
    assert PlanfactSourceService.extract_article_code("Услуга (05010101.4)") == "05010101.4"
    assert PlanfactSourceService.extract_article_code("Договор от 01.12.2015") == ""
    assert PlanfactSourceService.extract_article_code("БАНК ВТБ (ПАО) (КПП 784201001)") == ""


def test_pl_code_mode_does_not_fuzzy_match_different_codes() -> None:
    service = _service()

    match = service._match_fact_article(
        fact_key="220301011",
        fact_article="Аренда (220301011)",
        candidates=[{"article_key": "22030101", "article": "22030101 | Аренда"}],
        manual_map={},
        allow_fuzzy=False,
    )

    assert match == {"match_type": "unmatched", "confidence": 0.0}


def test_code_mapping_uses_last_valid_pair_for_duplicate_fact_code() -> None:
    service = _service()
    mapping = pd.DataFrame(
        {
            "Код 1С": [5010101.1, "35010400", "35010400", "BN-000019"],
            "Line Code PL": [5010100.0, "35010100", "35010400", "35010200"],
        }
    )

    assert service._detect_article_mapping(mapping) == [
        {"fact_article_key": "05010101.1", "plan_article_key": "05010100"},
        {"fact_article_key": "35010400", "plan_article_key": "35010400"},
    ]
