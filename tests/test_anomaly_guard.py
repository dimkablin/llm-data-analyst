from backend.agent.anomaly_guard import check_numeric_consistency
from backend.agent.services.message_builder import (
    ExecutionSystemPromptRequest,
    build_execution_system_prompt,
)
from backend.core.config import Settings
from backend.skills.registry import SkillRegistry


def _table(value: float, *, column: str = "Маржа") -> dict:
    return {
        "id": "category_metrics",
        "type": "table",
        "text": "Метрики по категориям",
        "data": {
            "format": "split",
            "data": {
                "columns": ["Категория", column],
                "index": [0],
                "data": [["Молоко", value]],
            },
        },
    }


def test_matches_rounded_millions_and_keeps_artifact_location() -> None:
    report = check_numeric_consistency(
        "Маржа категории составила 150 млн ₽.",
        [_table(150_100_000)],
    )

    assert report["status"] == "passed"
    [item] = report["items"]
    assert item["normalized_value"] == 150_000_000
    assert item["sources"][0] == {
        "value": 150_100_000.0,
        "percent": False,
        "artifact_id": "category_metrics",
        "artifact_title": "Метрики по категориям",
        "row": "Молоко",
        "column": "Маржа",
        "raw_value": 150_100_000,
        "difference_percent": 0.066667,
    }


def test_ignores_list_numbers_dates_years_and_identifiers() -> None:
    report = check_numeric_consistency(
        "1) На 2026-08-07 проверен SKU 12345. Заказов: 42 шт.",
        [_table(42, column="Заказы, шт.")],
    )

    assert [item["text"] for item in report["items"]] == ["42 шт."]
    assert report["status"] == "passed"


def test_ignores_markdown_emphasized_request_numbers() -> None:
    report = check_numeric_consistency(
        "Период: март **2026**. Проверить заявки № **17** и № **19**.",
        [],
    )

    assert report["items"] == []


def test_percent_can_match_ratio_and_unmatched_value_warns() -> None:
    matched = check_numeric_consistency("Rate жалоб — 12,5%.", [_table(0.125, column="Rate жалоб")])
    warning = check_numeric_consistency("Rate жалоб — 18%.", [_table(0.125, column="Rate жалоб")])

    assert matched["status"] == "passed"
    assert warning["status"] == "warning"
    assert warning["items"][0]["sources"] == []


def test_enabled_guard_adds_unambiguous_number_format_to_prompt() -> None:
    settings = Settings(anomaly_check_enabled=True)
    request = ExecutionSystemPromptRequest.model_construct(
        settings=settings,
        skill_registry=SkillRegistry.from_path(settings.skills_dir),
        capability_context={},
    )
    disabled = request.model_copy(update={"settings": Settings(anomaly_check_enabled=False)})

    prompt = build_execution_system_prompt(request)

    assert "NUMERIC CONSISTENCY FORMAT" in prompt
    assert "ISO `YYYY-MM-DD`" in prompt
    assert "NUMERIC CONSISTENCY FORMAT" not in build_execution_system_prompt(disabled)


def test_plot_series_is_a_linkable_numeric_source() -> None:
    artifact = {
        "id": "margin_chart",
        "type": "plot",
        "text": "Маржа по категориям",
        "data": {
            "format": "plotly-json",
            "data": {
                "data": [{"name": "Маржа", "x": ["Молоко"], "y": [150_100_000]}],
            },
        },
    }

    report = check_numeric_consistency("Маржа — 150 млн ₽.", [artifact])

    assert report["status"] == "passed"
    assert report["items"][0]["sources"][0]["artifact_id"] == "margin_chart"


def test_identifiers_match_request_and_numbers_inside_text_cells() -> None:
    artifact = {
        "id": "cfo_services_detail",
        "type": "table",
        "text": "Детализация услуг связи",
        "data": {
            "format": "split",
            "data": {
                "columns": ["cfo", "period", "article", "fact_counterparty"],
                "index": [0],
                "data": [[
                    "Департамент процессинга",
                    "2026-03",
                    "11030700 | Услуги связи",
                    "Т2 МОБАЙЛ ООО (КПП 775101001); Ростелеком КПП631543001",
                ]],
            },
        },
    }
    answer = "Департамент 11030700. Контрагенты: КПП 775101001 и КПП 631543001."

    report = check_numeric_consistency(
        answer,
        [artifact],
        "Дай детализацию по Департамент процессинга 11030700 | Услуги связи",
    )

    assert report["status"] == "passed"
    by_text = {item["text"]: item for item in report["items"]}
    assert {source["artifact_title"] for source in by_text["11030700"]["sources"]} == {
        "Детализация услуг связи",
        "Текст запроса",
    }
    assert by_text["775101001"]["sources"][0]["column"] == "fact_counterparty"
    assert by_text["631543001"]["sources"][0]["column"] == "fact_counterparty"
