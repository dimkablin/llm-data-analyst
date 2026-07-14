from __future__ import annotations

from backend.benchmark.benchmark_chat import _score_case


def test_score_penalizes_disabled_required_tool_answer() -> None:
    result = {
        "ok": True,
        "answer_text": "Не могу выполнить запрос: необходимый tool `forecast_tool` выключен.",
        "fallback_used": False,
        "tool_errors": 0,
        "plot_count": 0,
        "has_valid_plot": False,
        "artifact_count": 0,
        "tool_calls": 0,
        "duration_ms": 1000,
    }

    score, issues, _latency, _bucket = _score_case("Спрогнозируй продажи на 3 месяца", result)

    assert "required_tool_unavailable" in issues
    assert score <= 55


def test_score_penalizes_no_tool_evidence_for_artifact_question() -> None:
    result = {
        "ok": True,
        "answer_text": "Готово, построил таблицу.",
        "fallback_used": False,
        "tool_errors": 0,
        "plot_count": 0,
        "has_valid_plot": False,
        "artifact_count": 4,
        "tool_calls": 0,
        "duration_ms": 1000,
    }

    score, issues, _latency, _bucket = _score_case(
        "Построй таблицу продаж по месяцам",
        result,
    )

    assert "no_current_turn_evidence" in issues
    assert score <= 70
