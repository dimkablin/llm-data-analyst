from __future__ import annotations

import json
from datetime import date

from backend.benchmark.benchmark_chat import _judge_messages, _score_case


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


def test_judge_receives_successful_text_tool_observations() -> None:
    messages = _judge_messages(
        "What does the policy say?",
        {
            "answer_text": "It requires approval.",
            "tool_runs": [
                {
                    "tool_name": "rag_tool",
                    "status": "ok",
                    "input_preview": "approval policy",
                    "output_preview": "The policy requires approval. Source: policy.md",
                }
            ],
        },
        {},
    )

    payload = json.loads(messages[1]["content"].split("\n\n", maxsplit=1)[1])

    assert payload["current_date"] == date.today().isoformat()
    assert "successful current-run observations" in messages[0]["content"]
    assert payload["successful_tool_observations"] == [
        {
            "tool_name": "rag_tool",
            "input": "approval policy",
            "output": "The policy requires approval. Source: policy.md",
        }
    ]
