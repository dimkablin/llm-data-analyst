from __future__ import annotations

import importlib
import os
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
LIVE_LLM_TESTS_ENABLED = os.getenv("LIVE_LLM_TESTS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

pytestmark = [
    pytest.mark.live,
    pytest.mark.e2e,
    pytest.mark.skipif(
        not LIVE_LLM_TESTS_ENABLED,
        reason="Set LIVE_LLM_TESTS=1 to run live LLM dataset workflow tests.",
    ),
]


def _build_live_runner():
    load_dotenv(ROOT / ".env", override=False)

    import backend.core.config as config_module

    config_module = importlib.reload(config_module)
    from backend.agent import AgentRunner, ToolCollector

    settings = replace(
        config_module.Settings(),
        agent_cache_enabled=False,
        llm_warmup_enabled=False,
        agent_evaluate_enabled=False,
        backend_query_timeout_sec=120,
        agent_step_timeout_sec=60,
        agent_inner_recursion_limit=12,
    )
    runner = AgentRunner(settings)
    return runner, ToolCollector


def _require_live_llm(response) -> None:
    if getattr(response, "llm_unreachable", False):
        pytest.skip(
            "Live LLM endpoint returned unavailable/timeout during the test run."
        )


def test_live_llm_uses_tools_on_real_csv_dataset() -> None:
    runner, ToolCollector = _build_live_runner()
    df = pd.read_csv(ROOT / "examples" / "titanic" / "dataset.csv")
    tool_collector = ToolCollector(source_context={"source_type": "csv"})

    response = runner.run_query(
        df,
        (
            "Используя текущий датасет, посчитай количество строк и назови первые 3 столбца. "
            "Обязательно используй инструмент и верни хотя бы один артефакт."
        ),
        history=[],
        use_history=False,
        include_reasoning=False,
        callbacks=[tool_collector],
        trace_context={"request_kind": "live_test"},
        session_source={"source_type": "csv", "source_label": "titanic.csv"},
    )

    _require_live_llm(response)
    assert response.route == "analysis"
    assert response.tool_calls >= 1
    assert response.artifacts
    assert "pandas_tool" in response.tool_names


def test_live_llm_uses_explicit_skill_with_dataset() -> None:
    runner, ToolCollector = _build_live_runner()
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3],
            "date": [
                "2025-01-02",
                "2025-02-05",
                "2025-01-11",
                "2025-03-03",
                "2025-02-02",
                "2025-03-09",
            ],
            "revenue": [10, 12, 20, 25, 15, 18],
        }
    )
    tool_collector = ToolCollector(source_context={"source_type": "csv"})

    response = runner.run_query(
        df,
        (
            "Сделай когортный анализ по месяцам для этого датасета. "
            "Построй хотя бы одну таблицу с retention/cohort результатом через инструмент."
        ),
        history=[],
        use_history=False,
        include_reasoning=False,
        callbacks=[tool_collector],
        trace_context={"request_kind": "live_test"},
        session_source={"source_type": "csv", "source_label": "cohort.csv"},
        selected_skill_ids=["cohort_analysis"],
    )

    _require_live_llm(response)
    assert response.route == "analysis"
    assert response.tool_calls >= 1
    assert response.artifacts
    assert "pandas_tool" in response.tool_names
