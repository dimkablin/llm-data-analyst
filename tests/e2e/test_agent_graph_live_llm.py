from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

pytestmark = pytest.mark.e2e


def test_live_langgraph_chat_roundtrip(invoke_live_agent: Any) -> None:
    response, collector, _runner = invoke_live_agent(
        "hello. Reply with one short sentence.",
    )

    assert response.route == "chat"
    assert response.final_text.strip()
    assert response.tool_calls == 0
    assert collector.tool_calls == 0


def test_live_langgraph_uses_dataframe_tool_and_artifact(invoke_live_agent: Any) -> None:
    df = pd.DataFrame(
        {
            "region": ["EU", "US", "EU", "APAC", "US", "EU"],
            "revenue": [100, 200, 150, 300, 120, 250],
            "orders": [2, 4, 3, 5, 2, 4],
        },
    )

    response, collector, runner = invoke_live_agent(
        (
            "Use the current dataframe. You must call `value_tool` or `pandas_tool`; "
            "do not answer from memory. Count the rows and calculate the total revenue. "
            "In the final answer, explicitly include both numbers."
        ),
        df=df,
        session_source={"source_type": "csv", "source_label": "sales_e2e.csv"},
    )

    produced_text = _combined_response_text(response)
    assert response.route == "analysis"
    assert response.tool_calls >= 1
    assert collector.tool_calls >= 1
    assert {"value_tool", "pandas_tool"}.intersection(response.tool_names)
    assert response.artifacts
    assert "6" in produced_text
    assert "1120" in produced_text

    session_memory = runner.session_memory
    if session_memory is not None:
        assert session_memory.turn_count >= 1


def _combined_response_text(response: Any) -> str:
    parts = [str(getattr(response, "final_text", "") or "")]
    for artifact in getattr(response, "artifacts", []) or []:
        parts.append(str(getattr(artifact, "name", "") or ""))
        parts.append(str(getattr(artifact, "data", "") or ""))
    return "\n".join(parts)
