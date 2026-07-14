from __future__ import annotations

import pandas as pd

from backend.agent.callbacks import ToolCollector
from backend.agent.dependencies import AgentRuntimeDependencies
from backend.agent.graph.nodes.finalize import finalize_node
from backend.agent.models import AgentResponse
from backend.agent.services.finalization import fallback_text
from backend.agent.tool_loop import artifact_summary_text
from backend.artifacts.execution import ExecArtifactType, ExecutionArtifact


def test_finalize_preserves_agent_answer_without_text_rewrite() -> None:
    artifact = ExecutionArtifact(
        artifact_type=ExecArtifactType.PLOT,
        producer_tool="plotly_tool",
        name="chart",
        data=None,
    )
    text = (
        "**4. Графики и артефакты**\n"
        "- Keep this chart wording exactly: 123.456789.\n\n"
        "**5. Next**\n"
        "- Done"
    )
    response = AgentResponse(
        final_text=text,
        reasoning="agent reasoning",
        artifacts=[artifact],
        route="analysis",
        tool_calls=1,
        tool_names=["plotly_tool"],
    )

    result = finalize_node(
        {
            "response": response,
            "prompt": "what changed?",
            "callbacks": [],
            "trace_context": {},
            "step_index": 1,
            "max_steps": 1,
        },
        object(),
    )

    finalized = result["response"]
    assert finalized.final_text == text
    assert finalized.reasoning == "agent reasoning"


def test_finalize_runtime_dependencies_do_not_include_review_tool() -> None:
    assert "review_tool" not in AgentRuntimeDependencies.model_fields


def test_finalize_fills_empty_final_text_from_artifacts() -> None:
    artifact = ExecutionArtifact(
        artifact_type=ExecArtifactType.DATAFRAME,
        producer_tool="sql_tool",
        name="result_table",
        data=pd.DataFrame({"value": [10]}),
    )
    response = AgentResponse(
        final_text="",
        reasoning=None,
        artifacts=[artifact],
        route="analysis",
        tool_calls=1,
        tool_names=["sql_tool"],
    )

    result = finalize_node(
        {
            "response": response,
            "prompt": "Analyze values",
            "callbacks": [],
            "trace_context": {},
            "step_index": 1,
            "max_steps": 1,
        },
        object(),
    )

    assert result["response"].final_text == artifact_summary_text([artifact])


def test_finalize_creates_fallback_response_when_response_missing() -> None:
    result = finalize_node(
        {
            "response": None,
            "prompt": "Analyze values",
            "callbacks": [],
            "trace_context": {},
            "step_index": 1,
            "max_steps": 1,
        },
        object(),
    )

    response = result["response"]
    assert response.final_text == fallback_text("Analyze values")
    assert response.reasoning == "No response produced by graph."
    assert response.artifacts == []


def test_finalize_attaches_collected_artifacts_only_when_response_has_none() -> None:
    artifact = ExecutionArtifact(
        artifact_type=ExecArtifactType.DATAFRAME,
        producer_tool="sql_tool",
        name="collector_table",
        data=pd.DataFrame({"value": [10]}),
    )
    collector = ToolCollector()
    collector.artifacts.append(artifact)
    response = AgentResponse(
        final_text="Answer is ready.",
        reasoning=None,
        artifacts=[],
        route="analysis",
        tool_calls=1,
        tool_names=["sql_tool"],
    )

    result = finalize_node(
        {
            "response": response,
            "prompt": "Analyze values",
            "callbacks": [collector],
            "trace_context": {},
            "step_index": 1,
            "max_steps": 1,
        },
        object(),
    )

    assert result["response"].artifacts == [artifact]
