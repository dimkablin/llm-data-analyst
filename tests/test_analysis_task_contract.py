from __future__ import annotations

import pandas as pd
from langchain_core.messages import ToolMessage

from backend.agent.callbacks import ToolCollector
from backend.agent.contracts import (
    AnalysisTaskContract,
    AnalysisTaskContractDetector,
    RequiredOutput,
    validate_task_contract,
)
from backend.agent.models import AgentResponse
from backend.artifacts.execution import ExecArtifactType, ExecutionArtifact
from backend.tools.impl.pandas_tool import PandasTool
from backend.tools.sandbox import SessionSandbox


def test_task_contract_infers_generic_outputs_from_multistep_prompt() -> None:
    contract = AnalysisTaskContract.from_prompt(
        "Найди таблицы в DuckDB, определи схемы, сопоставь ключи, "
        "сначала агрегируй суммы, сделай join, построй график и дай выводы."
    )

    assert [item.kind for item in contract.required_outputs] == [
        "catalog",
        "table_schema",
        "aggregation",
        "joined_table",
        "plot",
        "brief",
    ]


def test_task_contract_uses_detector_for_prompt_inference() -> None:
    detector = AnalysisTaskContractDetector()

    contract = detector.detect("show tables and chart total sales")

    assert [item.kind for item in contract.required_outputs] == [
        "catalog",
        "aggregation",
        "plot",
    ]


def test_task_contract_validation_requires_plot_artifact() -> None:
    contract = AnalysisTaskContract(
        required_outputs=[
            RequiredOutput(kind="joined_table", reason="user requested join"),
            RequiredOutput(kind="plot", reason="user requested chart"),
        ]
    )
    response = AgentResponse(
        final_text="Суть: join выполнен. Ключевые цифры: 10.",
        reasoning=None,
        route="analysis",
        artifacts=[
            ExecutionArtifact(
                artifact_type=ExecArtifactType.DATAFRAME,
                producer_tool="sql_tool",
                name="joined_variance",
                data=pd.DataFrame({"segment": ["a"], "delta": [10]}),
                meta={"lineage": {"source_table_names": ["actuals", "plan"]}},
            )
        ],
        tool_names=["sql_tool"],
    )

    result = validate_task_contract(contract, response)

    assert result.passed is False
    assert result.missing_requirements == ["plot"]


def test_task_contract_validation_accepts_joined_table_lineage_and_plot() -> None:
    contract = AnalysisTaskContract(
        required_outputs=[
            RequiredOutput(kind="joined_table", reason="user requested join"),
            RequiredOutput(kind="plot", reason="user requested chart"),
        ]
    )
    response = AgentResponse(
        final_text="Суть: join выполнен. Ключевые цифры: 10.",
        reasoning=None,
        route="analysis",
        artifacts=[
            ExecutionArtifact(
                artifact_type=ExecArtifactType.DATAFRAME,
                producer_tool="sql_tool",
                name="joined_variance",
                data=pd.DataFrame({"segment": ["a"], "delta": [10]}),
                meta={"lineage": {"source_table_names": ["actuals", "plan"]}},
            ),
            ExecutionArtifact(
                artifact_type=ExecArtifactType.PLOT,
                producer_tool="plotly_tool",
                name="variance_chart",
            ),
        ],
        tool_names=["sql_tool", "plotly_tool"],
    )

    result = validate_task_contract(contract, response)

    assert result.passed is True
    assert result.missing_requirements == []


def test_task_contract_validation_accepts_pandas_merge_lineage_inferred_from_scope() -> None:
    contract = AnalysisTaskContract(
        required_outputs=[RequiredOutput(kind="joined_table", reason="user requested join")]
    )
    sandbox = SessionSandbox()
    sandbox.put("fact", pd.DataFrame({"project_id": [1], "fact": [100]}))
    sandbox.put("plan", pd.DataFrame({"project_id": [1], "plan": [80]}))
    tool = PandasTool(
        pd.DataFrame(),
        sandbox=sandbox,
        tool_cache_size=0,
    )

    text, payload = tool._run(
        """
joined = fact.merge(plan, on="project_id", how="inner")
tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"joined_plan_fact": joined},
}
tool_result
"""
    )
    collector = ToolCollector()
    collector.absorb_tool_message(
        "pandas_tool",
        ToolMessage(content=text, artifact=payload, tool_call_id="call-1"),
    )
    response = AgentResponse(
        final_text="Суть: join выполнен.",
        reasoning=None,
        route="analysis",
        artifacts=collector.artifacts,
        tool_names=["pandas_tool"],
    )

    result = validate_task_contract(contract, response)

    assert result.passed is True
    assert result.missing_requirements == []
    assert collector.artifacts[0].meta["lineage"]["source_table_names"] == ["fact", "plan"]
