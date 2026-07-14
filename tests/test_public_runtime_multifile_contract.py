from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
from langchain_core.messages import AIMessage

from backend.agent.callbacks import ToolCollector
from backend.agent.graph.nodes import agent as agent_node_module
from backend.agent.runner import AgentRunner
from backend.agent.tool_loop import ToolLoopRequest
from backend.artifacts.execution import artifact_type_label, is_tabular_artifact_type
from backend.core.config import Settings
from backend.data_access.csv_session_runtime import CSVSessionRuntime
from backend.data_access.tabular_upload_service import TabularUploadFile, TabularUploadService
from backend.notebook.manifest_store import ManifestStore
from backend.notebook.orchestrator import NotebookOrchestrator
from backend.notebook.store import NotebookStore
from backend.sessions.session_store import SessionStore
from backend.tools.impl.planner_tool import PlannerTool


class _FakeToolCallingLLM:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.invoke_count = 0

    def bind_tools(self, _tools: list[Any]) -> _FakeToolCallingLLM:
        return self

    def invoke(self, _messages: list[Any], config: dict[str, Any] | None = None) -> Any:
        self.invoke_count += 1
        if self._responses:
            return self._responses.pop(0)
        return _final_response("Суть: анализ завершен. Ключевой вывод: отклонения рассчитаны.")


def _tool_response(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def _final_response(content: str) -> AIMessage:
    return AIMessage(content=content)


def _upload_service(tmp_path: Path) -> tuple[TabularUploadService, SessionStore, CSVSessionRuntime, str]:
    store = SessionStore(str(tmp_path / "legacy"), ttl_days=7)
    session = store.create_session()
    csv_runtime = CSVSessionRuntime(default_ttl_sec=3600)
    service = TabularUploadService(
        store=store,
        csv_runtime=csv_runtime,
        manifest_store=ManifestStore(tmp_path),
        notebook_orchestrator=NotebookOrchestrator(NotebookStore(tmp_path)),
        storage_dir=tmp_path,
    )
    return service, store, csv_runtime, session.session_id


def test_public_runtime_multifile_join_aggregation_plot_contract(tmp_path: Path) -> None:
    service, store, csv_runtime, session_id = _upload_service(tmp_path)
    try:
        service.ingest_files(
            session_id=session_id,
            files=[
                TabularUploadFile(
                    file_name="actuals.csv",
                    content=(
                        b"account_code,department,actual_amount\n"
                        b"A-100,Sales,120\n"
                        b"A-100,Sales,80\n"
                        b"B-200,Marketing,50\n"
                        b"C-300,Support,40\n"
                    ),
                ),
                TabularUploadFile(
                    file_name="plan.csv",
                    content=(
                        b"acct_id,department,plan_amount\n"
                        b"A-100,Sales,150\n"
                        b"B-200,Marketing,90\n"
                        b"C-300,Support,70\n"
                    ),
                ),
            ],
            ttl_seconds=3600,
        )
        state = store.load_session(session_id)
        assert state is not None

        joined_sql = """
            WITH actuals_agg AS (
                SELECT account_code, department, SUM(actual_amount) AS actual_amount
                FROM actuals
                GROUP BY account_code, department
            ),
            plan_agg AS (
                SELECT acct_id, department, SUM(plan_amount) AS plan_amount
                FROM plan
                GROUP BY acct_id, department
            )
            SELECT
                COALESCE(a.account_code, p.acct_id) AS account_id,
                COALESCE(a.department, p.department) AS department,
                COALESCE(a.actual_amount, 0) AS actual_amount,
                COALESCE(p.plan_amount, 0) AS plan_amount,
                COALESCE(a.actual_amount, 0) - COALESCE(p.plan_amount, 0) AS variance_amount
            FROM actuals_agg AS a
            FULL OUTER JOIN plan_agg AS p
                ON a.account_code = p.acct_id
               AND a.department = p.department
            ORDER BY ABS(variance_amount) DESC
        """
        plot_code = """
fig = px.bar(
    joined_variance,
    x="department",
    y="variance_amount",
    color="department",
    title="Plan vs actual variance by department",
)
tool_result = chart.result(fig, artifact_name="variance_chart")
tool_result
"""
        responses = [
            _tool_response("sql_tool", {"mode": "catalog_tables", "artifact_name": "csv_tables"}, "call-1"),
            _tool_response(
                "sql_tool",
                {
                    "mode": "describe_table",
                    "table_names": ["actuals", "plan"],
                    "artifact_name": "table_schema",
                },
                "call-2",
            ),
            _tool_response(
                "sql_tool",
                {
                    "mode": "execute_sql",
                    "sql": joined_sql,
                    "artifact_name": "joined_variance",
                },
                "call-3",
            ),
            _tool_response("plotly_tool", {"code": plot_code}, "call-4"),
            _final_response(
                "Суть: факт и план сопоставлены по account_id/department. "
                "Ключевой вывод: Sales выше плана на 50, Marketing ниже плана на 40, Support ниже плана на 30."
            ),
        ]
        runner = AgentRunner(
            settings=replace(
                Settings(),
                storage_dir=str(tmp_path),
                agent_analysis_depth="medium",
                agent_cache_enabled=False,
                llm_warmup_enabled=False,
            ),
        )
        prompt = (
            "Найди таблицы в DuckDB, определи схемы, сделай join actuals.account_code = plan.acct_id, "
            "сначала агрегируй суммы, посчитай отклонение факт-план, построй график top отклонений "
            "и дай краткие выводы."
        )
        session_source = {
            "source_type": "csv",
            "source_label": state.dataset_name,
            "csv_loaded": True,
            "csv_session_id": session_id,
            "csv_table_names": list(state.csv_table_names or []),
        }
        callbacks = [ToolCollector(source_context=session_source)]

        fake_llm = _FakeToolCallingLLM(responses)
        loop_max_iterations: list[int] = []
        original_direct_tool_loop = agent_node_module.direct_tool_loop

        def _recording_direct_tool_loop(request: ToolLoopRequest) -> Any:
            loop_max_iterations.append(int(request.max_iterations))
            return original_direct_tool_loop(request)

        with (
            patch("backend.agent.tool_loop.build_runtime_llm", return_value=fake_llm),
            patch.object(PlannerTool, "_run", return_value="1. Catalog. 2. Schema. 3. Join. 4. Plot."),
            patch.object(
                agent_node_module,
                "direct_tool_loop",
                side_effect=_recording_direct_tool_loop,
            ),
        ):
            response = runner.run_query(
                None,
                prompt,
                history=[],
                use_history=False,
                include_reasoning=False,
                callbacks=callbacks,
                trace_context={"session_id": session_id},
                session_source=session_source,
            )

        tabular_artifacts = [
            artifact
            for artifact in response.artifacts
            if is_tabular_artifact_type(getattr(artifact, "artifact_type", ""))
        ]
        plot_artifacts = [
            artifact
            for artifact in response.artifacts
            if artifact_type_label(getattr(artifact, "artifact_type", "")) == "plot"
        ]
        artifact_names = [str(getattr(artifact, "name", "")) for artifact in response.artifacts]
        assert loop_max_iterations and loop_max_iterations[0] > 1, loop_max_iterations
        assert not (response.reasoning or "").startswith("Agent step failed"), response.reasoning
        assert "joined_variance" in artifact_names, {
            "artifact_names": artifact_names,
            "tool_names": response.tool_names,
            "final_text": response.final_text,
            "reasoning": response.reasoning,
            "invoke_count": fake_llm.invoke_count,
            "loop_max_iterations": loop_max_iterations,
            "events": callbacks[0].events,
        }
        joined = next(artifact for artifact in tabular_artifacts if artifact.name == "joined_variance")
        joined_frame = joined.data

        assert response.route == "analysis"
        assert "не удалось завершить анализ" not in response.final_text.lower()
        assert {"sql_tool", "plotly_tool"} <= set(response.tool_names)
        assert any((artifact.meta or {}).get("catalog_listing") for artifact in tabular_artifacts)
        assert any((artifact.meta or {}).get("schema_description") for artifact in tabular_artifacts)
        assert len(plot_artifacts) >= 1
        assert isinstance(joined_frame, pd.DataFrame)
        assert set(joined.meta["lineage"]["source_table_names"]) == {"actuals", "plan"}
        assert joined_frame["variance_amount"].tolist() == [50.0, -40.0, -30.0]
    finally:
        csv_runtime.delete_session(session_id)
