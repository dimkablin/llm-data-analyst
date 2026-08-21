from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pandas as pd

from backend.agent.models import AgentResponse
from backend.agent.runtime_contracts import AgentRunRequest, AgentRunResult
from backend.api.models import QueryRequest
from backend.api.services.query_execution import QueryExecutionDependencies, QueryExecutionService
from backend.auth.auth_db import AuthDB, AuthUser, UserSettings
from backend.core.config import Settings
from backend.data_access.csv_session_runtime import CSVSessionRuntime
from backend.data_access.planfact_source_service import PlanfactSourceService
from backend.data_access.session_source_service import SessionSourceService
from backend.data_access.source_inventory import build_source_inventory
from backend.data_access.sql_table_service import SQLTableService
from backend.notebook.manifest_store import ManifestStore
from backend.notebook.orchestrator import NotebookOrchestrator
from backend.notebook.store import NotebookStore
from backend.sessions.session_store import SessionStore
from backend.skills.registry import SkillRegistry
from backend.tools.impl.sql_tool import SQLTool


class _AuthDB(AuthDB):
    def __init__(self) -> None:
        return None

    def is_session_owner(self, _session_id: str, _user_id: int) -> bool:
        return True

    def touch_session(self, _session_id: str) -> None:
        return None

    def list_user_tool_settings(self, _user_id: int) -> dict[str, bool]:
        return {}

    def list_user_skill_settings(self, _user_id: int) -> dict[str, bool]:
        return {}

    def get_user_settings(self, _user_id: int) -> UserSettings:
        return UserSettings(
            theme="light",
            default_include_reasoning=False,
            default_answer_style="concise",
            analysis_mode="fast",
            analysis_depth="medium",
            llm_temperature_chat=0.1,
            llm_temperature_tool=0.1,
            llm_max_tokens_default=2400,
            llm_max_tokens_reasoning=3200,
            backend_query_timeout_sec=30,
            agent_max_steps=9,
            agent_step_timeout_sec=30,
            agent_inner_recursion_limit=10,
            llm_streaming=False,
            show_thinking=False,
        )


class _SkillRegistry(SkillRegistry):
    def __init__(self, _path: Path) -> None:
        return None

    def list_skills(self):
        return []

    def resolve_selection(self, _skill_ids):
        return []


class _Runner:
    instances: ClassVar[list[_Runner]] = []

    def __init__(self, _settings: Settings, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.run_request: AgentRunRequest | None = None
        self.instances.append(self)

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        self.run_request = request
        return AgentRunResult(response=AgentResponse(final_text="ok", artifacts=[]))


def _user() -> AuthUser:
    return AuthUser(
        id=7,
        username="user",
        is_admin=False,
        created_at="2026-01-01T00:00:00+00:00",
    )


def _confirm_planfact(
    *,
    tmp_path: Path,
    store: SessionStore,
    csv_runtime: CSVSessionRuntime,
    manifest_store: ManifestStore,
    session_id: str,
) -> None:
    service = PlanfactSourceService(
        store=store,
        csv_runtime=csv_runtime,
        manifest_store=manifest_store,
        storage_dir=tmp_path,
    )
    plan = pd.DataFrame(
        {
            "CFO": ["A", "B"],
            "Article": ["Rent", "Ops"],
            "CF Mar": [100, 70],
        }
    )
    fact = pd.DataFrame(
        {
            "DocDate": ["2026-03-05", "2026-03-21"],
            "CFO Doc": ["A", "C"],
            "FactArticle": ["Rent", "Ops"],
            "Amount": [130, 90],
        }
    )
    service._write_pending_file(session_id, "plan", "plan.xlsx", b"plan")
    service._write_pending_file(session_id, "fact", "fact.xlsx", b"fact")
    service._read_dataframe = lambda file_name, _content: plan if file_name == "plan.xlsx" else fact
    service.confirm(
        session_id=session_id,
        config={
            "source_type": "planfact",
            "plan": {
                "cfo_column": "CFO",
                "article_column": "Article",
                "monthly_metric": "CF",
                "monthly_columns": {"2026-03": "CF Mar"},
            },
            "fact": {
                "date_column": "DocDate",
                "cfo_column": "CFO Doc",
                "article_column": "FactArticle",
                "amount_column": "Amount",
            },
        },
        ttl_seconds=3600,
    )


def _query_service(
    *,
    tmp_path: Path,
    store: SessionStore,
    csv_runtime: CSVSessionRuntime,
    manifest_store: ManifestStore,
) -> QueryExecutionService:
    return QueryExecutionService(
        dependencies=QueryExecutionDependencies(
            auth_db=_AuthDB(),
            store=store,
            skill_registry=_SkillRegistry(tmp_path),
            db_runtime_service=None,
            forecast_integration_service=SimpleNamespace(source_descriptor=lambda: {"key": "forecast"}),
            anomaly_planfact_integration_service=SimpleNamespace(source_descriptor=lambda: {"key": "anomaly"}),
            rag_service=SimpleNamespace(source_descriptor=lambda: {"key": "rag"}),
            user_memory_service=SimpleNamespace(load=lambda _user_id: None),
            build_trace_context_fn=lambda **kwargs: kwargs,
            query_trace_context_fn=lambda **_kwargs: SimpleNamespace(__enter__=lambda: None, __exit__=lambda *_: None),
            settings=Settings(backend_query_timeout_sec=30, storage_dir=str(tmp_path)),
            csv_runtime=csv_runtime,
            manifest_store=manifest_store,
            storage_dir=tmp_path,
            llm_text_collector_cls=lambda: object(),
            tool_collector_cls=lambda: object(),
            agent_runner_cls=_Runner,
            effective_enabled_tool_keys_fn=lambda _catalog: {"sql_tool", "data_catalog_tool"},
            build_tool_catalog_fn=lambda **_kwargs: [],
        )
    )


def _sql_count(prepared, csv_runtime: CSVSessionRuntime) -> int:
    tool = SQLTool(
        csv_loaded=bool(prepared.session_source.get("csv_loaded")),
        csv_session_id=prepared.session_source.get("csv_session_id"),
        storage_dir=str(Path(prepared.runtime_settings.storage_dir)),
    )
    tool._service.csv_runtime = csv_runtime
    _text, payload = tool._run(
        mode="execute_sql",
        sql="SELECT COUNT(*) AS rows FROM planfact_by_cfo_period",
        artifact_name="planfact_count",
    )
    df = payload["items"]["planfact_count"]
    return int(df.loc[0, "rows"])


def test_planfact_prepare_runtime_exposes_duckdb_tables_and_restores_after_restart(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path / "sessions_state"), ttl_days=7)
    session = store.create_session()
    csv_runtime = CSVSessionRuntime(base_dir=tmp_path / "duckdb", default_ttl_sec=3600)
    manifest_store = ManifestStore(tmp_path)
    _confirm_planfact(
        tmp_path=tmp_path,
        store=store,
        csv_runtime=csv_runtime,
        manifest_store=manifest_store,
        session_id=session.session_id,
    )
    service = _query_service(
        tmp_path=tmp_path,
        store=store,
        csv_runtime=csv_runtime,
        manifest_store=manifest_store,
    )

    prepared = service._prepare_agent_runtime(
        session_id=session.session_id,
        payload=QueryRequest(query="show planfact"),
        current_user=_user(),
        request_kind="query",
    )

    assert prepared.has_active_source is True
    assert prepared.session_source["csv_loaded"] is True
    assert prepared.session_source["csv_session_id"] == session.session_id
    assert set(prepared.session_source["csv_table_names"]) >= {
        "planfact_plan_raw",
        "planfact_fact_raw",
        "planfact_by_cfo_period",
        "planfact_by_cfo_article_period",
    }
    inventory = build_source_inventory(
        session_id=session.session_id,
        session_source=prepared.session_source,
        manifest_store=manifest_store,
        csv_runtime=csv_runtime,
        db_runtime=None,
    )
    assert "planfact_by_cfo_period" in {table.qualified_name for table in inventory.tables}
    assert {table.source_type for table in inventory.tables} == {"planfact"}
    assert _sql_count(prepared, csv_runtime) == 3

    csv_runtime.delete_session(session.session_id)
    store.clear_csv_runtime_state(session.session_id)

    restored = service._prepare_agent_runtime(
        session_id=session.session_id,
        payload=QueryRequest(query="show planfact again"),
        current_user=_user(),
        request_kind="query",
    )

    assert restored.has_active_source is True
    assert restored.session_source["csv_loaded"] is True
    assert restored.session_source["csv_session_id"] == session.session_id
    assert set(restored.session_source["csv_table_names"]) >= {
        "planfact_plan_raw",
        "planfact_fact_raw",
        "planfact_by_cfo_period",
        "planfact_by_cfo_article_period",
    }
    assert _sql_count(restored, csv_runtime) == 3


def test_remove_planfact_source_clears_duckdb_runtime_files_and_active_state(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path / "sessions_state"), ttl_days=7)
    session = store.create_session()
    csv_runtime = CSVSessionRuntime(base_dir=tmp_path / "duckdb", default_ttl_sec=3600)
    manifest_store = ManifestStore(tmp_path)
    _confirm_planfact(
        tmp_path=tmp_path,
        store=store,
        csv_runtime=csv_runtime,
        manifest_store=manifest_store,
        session_id=session.session_id,
    )
    source = manifest_store.load(session.session_id).source_by_alias("planfact")
    assert source is not None
    table_paths = (source.preprocessing_summary or {})["duckdb_table_paths"]
    parquet_paths = [
        tmp_path / "sessions" / session.session_id / source.parquet_path,
        *[
            tmp_path / "sessions" / session.session_id / rel_path
            for rel_path in table_paths.values()
        ],
    ]
    assert {row["table_name"] for row in csv_runtime.list_tables(session.session_id)} >= {
        "planfact_plan_raw",
        "planfact_fact_raw",
        "planfact_by_cfo_period",
        "planfact_by_cfo_article_period",
    }
    assert all(path.is_file() for path in parquet_paths)

    source_service = SessionSourceService(
        store=store,
        csv_runtime=csv_runtime,
        manifest_store=manifest_store,
        notebook_orchestrator=NotebookOrchestrator(NotebookStore(tmp_path)),
        storage_dir=tmp_path,
    )
    removed = source_service.remove_source(session_id=session.session_id, alias="planfact")

    assert removed.source_type == "planfact"
    assert manifest_store.load(session.session_id).sources == []
    assert csv_runtime.list_tables(session.session_id) == []
    assert all(not path.exists() for path in parquet_paths)
    state = store.load_session(session.session_id)
    assert state is not None
    assert state.csv_loaded is False
    assert state.csv_session_id is None
    assert state.csv_table_names == []
    assert state.source_type is None
    assert state.source_ref_id is None
    assert state.source_label is None
    assert state.source_mode is None

    prepared = _query_service(
        tmp_path=tmp_path,
        store=store,
        csv_runtime=csv_runtime,
        manifest_store=manifest_store,
    )._prepare_agent_runtime(
        session_id=session.session_id,
        payload=QueryRequest(query="show planfact after removal"),
        current_user=_user(),
        request_kind="query",
    )

    assert prepared.has_active_source is False
    assert prepared.session_source["csv_loaded"] is False
    assert prepared.session_source["csv_session_id"] is None
    assert prepared.session_source["csv_table_names"] == []


def test_planfact_sql_candidates_include_manifest_metadata(tmp_path: Path) -> None:
    store = SessionStore(str(tmp_path / "sessions_state"), ttl_days=7)
    session = store.create_session()
    csv_runtime = CSVSessionRuntime(base_dir=tmp_path / "duckdb", default_ttl_sec=3600)
    manifest_store = ManifestStore(tmp_path)
    _confirm_planfact(
        tmp_path=tmp_path,
        store=store,
        csv_runtime=csv_runtime,
        manifest_store=manifest_store,
        session_id=session.session_id,
    )
    sql_service = SQLTableService(
        csv_loaded=True,
        csv_session_id=session.session_id,
        storage_dir=tmp_path,
    )
    sql_service.csv_runtime = csv_runtime

    candidates = {candidate.table_name: candidate for candidate in sql_service.collect_candidates()}
    candidate = candidates["planfact_by_cfo_period"]

    assert candidate.display_name == "План-факт"
    assert candidate.file_name == "plan.xlsx + fact.xlsx"
    assert candidate.source_alias == "planfact"
    assert candidate.source_label != f"CSV session {session.session_id}"
    assert "planfact_by_cfo_period" in candidate.schema_hint
    assert "planfact_config" in candidate.preprocessing_summary
    assert "duckdb_table_paths" in candidate.preprocessing_summary
    descriptor = sql_service._candidate_descriptor(candidate)
    assert "planfact_config" not in descriptor["preprocessing_summary"]
    assert "duckdb_table_paths" in descriptor["preprocessing_summary"]
    assert "article_mapping" not in str(descriptor)
    assert candidate.row_count == 3
    assert candidate.column_count is not None
