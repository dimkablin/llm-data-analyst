from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse, urlunparse

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from backend.api.deps import get_current_user
from backend.api.models import (
    OpenProjectProjectsResponse,
    OpenProjectSyncRequest,
    OpenProjectSyncResponse,
    SessionBindDBConnectionSourceRequest,
    SessionSourceResponse,
    SessionSourceStateResponse,
    SourceDescriptorResponse,
)
from backend.auth.auth_db import AuthDB, AuthUser
from backend.core.config import settings
from backend.core.json_utils import make_json_safe
from backend.data_access.csv_runtime_state_service import (
    CSVRuntimeStateError,
    CSVRuntimeStateService,
)
from backend.data_access.csv_session_runtime import CSVSessionRuntime
from backend.data_access.session_source_service import (
    SessionSourceError,
    SessionSourceService,
)
from backend.integrations.openproject import OpenProjectSyncOptions
from backend.notebook.cell_builder import build_source_binding_cell
from backend.notebook.manifest_store import ManifestStore
from backend.notebook.orchestrator import CellOp, NotebookEdit, NotebookOrchestrator
from backend.notebook.session_source import (
    SessionSource,
    alias_to_variable_name,
    make_source_alias,
)
from backend.sessions.session_store import SessionState, SessionStore

router = APIRouter(tags=["Источники"])

# Singletons set during app startup
_auth_db: AuthDB = None  # type: ignore
_store: SessionStore = None  # type: ignore
_db_connections_service = None  # type: ignore
_integration_source_descriptors_fn = None  # type: ignore
_csv_runtime: CSVSessionRuntime = None  # type: ignore
_manifest_store: ManifestStore = None  # type: ignore
_orchestrator: NotebookOrchestrator = None  # type: ignore
_db_runtime_service = None  # type: ignore
_openproject_sync_service = None  # type: ignore
_storage_dir: Path | None = None
_semantic_catalog_service = None  # type: ignore


def setup(
    auth_db: AuthDB,
    store: SessionStore,
    db_connections_service,
    integration_source_descriptors_fn,
    csv_runtime: CSVSessionRuntime,
    manifest_store: ManifestStore,
    notebook_orchestrator: NotebookOrchestrator,
    db_runtime_service=None,
    storage_dir: str | Path | None = None,
    openproject_sync_service=None,
    semantic_catalog_service=None,
) -> None:
    global _auth_db, _store, _db_connections_service
    global _integration_source_descriptors_fn, _csv_runtime
    global _manifest_store, _orchestrator, _db_runtime_service, _storage_dir
    global _openproject_sync_service, _semantic_catalog_service
    _auth_db = auth_db
    _store = store
    _db_connections_service = db_connections_service
    _integration_source_descriptors_fn = integration_source_descriptors_fn
    _csv_runtime = csv_runtime
    _manifest_store = manifest_store
    _orchestrator = notebook_orchestrator
    _db_runtime_service = db_runtime_service
    _storage_dir = Path(storage_dir) if storage_dir is not None else Path(settings.storage_dir)
    _openproject_sync_service = openproject_sync_service
    _semantic_catalog_service = semantic_catalog_service


def _load_owned_session(session_id: str, current_user: AuthUser) -> SessionState:
    if not _auth_db.is_session_owner(session_id, current_user.id):
        raise HTTPException(status_code=404, detail="Session not found")
    state = _store.load_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return state


def _session_source_type(state: SessionState) -> str:
    return str(state.source_type or "").strip().lower()


def _ensure_csv_runtime_state(session_id: str, state: SessionState) -> SessionState:
    if _session_source_type(state) != "csv":
        return state
    try:
        refreshed = CSVRuntimeStateService(
            store=_store,
            csv_runtime=_csv_runtime,
            manifest_store=_manifest_store,
            storage_dir=_storage_dir or Path(settings.storage_dir),
        ).ensure_csv_runtime(
            session_id=session_id,
            ttl_seconds=settings.csv_session_ttl_sec,
        )
    except CSVRuntimeStateError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize CSV runtime: {exc}",
        ) from exc
    try:
        from backend.data_access.catalog_refresh import refresh_session_catalog

        refresh_session_catalog(
            _store,
            session_id,
            csv_runtime=_csv_runtime,
        )
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning(
            "Failed to refresh data catalog after CSV runtime init: %s", exc
        )
    return refreshed


def _refresh_semantic_catalog(session_id: str, user_id: int, *, reason: str) -> None:
    if _semantic_catalog_service is None or not settings.semantic_layer_enabled:
        return
    try:
        _semantic_catalog_service.refresh(session_id=session_id, user_id=user_id)
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning(
            "Failed to refresh semantic catalog after %s: %s",
            reason,
            exc,
        )


def _to_session_source_response(state: SessionState) -> SessionSourceStateResponse:
    return SessionSourceStateResponse(
        source_type=state.source_type,
        source_ref_id=state.source_ref_id,
        source_label=state.source_label,
        source_mode=state.source_mode,
    )


# ── Legacy single-source endpoints (backward compat) ────────────────────────


@router.get("/sources", response_model=list[SourceDescriptorResponse])
def list_available_sources(
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> list[SourceDescriptorResponse]:
    _ = current_user
    return [SourceDescriptorResponse(**item) for item in _integration_source_descriptors_fn()]


@router.post(
    "/sessions/{session_id}/source/db-connection",
    response_model=SessionSourceStateResponse,
)
def bind_session_db_connection_source(
    session_id: str,
    payload: SessionBindDBConnectionSourceRequest,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SessionSourceStateResponse:
    _load_owned_session(session_id, current_user)
    connection = _db_connections_service.get_connection(
        current_user.id,
        payload.connection_id,
    )
    _store.bind_db_connection_source(
        session_id,
        connection_id=connection.id,
        label=connection.name,
        source_mode=payload.source_mode,
    )

    # Also register in manifest.
    _add_source_to_manifest(
        session_id,
        source_type="db_connection",
        display_name=connection.name,
        connection_id=connection.id,
        connection_name=connection.name,
    )

    if _db_runtime_service is not None:
        try:
            from backend.data_access.catalog_refresh import refresh_session_catalog

            runtime = _db_runtime_service.get_runtime_config(
                user_id=current_user.id,
                connection_id=connection.id,
            )
            refresh_session_catalog(
                _store,
                session_id,
                csv_runtime=_csv_runtime,
                db_runtime=runtime,
            )
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning(
                "Failed to refresh data catalog after DB bind: %s", exc
            )
    _refresh_semantic_catalog(session_id, current_user.id, reason="DB bind")

    refreshed = _load_owned_session(session_id, current_user)
    return _to_session_source_response(refreshed)


@router.post(
    "/sessions/{session_id}/source/clear",
    response_model=SessionSourceStateResponse,
)
def clear_session_source(
    session_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SessionSourceStateResponse:
    _load_owned_session(session_id, current_user)
    _store.set_source(
        session_id,
        source_type=None,
        source_ref_id=None,
        source_label=None,
        source_mode=None,
    )
    _store.clear_semantic_catalog(session_id)
    refreshed = _load_owned_session(session_id, current_user)
    return _to_session_source_response(refreshed)


@router.post(
    "/sessions/{session_id}/source/csv",
    response_model=SessionSourceStateResponse,
)
def bind_session_csv_source(
    session_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SessionSourceStateResponse:
    state = _load_owned_session(session_id, current_user)
    if not state.df_path or not state.dataset_name:
        raise HTTPException(
            status_code=400,
            detail="No CSV dataset is attached to this session",
        )
    _store.bind_csv_source(session_id, filename=state.dataset_name)
    refreshed = _load_owned_session(session_id, current_user)
    refreshed = _ensure_csv_runtime_state(session_id, refreshed)
    _refresh_semantic_catalog(session_id, current_user.id, reason="CSV bind")
    return _to_session_source_response(refreshed)


def _openproject_table_artifacts(
    session_id: str,
    dataframes: dict[str, pd.DataFrame],
    *,
    schema_name: str,
    synced_at: str,
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for table_name, df in dataframes.items():
        if not isinstance(df, pd.DataFrame):
            continue
        preview = df.head(100).copy()
        artifact_id = f"openproject_{session_id}_{table_name}"
        artifacts.append(
            make_json_safe(
                {
                    "id": artifact_id,
                    "type": "table",
                    "text": f"OpenProject · {table_name}",
                    "role": "ai",
                    "meta": {
                        "producer_tool": "openproject_sync",
                        "source_type": "openproject",
                        "schema": schema_name,
                        "table": table_name,
                        "row_count": len(df),
                        "preview_rows": len(preview),
                        "synced_at": synced_at,
                    },
                    "timestamp": datetime.now(UTC).isoformat(),
                    "data": {
                        "format": "split",
                        "data": preview.where(pd.notna(preview), None).to_dict(orient="split"),
                    },
                }
            )
        )
    return artifacts


def _split_table(df: pd.DataFrame) -> dict[str, Any]:
    preview = df.where(pd.notna(df), None)
    return preview.to_dict(orient="split")


def _ru_table_artifact(
    *,
    session_id: str,
    artifact_id: str,
    title: str,
    df: pd.DataFrame,
    synced_at: str,
) -> dict[str, Any]:
    return make_json_safe(
        {
            "id": f"openproject_report_{session_id}_{artifact_id}",
            "type": "table",
            "text": title,
            "role": "ai",
            "meta": {
                "producer_tool": "openproject_sync",
                "source_type": "openproject",
                "openproject_report": True,
                "report_kind": "table",
                "synced_at": synced_at,
            },
            "timestamp": datetime.now(UTC).isoformat(),
            "data": {"format": "split", "data": _split_table(df)},
        }
    )


def _plot_artifact(
    *,
    session_id: str,
    artifact_id: str,
    title: str,
    traces: list[dict[str, Any]],
    layout: dict[str, Any] | None = None,
    synced_at: str,
) -> dict[str, Any]:
    return make_json_safe(
        {
            "id": f"openproject_report_{session_id}_{artifact_id}",
            "type": "plot",
            "text": title,
            "role": "ai",
            "meta": {
                "producer_tool": "openproject_sync",
                "source_type": "openproject",
                "openproject_report": True,
                "report_kind": "chart",
                "synced_at": synced_at,
            },
            "timestamp": datetime.now(UTC).isoformat(),
            "data": {
                "format": "plotly-json",
                "data": {
                    "data": traces,
                    "layout": {
                        "title": {"text": title},
                        "margin": {"l": 52, "r": 24, "t": 56, "b": 56},
                        "height": 360,
                        **(layout or {}),
                    },
                },
            },
        }
    )


def _series_by(df: pd.DataFrame, group_col: str, value_col: str, *, limit: int = 12) -> pd.Series:
    if df.empty or group_col not in df.columns or value_col not in df.columns:
        return pd.Series(dtype=float)
    grouped = (
        df.assign(**{group_col: df[group_col].fillna("Не указано").astype(str)})
        .groupby(group_col, dropna=False)[value_col]
        .sum()
        .sort_values(ascending=False)
        .head(limit)
    )
    return grouped


def _count_by(df: pd.DataFrame, group_col: str, *, limit: int = 12) -> pd.Series:
    if df.empty or group_col not in df.columns:
        return pd.Series(dtype=int)
    return (
        df.assign(**{group_col: df[group_col].fillna("Не указано").astype(str)})
        .groupby(group_col, dropna=False)
        .size()
        .sort_values(ascending=False)
        .head(limit)
    )


def _openproject_report_artifacts(
    session_id: str,
    dataframes: dict[str, pd.DataFrame],
    *,
    synced_at: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    projects = dataframes.get("projects", pd.DataFrame()).copy()
    work_packages = dataframes.get("work_packages", pd.DataFrame()).copy()
    time_entries = dataframes.get("time_entries", pd.DataFrame()).copy()
    personnel = dataframes.get("personnel", pd.DataFrame()).copy()

    for frame, column in (
        (projects, "spent_hours"),
        (work_packages, "spent_hours"),
        (time_entries, "hours"),
    ):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    status_counts = _count_by(work_packages, "status", limit=20)
    project_hours = _series_by(time_entries, "project", "hours", limit=12)
    user_hours = _series_by(time_entries, "user", "hours", limit=15)
    activity_hours = _series_by(time_entries, "activity", "hours", limit=12)

    project_summary = pd.DataFrame(
        {
            "Проект": projects.get("name", pd.Series(dtype=str)),
            "Идентификатор": projects.get("identifier", pd.Series(dtype=str)),
            "Статус": projects.get("status", pd.Series(dtype=str)),
            "Задач": projects.get("work_packages_count", pd.Series(dtype=int)),
            "План, ч": projects.get("planned_hours", pd.Series(dtype=float)),
            "Списано по задачам, ч": projects.get("spent_hours", pd.Series(dtype=float)),
            "Списано по таймшитам, ч": projects.get("time_entry_hours", pd.Series(dtype=float)),
            "Отклонение, ч": projects.get("variance_hours", pd.Series(dtype=float)),
        }
    ).fillna("")

    tasks_source = work_packages.head(80)
    task_list = pd.DataFrame(
        {
            "Код": tasks_source.get("id", pd.Series(dtype=str)),
            "Тема": tasks_source.get("subject", pd.Series(dtype=str)),
            "Проект": tasks_source.get("project", pd.Series(dtype=str)),
            "Тип": tasks_source.get("type", pd.Series(dtype=str)),
            "Статус": tasks_source.get("status", pd.Series(dtype=str)),
            "Исполнитель": tasks_source.get("assignee", pd.Series(dtype=str)),
            "Приоритет": tasks_source.get("priority", pd.Series(dtype=str)),
            "План, ч": tasks_source.get("planned_hours", pd.Series(dtype=float)),
            "Списано, ч": tasks_source.get("spent_hours", pd.Series(dtype=float)),
            "Готовность, %": tasks_source.get("percentage_done", pd.Series(dtype=float)),
            "Создано": tasks_source.get("created_at", pd.Series(dtype=str)),
        }
    ).fillna("")

    project_total = float(project_hours.sum()) if len(project_hours) else 0.0
    user_total = float(user_hours.sum()) if len(user_hours) else 0.0
    writeoff_by_project = pd.DataFrame(
        {
            "Проект": project_hours.index,
            "Списано, ч": project_hours.values,
            "Доля, %": (
                (project_hours / project_total * 100).round(1).values
                if project_total
                else [0 for _ in range(len(project_hours))]
            ),
        }
    )
    writeoff_by_user = pd.DataFrame(
        {
            "Сотрудник": user_hours.index,
            "Списано, ч": user_hours.values,
            "Доля, %": (
                (user_hours / user_total * 100).round(1).values
                if user_total
                else [0 for _ in range(len(user_hours))]
            ),
        }
    )
    personnel_table = pd.DataFrame(
        {
            "Сотрудник": personnel.get("name", pd.Series(dtype=str)),
            "Логин": personnel.get("login", pd.Series(dtype=str)),
            "Email": personnel.get("email", pd.Series(dtype=str)),
            "Статус": personnel.get("status", pd.Series(dtype=str)),
            "Администратор": personnel.get("admin", pd.Series(dtype=bool)),
        }
    ).fillna("")

    tables = [
        _ru_table_artifact(
            session_id=session_id,
            artifact_id="projects",
            title="Проекты: сводка",
            df=project_summary,
            synced_at=synced_at,
        ),
        _ru_table_artifact(
            session_id=session_id,
            artifact_id="tasks",
            title="Клиентские заявки: список задач",
            df=task_list,
            synced_at=synced_at,
        ),
        _ru_table_artifact(
            session_id=session_id,
            artifact_id="writeoff_projects",
            title="Списания по проектам",
            df=writeoff_by_project,
            synced_at=synced_at,
        ),
        _ru_table_artifact(
            session_id=session_id,
            artifact_id="writeoff_users",
            title="Списания по сотрудникам",
            df=writeoff_by_user,
            synced_at=synced_at,
        ),
        _ru_table_artifact(
            session_id=session_id,
            artifact_id="personnel",
            title="Участники OpenProject",
            df=personnel_table,
            synced_at=synced_at,
        ),
    ]

    charts = [
        _plot_artifact(
            session_id=session_id,
            artifact_id="chart_statuses",
            title="Задачи по статусам",
            traces=[
                {
                    "type": "bar",
                    "x": status_counts.index.tolist(),
                    "y": status_counts.values.tolist(),
                    "name": "Задач",
                }
            ],
            layout={
                "xaxis": {"title": {"text": "Статус"}},
                "yaxis": {"title": {"text": "Количество"}},
            },
            synced_at=synced_at,
        ),
        _plot_artifact(
            session_id=session_id,
            artifact_id="chart_project_hours",
            title="Списания по проектам",
            traces=[
                {
                    "type": "bar",
                    "orientation": "h",
                    "y": project_hours.index.tolist()[::-1],
                    "x": project_hours.values.tolist()[::-1],
                    "name": "Часы",
                }
            ],
            layout={
                "xaxis": {"title": {"text": "Часы"}},
                "yaxis": {"title": {"text": "Проект"}},
            },
            synced_at=synced_at,
        ),
        _plot_artifact(
            session_id=session_id,
            artifact_id="chart_user_hours",
            title="Списания по сотрудникам",
            traces=[
                {
                    "type": "bar",
                    "orientation": "h",
                    "y": user_hours.index.tolist()[::-1],
                    "x": user_hours.values.tolist()[::-1],
                    "name": "Часы",
                }
            ],
            layout={
                "xaxis": {"title": {"text": "Часы"}},
                "yaxis": {"title": {"text": "Сотрудник"}},
            },
            synced_at=synced_at,
        ),
        _plot_artifact(
            session_id=session_id,
            artifact_id="chart_activity_hours",
            title="Типы списаний",
            traces=[
                {
                    "type": "pie",
                    "labels": activity_hours.index.tolist(),
                    "values": activity_hours.values.tolist(),
                    "hole": 0.35,
                }
            ],
            layout={"showlegend": True},
            synced_at=synced_at,
        ),
    ]
    return tables, charts


def _normalize_openproject_url_for_backend(
    base_url: str | None,
    host_header: str | None,
) -> tuple[str | None, str | None]:
    clean_url = str(base_url or "").strip() or None
    clean_host_header = str(host_header or "").strip() or None
    if not clean_url:
        return None, clean_host_header
    parsed = urlparse(clean_url)
    host = (parsed.hostname or "").lower()
    if host not in {"localhost", "127.0.0.1", "::1"}:
        return clean_url, clean_host_header
    userinfo = ""
    if parsed.username:
        userinfo = parsed.username
        if parsed.password:
            userinfo += f":{parsed.password}"
        userinfo += "@"
    port = f":{parsed.port}" if parsed.port else ""
    netloc = f"{userinfo}host.docker.internal{port}"
    return urlunparse(parsed._replace(netloc=netloc)), clean_host_header or (parsed.netloc or None)


def _openproject_options_from_payload(payload: OpenProjectSyncRequest | None) -> OpenProjectSyncOptions:
    base_url, host_header = _normalize_openproject_url_for_backend(
        payload.base_url if payload else None,
        payload.host_header if payload else None,
    )
    return OpenProjectSyncOptions(
        base_url=base_url,
        api_key=(payload.api_key if payload else None),
        host_header=host_header,
        project=(payload.project if payload else None),
        all_projects=bool(payload.all_projects) if payload else False,
        days=(payload.days if payload else None),
        max_items=(payload.max_items if payload else None),
    )


@router.post(
    "/sessions/{session_id}/source/openproject/projects",
    response_model=OpenProjectProjectsResponse,
)
def list_openproject_projects(
    session_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
    payload: OpenProjectSyncRequest | None = None,
) -> OpenProjectProjectsResponse:
    _load_owned_session(session_id, current_user)
    if _openproject_sync_service is None:
        raise HTTPException(status_code=503, detail="OpenProject sync service is not configured")
    try:
        projects = _openproject_sync_service.list_projects(_openproject_options_from_payload(payload))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"OpenProject projects load failed: {exc}",
        ) from exc
    return OpenProjectProjectsResponse(projects=projects)


@router.post(
    "/sessions/{session_id}/source/openproject",
    response_model=OpenProjectSyncResponse,
)
def bind_session_openproject_source(
    session_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
    payload: OpenProjectSyncRequest | None = None,
) -> OpenProjectSyncResponse:
    _load_owned_session(session_id, current_user)
    if _openproject_sync_service is None:
        raise HTTPException(status_code=503, detail="OpenProject sync service is not configured")

    try:
        sync_result = _openproject_sync_service.sync(
            user_id=current_user.id,
            options=_openproject_options_from_payload(payload),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"OpenProject sync failed: {exc}",
        ) from exc

    connection = sync_result.connection
    _store.set_source(
        session_id,
        source_type="openproject",
        source_ref_id=connection.id,
        source_label="OpenProject",
        source_mode="postgres_sync",
    )

    _add_source_to_manifest(
        session_id,
        source_type="openproject",
        display_name="OpenProject",
        connection_id=connection.id,
        connection_name=connection.name,
        schema_hint={table: sync_result.schema for table in sync_result.rows_by_table},
    )

    if _db_runtime_service is not None:
        try:
            from backend.data_access.catalog_refresh import refresh_session_catalog

            runtime = _db_runtime_service.get_runtime_config(
                user_id=current_user.id,
                connection_id=connection.id,
            )
            refresh_session_catalog(
                _store,
                session_id,
                csv_runtime=_csv_runtime,
                db_runtime=runtime,
            )
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning(
                "Failed to refresh data catalog after OpenProject sync: %s", exc
            )
    _refresh_semantic_catalog(session_id, current_user.id, reason="OpenProject sync")

    table_artifacts, chart_artifacts = _openproject_report_artifacts(
        session_id,
        sync_result.dataframes,
        synced_at=sync_result.synced_at,
    )
    report_artifacts = [*table_artifacts, *chart_artifacts]
    _store.add_serialized_artifacts(session_id, report_artifacts)
    _store.add_chat_message(
        session_id,
        "assistant",
        "Отчет OpenProject обновлен. Ниже доступны графики по текущей выгрузке.",
        artifacts=chart_artifacts,
    )
    artifact_ids = [str(item["id"]) for item in report_artifacts]

    return OpenProjectSyncResponse(
        source_ref_id=connection.id,
        source_label="OpenProject",
        connection_id=connection.id,
        connection_name=connection.name,
        schema_name=sync_result.schema,
        tables=dict(sync_result.rows_by_table),
        artifact_ids=artifact_ids,
        synced_at=sync_result.synced_at,
    )


# ── Multi-source endpoints ──────────────────────────────────────────────────


@router.get(
    "/sessions/{session_id}/sources",
    response_model=list[SessionSourceResponse],
)
def list_session_sources(
    session_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> list[SessionSourceResponse]:
    """List all sources bound to a session."""
    _load_owned_session(session_id, current_user)
    manifest = _manifest_store.load(session_id)
    return [
        SessionSourceResponse(
            alias=s.alias,
            source_type=s.source_type,
            display_name=s.display_name,
            variable_name=s.variable_name,
            file_name=s.file_name,
            connection_id=s.connection_id,
            connection_name=s.connection_name,
            bound_at=s.bound_at,
            csv_table_names=list(s.csv_table_names or []),
            schema_hint=s.schema_hint,
        )
        for s in manifest.sources
    ]


@router.delete(
    "/sessions/{session_id}/sources/{alias}",
    response_model=list[SessionSourceResponse],
)
def remove_session_source(
    session_id: str,
    alias: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> list[SessionSourceResponse]:
    """Remove a specific source from a session by alias."""
    state = _load_owned_session(session_id, current_user)
    db_runtime = None
    if (
        _db_runtime_service is not None
        and _session_source_type(state) == "db_connection"
        and state.source_ref_id
    ):
        try:
            db_runtime = _db_runtime_service.get_runtime_config(
                user_id=current_user.id,
                connection_id=state.source_ref_id,
            )
        except Exception:
            db_runtime = None

    service = SessionSourceService(
        store=_store,
        csv_runtime=_csv_runtime,
        manifest_store=_manifest_store,
        notebook_orchestrator=_orchestrator,
        storage_dir=_storage_dir or Path(settings.storage_dir),
        db_runtime=db_runtime,
    )
    try:
        service.remove_source(session_id=session_id, alias=alias)
    except SessionSourceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    _store.clear_semantic_catalog(session_id)

    return list_session_sources(session_id, current_user)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _add_source_to_manifest(
    session_id: str,
    *,
    source_type: str,
    display_name: str,
    file_name: str | None = None,
    parquet_path: str | None = None,
    connection_id: str | None = None,
    connection_name: str | None = None,
    csv_session_id: str | None = None,
    csv_table_names: list[str] | None = None,
    csv_expires_at: int | None = None,
    schema_hint: dict[str, str] | None = None,
) -> SessionSource:
    """Add a source to the manifest and create a source_binding cell."""
    manifest = _manifest_store.load(session_id)

    existing_aliases = [s.alias for s in manifest.sources]
    alias = make_source_alias(display_name, source_type, existing_aliases)
    var_name = alias_to_variable_name(alias)

    source = SessionSource(
        alias=alias,
        source_type=source_type,
        display_name=display_name,
        variable_name=var_name,
        file_name=file_name,
        parquet_path=parquet_path,
        connection_id=connection_id,
        connection_name=connection_name,
        csv_session_id=csv_session_id,
        csv_table_names=csv_table_names or [],
        csv_expires_at=csv_expires_at,
        schema_hint=schema_hint or {},
    )
    manifest.add_source(source)
    _manifest_store.save(session_id, manifest)

    # Create source_binding notebook cell.
    if source_type == "csv":
        load_code = f'{var_name} = pd.read_parquet("{parquet_path or "data.parquet"}")'
    elif source_type == "openproject":
        load_code = f'{var_name} = _restore_db_connection("{alias}")  # OpenProject PostgreSQL sync'
    else:
        load_code = f'{var_name} = _restore_db_connection("{alias}")'

    cell = build_source_binding_cell(
        alias=alias,
        variable_name=var_name,
        source_type=source_type,
        display_name=display_name,
        load_code=load_code,
    )
    _orchestrator.apply(session_id, NotebookEdit(op=CellOp.INSERT, cell=cell))

    return source
