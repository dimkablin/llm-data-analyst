from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
from typing import Any, Iterable

import httpx
import pandas as pd

from backend.auth.auth_db import DBConnectionRecord
from backend.core.config import Settings
from backend.data_access.db_connections_service import DBConnectionsService

_ISO_TIME_DURATION = re.compile(
    r"^P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?$",
    re.IGNORECASE,
)

OPENPROJECT_TABLES = ("projects", "work_packages", "time_entries", "personnel")


@dataclass(frozen=True)
class OpenProjectSyncResult:
    connection: DBConnectionRecord
    schema: str
    rows_by_table: dict[str, int]
    dataframes: dict[str, pd.DataFrame]
    synced_at: str


@dataclass(frozen=True)
class OpenProjectSyncOptions:
    base_url: str | None = None
    api_key: str | None = None
    host_header: str | None = None
    project: str | None = None
    all_projects: bool = False
    days: int | None = None
    max_items: int | None = None


class OpenProjectIntegrationError(RuntimeError):
    pass


def _parse_hours_to_seconds(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(float(value) * 3600)
    text = str(value).strip()
    if not text:
        return 0
    if text.upper().startswith("P"):
        match = _ISO_TIME_DURATION.match(text)
        if not match:
            return 0
        days = int(match.group("days") or 0)
        hours = int(match.group("hours") or 0)
        minutes = int(match.group("minutes") or 0)
        seconds = float(match.group("seconds") or 0)
        return int((days * 24 + hours) * 3600 + minutes * 60 + seconds)
    return int(float(text) * 3600)


def _hours_float(value: Any) -> float:
    return round(_parse_hours_to_seconds(value) / 3600.0, 4)


def _href_tail(href: str | None) -> str:
    return href.rstrip("/").split("/")[-1] if href else ""


def _link_info(links: dict[str, Any], key: str) -> tuple[str, str, str]:
    block = links.get(key) or {}
    if isinstance(block, list):
        return "", "", ""
    href = str(block.get("href") or "")
    ident = _href_tail(href)
    title = str(block.get("title") or ident)
    return ident, title, href


def _formattable_raw(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("raw") or "")
    return "" if value is None else str(value)


def _date_range_filter(field: str, days: int) -> dict[str, Any]:
    since = (date.today() - timedelta(days=days)).isoformat()
    until = date.today().isoformat()
    return {field: {"operator": "<>d", "values": [since, until]}}


class OpenProjectAPI:
    def __init__(self, *, base_url: str, api_key: str, host_header: str | None = None) -> None:
        if not base_url:
            raise OpenProjectIntegrationError("OPENPROJECT_BASE_URL is not configured.")
        if not api_key:
            raise OpenProjectIntegrationError("OPENPROJECT_API_KEY is not configured.")
        self.base_url = base_url.rstrip("/")
        self.api = f"{self.base_url}/api/v3"
        headers = {"Accept": "application/json"}
        if host_header:
            headers["Host"] = host_header
        self._client = httpx.Client(
            auth=("apikey", api_key),
            headers=headers,
            timeout=60.0,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OpenProjectAPI":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = path if path.startswith("http") else f"{self.api}/{path.lstrip('/')}"
        response = self._client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise OpenProjectIntegrationError("OpenProject returned an unexpected response.")
        return data

    def iter_collection(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        page_size: int = 100,
        max_items: int | None = None,
    ) -> Iterable[dict[str, Any]]:
        query = dict(params or {})
        query.setdefault("pageSize", page_size)
        query.setdefault("offset", 1)
        yielded = 0
        while True:
            data = self.get_json(path, query)
            elements = data.get("_embedded", {}).get("elements", [])
            if not elements:
                break
            for element in elements:
                if isinstance(element, dict):
                    yield element
                    yielded += 1
                if max_items is not None and yielded >= max_items:
                    return
            total = int(data.get("total") or 0)
            count = int(data.get("count") or len(elements))
            if yielded >= total or count == 0:
                break
            query["offset"] = int(query.get("offset") or 1) + 1

    def resolve_project_id(self, project: str) -> str | None:
        if project.isdigit():
            return project
        for item in self.iter_collection("projects", page_size=100):
            if item.get("identifier") == project or item.get("name") == project:
                return str(item.get("id"))
        return None


def _project_to_row(project: dict[str, Any]) -> dict[str, Any]:
    links = project.get("_links", {})
    parent_id, parent_name, _ = _link_info(links, "parent")
    status_id, status_name, _ = _link_info(links, "status")
    return {
        "id": project.get("id"),
        "identifier": project.get("identifier"),
        "name": project.get("name"),
        "active": project.get("active"),
        "public": project.get("public"),
        "parent_id": parent_id,
        "parent": parent_name,
        "status_id": status_id,
        "status": status_name,
        "description": _formattable_raw(project.get("description")),
        "status_explanation": _formattable_raw(project.get("statusExplanation")),
        "created_at": project.get("createdAt"),
        "updated_at": project.get("updatedAt"),
    }


def _work_package_to_row(work_package: dict[str, Any]) -> dict[str, Any]:
    links = work_package.get("_links", {})
    project_id, project_name, _ = _link_info(links, "project")
    type_id, type_name, _ = _link_info(links, "type")
    status_id, status_name, _ = _link_info(links, "status")
    assignee_id, assignee_name, _ = _link_info(links, "assignee")
    author_id, author_name, _ = _link_info(links, "author")
    responsible_id, responsible_name, _ = _link_info(links, "responsible")
    priority_id, priority_name, _ = _link_info(links, "priority")
    parent_id, parent_name, _ = _link_info(links, "parent")
    version_id, version_name, _ = _link_info(links, "version")
    estimated_time = work_package.get("estimatedTime")
    remaining_time = work_package.get("remainingTime") or work_package.get("derivedRemainingTime")
    spent_time = work_package.get("spentTime")
    return {
        "id": work_package.get("id"),
        "subject": work_package.get("subject"),
        "project_id": project_id,
        "project": project_name,
        "type_id": type_id,
        "type": type_name,
        "status_id": status_id,
        "status": status_name,
        "assignee_id": assignee_id,
        "assignee": assignee_name,
        "author_id": author_id,
        "author": author_name,
        "responsible_id": responsible_id,
        "responsible": responsible_name,
        "priority_id": priority_id,
        "priority": priority_name,
        "parent_id": parent_id,
        "parent": parent_name,
        "version_id": version_id,
        "version": version_name,
        "start_date": work_package.get("startDate") or work_package.get("derivedStartDate"),
        "due_date": work_package.get("dueDate") or work_package.get("derivedDueDate"),
        "date": work_package.get("date"),
        "duration": work_package.get("duration"),
        "estimated_time": estimated_time,
        "planned_hours": _hours_float(estimated_time),
        "estimated_hours": _hours_float(estimated_time),
        "spent_time": spent_time,
        "spent_hours": _hours_float(spent_time),
        "remaining_time": remaining_time,
        "remaining_hours": _hours_float(remaining_time),
        "variance_hours": round(_hours_float(spent_time) - _hours_float(estimated_time), 4),
        "percentage_done": work_package.get("percentageDone") or work_package.get("derivedPercentageDone"),
        "description": _formattable_raw(work_package.get("description")),
        "created_at": work_package.get("createdAt"),
        "updated_at": work_package.get("updatedAt"),
    }


def _time_entry_to_row(time_entry: dict[str, Any]) -> dict[str, Any]:
    links = time_entry.get("_links", {})
    project_id, project_name, _ = _link_info(links, "project")
    wp_id, wp_name, _ = _link_info(links, "entity")
    if not wp_id:
        wp_id, wp_name, _ = _link_info(links, "workPackage")
    user_id, user_name, _ = _link_info(links, "user")
    activity_id, activity_name, _ = _link_info(links, "activity")
    hours = time_entry.get("hours")
    return {
        "id": time_entry.get("id"),
        "spent_on": time_entry.get("spentOn"),
        "hours": _hours_float(hours),
        "hours_raw": hours,
        "project_id": project_id,
        "project": project_name,
        "work_package_id": wp_id,
        "work_package": wp_name,
        "user_id": user_id,
        "user": user_name,
        "activity_id": activity_id,
        "activity": activity_name,
        "comment": _formattable_raw(time_entry.get("comment")),
        "created_at": time_entry.get("createdAt"),
        "updated_at": time_entry.get("updatedAt"),
    }


def _user_to_row(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_id": user.get("id"),
        "login": user.get("login"),
        "name": user.get("name"),
        "first_name": user.get("firstName"),
        "last_name": user.get("lastName"),
        "email": user.get("email"),
        "status": user.get("status"),
        "admin": user.get("admin"),
        "created_at": user.get("createdAt"),
        "updated_at": user.get("updatedAt"),
    }


_TABLE_SPECS = {
    "projects": {"endpoint": "projects", "row": _project_to_row, "params": {}},
    "work_packages": {"endpoint": "work_packages", "row": _work_package_to_row, "params": {"filters": "[]"}},
    "time_entries": {"endpoint": "time_entries", "row": _time_entry_to_row, "params": {}},
    "personnel": {"endpoint": "users", "row": _user_to_row, "params": {"filters": "[]"}},
}


def _project_filters(project_id: str | None) -> list[dict[str, Any]]:
    if not project_id:
        return []
    return [{"project": {"operator": "=", "values": [project_id]}}]


def _load_rows(
    api: OpenProjectAPI,
    table: str,
    *,
    project_id: str | None,
    days: int | None,
    max_items: int | None,
) -> list[dict[str, Any]]:
    spec = _TABLE_SPECS[table]
    params = dict(spec["params"])
    filters: list[dict[str, Any]] = []
    if table not in {"projects", "personnel"}:
        filters.extend(_project_filters(project_id))
    if table == "time_entries" and days:
        filters.append(_date_range_filter("spentOn", days))
    if filters:
        params["filters"] = json.dumps(filters)
    rows = [
        spec["row"](item)
        for item in api.iter_collection(
            spec["endpoint"],
            params=params,
            page_size=100,
            max_items=max_items,
        )
    ]
    if table == "projects" and project_id:
        rows = [row for row in rows if str(row.get("id")) == str(project_id)]
    return rows


def _augment_projects(
    project_rows: list[dict[str, Any]],
    work_package_rows: list[dict[str, Any]],
    time_entry_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    planned_by_project: dict[str, float] = {}
    spent_by_project_from_wp: dict[str, float] = {}
    tasks_by_project: dict[str, int] = {}
    spent_by_project_from_te: dict[str, float] = {}
    entries_by_project: dict[str, int] = {}
    for row in work_package_rows:
        project_id = str(row.get("project_id") or "")
        if not project_id:
            continue
        planned_by_project[project_id] = planned_by_project.get(project_id, 0.0) + float(row.get("planned_hours") or 0.0)
        spent_by_project_from_wp[project_id] = spent_by_project_from_wp.get(project_id, 0.0) + float(row.get("spent_hours") or 0.0)
        tasks_by_project[project_id] = tasks_by_project.get(project_id, 0) + 1
    for row in time_entry_rows:
        project_id = str(row.get("project_id") or "")
        if not project_id:
            continue
        spent_by_project_from_te[project_id] = spent_by_project_from_te.get(project_id, 0.0) + float(row.get("hours") or 0.0)
        entries_by_project[project_id] = entries_by_project.get(project_id, 0) + 1
    for row in project_rows:
        project_id = str(row.get("id") or "")
        planned = round(planned_by_project.get(project_id, 0.0), 4)
        spent = round(spent_by_project_from_wp.get(project_id, 0.0), 4)
        spent_in_period = round(spent_by_project_from_te.get(project_id, 0.0), 4)
        row["task_count"] = tasks_by_project.get(project_id, 0)
        row["planned_hours"] = planned
        row["spent_hours"] = spent
        row["spent_hours_in_period"] = spent_in_period
        row["time_entry_count_in_period"] = entries_by_project.get(project_id, 0)
        row["variance_hours"] = round(spent - planned, 4)
        row["spent_ratio_percent"] = round(spent / planned * 100, 2) if planned else 0.0
    return project_rows


def _augment_personnel(
    user_rows: list[dict[str, Any]],
    work_package_rows: list[dict[str, Any]],
    time_entry_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    planned_by_user: dict[str, float] = {}
    assigned_tasks_by_user: dict[str, int] = {}
    spent_by_user: dict[str, float] = {}
    entries_by_user: dict[str, int] = {}
    for row in work_package_rows:
        user_id = str(row.get("assignee_id") or "")
        if not user_id:
            continue
        planned_by_user[user_id] = planned_by_user.get(user_id, 0.0) + float(row.get("planned_hours") or 0.0)
        assigned_tasks_by_user[user_id] = assigned_tasks_by_user.get(user_id, 0) + 1
    for row in time_entry_rows:
        user_id = str(row.get("user_id") or "")
        if not user_id:
            continue
        spent_by_user[user_id] = spent_by_user.get(user_id, 0.0) + float(row.get("hours") or 0.0)
        entries_by_user[user_id] = entries_by_user.get(user_id, 0) + 1
    for row in user_rows:
        user_id = str(row.get("user_id") or "")
        planned = round(planned_by_user.get(user_id, 0.0), 4)
        spent = round(spent_by_user.get(user_id, 0.0), 4)
        row["assigned_tasks"] = assigned_tasks_by_user.get(user_id, 0)
        row["planned_hours"] = planned
        row["spent_hours"] = spent
        row["time_entry_count"] = entries_by_user.get(user_id, 0)
        row["variance_hours"] = round(spent - planned, 4)
        row["load_ratio_percent"] = round(spent / planned * 100, 2) if planned else 0.0
    return user_rows


class OpenProjectSyncService:
    def __init__(self, settings: Settings, db_connections_service: DBConnectionsService) -> None:
        self.settings = settings
        self.db_connections_service = db_connections_service

    @classmethod
    def from_settings(
        cls,
        *,
        settings: Settings,
        db_connections_service: DBConnectionsService,
    ) -> "OpenProjectSyncService":
        return cls(settings, db_connections_service)

    @property
    def is_enabled(self) -> bool:
        return bool(self.settings.openproject_base_url and self.settings.openproject_api_key)

    def source_descriptor(self) -> dict[str, Any]:
        return {
            "source_type": "openproject",
            "source_ref_id": "openproject",
            "source_label": "OpenProject",
            "display_name_ru": "OpenProject",
            "source_mode": "postgres_sync",
            "enabled": self.is_enabled,
            "available": bool(self.settings.openproject_pg_host and self.settings.openproject_pg_database),
            "status": "available" if self.is_enabled else "disabled",
            "description": "Sync OpenProject API tables into PostgreSQL and use them for analysis.",
            "description_ru": "Выгрузка таблиц OpenProject в PostgreSQL для анализа.",
            "capabilities": ["projects", "work_packages", "time_entries", "personnel", "postgresql"],
            "requires_session_data": False,
            "timeout_hint_sec": 60,
        }

    def _load_dataframes(self, options: OpenProjectSyncOptions | None = None) -> dict[str, pd.DataFrame]:
        options = options or OpenProjectSyncOptions()
        max_items = (
            options.max_items
            if options.max_items is not None
            else self.settings.openproject_max_items
        ) or None
        project = None if options.all_projects else (
            options.project if options.project is not None else self.settings.openproject_project
        )
        project = project or None
        days = options.days if options.days is not None else self.settings.openproject_days
        base_url = options.base_url or self.settings.openproject_base_url
        api_key = options.api_key or self.settings.openproject_api_key
        host_header = options.host_header or self.settings.openproject_host_header or None
        with OpenProjectAPI(
            base_url=base_url,
            api_key=api_key,
            host_header=host_header,
        ) as api:
            project_id: str | None = None
            if project:
                project_id = api.resolve_project_id(project)
                if not project_id:
                    raise OpenProjectIntegrationError(f"OpenProject project not found: {project}")
            base_rows = {
                table: _load_rows(
                    api,
                    table,
                    project_id=project_id,
                    days=days,
                    max_items=max_items,
                )
                for table in OPENPROJECT_TABLES
            }
        base_rows["projects"] = _augment_projects(
            base_rows["projects"],
            base_rows["work_packages"],
            base_rows["time_entries"],
        )
        base_rows["personnel"] = _augment_personnel(
            base_rows["personnel"],
            base_rows["work_packages"],
            base_rows["time_entries"],
        )
        return {name: pd.DataFrame(rows) for name, rows in base_rows.items()}

    def list_projects(self, options: OpenProjectSyncOptions | None = None) -> list[dict[str, str]]:
        options = options or OpenProjectSyncOptions()
        base_url = options.base_url or self.settings.openproject_base_url
        api_key = options.api_key or self.settings.openproject_api_key
        host_header = options.host_header or self.settings.openproject_host_header or None
        if not base_url:
            raise OpenProjectIntegrationError("OpenProject base URL is not configured.")
        if not api_key:
            raise OpenProjectIntegrationError("OpenProject API key is not configured.")
        with OpenProjectAPI(
            base_url=base_url,
            api_key=api_key,
            host_header=host_header,
        ) as api:
            projects: list[dict[str, str]] = []
            for item in api.iter_collection("projects", page_size=100):
                project_id = str(item.get("id") or "").strip()
                identifier = str(item.get("identifier") or "").strip()
                name = str(item.get("name") or identifier or project_id).strip()
                if project_id or identifier:
                    projects.append(
                        {
                            "id": project_id,
                            "identifier": identifier or project_id,
                            "name": name,
                        }
                    )
            return projects

    def _write_postgres(self, dataframes: dict[str, pd.DataFrame]) -> dict[str, int]:
        import psycopg
        from psycopg import sql

        schema = self.settings.openproject_pg_schema or "openproject"
        rows_by_table: dict[str, int] = {}
        conn_kwargs = {
            "host": self.settings.openproject_pg_host,
            "port": self.settings.openproject_pg_port,
            "dbname": self.settings.openproject_pg_database,
            "user": self.settings.openproject_pg_username,
            "password": self.settings.openproject_pg_password or None,
            "connect_timeout": self.settings.db_connections_test_timeout_sec,
        }
        sslmode = self.settings.openproject_pg_sslmode
        if sslmode:
            conn_kwargs["sslmode"] = sslmode

        with psycopg.connect(**conn_kwargs) as conn:
            with conn.cursor() as cur:
                cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
                for table_name, df in dataframes.items():
                    clean_df = df.where(pd.notna(df), None).copy()
                    columns = [str(column) for column in clean_df.columns]
                    cur.execute(
                        sql.SQL("DROP TABLE IF EXISTS {}.{}").format(
                            sql.Identifier(schema),
                            sql.Identifier(table_name),
                        )
                    )
                    column_defs = [
                        sql.SQL("{} {}").format(sql.Identifier(column), sql.SQL(_postgres_type_for_series(clean_df[column])))
                        for column in columns
                    ]
                    if not column_defs:
                        column_defs = [sql.SQL("{} text").format(sql.Identifier("_empty"))]
                    cur.execute(
                        sql.SQL("CREATE TABLE {}.{} ({})").format(
                            sql.Identifier(schema),
                            sql.Identifier(table_name),
                            sql.SQL(", ").join(column_defs),
                        )
                    )
                    if columns and not clean_df.empty:
                        placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in columns)
                        insert_sql = sql.SQL("INSERT INTO {}.{} ({}) VALUES ({})").format(
                            sql.Identifier(schema),
                            sql.Identifier(table_name),
                            sql.SQL(", ").join(sql.Identifier(column) for column in columns),
                            placeholders,
                        )
                        cur.executemany(insert_sql, [tuple(row) for row in clean_df.itertuples(index=False, name=None)])
                    rows_by_table[table_name] = int(len(clean_df))
                conn.commit()
        return rows_by_table

    def _ensure_connection(self, user_id: int) -> DBConnectionRecord:
        schema = self.settings.openproject_pg_schema or "openproject"
        name = "OpenProject Analytics"
        existing = next(
            (item for item in self.db_connections_service.list_connections(user_id) if item.name == name),
            None,
        )
        payload = {
            "name": name,
            "db_type": "postgresql",
            "host": self.settings.openproject_pg_host,
            "port": self.settings.openproject_pg_port,
            "database": self.settings.openproject_pg_database,
            "username": self.settings.openproject_pg_username,
            "password": self.settings.openproject_pg_password or None,
            "options_json": {
                "sslmode": self.settings.openproject_pg_sslmode or "prefer",
                "schema": schema,
                "source_type": "openproject",
                "hidden": True,
            },
        }
        if existing is None:
            return self.db_connections_service.create_connection(user_id, **payload)
        return self.db_connections_service.update_connection(
            user_id,
            existing.id,
            clear_password=False,
            options_json_set=True,
            **payload,
        )

    def sync(
        self,
        *,
        user_id: int,
        options: OpenProjectSyncOptions | None = None,
    ) -> OpenProjectSyncResult:
        options = options or OpenProjectSyncOptions()
        if not (options.base_url or self.settings.openproject_base_url):
            raise OpenProjectIntegrationError("OpenProject base URL is not configured.")
        if not (options.api_key or self.settings.openproject_api_key):
            raise OpenProjectIntegrationError("OpenProject API key is not configured.")
        dataframes = self._load_dataframes(options)
        rows_by_table = self._write_postgres(dataframes)
        connection = self._ensure_connection(user_id)
        return OpenProjectSyncResult(
            connection=connection,
            schema=self.settings.openproject_pg_schema or "openproject",
            rows_by_table=rows_by_table,
            dataframes=dataframes,
            synced_at=datetime.now(timezone.utc).isoformat(),
        )


def _postgres_type_for_series(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "bigint"
    if pd.api.types.is_float_dtype(series):
        return "double precision"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "timestamptz"
    return "text"
