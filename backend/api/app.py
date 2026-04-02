from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import anyio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from backend.agent.callbacks import (
    AgentProgressCollector,
    LLMTextCollector,
    PhaseCollector,
    PhaseTokenStreamHandler,
    TokenStreamCallbackHandler,
    ToolCollector,
)
from backend.agent.runner import AgentRunner
from backend.api import deps
from backend.api.models import ToolAvailabilityResponse
from backend.api.routes import (
    admin,
    auth,
    data,
    db_connections,
    health,
    observability_route,
    query,
    sessions,
    skills,
    sources,
)
from backend.auth.auth_db import AuthDB
from backend.auth.user_memory import UserMemoryService
from backend.core.config import settings
from backend.data_access.csv_session_runtime import CSVSessionRuntime
from backend.data_access.db_connections_service import DBConnectionsService
from backend.data_access.db_runtime_service import DBRuntimeService
from backend.integrations.anomaly_planfact import AnomalyPlanfactIntegrationService
from backend.integrations.forecast import ForecastIntegrationService
from backend.integrations.rag import RAGService
from backend.integrations.search import SearchIntegrationService
from backend.observability.phoenix import (
    build_trace_context,
    initialize_phoenix,
    query_trace_context,
)
from backend.observability.service import PhoenixObservabilityService
from backend.sessions.session_store import SessionStore
from backend.tools.catalog import KNOWN_TOOL_KEYS, build_tool_catalog
from backend.tools.policy import effective_enabled_tool_keys


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    await anyio.to_thread.run_sync(runner.warmup)
    yield


_OPENAPI_TAGS: list[dict[str, str]] = [
    {"name": "Сервис", "description": "Проверка доступности и параметры runtime."},
    {"name": "Аутентификация", "description": "Регистрация, вход, профиль, настройки и инструменты."},
    {"name": "Администрирование", "description": "Управление пользователями (только администратор)."},
    {"name": "Сессии", "description": "Чаты и состояние сессий."},
    {"name": "Данные", "description": "Загрузка данных в сессию."},
    {"name": "Источники", "description": "Источники данных и привязка к сессии."},
    {"name": "Запросы и агент", "description": "Запросы к агенту и стриминг ответов."},
    {"name": "Навыки", "description": "Просмотр доступных markdown skills и их идентификаторов."},
    {"name": "Подключения к БД", "description": "Сохранённые подключения к БД и схемы."},
    {"name": "Наблюдаемость", "description": "Обзор Phoenix / трассировки."},
]

app = FastAPI(
    title="LLM Data Analyst Backend",
    version="0.2.0",
    lifespan=_lifespan,
    openapi_tags=_OPENAPI_TAGS,
)

origins = (
    ["*"]
    if settings.cors_allow_origins.strip() == "*"
    else [o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = SessionStore(settings.storage_dir, settings.session_ttl_days)
auth_db = AuthDB(settings.auth_db_path, settings.auth_token_ttl_days)
auth_db.ensure_default_admin(
    settings.auth_default_admin_username,
    settings.auth_default_admin_password,
)
user_memory_service = UserMemoryService(auth_db)
db_connections_service = DBConnectionsService(auth_db, settings)
db_runtime_service = DBRuntimeService(db_connections_service)
csv_runtime = CSVSessionRuntime(default_ttl_sec=settings.csv_session_ttl_sec)
search_integration_service = SearchIntegrationService.from_env()
forecast_integration_service = ForecastIntegrationService.from_env()
anomaly_planfact_integration_service = AnomalyPlanfactIntegrationService.from_env()
rag_service = RAGService.from_env()
runner = AgentRunner(
    settings,
    db_runtime_service=db_runtime_service,
    search_service=search_integration_service,
    forecast_service=forecast_integration_service,
    anomaly_planfact_service=anomaly_planfact_integration_service,
    rag_service=rag_service,
)
phoenix_observability_service = PhoenixObservabilityService(settings)
initialize_phoenix()


def _integration_source_descriptors() -> list[dict[str, Any]]:
    return [
        search_integration_service.source_descriptor(),
        rag_service.source_descriptor(),
        forecast_integration_service.source_descriptor(),
        anomaly_planfact_integration_service.source_descriptor(),
    ]


def _tool_catalog_payload(user_id: int) -> list[dict[str, Any]]:
    return build_tool_catalog(
        source_descriptors=_integration_source_descriptors(),
        user_settings=auth_db.list_user_tool_settings(user_id),
    )


def _tool_catalog_response(user_id: int) -> list[ToolAvailabilityResponse]:
    return [ToolAvailabilityResponse(**item) for item in _tool_catalog_payload(user_id)]


def _configure_routes() -> None:
    deps.set_auth_db(auth_db)

    auth.setup(
        auth_db=auth_db,
        user_memory_service=user_memory_service,
        tool_catalog_response_fn=_tool_catalog_response,
        tool_catalog_payload_fn=_tool_catalog_payload,
        known_tool_keys=KNOWN_TOOL_KEYS,
    )
    admin.setup(auth_db=auth_db)
    sessions.setup(
        auth_db=auth_db,
        store=store,
        runner=runner,
        build_trace_context_fn=build_trace_context,
        query_trace_context_fn=query_trace_context,
    )
    data.setup(auth_db=auth_db, store=store, csv_runtime=csv_runtime)
    sources.setup(
        auth_db=auth_db,
        store=store,
        db_connections_service=db_connections_service,
        integration_source_descriptors_fn=_integration_source_descriptors,
        csv_runtime=csv_runtime,
    )
    db_connections.setup(
        auth_db=auth_db,
        db_connections_service=db_connections_service,
        db_runtime_service=db_runtime_service,
    )
    query.setup(
        auth_db=auth_db,
        store=store,
        runner=runner,
        db_runtime_service=db_runtime_service,
        search_integration_service=search_integration_service,
        forecast_integration_service=forecast_integration_service,
        anomaly_planfact_integration_service=anomaly_planfact_integration_service,
        rag_service=rag_service,
        user_memory_service=user_memory_service,
        build_trace_context_fn=build_trace_context,
        query_trace_context_fn=query_trace_context,
        app_settings=settings,
        csv_runtime=csv_runtime,
        LLMTextCollector=LLMTextCollector,
        ToolCollector=ToolCollector,
        AgentProgressCollector=AgentProgressCollector,
        PhaseCollector=PhaseCollector,
        TokenStreamCallbackHandler=TokenStreamCallbackHandler,
        PhaseTokenStreamHandler=PhaseTokenStreamHandler,
        AgentRunner=AgentRunner,
        effective_enabled_tool_keys_fn=effective_enabled_tool_keys,
        build_tool_catalog_fn=build_tool_catalog,
        known_tool_keys=KNOWN_TOOL_KEYS,
    )
    skills.setup(runner=runner)
    observability_route.setup(
        phoenix_observability_service=phoenix_observability_service,
    )

    app.include_router(health.router)
    app.include_router(observability_route.router)
    app.include_router(auth.router)
    app.include_router(admin.router)
    app.include_router(sessions.router)
    app.include_router(skills.router)
    app.include_router(data.router)
    app.include_router(sources.router)
    app.include_router(query.router)
    app.include_router(db_connections.router)


_configure_routes()


def _custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        tags=_OPENAPI_TAGS,
        routes=app.routes,
    )
    schema.setdefault("components", {}).setdefault("securitySchemes", {})["BearerAuth"] = {
        "type": "http",
        "scheme": "bearer",
    }
    for path in schema.get("paths", {}).values():
        for operation in path.values():
            operation.setdefault("security", [{"BearerAuth": []}])
    app.openapi_schema = schema
    return schema


app.openapi = _custom_openapi
