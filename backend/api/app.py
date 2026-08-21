from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager, suppress
from typing import Any

import anyio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from backend.agent.callbacks import (
    ContextUsageCollector,
    LLMTextCollector,
    PhaseCollector,
    TokenStreamCallbackHandler,
    ToolCollector,
)
from backend.agent.runner import AgentRunner
from backend.agent.services.chat_title import ChatTitleService
from backend.api import deps
from backend.api.models import ToolAvailabilityResponse
from backend.api.routes import (
    admin,
    admin_skills,
    auth,
    data,
    db_connections,
    health,
    mcp_servers,
    observability_route,
    query,
    rag_documents,
    reports,
    semantic_catalog,
    sessions,
    skills,
    sources,
)
from backend.auth.app_data_postgres import AppDataPostgresStore
from backend.auth.auth_db import PostgresAuthDB
from backend.auth.blob_store import PostgresBlobStore
from backend.auth.user_memory import UserMemoryService
from backend.auth.user_settings_defaults import user_settings_defaults_from_runtime
from backend.core.config import settings
from backend.data_access.csv_session_runtime import CSVSessionRuntime
from backend.data_access.db_connections_service import DBConnectionsService
from backend.data_access.db_runtime_service import DBRuntimeService
from backend.data_access.semantic_catalog_service import SemanticCatalogService
from backend.data_access.semantic_catalog_store import semantic_catalog_store_from_settings
from backend.data_access.semantic_context import SemanticContextBuilder
from backend.data_access.semantic_generation_service import SemanticCatalogGenerationService
from backend.data_access.semantic_scenario_service import SemanticScenarioService
from backend.data_access.semantic_vector_store import SemanticVectorStore
from backend.integrations.anomaly_planfact import AnomalyPlanfactIntegrationService
from backend.integrations.forecast import ForecastIntegrationService
from backend.integrations.openproject import OpenProjectSyncService
from backend.integrations.rag import RAGService
from backend.mcp.service import MCPServerService
from backend.notebook.kernel_manager import KernelManager
from backend.notebook.manifest_store import PostgresManifestStore
from backend.notebook.orchestrator import NotebookOrchestrator
from backend.notebook.store import PostgresNotebookStore
from backend.observability.phoenix import (
    build_trace_context,
    initialize_phoenix,
    query_trace_context,
)
from backend.observability.service import PhoenixObservabilityService
from backend.sessions.postgres_session_store import PostgresSessionStore
from backend.skills.override_store import PostgresSkillOverrideStore
from backend.skills.registry import SkillRegistry
from backend.tools.catalog import KNOWN_TOOL_KEYS, build_tool_catalog
from backend.tools.policy import effective_enabled_tool_keys
from backend.tools.sandbox_manager import SandboxManager

_log = logging.getLogger(__name__)


def _validate_startup_config() -> None:
    """Abort on obviously insecure or broken configurations before serving traffic."""
    if settings.auth_default_admin_password == "admin":
        sys.exit(
            "FATAL: AUTH_DEFAULT_ADMIN_PASSWORD is set to the insecure default 'admin'. "
            "Set a strong password via the AUTH_DEFAULT_ADMIN_PASSWORD environment variable "
            "before starting the server."
        )

    if settings.cors_allow_origins.strip() == "*":
        _log.warning(
            "BACKEND_CORS_ALLOW_ORIGINS='*' combined with allow_credentials=True is rejected "
            "by browsers (Fetch standard). allow_credentials will be forced to False. "
            "Set explicit origins in BACKEND_CORS_ALLOW_ORIGINS to enable credentialed requests."
        )


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _validate_startup_config()
    await anyio.to_thread.run_sync(auth_db.initialize)
    await anyio.to_thread.run_sync(
        lambda: auth_db.ensure_default_admin(
            settings.auth_default_admin_username,
            settings.auth_default_admin_password,
        )
    )
    await anyio.to_thread.run_sync(store.initialize)
    await anyio.to_thread.run_sync(override_store.initialize)
    skill_registry.override_store = override_store
    await anyio.to_thread.run_sync(skill_registry.reload)
    await anyio.to_thread.run_sync(runner.warmup)

    # Periodically evict sandboxes for sessions that have been idle for >2 h.
    async def _sandbox_cleanup_loop() -> None:
        while True:
            await anyio.sleep(3600)
            evicted = await anyio.to_thread.run_sync(
                lambda: SandboxManager.get_instance().cleanup_expired(ttl_sec=7200)
            )
            if evicted:
                _log.info("Lifespan: evicted %d idle sandbox(es)", evicted)

    cleanup_task = asyncio.create_task(_sandbox_cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task


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

_cors_wildcard = settings.cors_allow_origins.strip() == "*"
origins = (
    ["*"] if _cors_wildcard else [o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()]
)
# allow_credentials=True is incompatible with wildcard origins per the Fetch standard.
# When origins are explicit, credentials (cookies / Authorization header) are allowed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=not _cors_wildcard,
    allow_methods=["*"],
    allow_headers=["*"],
)

semantic_catalog_store = semantic_catalog_store_from_settings(settings)
app_data_store = AppDataPostgresStore(
    settings.app_data_postgres_dsn,
    schema=settings.app_data_postgres_schema,
)
blob_store = PostgresBlobStore(app_data_store)
store = PostgresSessionStore(
    settings.storage_dir,
    app_data_store=app_data_store,
    data_catalog_store=semantic_catalog_store,
    artifact_blob_store=blob_store,
)
auth_db = PostgresAuthDB(
    settings.app_data_postgres_dsn,
    schema=settings.app_data_postgres_schema,
    token_ttl_days=settings.auth_token_ttl_days,
    user_settings_defaults=user_settings_defaults_from_runtime(settings),
)
override_store = PostgresSkillOverrideStore(
    settings.app_data_postgres_dsn,
    schema=settings.app_data_postgres_schema,
)
skill_registry = SkillRegistry.from_path(settings.skills_dir)
user_memory_service = UserMemoryService(auth_db)
db_connections_service = DBConnectionsService(auth_db, settings)
db_runtime_service = DBRuntimeService(db_connections_service)
csv_runtime = CSVSessionRuntime(default_ttl_sec=settings.csv_session_ttl_sec)
semantic_vector_store = SemanticVectorStore.from_settings(settings)
semantic_catalog_service = SemanticCatalogService(
    store=store,
    vector_store=semantic_vector_store if settings.semantic_layer_enabled else None,
    settings=settings,
)
semantic_generation_service = SemanticCatalogGenerationService(
    store=store,
    catalog_service=semantic_catalog_service,
    db_runtime_service=db_runtime_service,
    settings=settings,
)
semantic_scenario_service = SemanticScenarioService(
    catalog_service=semantic_catalog_service,
    settings=settings,
)
semantic_context_builder = SemanticContextBuilder(
    store=store,
    vector_store=semantic_vector_store if settings.semantic_layer_enabled else None,
    catalog_service=semantic_catalog_service,
    top_k=settings.semantic_top_k,
)
notebook_store = PostgresNotebookStore(app_data_store)
manifest_store = PostgresManifestStore(app_data_store)
notebook_orchestrator = NotebookOrchestrator(notebook_store)
kernel_manager = KernelManager(
    notebook_store=notebook_store,
    manifest_store=manifest_store,
    storage_dir=settings.storage_dir,
)
forecast_integration_service = ForecastIntegrationService.from_env(settings=settings)
anomaly_planfact_integration_service = AnomalyPlanfactIntegrationService.from_env(settings=settings)
openproject_sync_service = OpenProjectSyncService.from_settings(
    settings=settings,
    db_connections_service=db_connections_service,
)
rag_service = RAGService.from_env()
mcp_service = MCPServerService(auth_db=auth_db)
runner = AgentRunner(
    settings,
    db_runtime_service=db_runtime_service,
    forecast_service=forecast_integration_service,
    anomaly_planfact_service=anomaly_planfact_integration_service,
    rag_service=rag_service,
    skill_registry=skill_registry,
    semantic_catalog_service=semantic_catalog_service,
    semantic_generation_service=semantic_generation_service,
    manifest_store=manifest_store,
    session_store=store,
    blob_store=blob_store,
)
chat_title_service = ChatTitleService(settings=settings)
phoenix_observability_service = PhoenixObservabilityService(settings)
initialize_phoenix()


def _integration_source_descriptors() -> list[dict[str, Any]]:
    return [
        rag_service.source_descriptor(),
        forecast_integration_service.source_descriptor(),
        anomaly_planfact_integration_service.source_descriptor(),
        openproject_sync_service.source_descriptor(),
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
    mcp_servers.setup(auth_db=auth_db, mcp_service=mcp_service)
    admin_skills.setup(
        skill_registry=skill_registry,
        auth_db=auth_db,
        override_store=override_store,
    )
    sessions.setup(
        auth_db=auth_db,
        store=store,
        title_service=chat_title_service,
        build_trace_context_fn=build_trace_context,
        query_trace_context_fn=query_trace_context,
        manifest_store=manifest_store,
        semantic_catalog_service=semantic_catalog_service,
    )
    data.setup(
        auth_db=auth_db,
        store=store,
        csv_runtime=csv_runtime,
        manifest_store=manifest_store,
        notebook_orchestrator=notebook_orchestrator,
        storage_dir=settings.storage_dir,
        semantic_catalog_service=semantic_catalog_service,
        blob_store=blob_store,
    )
    sources.setup(
        db_runtime_service=db_runtime_service,
        auth_db=auth_db,
        store=store,
        db_connections_service=db_connections_service,
        integration_source_descriptors_fn=_integration_source_descriptors,
        csv_runtime=csv_runtime,
        manifest_store=manifest_store,
        notebook_orchestrator=notebook_orchestrator,
        storage_dir=settings.storage_dir,
        openproject_sync_service=openproject_sync_service,
        semantic_catalog_service=semantic_catalog_service,
        blob_store=blob_store,
    )
    semantic_catalog.setup(
        auth_db=auth_db,
        store=store,
        semantic_catalog_service=semantic_catalog_service,
        semantic_generation_service=semantic_generation_service,
        semantic_scenario_service=semantic_scenario_service,
        db_runtime_service=db_runtime_service,
    )
    rag_documents.setup(
        auth_db=auth_db,
        store=store,
        rag_service=rag_service,
    )
    db_connections.setup(
        auth_db=auth_db,
        db_connections_service=db_connections_service,
        db_runtime_service=db_runtime_service,
        semantic_catalog_service=semantic_catalog_service,
    )
    query.setup(
        auth_db=auth_db,
        store=store,
        skill_registry=skill_registry,
        db_runtime_service=db_runtime_service,
        forecast_integration_service=forecast_integration_service,
        anomaly_planfact_integration_service=anomaly_planfact_integration_service,
        rag_service=rag_service,
        user_memory_service=user_memory_service,
        build_trace_context_fn=build_trace_context,
        query_trace_context_fn=query_trace_context,
        app_settings=settings,
        csv_runtime=csv_runtime,
        manifest_store=manifest_store,
        blob_store=blob_store,
        storage_dir=settings.storage_dir,
        LLMTextCollector=LLMTextCollector,
        ToolCollector=ToolCollector,
        PhaseCollector=PhaseCollector,
        TokenStreamCallbackHandler=TokenStreamCallbackHandler,
        ContextUsageCollector=ContextUsageCollector,
        AgentRunner=AgentRunner,
        effective_enabled_tool_keys_fn=effective_enabled_tool_keys,
        build_tool_catalog_fn=build_tool_catalog,
        known_tool_keys=KNOWN_TOOL_KEYS,
        mcp_service=mcp_service,
        semantic_context_builder=semantic_context_builder,
        semantic_catalog_service=semantic_catalog_service,
        semantic_generation_service=semantic_generation_service,
    )
    skills.setup(skill_registry=skill_registry, auth_db=auth_db)
    observability_route.setup(
        phoenix_observability_service=phoenix_observability_service,
    )
    reports.setup(
        auth_db=auth_db,
        store=store,
        csv_runtime=csv_runtime,
        manifest_store=manifest_store,
        blob_store=blob_store,
    )

    app.include_router(health.router)
    app.include_router(observability_route.router)
    app.include_router(auth.router)
    app.include_router(admin.router)
    app.include_router(mcp_servers.router)
    app.include_router(admin_skills.router)
    app.include_router(sessions.router)
    app.include_router(skills.router)
    app.include_router(data.router)
    app.include_router(sources.router)
    app.include_router(semantic_catalog.router)
    app.include_router(rag_documents.router)
    app.include_router(query.router)
    app.include_router(db_connections.router)
    app.include_router(reports.router)


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
