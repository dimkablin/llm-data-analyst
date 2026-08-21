from __future__ import annotations

import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import patch

import pandas as pd
from cryptography.fernet import Fernet

from backend.agent.models import AgentResponse
from backend.agent.runtime_contracts import AgentRunRequest, AgentRunResult
from backend.api.models import QueryRequest
from backend.api.services.query_execution import (
    QueryExecutionDependencies,
    QueryExecutionRequest,
    QueryExecutionService,
)
from backend.auth import AuthDB
from backend.auth.auth_db import AuthUser, UserSettings
from backend.core.config import Settings
from backend.data_access.csv_session_runtime import CSVSessionRuntime
from backend.mcp.models import (
    AdminMCPServerConfigResponse,
    MCPServerConfig,
    MCPServerCreateRequest,
    MCPServerTransport,
    MCPServerUpdateRequest,
    MCPToolBindingConfig,
    MCPToolCallResult,
    MCPToolDescriptor,
)
from backend.mcp.service import MCPServerService
from backend.mcp.transport import SDKMCPToolProvider
from backend.notebook.manifest_store import ManifestStore
from backend.sessions.session_store import SessionState, SessionStore
from backend.skills.registry import SkillRegistry
from backend.tools.context import ToolBuildContext
from backend.tools.registry import ToolRegistry


class _FakeMCPProvider:
    def __init__(self) -> None:
        self.tools_by_server: dict[str, list[MCPToolDescriptor]] = {}
        self.list_tool_calls: list[str] = []

    def list_tools(self, config: MCPServerConfig) -> list[MCPToolDescriptor]:
        self.list_tool_calls.append(config.server_id)
        return list(self.tools_by_server.get(config.server_id, []))

    def call_tool(
        self,
        *,
        config: MCPServerConfig,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        return f"{config.server_id}:{tool_name}:{arguments}"


class MCPServerSupportTests(unittest.TestCase):
    def test_chronos_config_projects_discovered_tool_to_forecast_capability(self) -> None:
        provider = _FakeMCPProvider()
        config = MCPServerConfig(
            server_id="chronos",
            name="Chronos",
            transport=MCPServerTransport.streamable_http,
            url="http://127.0.0.1:8810/mcp",
            tool_bindings={
                "forecast": MCPToolBindingConfig(
                    capability_key="forecast",
                    provider_identity="chronos",
                    preferred=True,
                )
            },
        )
        provider.tools_by_server["chronos"] = [
            MCPToolDescriptor.from_mcp_tool(
                server_id="chronos",
                tool_name="forecast",
            )
        ]

        [descriptor] = MCPServerService(
            configs=[config],
            provider=provider,
        ).enabled_tool_descriptors()

        self.assertEqual(descriptor.capability_key, "forecast")
        self.assertEqual(descriptor.provider_identity, "chronos")
        self.assertTrue(descriptor.binding_preferred)

    def test_active_registry_resolves_capability_to_highest_priority_binding(self) -> None:
        provider = _FakeMCPProvider()
        config = MCPServerConfig(
            server_id="provider-b",
            name="Provider B",
            transport=MCPServerTransport.streamable_http,
            url="http://127.0.0.1:8810/mcp",
        )
        descriptor = MCPToolDescriptor.from_mcp_tool(
            server_id=config.server_id,
            tool_name="forecast",
            description="Specialized projection",
            capability_key="forecast",
            provider_identity="provider-b",
            binding_preferred=True,
        )
        registry = ToolRegistry.from_services(
            forecast_service=SimpleNamespace(is_enabled=True),
            mcp_tool_provider=provider,
            mcp_server_configs={config.server_id: config},
            mcp_tool_descriptors=[descriptor],
        )
        surface = registry.build_active_surface(
            ToolBuildContext(
                settings=Settings(),
                allowed_tool_keys={"forecast_tool", descriptor.tool_key},
                df=pd.DataFrame({"value": [1]}),
            )
        )

        self.assertIn(descriptor.tool_key, surface.catalog.tool_keys)
        self.assertNotIn("forecast_tool", surface.catalog.tool_keys)
        self.assertEqual(surface.catalog.capability_for_tool(descriptor.tool_key).key, "forecast")

    def test_bearer_token_is_encrypted_hidden_and_preserved_on_update(self) -> None:
        tmpdir = tempfile.mkdtemp()
        try:
            auth_db = AuthDB(str(Path(tmpdir) / "app.db"))
            admin = auth_db.create_user("chronos_admin", "secret", is_admin=True)
            settings = Settings(
                db_connections_encryption_key_current=Fernet.generate_key().decode(),
            )
            service = MCPServerService(
                auth_db=auth_db,
                provider=_FakeMCPProvider(),
                settings=settings,
            )
            created = service.upsert_config(
                MCPServerCreateRequest(
                    server_id="chronos",
                    name="Chronos",
                    url="http://127.0.0.1:8765/mcp",
                    bearer_token="chronos-secret",
                ),
                updated_by=admin.id,
            )

            blob = auth_db.get_mcp_server_secret_blob("chronos")
            response = AdminMCPServerConfigResponse(
                **created.model_dump(),
                secret_configured=bool(created.bearer_token),
            )
            updated = service.update_config(
                "chronos",
                MCPServerUpdateRequest(name="Chronos MCP"),
                updated_by=admin.id,
            )

            self.assertIsNotNone(blob)
            self.assertNotIn("chronos-secret", str(blob))
            self.assertNotIn("bearer_token", response.model_dump())
            self.assertTrue(response.secret_configured)
            self.assertEqual(updated.bearer_token, "chronos-secret")
        finally:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_http_transport_sends_bearer_authorization_header(self) -> None:
        captured: dict[str, object] = {}

        @asynccontextmanager
        async def fake_client(url: str, **kwargs):
            captured.update(url=url, **kwargs)
            yield object(), object(), lambda: None

        class FakeSession:
            def __init__(self, *_args) -> None:
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return None

            async def initialize(self) -> None:
                pass

        config = MCPServerConfig(
            server_id="chronos",
            name="Chronos",
            transport=MCPServerTransport.streamable_http,
            url="http://127.0.0.1:8765/mcp",
            bearer_token="chronos-secret",
        )

        async def connect() -> None:
            async with SDKMCPToolProvider._http_session(config):
                pass

        with (
            patch("backend.mcp.transport.streamablehttp_client", fake_client),
            patch("backend.mcp.transport.ClientSession", FakeSession),
        ):
            anyio_run(connect())

        self.assertEqual(captured["headers"], {"Authorization": "Bearer chronos-secret"})

    def test_auth_db_persists_user_mcp_server_settings_and_admin_configs(self) -> None:
        tmpdir = tempfile.mkdtemp()
        try:
            auth_db = AuthDB(str(Path(tmpdir) / "app.db"), token_ttl_days=30)
            admin = auth_db.create_user("admin_mcp", "secret", is_admin=True)
            user = auth_db.create_user("user_mcp", "secret", is_admin=False)

            self.assertEqual(auth_db.list_user_mcp_server_settings(user.id), {})

            config = MCPServerCreateRequest(
                server_id="finance-research",
                name="Finance Research",
                transport=MCPServerTransport.streamable_http,
                url="http://127.0.0.1:8765/mcp",
                enabled=True,
                enabled_by_default=False,
                tool_bindings={
                    "risk": MCPToolBindingConfig(
                        capability_key="portfolio_risk",
                        preferred=True,
                    )
                },
            )
            stored = auth_db.upsert_mcp_server_config(config, updated_by=admin.id)
            auth_db.set_user_mcp_server_enabled(user.id, "finance-research", True)

            self.assertEqual(stored.server_id, "finance-research")
            self.assertEqual(stored.updated_by, admin.id)
            self.assertTrue(auth_db.list_user_mcp_server_settings(user.id)["finance-research"])
            self.assertEqual(auth_db.get_mcp_server_config("finance-research").name, "Finance Research")
            self.assertEqual(
                auth_db.get_mcp_server_config("finance-research")
                .tool_bindings["risk"]
                .capability_key,
                "portfolio_risk",
            )
        finally:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_mcp_catalog_combines_global_availability_and_user_toggle(self) -> None:
        provider = _FakeMCPProvider()
        config = MCPServerConfig(
            server_id="finance-research",
            name="Finance Research",
            transport=MCPServerTransport.streamable_http,
            url="http://127.0.0.1:8765/mcp",
            enabled=True,
            enabled_by_default=True,
        )
        provider.tools_by_server[config.server_id] = [
            MCPToolDescriptor.from_mcp_tool(
                server_id=config.server_id,
                tool_name="portfolio.risk",
                description="Portfolio risk",
                input_schema={"type": "object", "properties": {"ticker": {"type": "string"}}},
            )
        ]
        service = MCPServerService(configs=[config], provider=provider)

        [enabled] = service.list_catalog(user_settings={})
        [disabled] = service.list_catalog(user_settings={"finance-research": False})

        self.assertTrue(enabled.available_globally)
        self.assertTrue(enabled.effective_enabled)
        self.assertEqual(enabled.tool_count, 1)
        self.assertEqual(enabled.tools[0].tool_key, "mcp__finance_research__portfolio_risk")
        self.assertFalse(disabled.effective_enabled)

    def test_tool_registry_builds_namespaced_mcp_tool_factory(self) -> None:
        provider = _FakeMCPProvider()
        config = MCPServerConfig(
            server_id="finance-research",
            name="Finance Research",
            transport=MCPServerTransport.streamable_http,
            url="http://127.0.0.1:8765/mcp",
        )
        descriptor = MCPToolDescriptor.from_mcp_tool(
            server_id=config.server_id,
            tool_name="portfolio.risk",
            description="Portfolio risk",
            input_schema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "request": {
                        "type": "object",
                        "properties": {
                            "frequency": {"enum": ["daily", "monthly"]},
                        },
                        "required": ["frequency"],
                    },
                },
                "required": ["ticker"],
            },
        )
        registry = ToolRegistry.from_services(
            mcp_tool_provider=provider,
            mcp_server_configs={config.server_id: config},
            mcp_tool_descriptors=[descriptor],
        )
        context = ToolBuildContext(
            settings=Settings(),
            allowed_tool_keys={descriptor.tool_key},
        )

        [tool] = [item for item in registry.build_tools(context) if item.name == descriptor.tool_key]
        result = tool.invoke({"ticker": "AAPL"})
        tool_call_result = tool.invoke(
            {
                "name": descriptor.tool_key,
                "args": {"ticker": "MSFT"},
                "id": "call-1",
                "type": "tool_call",
            }
        )

        self.assertEqual(tool.name, "mcp__finance_research__portfolio_risk")
        self.assertEqual(
            tool.tool_call_schema["properties"]["request"]["properties"]["frequency"]["enum"],
            ["daily", "monthly"],
        )
        self.assertEqual(result, "finance-research:portfolio.risk:{'ticker': 'AAPL'}")
        self.assertEqual(
            getattr(tool_call_result, "content", tool_call_result),
            "finance-research:portfolio.risk:{'ticker': 'MSFT'}",
        )

    def test_mcp_error_result_becomes_failed_tool_call(self) -> None:
        class ErrorProvider(_FakeMCPProvider):
            def call_tool(self, **_kwargs) -> MCPToolCallResult:
                return MCPToolCallResult(
                    structured_content={"code": "INVALID_ARGUMENT", "field": "source"},
                    is_error=True,
                )

        provider = ErrorProvider()
        config = MCPServerConfig(
            server_id="provider-b",
            name="Provider B",
            transport=MCPServerTransport.streamable_http,
            url="http://127.0.0.1:8810/mcp",
        )
        descriptor = MCPToolDescriptor.from_mcp_tool(
            server_id=config.server_id,
            tool_name="forecast",
            description="Specialized projection",
        )
        registry = ToolRegistry.from_services(
            mcp_tool_provider=provider,
            mcp_server_configs={config.server_id: config},
            mcp_tool_descriptors=[descriptor],
        )
        tool = next(
            item
            for item in registry.build_tools(
                ToolBuildContext(
                    settings=Settings(),
                    allowed_tool_keys={descriptor.tool_key},
                )
            )
            if item.name == descriptor.tool_key
        )

        with self.assertRaisesRegex(RuntimeError, "INVALID_ARGUMENT"):
            tool.invoke({})

    def test_mcp_service_updates_existing_config_without_create_payload_extras(self) -> None:
        service = MCPServerService(
            configs=[
                MCPServerConfig(
                    server_id="finance-research",
                    name="Finance Research",
                    transport=MCPServerTransport.streamable_http,
                    url="http://127.0.0.1:8765/mcp",
                    created_at="2026-01-01T00:00:00+00:00",
                    updated_at="2026-01-01T00:00:00+00:00",
                    updated_by=1,
                )
            ],
            provider=_FakeMCPProvider(),
        )

        updated = service.update_config(
            "finance-research",
            MCPServerUpdateRequest(enabled=False),
            updated_by=7,
        )

        self.assertIsNotNone(updated)
        assert updated is not None
        self.assertFalse(updated.enabled)
        self.assertEqual(updated.updated_by, 7)

    def test_query_execution_passes_only_enabled_mcp_tools_to_runner(self) -> None:
        provider = _FakeMCPProvider()
        enabled_config = MCPServerConfig(
            server_id="finance-research",
            name="Finance Research",
            transport=MCPServerTransport.streamable_http,
            url="http://127.0.0.1:8765/mcp",
            enabled=True,
            enabled_by_default=True,
        )
        disabled_config = MCPServerConfig(
            server_id="offline-research",
            name="Offline Research",
            transport=MCPServerTransport.streamable_http,
            url="http://127.0.0.1:8766/mcp",
            enabled=True,
            enabled_by_default=True,
        )
        chronos_config = MCPServerConfig(
            server_id="chronos",
            name="Chronos MCP",
            transport=MCPServerTransport.streamable_http,
            url="http://127.0.0.1:8810/mcp",
            enabled=True,
            enabled_by_default=True,
        )
        provider.tools_by_server[enabled_config.server_id] = [
            MCPToolDescriptor.from_mcp_tool(
                server_id=enabled_config.server_id,
                tool_name="portfolio.risk",
                description="Portfolio risk",
                input_schema={"type": "object", "properties": {"ticker": {"type": "string"}}},
            )
        ]
        provider.tools_by_server[disabled_config.server_id] = [
            MCPToolDescriptor.from_mcp_tool(
                server_id=disabled_config.server_id,
                tool_name="hidden.tool",
                description="Hidden",
                input_schema={"type": "object"},
            )
        ]
        provider.tools_by_server[chronos_config.server_id] = [
            MCPToolDescriptor.from_mcp_tool(
                server_id=chronos_config.server_id,
                tool_name="forecast",
                description="Chronos forecast",
                input_schema={"type": "object"},
            )
        ]
        mcp_service = MCPServerService(
            configs=[enabled_config, disabled_config, chronos_config],
            provider=provider,
        )

        auth_db = AuthDB.__new__(AuthDB)
        store = SessionStore.__new__(SessionStore)
        registry = SkillRegistry(skills_dir=Path("."))
        state = SessionState(
            session_id="session-1",
            created_at="2026-01-01T00:00:00+00:00",
            last_access="2026-01-01T00:00:00+00:00",
            chat_history=[],
            artifacts=[],
            source_type=None,
        )
        calls: dict[str, object] = {}

        auth_db.is_session_owner = MethodType(lambda _self, _sid, _uid: True, auth_db)
        auth_db.touch_session = MethodType(lambda _self, _sid: None, auth_db)
        auth_db.update_session_after_reply = MethodType(lambda _self, *_args, **_kwargs: None, auth_db)
        auth_db.list_user_tool_settings = MethodType(lambda _self, _uid: {}, auth_db)
        auth_db.list_user_skill_settings = MethodType(lambda _self, _uid: {}, auth_db)
        auth_db.list_user_mcp_server_settings = MethodType(
            lambda _self, _uid: {"offline-research": False},
            auth_db,
        )
        auth_db.get_user_settings = MethodType(lambda _self, _uid: _user_settings(), auth_db)

        store.load_session = MethodType(lambda _self, _sid: state, store)
        store.get_dataframe = MethodType(lambda _self, _sid: None, store)
        store.load_data_catalog = MethodType(lambda _self, _sid: None, store)
        store.get_structured_memory = MethodType(lambda _self, _sid: None, store)
        store.add_artifacts = MethodType(lambda _self, *_args: None, store)
        store.set_selected_skill_ids = MethodType(lambda _self, *_args: None, store)
        store.add_chat_message = MethodType(lambda _self, *_args, **_kwargs: None, store)
        registry.list_skills = MethodType(lambda _self: [], registry)
        registry.resolve_selection = MethodType(lambda _self, _ids: [], registry)

        class _FakeRunner:
            instances: ClassVar[list[_FakeRunner]] = []

            def __init__(self, _settings, **kwargs) -> None:
                self.kwargs = kwargs
                self.instances.append(self)

            def run(self, request: AgentRunRequest) -> AgentRunResult:
                calls["request"] = request
                return AgentRunResult(
                    response=AgentResponse(
                        final_text="answer",
                        reasoning=None,
                        artifacts=[],
                        route="analysis",
                    )
                )

        service = QueryExecutionService(
            dependencies=QueryExecutionDependencies(
                auth_db=auth_db,
                store=store,
                skill_registry=registry,
                db_runtime_service=None,
                forecast_integration_service=SimpleNamespace(source_descriptor=lambda: {}),
                anomaly_planfact_integration_service=SimpleNamespace(source_descriptor=lambda: {}),
                rag_service=SimpleNamespace(source_descriptor=lambda: {}),
                user_memory_service=SimpleNamespace(load=lambda _uid: None),
                build_trace_context_fn=lambda **kwargs: kwargs,
                query_trace_context_fn=lambda **_kwargs: _NullContext(),
                settings=Settings(backend_query_timeout_sec=30),
                csv_runtime=CSVSessionRuntime.__new__(CSVSessionRuntime),
                manifest_store=ManifestStore.__new__(ManifestStore),
                storage_dir=Path("."),
                llm_text_collector_cls=lambda: object(),
                tool_collector_cls=_ToolCollector,
                agent_runner_cls=_FakeRunner,
                effective_enabled_tool_keys_fn=lambda _catalog: {
                    "planner_tool",
                    "forecast_tool",
                },
                build_tool_catalog_fn=lambda **_kwargs: [],
                mcp_service=mcp_service,
            )
        )

        anyio_run(
            service.execute(
                QueryExecutionRequest(
                    session_id="session-1",
                    payload=QueryRequest(
                        query="analyze",
                        requested_tool_key="mcp__finance_research__portfolio_risk",
                    ),
                    current_user=AuthUser(id=7, username="user", is_admin=False, created_at="now"),
                    persist=False,
                )
            )
        )

        runner = _FakeRunner.instances[0]
        self.assertIn("mcp__finance_research__portfolio_risk", runner.kwargs["allowed_tool_keys"])
        self.assertIn("mcp__chronos__forecast", runner.kwargs["allowed_tool_keys"])
        self.assertNotIn("mcp__offline_research__hidden_tool", runner.kwargs["allowed_tool_keys"])
        self.assertEqual(
            calls["request"].requested_tool_key,
            "mcp__finance_research__portfolio_risk",
        )
        self.assertIn("forecast_tool", runner.kwargs["allowed_tool_keys"])
        self.assertEqual(
            [tool.tool_name for tool in runner.kwargs["mcp_tool_descriptors"]],
            ["portfolio.risk", "forecast"],
        )
        self.assertEqual(provider.list_tool_calls, ["finance-research", "chronos"])


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exc: object) -> None:
        return None


class _ToolCollector:
    def __init__(self, **_kwargs) -> None:
        self.tool_names: list[str] = []
        self.tool_calls = 0
        self.events: list[dict[str, object]] = []


def _user_settings() -> UserSettings:
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
    )


def anyio_run(awaitable):
    import anyio

    return anyio.run(lambda: awaitable)
