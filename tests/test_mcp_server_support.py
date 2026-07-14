from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, ClassVar

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
    MCPServerConfig,
    MCPServerCreateRequest,
    MCPServerTransport,
    MCPServerUpdateRequest,
    MCPToolDescriptor,
)
from backend.mcp.service import MCPServerService
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
            )
            stored = auth_db.upsert_mcp_server_config(config, updated_by=admin.id)
            auth_db.set_user_mcp_server_enabled(user.id, "finance-research", True)

            self.assertEqual(stored.server_id, "finance-research")
            self.assertEqual(stored.updated_by, admin.id)
            self.assertTrue(auth_db.list_user_mcp_server_settings(user.id)["finance-research"])
            self.assertEqual(auth_db.get_mcp_server_config("finance-research").name, "Finance Research")
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
                "properties": {"ticker": {"type": "string"}},
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

        [tool] = [
            item
            for item in registry.build_tools(context)
            if item.name == descriptor.tool_key
        ]
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
        self.assertEqual(result, "finance-research:portfolio.risk:{'ticker': 'AAPL'}")
        self.assertEqual(
            getattr(tool_call_result, "content", tool_call_result),
            "finance-research:portfolio.risk:{'ticker': 'MSFT'}",
        )

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
        mcp_service = MCPServerService(
            configs=[enabled_config, disabled_config],
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
                search_integration_service=SimpleNamespace(source_descriptor=lambda: {}),
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
                effective_enabled_tool_keys_fn=lambda _catalog: {"planner_tool"},
                build_tool_catalog_fn=lambda **_kwargs: [],
                mcp_service=mcp_service,
            )
        )

        anyio_run(
            service.execute(
                QueryExecutionRequest(
                    session_id="session-1",
                    payload=QueryRequest(query="analyze"),
                    current_user=AuthUser(id=7, username="user", is_admin=False, created_at="now"),
                    persist=False,
                )
            )
        )

        runner = _FakeRunner.instances[0]
        self.assertIn("mcp__finance_research__portfolio_risk", runner.kwargs["allowed_tool_keys"])
        self.assertNotIn("mcp__offline_research__hidden_tool", runner.kwargs["allowed_tool_keys"])
        self.assertEqual([tool.tool_name for tool in runner.kwargs["mcp_tool_descriptors"]], ["portfolio.risk"])
        self.assertEqual(provider.list_tool_calls, ["finance-research"])


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
