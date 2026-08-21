from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, SkipValidation

from backend.agent.context_manager import AgentContextBuilder
from backend.auth.user_memory import UserMemory
from backend.core.config import Settings
from backend.data_access.db_runtime_service import DBRuntimeService
from backend.domain_extensions import DomainExtensionRegistry
from backend.integrations.anomaly_planfact import AnomalyPlanfactIntegrationService
from backend.integrations.forecast import ForecastIntegrationService
from backend.integrations.rag import RAGService
from backend.mcp.models import MCPServerConfig, MCPToolDescriptor
from backend.mcp.service import MCPToolProvider
from backend.sessions.session_memory import SessionMemory
from backend.skills import SkillRegistry
from backend.tools.registry import ToolRegistry


class AgentRuntimeDependencies(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    settings: Settings
    tool_registry: ToolRegistry
    skill_registry: SkillRegistry
    domain_extension_registry: DomainExtensionRegistry
    user_memory: UserMemory
    session_memory: SessionMemory
    depth_profile: dict[str, Any]
    context_builder: AgentContextBuilder | None = None
    db_runtime_service: SkipValidation[DBRuntimeService | None] = None
    forecast_service: SkipValidation[ForecastIntegrationService | None] = None
    anomaly_planfact_service: SkipValidation[AnomalyPlanfactIntegrationService | None] = None
    rag_service: SkipValidation[RAGService | None] = None
    semantic_catalog_service: SkipValidation[Any | None] = None
    semantic_generation_service: SkipValidation[Any | None] = None
    manifest_store: SkipValidation[Any | None] = None
    session_store: SkipValidation[Any | None] = None
    blob_store: SkipValidation[Any | None] = None
    allowed_tool_keys: set[str] | None = None
    enabled_analytical_skill_ids: set[str] | None = None
    mcp_tool_provider: SkipValidation[MCPToolProvider | None] = None
    mcp_server_configs: SkipValidation[dict[str, MCPServerConfig] | None] = None
    mcp_tool_descriptors: SkipValidation[list[MCPToolDescriptor] | None] = None
