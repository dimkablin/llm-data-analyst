from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from backend.auth.user_memory import UserMemory
from backend.core.config import Settings
from backend.data_access.db_runtime_service import DBRuntimeService
from backend.integrations.anomaly_planfact import AnomalyPlanfactIntegrationService
from backend.integrations.forecast import ForecastIntegrationService
from backend.integrations.rag import RAGService
from backend.integrations.search import SearchIntegrationService
from backend.sessions.session_memory import SessionMemory
from backend.skills import SkillRegistry
from backend.tools.policy import normalize_allowed_tool_keys
from backend.tools.registry import ToolRegistry


@dataclass(slots=True)
class AgentRuntimeServices:
    """Service container for the LangGraph agent runtime.

    This replaces the old pattern where graph nodes reached back into a large
    runner object.  Dependencies are explicit, typed and replaceable in tests.
    """

    settings: Settings
    tool_registry: ToolRegistry
    skill_registry: SkillRegistry
    db_runtime_service: DBRuntimeService | None = None
    search_service: SearchIntegrationService | None = None
    forecast_service: ForecastIntegrationService | None = None
    anomaly_planfact_service: AnomalyPlanfactIntegrationService | None = None
    rag_service: RAGService | None = None
    allowed_tool_keys: set[str] | None = None
    user_memory: UserMemory | None = None
    session_memory: SessionMemory | None = None
    enabled_analytical_skill_ids: set[str] | None = None
    memory_note_callback: Callable[[str], None] | None = None
    session_note_callback: Callable[[str], None] | None = None

    @classmethod
    def create(
        cls,
        *,
        settings: Settings,
        db_runtime_service: DBRuntimeService | None = None,
        search_service: SearchIntegrationService | None = None,
        forecast_service: ForecastIntegrationService | None = None,
        anomaly_planfact_service: AnomalyPlanfactIntegrationService | None = None,
        rag_service: RAGService | None = None,
        allowed_tool_keys: set[str] | None = None,
        user_memory: UserMemory | None = None,
        session_memory: SessionMemory | None = None,
        skill_registry: SkillRegistry | None = None,
        enabled_analytical_skill_ids: set[str] | None = None,
        memory_note_callback: Callable[[str], None] | None = None,
        session_note_callback: Callable[[str], None] | None = None,
    ) -> AgentRuntimeServices:
        resolved_skill_registry = skill_registry or SkillRegistry.from_path(settings.skills_dir)
        resolved_skill_registry.load()

        normalized_allowed_tool_keys = normalize_allowed_tool_keys(allowed_tool_keys)
        resolved_tool_registry = ToolRegistry.from_services(
            search_service=search_service,
            forecast_service=forecast_service,
            anomaly_planfact_service=anomaly_planfact_service,
            rag_service=rag_service,
            memory_note_callback=memory_note_callback or (lambda _: None),
            session_note_callback=session_note_callback or (lambda _: None),
            skill_registry=resolved_skill_registry,
        )

        return cls(
            settings=settings,
            tool_registry=resolved_tool_registry,
            skill_registry=resolved_skill_registry,
            db_runtime_service=db_runtime_service,
            search_service=search_service,
            forecast_service=forecast_service,
            anomaly_planfact_service=anomaly_planfact_service,
            rag_service=rag_service,
            allowed_tool_keys=normalized_allowed_tool_keys,
            user_memory=user_memory,
            session_memory=session_memory,
            enabled_analytical_skill_ids=(
                set(enabled_analytical_skill_ids)
                if enabled_analytical_skill_ids is not None
                else None
            ),
            memory_note_callback=memory_note_callback,
            session_note_callback=session_note_callback,
        )
