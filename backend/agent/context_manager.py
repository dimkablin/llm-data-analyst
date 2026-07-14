from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.agent.contracts import AnalysisTaskContract
from backend.agent.models import AgentResponse
from backend.agent.services.runtime_context import (
    csv_table_descriptors_from_manifest,
    is_rag_session_source,
    resolve_csv_runtime_state,
    resolve_tool_db_runtime_config,
)
from backend.agent.working_memory import AnalysisWorkingMemory
from backend.data_access.csv_session_runtime import CSVSessionRuntime
from backend.data_access.data_catalog import catalog_cache_key
from backend.data_access.source_inventory import (
    build_source_inventory,
    format_source_inventory_prompt,
)
from backend.notebook.manifest_store import ManifestStore
from backend.skills.contracts import (
    SkillExecutionRequirement,
    SkillPermissionValidationResult,
    validate_skill_tool_permissions,
)
from backend.tools.capabilities import build_runtime_capability_context
from backend.tools.context import ToolBuildContext
from backend.tools.sandbox_manager import SandboxManager

logger = logging.getLogger(__name__)

ContextBudgetStrategy = Literal["disabled", "token_limit"]
ContextBudgetStatus = Literal["disabled", "not_configured", "planned"]
ContextRetrievalStrategy = Literal["disabled", "vector_similarity"]

_HIDDEN_FROM_AGENT: frozenset[str] = frozenset()


class ContextBudgetPolicy(BaseModel):
    """Configuration contract for future prompt-length control."""

    strategy: ContextBudgetStrategy = "disabled"
    max_context_tokens: int | None = Field(default=None, ge=1)
    reserved_response_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_strategy(self) -> ContextBudgetPolicy:
        if self.strategy == "token_limit" and self.max_context_tokens is None:
            raise ValueError("max_context_tokens is required for token_limit strategy")
        return self


class ContextBudget(BaseModel):
    """Budget metadata produced while preparing a graph turn."""

    strategy: ContextBudgetStrategy = "disabled"
    status: ContextBudgetStatus = "disabled"
    max_context_tokens: int | None = Field(default=None, ge=1)
    reserved_response_tokens: int = Field(default=0, ge=0)
    estimated_context_tokens: int | None = Field(default=None, ge=0)
    usage_ratio: float | None = Field(default=None, ge=0.0)
    usage_percent: int | None = Field(default=None, ge=0)
    remaining_context_tokens: int | None = Field(default=None, ge=0)
    truncated_messages_count: int = Field(default=0, ge=0)
    overflow: bool = False


class RetrievedContextMessage(BaseModel):
    """A chat-history message selected by future semantic retrieval."""

    role: str
    content: str
    score: float | None = None
    source_index: int | None = Field(default=None, ge=0)


class ContextRetrievalPolicy(BaseModel):
    """Configuration contract for future semantic chat-history retrieval."""

    strategy: ContextRetrievalStrategy = "disabled"
    top_k: int = Field(default=0, ge=0, le=50)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    provider: str | None = None

    @model_validator(mode="after")
    def _validate_strategy(self) -> ContextRetrievalPolicy:
        if self.strategy == "vector_similarity" and self.top_k <= 0:
            raise ValueError("top_k must be positive for vector_similarity strategy")
        return self


class ContextRetrievalResult(BaseModel):
    """Semantic retrieval metadata produced while preparing a graph turn."""

    strategy: ContextRetrievalStrategy = "disabled"
    query: str = ""
    requested_top_k: int = Field(default=0, ge=0)
    provider: str | None = None
    skipped_reason: str | None = None
    messages: list[RetrievedContextMessage] = Field(default_factory=list)


class AgentContextManagerConfig(BaseModel):
    """Public context-manager configuration for budget and retrieval policies."""

    budget_policy: ContextBudgetPolicy = Field(default_factory=ContextBudgetPolicy)
    retrieval_policy: ContextRetrievalPolicy = Field(default_factory=ContextRetrievalPolicy)


class AgentContextRequest(BaseModel):
    """Input snapshot consumed by the agent context builder."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    state: dict[str, Any]
    context_config: AgentContextManagerConfig | None = None


class AgentPreparedContext(BaseModel):
    """Prepared graph state plus future context-management metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    state_update: dict[str, Any]
    context_budget: ContextBudget = Field(default_factory=ContextBudget)
    retrieved_context: ContextRetrievalResult = Field(default_factory=ContextRetrievalResult)


class _RuntimeSourceResolver(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    db_runtime_service: Any | None = None

    def resolve(
        self,
        *,
        session_source: dict[str, Any] | None,
        trace_context: dict[str, Any] | None,
    ) -> Any | None:
        return resolve_tool_db_runtime_config(
            db_runtime_service=self.db_runtime_service,
            session_source=session_source,
            trace_context=trace_context,
        )


class _SkillExecutionGate(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dependencies: Any

    def resolve_requirements(
        self,
        *,
        selected_skill_ids: list[str] | tuple[str, ...] | None,
    ) -> tuple[SkillExecutionRequirement, ...]:
        return self.dependencies.skill_registry.execution_requirements_for_prompt(
            selected_skill_ids=selected_skill_ids,
            enabled_skill_ids=self.dependencies.enabled_analytical_skill_ids,
        )

    def validate(
        self,
        requirements: tuple[SkillExecutionRequirement, ...],
    ) -> SkillPermissionValidationResult:
        return validate_skill_tool_permissions(
            requirements,
            self.dependencies.allowed_tool_keys,
        )

    @staticmethod
    def denied_response(result: SkillPermissionValidationResult) -> AgentResponse:
        reason = (result.reason or "skill required tools are disabled").strip()
        tools = ", ".join(f"`{tool}`" for tool in result.missing_tool_keys)
        tools_text = tools or "из execution contract навыка"
        return AgentResponse(
            final_text=(
                f"Не могу выполнить запрос: необходимый tool {tools_text} "
                f"выключен или недоступен. Детали: {reason}."
            ),
            reasoning=f"Skill permission denied: {reason}",
            artifacts=[],
            route="analysis",
        )


class _ContextBudgetPlanner(BaseModel):
    def plan(
        self,
        request: AgentContextRequest,
        policy: ContextBudgetPolicy,
    ) -> ContextBudget:
        del request
        if policy.strategy == "disabled":
            return ContextBudget()
        return ContextBudget(
            strategy=policy.strategy,
            status="not_configured",
            max_context_tokens=policy.max_context_tokens,
            reserved_response_tokens=policy.reserved_response_tokens,
        )


class _RelevantHistoryRetriever(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    retriever: Any | None = None

    def retrieve(
        self,
        request: AgentContextRequest,
        policy: ContextRetrievalPolicy,
    ) -> ContextRetrievalResult:
        prompt = str(request.state.get("prompt") or "")
        if policy.strategy == "disabled":
            return ContextRetrievalResult(query=prompt)
        if self.retriever is None:
            return ContextRetrievalResult(
                strategy=policy.strategy,
                query=prompt,
                requested_top_k=policy.top_k,
                provider=policy.provider,
                skipped_reason="retriever_not_configured",
            )
        raw_messages = self.retriever.retrieve(
            query=prompt,
            history=request.state.get("history") or [],
            top_k=policy.top_k,
            min_score=policy.min_score,
        )
        return ContextRetrievalResult(
            strategy=policy.strategy,
            query=prompt,
            requested_top_k=policy.top_k,
            provider=policy.provider,
            messages=[
                message
                if isinstance(message, RetrievedContextMessage)
                else RetrievedContextMessage.model_validate(message)
                for message in raw_messages or []
            ],
        )


class _ToolContextBuilder(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dependencies: Any

    def build(
        self,
        *,
        state: dict[str, Any],
        df: Any,
        tool_df: Any,
        tool_db_runtime: Any,
        csv_loaded: bool,
        csv_session_id: str | None,
        trace_context: dict[str, Any],
        sandbox: Any,
        source_inventory: Any,
    ) -> tuple[list[Any], str]:
        session_id = trace_context.get("session_id", "default")
        source_context = state.get("session_source") or {}
        candidates_key = catalog_cache_key(
            session_id=str(session_id),
            source_type=source_context.get("source_type"),
            source_ref_id=source_context.get("source_ref_id"),
            csv_session_id=csv_session_id,
        )
        tool_context = ToolBuildContext(
            settings=self.dependencies.settings,
            allowed_tool_keys=self.dependencies.allowed_tool_keys,
            allowed_skill_ids=self.dependencies.enabled_analytical_skill_ids,
            df=tool_df,
            tool_db_runtime=tool_db_runtime,
            csv_loaded=csv_loaded,
            csv_session_id=csv_session_id,
            sandbox=sandbox,
            candidates_cache_key=candidates_key,
            source_inventory=source_inventory,
            semantic_context_prompt=str(source_context.get("semantic_context_prompt") or ""),
            semantic_hints=dict(source_context.get("semantic_context_hints") or {}),
            history=state.get("history") or [],
            session_notes=str(getattr(self.dependencies.session_memory, "notes", "") or ""),
            trace_context=trace_context,
        )
        del df
        tools = self.dependencies.tool_registry.build_tools(tool_context)
        tool_descriptions = self.dependencies.tool_registry.describe_available_tools(
            tool_context
        )
        self._configure_planner_descriptions(tools, tool_descriptions)
        return tools, tool_descriptions

    def _configure_planner_descriptions(
        self,
        tools: list[Any],
        tool_descriptions: str,
    ) -> None:
        planner_descriptions = "\n".join(
            line for line in tool_descriptions.splitlines()
            if "planner_tool" not in line
        ).strip()
        analytical_block = (
            self.dependencies.skill_registry.build_analytical_skills_brief_block(
                enabled_skill_ids=self.dependencies.enabled_analytical_skill_ids,
            )
        )
        if analytical_block:
            planner_descriptions = planner_descriptions + "\n\n" + analytical_block
        for tool in tools:
            if hasattr(tool, "set_tool_descriptions"):
                tool.set_tool_descriptions(planner_descriptions)


class _CapabilityContextBuilder(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dependencies: Any

    def build(
        self,
        *,
        state: dict[str, Any],
        tools: list[Any],
        tool_descriptions: str,
        tool_df: Any,
        tool_db_runtime: Any,
        csv_duckdb_mode: bool,
        csv_session_id: str | None,
        session_id: str,
        source_inventory: Any,
        source_context: dict[str, Any],
    ) -> dict[str, Any]:
        tool_keys = [
            str(getattr(tool, "name", "")).strip()
            for tool in tools
            if str(getattr(tool, "name", "")).strip()
            and str(getattr(tool, "name", "")).strip() not in _HIDDEN_FROM_AGENT
        ]
        csv_table_names = list((state.get("session_source") or {}).get("csv_table_names") or [])
        csv_table_descriptors = csv_table_descriptors_from_manifest(
            storage_dir=self.dependencies.settings.storage_dir,
            csv_session_id=csv_session_id,
            session_id=session_id,
            session_source=state.get("session_source") or {},
        )
        capability_context = build_runtime_capability_context(
            available_tool_keys=tool_keys,
            has_dataframe=tool_df is not None,
            has_db_source=(tool_db_runtime is not None) or csv_duckdb_mode,
            has_knowledge_base=is_rag_session_source(source_context),
            csv_table_names=csv_table_names or None,
            csv_table_descriptors=csv_table_descriptors or None,
            source_table_count=(
                len(source_inventory.tables) if source_inventory is not None else 0
            ),
            source_count=(
                len(source_inventory.sources) if source_inventory is not None else 0
            ),
        )
        if source_inventory is not None:
            inventory_prompt = format_source_inventory_prompt(source_inventory)
            if inventory_prompt:
                capability_context["source_inventory"] = source_inventory.model_dump()
                capability_context["prompt_block"] = (
                    str(capability_context.get("prompt_block") or "").rstrip()
                    + "\n"
                    + inventory_prompt
                ).strip()
        description_lines = [
            line for line in tool_descriptions.splitlines()
            if not any(("`" + key + "`") in line for key in _HIDDEN_FROM_AGENT)
        ]
        capability_context["tool_descriptions"] = "\n".join(description_lines).strip()
        return capability_context


class AgentContextBuilder(BaseModel):
    """Build prepared state for the generic LangGraph runtime."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dependencies: Any
    context_config: AgentContextManagerConfig = Field(
        default_factory=AgentContextManagerConfig
    )
    budget_planner: _ContextBudgetPlanner = Field(default_factory=_ContextBudgetPlanner)
    history_retriever: _RelevantHistoryRetriever = Field(
        default_factory=_RelevantHistoryRetriever
    )

    def build(self, request: AgentContextRequest) -> AgentPreparedContext:
        state = dict(request.state)
        df = state.get("df")
        prompt = state.get("prompt", "")
        trace_context = state.get("trace_context") or {}
        session_source = dict(state.get("session_source") or {})
        context_config = request.context_config or self.context_config

        context_budget = self.budget_planner.plan(
            request,
            context_config.budget_policy,
        )
        retrieved_context = self.history_retriever.retrieve(
            request,
            context_config.retrieval_policy,
        )

        tool_db_runtime = _RuntimeSourceResolver(
            db_runtime_service=self.dependencies.db_runtime_service
        ).resolve(
            session_source=session_source,
            trace_context=trace_context,
        )
        csv_loaded, csv_session_id = resolve_csv_runtime_state(
            session_source,
            trace_context,
        )
        session_source = self._ensure_csv_table_descriptors(
            session_source=session_source,
            csv_loaded=csv_loaded,
            csv_session_id=csv_session_id,
            trace_context=trace_context,
        )
        state["session_source"] = session_source

        skill_gate = _SkillExecutionGate(dependencies=self.dependencies)
        skill_requirements = skill_gate.resolve_requirements(
            selected_skill_ids=state.get("selected_skill_ids") or [],
        )
        permission_result = skill_gate.validate(skill_requirements)
        if not permission_result.passed:
            return AgentPreparedContext(
                state_update={
                    "response": skill_gate.denied_response(permission_result),
                    "done": True,
                    "stop_reason": "skill_permission_denied",
                    "skill_execution_requirements": list(skill_requirements),
                    "context_budget": context_budget,
                    "retrieved_context": retrieved_context,
                },
                context_budget=context_budget,
                retrieved_context=retrieved_context,
            )

        prepared = self._prepare_agent_context(
            state=state,
            df=df,
            prompt=prompt,
            trace_context=trace_context,
            tool_db_runtime=tool_db_runtime,
            csv_loaded=csv_loaded,
            csv_session_id=csv_session_id,
            skill_requirements=skill_requirements,
            context_budget=context_budget,
            retrieved_context=retrieved_context,
        )
        return prepared

    def _prepare_agent_context(
        self,
        *,
        state: dict[str, Any],
        df: Any,
        prompt: str,
        trace_context: dict[str, Any],
        tool_db_runtime: Any,
        csv_loaded: bool,
        csv_session_id: str | None,
        skill_requirements: tuple[SkillExecutionRequirement, ...],
        context_budget: ContextBudget,
        retrieved_context: ContextRetrievalResult,
    ) -> AgentPreparedContext:
        csv_duckdb_mode = bool(csv_loaded and str(csv_session_id or "").strip())
        tool_df = None if csv_duckdb_mode else df

        session_id = trace_context.get("session_id", "default")
        sandbox = SandboxManager.get_instance().get_or_create(session_id)
        sandbox.ensure_storage_dir(Path(self.dependencies.settings.storage_dir) / session_id)
        if df is not None:
            source_label = str(trace_context.get("dataset_name", "") or "")
            sandbox.bind_dataframe(
                df,
                source_label=source_label,
                db_runtime_config=tool_db_runtime,
            )

        source_context = state.get("session_source") or {}
        source_inventory = self._build_source_inventory(
            session_id=str(session_id),
            source_context=source_context,
            tool_db_runtime=tool_db_runtime,
        )
        tools, tool_descriptions = _ToolContextBuilder(
            dependencies=self.dependencies
        ).build(
            state=state,
            df=df,
            tool_df=tool_df,
            tool_db_runtime=tool_db_runtime,
            csv_loaded=csv_loaded,
            csv_session_id=csv_session_id,
            trace_context=trace_context,
            sandbox=sandbox,
            source_inventory=source_inventory,
        )
        max_steps = self._max_steps()
        capability_context = _CapabilityContextBuilder(
            dependencies=self.dependencies
        ).build(
            state=state,
            tools=tools,
            tool_descriptions=tool_descriptions,
            tool_df=tool_df,
            tool_db_runtime=tool_db_runtime,
            csv_duckdb_mode=csv_duckdb_mode,
            csv_session_id=csv_session_id,
            session_id=str(session_id),
            source_inventory=source_inventory,
            source_context=source_context,
        )
        task_contract = AnalysisTaskContract.from_prompt(prompt)

        return AgentPreparedContext(
            state_update={
                "prompt": prompt,
                "max_steps": max_steps,
                "done": False,
                "stop_reason": "",
                "tools": tools,
                "step_index": 0,
                "sandbox": sandbox,
                "capability_context": capability_context,
                "task_contract": task_contract,
                "skill_execution_requirements": list(skill_requirements),
                "llm_unreachable": False,
                "tool_db_runtime": tool_db_runtime,
                "working_memory": AnalysisWorkingMemory(goal=prompt),
                "session_source": state.get("session_source") or {},
                "context_budget": context_budget,
                "retrieved_context": retrieved_context,
            },
            context_budget=context_budget,
            retrieved_context=retrieved_context,
        )

    def _max_steps(self) -> int:
        depth_inner_limit = self.dependencies.depth_profile.get("inner_recursion_limit")
        return max(
            1,
            depth_inner_limit if isinstance(depth_inner_limit, int)
            else self.dependencies.settings.agent_inner_recursion_limit,
        )

    def _build_source_inventory(
        self,
        *,
        session_id: str,
        source_context: dict[str, Any],
        tool_db_runtime: Any,
    ) -> Any | None:
        try:
            return build_source_inventory(
                session_id=session_id,
                session_source=source_context,
                manifest_store=ManifestStore(self.dependencies.settings.storage_dir),
                csv_runtime=CSVSessionRuntime(),
                db_runtime=tool_db_runtime,
            )
        except Exception as exc:
            logger.debug("Source inventory build failed: %s", exc)
            return None

    def _ensure_csv_table_descriptors(
        self,
        *,
        session_source: dict[str, Any],
        csv_loaded: bool,
        csv_session_id: str | None,
        trace_context: dict[str, Any],
    ) -> dict[str, Any]:
        if not csv_loaded or session_source.get("csv_table_descriptors"):
            return session_source
        csv_table_descriptors = csv_table_descriptors_from_manifest(
            storage_dir=self.dependencies.settings.storage_dir,
            csv_session_id=csv_session_id,
            session_id=str(trace_context.get("session_id") or "") or None,
            session_source=session_source,
        )
        if not csv_table_descriptors:
            return session_source
        updated = dict(session_source)
        updated["csv_table_descriptors"] = csv_table_descriptors
        return updated
