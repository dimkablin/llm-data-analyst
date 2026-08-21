from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Any, Literal

import pandas as pd
from langchain_core.messages import HumanMessage

from backend.agent.context_manager import AgentContextBuilder, AgentContextRequest
from backend.agent.dependencies import AgentRuntimeDependencies
from backend.agent.graph.builder import build_query_graph
from backend.agent.llm_client import AnyReasoningLLM
from backend.agent.models import AgentOutcome, AgentResponse, ErrorCategory, QueryCacheEntry
from backend.agent.runtime_contracts import AgentRunRequest, AgentRunResult
from backend.agent.runtime_llm import build_runtime_llm
from backend.agent.services.finalization import fallback_text
from backend.agent.services.message_builder import truncate
from backend.agent.services.runtime_effects import (
    RuntimeEffectsBuilder,
    RuntimeEffectsRequest,
)
from backend.artifacts.bridge import execution_data_is_complete
from backend.artifacts.execution import ExecutionArtifact
from backend.auth.user_memory import UserMemory
from backend.core import redis_cache
from backend.core.config import DEPTH_PROFILES, Settings
from backend.core.config import settings as default_settings
from backend.data_access.db_runtime_service import DBRuntimeService
from backend.domain_extensions import DomainExtensionRegistry, get_domain_extension_registry
from backend.integrations.anomaly_planfact import AnomalyPlanfactIntegrationService
from backend.integrations.forecast import ForecastIntegrationService
from backend.integrations.rag import RAGService
from backend.observability.phoenix import record_llm_usage_on_active_span
from backend.sessions.session_memory import SessionArtifactRef, SessionMemory, StructuredSessionMemory
from backend.skills import SkillRegistry
from backend.tools.policy import (
    normalize_allowed_tool_keys,
)
from backend.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)
_QUERY_CACHE_PREFIX = "agent:query:"

_INFRASTRUCTURE_FINDING_TOOL_NAMES = frozenset(
    {"database_tool", "planner_tool", "get_tool_instructions", "update_plan"}
)


def _extract_findings_from_actions(
    actions: list[str],
    turn_index: int,
) -> list[str]:
    findings: list[str] = []
    for action in actions:
        clean_action = str(action or "").strip()
        tool_name = clean_action.split("→", 1)[0].strip()
        if tool_name in _INFRASTRUCTURE_FINDING_TOOL_NAMES:
            continue
        if clean_action:
            findings.append(f"[turn {turn_index}] {clean_action}")
    return findings


def _durable_artifact_refs(
    artifacts: list[Any],
    handles: list[Any],
    *,
    turn_index: int,
) -> list[SessionArtifactRef]:
    summaries = {handle.name: handle.summary for handle in handles}
    refs: list[SessionArtifactRef] = []
    for artifact in artifacts:
        if not isinstance(artifact, ExecutionArtifact):
            continue
        if not execution_data_is_complete(artifact) or not artifact.reusable:
            continue
        schema = artifact.schema or artifact.build_schema()
        refs.append(
            SessionArtifactRef(
                id=str(artifact.id),
                name=str(artifact.name),
                type="table",
                turn_index=turn_index,
                schema=dict(schema.dtypes) if schema is not None else None,
                row_count=int(schema.row_count) if schema is not None else None,
                summary=summaries.get(artifact.name),
                producer_tool=str(artifact.producer_tool or "") or None,
                parent_ids=list(artifact.parent_ids),
            )
        )
    return refs


class AgentRunner:
    def __init__(
        self,
        settings: Settings | None = None,
        db_runtime_service: DBRuntimeService | None = None,
        forecast_service: ForecastIntegrationService | None = None,
        anomaly_planfact_service: AnomalyPlanfactIntegrationService | None = None,
        rag_service: RAGService | None = None,
        allowed_tool_keys: set[str] | None = None,
        user_memory: UserMemory | None = None,
        session_memory: SessionMemory | None = None,
        skill_registry: SkillRegistry | None = None,
        domain_extension_registry: DomainExtensionRegistry | None = None,
        enabled_analytical_skill_ids: set[str] | None = None,
        mcp_tool_provider: Any | None = None,
        mcp_server_configs: dict[str, Any] | None = None,
        mcp_tool_descriptors: list[Any] | None = None,
        semantic_catalog_service: Any | None = None,
        semantic_generation_service: Any | None = None,
        manifest_store: Any | None = None,
        session_store: Any | None = None,
        blob_store: Any | None = None,
        override_store: Any | None = None,
    ) -> None:
        self.settings = settings or default_settings
        self.enabled_analytical_skill_ids = (
            set(enabled_analytical_skill_ids) if enabled_analytical_skill_ids is not None else None
        )
        self.db_runtime_service = db_runtime_service
        self.forecast_service = forecast_service
        self.anomaly_planfact_service = anomaly_planfact_service
        self.rag_service = rag_service
        self.allowed_tool_keys = normalize_allowed_tool_keys(allowed_tool_keys)
        self.mcp_tool_provider = mcp_tool_provider
        self.mcp_server_configs = dict(mcp_server_configs or {})
        self.mcp_tool_descriptors = list(mcp_tool_descriptors or [])
        self.semantic_catalog_service = semantic_catalog_service
        self.semantic_generation_service = semantic_generation_service
        self.manifest_store = manifest_store
        self.session_store = session_store
        self.blob_store = blob_store
        self.user_memory: UserMemory = user_memory or UserMemory(profile="", notes="")
        self.session_memory: SessionMemory = session_memory or SessionMemory()
        self._user_memory_buffer: list[str] = []
        self._session_memory_buffer: list[str] = []
        self.skill_registry = skill_registry or SkillRegistry.from_path(self.settings.skills_dir)
        if override_store is not None:
            self.skill_registry.override_store = override_store
        self.skill_registry.load()
        self.domain_extension_registry = domain_extension_registry or get_domain_extension_registry()
        self._tool_registry = ToolRegistry.from_services(
            forecast_service=forecast_service,
            anomaly_planfact_service=anomaly_planfact_service,
            rag_service=rag_service,
            memory_note_callback=self._user_memory_buffer.append,
            session_note_callback=self._session_memory_buffer.append,
            skill_registry=self.skill_registry,
            mcp_tool_provider=self.mcp_tool_provider,
            mcp_server_configs=self.mcp_server_configs,
            mcp_tool_descriptors=self.mcp_tool_descriptors,
            semantic_catalog_service=self.semantic_catalog_service,
            semantic_generation_service=self.semantic_generation_service,
        )
        self._depth_profile = self._resolve_depth_profile()
        self.dependencies = AgentRuntimeDependencies(
            settings=self.settings,
            tool_registry=self._tool_registry,
            skill_registry=self.skill_registry,
            domain_extension_registry=self.domain_extension_registry,
            user_memory=self.user_memory,
            session_memory=self.session_memory,
            depth_profile=self._depth_profile,
            db_runtime_service=self.db_runtime_service,
            forecast_service=self.forecast_service,
            anomaly_planfact_service=self.anomaly_planfact_service,
            rag_service=self.rag_service,
            semantic_catalog_service=self.semantic_catalog_service,
            semantic_generation_service=self.semantic_generation_service,
            manifest_store=self.manifest_store,
            session_store=self.session_store,
            blob_store=self.blob_store,
            allowed_tool_keys=self.allowed_tool_keys,
            enabled_analytical_skill_ids=self.enabled_analytical_skill_ids,
            mcp_tool_provider=self.mcp_tool_provider,
            mcp_server_configs=self.mcp_server_configs,
            mcp_tool_descriptors=self.mcp_tool_descriptors,
        )
        self.dependencies.context_builder = AgentContextBuilder(
            dependencies=self.dependencies,
        )
        self._query_cache: OrderedDict[str, QueryCacheEntry] = OrderedDict()
        self._graph = build_query_graph(self.dependencies).compile()
        self._runtime_effects_builder = RuntimeEffectsBuilder()

    def _resolve_depth_profile(self) -> dict[str, Any]:
        depth = self.settings.agent_analysis_depth
        return DEPTH_PROFILES.get(depth, DEPTH_PROFILES["light"])

    # ── Utility: LLM / data context ──────────────────────────────────────────

    def _build_llm(
        self,
        *,
        role: Literal["chat", "tool"],
        include_reasoning: bool,
        timeout_sec: int | None = None,
        max_tokens_override: int | None = None,
    ) -> AnyReasoningLLM:
        return build_runtime_llm(
            self.settings,
            role=role,
            include_reasoning=include_reasoning,
            timeout_sec=timeout_sec,
            max_tokens_override=max_tokens_override,
        )

    # ── Utility: history / artifacts ─────────────────────────────────────────

    # -- Utility: cache --

    def _dataset_signature(self, df: pd.DataFrame | None) -> str:
        if df is None:
            return "no-dataset"
        head = df.head(6).to_csv(index=False)
        tail = df.tail(6).to_csv(index=False)
        columns = ",".join(str(c) for c in df.columns[:64])
        payload = f"{df.shape}|{columns}|{head}|{tail}"
        return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()

    def _history_cache_signature(
        self,
        history: list[dict[str, Any]],
        use_history: bool,
    ) -> str:
        if not use_history:
            return "no-history"
        max_msgs = max(0, self.settings.agent_history_max_messages)
        recent = history[-max_msgs:] if max_msgs > 0 else []
        normalized = [
            {
                "role": str(item.get("role", "assistant")),
                "content": truncate(str(item.get("content", "")), 220),
            }
            for item in recent
        ]
        payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()

    def _query_cache_key(
        self,
        *,
        df: pd.DataFrame | None,
        prompt: str,
        history: list[dict[str, Any]],
        use_history: bool,
        include_reasoning: bool,
        selected_skill_ids: list[str] | None = None,
        requested_tool_key: str | None = None,
        trace_context: dict[str, Any] | None = None,
        session_source: dict[str, Any] | None = None,
        registry_snapshot_fingerprint: str = "",
    ) -> str:
        source_json = json.dumps(
            session_source or {},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        payload = {
            "model": self.settings.llm_model,
            "dataset": self._dataset_signature(df),
            "user_id": (trace_context or {}).get("user_id"),
            "session_id": (trace_context or {}).get("session_id"),
            "source": hashlib.sha1(source_json.encode("utf-8", errors="ignore")).hexdigest(),
            "prompt": truncate(prompt, 600),
            "history": self._history_cache_signature(history, use_history),
            "use_history": bool(use_history),
            "include_reasoning": bool(include_reasoning),
            "analysis_depth": str(self.settings.agent_analysis_depth or "light"),
            "selected_skill_ids": list(selected_skill_ids or []),
            "requested_tool_key": requested_tool_key,
            "registry_snapshot": registry_snapshot_fingerprint,
            "max_steps": self.settings.agent_max_steps,
            "anomaly_check_enabled": self.settings.anomaly_check_enabled,
            "always_use_analysis_plan": self.settings.always_use_analysis_plan,
            "step_timeout": self.settings.agent_step_timeout_sec,
            "inner_recursion_limit": self.settings.agent_inner_recursion_limit,
            "max_tools_per_cycle": self.settings.max_tools_per_cycle,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()

    def _cache_get(self, key: str) -> AgentResponse | None:
        if not self.settings.agent_cache_enabled:
            return None
        entry = self._query_cache.get(key)
        if entry is None:
            entry = redis_cache.get_pickle(f"{_QUERY_CACHE_PREFIX}{key}")
            if not isinstance(entry, QueryCacheEntry):
                return None
            self._query_cache[key] = entry
            self._trim_query_cache()
        age_sec = time.time() - entry.created_at
        if age_sec > max(1, self.settings.agent_cache_ttl_sec):
            self._query_cache.pop(key, None)
            redis_cache.delete(f"{_QUERY_CACHE_PREFIX}{key}")
            return None
        self._query_cache.move_to_end(key)
        logger.debug("agent query cache hit key=%s", key)
        return copy.deepcopy(entry.response)

    def _cache_set(self, key: str, response: AgentResponse) -> None:
        if not self.settings.agent_cache_enabled:
            return
        self._query_cache[key] = QueryCacheEntry(created_at=time.time(), response=copy.deepcopy(response))
        redis_cache.set_pickle(
            f"{_QUERY_CACHE_PREFIX}{key}",
            self._query_cache[key],
            ttl_sec=self.settings.agent_cache_ttl_sec,
        )
        self._query_cache.move_to_end(key)
        self._trim_query_cache()

    def _trim_query_cache(self) -> None:
        max_size = max(8, self.settings.agent_cache_size)
        while len(self._query_cache) > max_size:
            self._query_cache.popitem(last=False)

    # -- Public API --

    def is_tool_available(
        self,
        tool_key: str,
        *,
        df: pd.DataFrame | None,
        session_source: dict[str, Any],
        trace_context: dict[str, Any],
    ) -> bool:
        prepared = self.dependencies.context_builder.build(
            AgentContextRequest(
                state={
                    "df": df,
                    "prompt": "",
                    "history": [],
                    "trace_context": trace_context,
                    "session_source": session_source,
                    "selected_skill_ids": [],
                }
            )
        )
        return any(getattr(tool, "name", None) == tool_key for tool in prepared.state_update.get("tools", []))

    def run_query(
        self,
        df: pd.DataFrame | None,
        prompt: str,
        history: list[dict[str, Any]],
        use_history: bool,
        include_reasoning: bool,
        callbacks: list,
        trace_context: dict[str, Any] | None = None,
        session_source: dict[str, Any] | None = None,
        selected_skill_ids: list[str] | None = None,
        requested_tool_key: str | None = None,
    ) -> AgentResponse:
        return self.run(
            AgentRunRequest(
                df=df,
                prompt=prompt,
                history=history,
                use_history=use_history,
                include_reasoning=include_reasoning,
                callbacks=callbacks,
                trace_context=trace_context or {},
                session_source=session_source or {},
                selected_skill_ids=selected_skill_ids or [],
                requested_tool_key=requested_tool_key,
            )
        ).response

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        df = request.df
        prompt = request.prompt
        history = request.history
        use_history = request.use_history
        include_reasoning = request.include_reasoning
        callbacks = request.callbacks
        trace_context = request.trace_context
        session_source = request.session_source
        selected_skill_ids = request.selected_skill_ids
        requested_tool_key = request.requested_tool_key
        cancel_event = request.cancel_event

        resolved_skill_ids = [
            skill.skill_id for skill in self.skill_registry.resolve_selection(selected_skill_ids)
        ]
        request_kind = str((trace_context or {}).get("request_kind", "")).strip().lower()
        # Cache only for persistent queries (/query endpoint).
        # /evaluate (persist=False) and /stream both bypass the cache:
        # evaluate is designed for preview without side-effects, and stream is real-time.
        cache_allowed = self.settings.agent_cache_enabled and request_kind == "query"

        try:
            prepared = self._prepare_request(
                request.model_copy(update={"selected_skill_ids": resolved_skill_ids})
            )
        except Exception:
            logger.exception("agent context preparation failed for prompt=%r", prompt[:60])
            return AgentRunResult(
                response=AgentResponse(
                    final_text=fallback_text(prompt, df),
                    reasoning="Agent context preparation failed.",
                    artifacts=[],
                    route="analysis",
                    outcome=AgentOutcome.failed(ErrorCategory.INTERNAL),
                )
            )
        snapshot = prepared.state_update.get("registry_snapshot")
        snapshot_fingerprint = str(getattr(snapshot, "fingerprint", "") or "")
        cache_key = self._query_cache_key(
            df=df,
            prompt=prompt,
            history=history,
            use_history=use_history,
            include_reasoning=include_reasoning,
            selected_skill_ids=resolved_skill_ids,
            requested_tool_key=requested_tool_key,
            trace_context=trace_context,
            session_source=session_source,
            registry_snapshot_fingerprint=snapshot_fingerprint,
        )
        if cache_allowed:
            cached = self._cache_get(cache_key)
            if cached is not None:
                return AgentRunResult(response=cached)

        # Graph: prepare → agent → finalize (3 supersteps max).
        graph_state = {
            "df": df,
            "prompt": prompt,
            "history": history,
            "use_history": use_history,
            "include_reasoning": include_reasoning,
            "callbacks": callbacks,
            "trace_context": trace_context or {},
            "session_source": session_source or {},
            "selected_skill_ids": resolved_skill_ids,
            "requested_tool_key": requested_tool_key,
            "cancel_event": cancel_event,
        }
        graph_state.update(prepared.state_update)
        try:
            result = self._graph.invoke(
                graph_state,
                config={"recursion_limit": 20},
            )
        except Exception:
            logger.exception("graph.invoke failed for prompt=%r", prompt[:60])
            fallback = AgentResponse(
                final_text=fallback_text(prompt, df),
                reasoning=None,
                artifacts=[],
                route="analysis",
                outcome=AgentOutcome.failed(ErrorCategory.GRAPH),
            )
            fallback.runtime_effects = self._runtime_effects_builder.build(
                RuntimeEffectsRequest(
                    session_memory=self.session_memory,
                    user_memory_notes=self._user_memory_buffer,
                    session_memory_notes=self._session_memory_buffer,
                )
            )
            return AgentRunResult(response=fallback)

        response = result.get("response")

        if not isinstance(response, AgentResponse):
            response = AgentResponse(
                final_text=fallback_text(prompt, df),
                reasoning=None,
                artifacts=[],
                route="analysis",
                outcome=AgentOutcome.failed(ErrorCategory.GRAPH),
            )

        # Flush working_memory → StructuredSessionMemory (Task 4)
        # Only runs on a valid AgentResponse to avoid incrementing turn_count on failure.
        working_memory = result.get("working_memory")
        if working_memory is not None and isinstance(self.session_memory, StructuredSessionMemory):
            structured = self.session_memory
            existing_ids = {r.id for r in structured.artifact_index}
            for ref in _durable_artifact_refs(
                response.artifacts,
                working_memory.artifact_handles,
                turn_index=structured.turn_count,
            ):
                if ref.id in existing_ids:
                    continue
                structured.artifact_index.append(ref)
                existing_ids.add(ref.id)
            # Cap artifact_index at 100 entries (oldest evicted)
            structured.artifact_index = structured.artifact_index[-100:]
            new_findings = _extract_findings_from_actions(
                working_memory.completed_actions,
                turn_index=structured.turn_count,
            )
            structured.key_findings = (structured.key_findings + new_findings)[-30:]
            structured.turn_count += 1

        if cache_allowed and response.outcome.cacheable_success and not response.llm_unreachable:
            self._cache_set(cache_key, response)
        response.runtime_effects = self._runtime_effects_builder.build(
            RuntimeEffectsRequest(
                session_memory=self.session_memory,
                user_memory_notes=self._user_memory_buffer,
                session_memory_notes=self._session_memory_buffer,
            )
        )
        return AgentRunResult(response=response)

    def _prepare_request(self, request: AgentRunRequest):
        if self.dependencies.context_builder is None:
            raise RuntimeError("Agent context builder is not configured")
        return self.dependencies.context_builder.build(AgentContextRequest(state=request.model_dump()))

    def warmup(self) -> None:
        if not self.settings.llm_warmup_enabled:
            return
        try:
            llm = self._build_llm(
                role="chat",
                include_reasoning=False,
                timeout_sec=max(3, self.settings.llm_warmup_timeout_sec),
            )
            response = llm.invoke([HumanMessage(content="ping")])
            record_llm_usage_on_active_span(
                response,
                fallback_model=self.settings.llm_model,
                fallback_provider=self.settings.llm_provider,
            )
        except Exception:
            # Warmup is best-effort; backend stays available even if model is cold.
            return
