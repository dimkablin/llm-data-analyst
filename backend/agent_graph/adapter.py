from __future__ import annotations

import copy
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from backend.agent.llm_client import make_reasoning_llm
from backend.agent_graph.models import AgentGraphRequest
from backend.agent_graph.routing import RouteClassifier
from backend.agent_graph.runner import AgentGraphRunner
from backend.agent_graph.runtime import GraphRuntimeContext
from backend.agent_graph.services import AgentRuntimeServices
from backend.core.config import Settings
from backend.observability.phoenix import record_llm_usage_on_active_span
from backend.tools.policy import (
    detect_data_access_mode,
    has_enabled_data_tools,
    normalize_allowed_tool_keys,
)


@dataclass
class AgentResponse:
    final_text: str
    reasoning: str | None
    artifacts: list
    route: str = "chat"
    tool_calls: int = 0
    tool_names: list[str] = field(default_factory=list)
    llm_unreachable: bool = False
    reasoning_steps: list[str] = field(default_factory=list)


@dataclass
class QueryCacheEntry:
    created_at: float
    response: AgentResponse


@dataclass
class AgentPostRunReport:
    """Result of flushing runtime side effects after a query run."""

    user_memory_notes: int = 0
    session_memory_notes: int = 0
    structured_session_memory_persisted: bool = False
    failed_steps: list[str] = field(default_factory=list)

    @property
    def has_failures(self) -> bool:
        return bool(self.failed_steps)


class AgentGraphQueryRunner:
    """Compatibility boundary between FastAPI query routes and AgentGraphRunner.

    The API layer still expects the old ``run_query`` signature and
    ``AgentResponse`` shape.  This adapter keeps that surface stable while the
    implementation path moves to ``backend.agent_graph``.
    """

    def __init__(
        self,
        settings: Settings,
        db_runtime_service: Any | None = None,
        search_service: Any | None = None,
        forecast_service: Any | None = None,
        anomaly_planfact_service: Any | None = None,
        rag_service: Any | None = None,
        allowed_tool_keys: set[str] | None = None,
        user_memory: Any | None = None,
        session_memory: Any | None = None,
        skill_registry: Any | None = None,
        enabled_analytical_skill_ids: set[str] | None = None,
    ) -> None:
        self.settings = settings
        self.allowed_tool_keys = normalize_allowed_tool_keys(allowed_tool_keys)
        self.user_memory = user_memory
        self.session_memory = session_memory
        self._user_memory_buffer: list[str] = []
        self._session_memory_buffer: list[str] = []
        self.services = AgentRuntimeServices.create(
            settings=settings,
            db_runtime_service=db_runtime_service,
            search_service=search_service,
            forecast_service=forecast_service,
            anomaly_planfact_service=anomaly_planfact_service,
            rag_service=rag_service,
            allowed_tool_keys=self.allowed_tool_keys,
            user_memory=user_memory,
            session_memory=session_memory,
            skill_registry=skill_registry,
            enabled_analytical_skill_ids=enabled_analytical_skill_ids,
            memory_note_callback=self._user_memory_buffer.append,
            session_note_callback=self._session_memory_buffer.append,
        )
        self.skill_registry = self.services.skill_registry
        self._graph_runner = AgentGraphRunner()
        self._route_classifier = RouteClassifier()
        self._query_cache: OrderedDict[str, QueryCacheEntry] = OrderedDict()

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
            return

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
    ) -> AgentResponse:
        resolved_skill_ids = [
            skill.skill_id for skill in self.skill_registry.resolve_selection(selected_skill_ids)
        ]
        request_kind = str((trace_context or {}).get("request_kind", "")).strip().lower()
        cache_allowed = self.settings.agent_cache_enabled and request_kind == "query"
        cache_key = self._query_cache_key(
            df=df,
            prompt=prompt,
            history=history,
            use_history=use_history,
            include_reasoning=include_reasoning,
            selected_skill_ids=resolved_skill_ids,
        )
        if cache_allowed:
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached

        guardrail = self._data_tools_disabled_response(
            df=df,
            prompt=prompt,
            session_source=session_source or {},
        )
        if guardrail is not None:
            if cache_allowed:
                self._cache_set(cache_key, guardrail)
            return guardrail

        result = self._graph_runner.run(
            AgentGraphRequest(
                prompt=prompt,
                history=history,
                use_history=use_history,
                include_reasoning=include_reasoning,
                trace_context=trace_context or {},
                session_source=session_source or {},
                selected_skill_ids=resolved_skill_ids,
                max_steps=self.settings.agent_inner_recursion_limit,
            ),
            runtime_context=GraphRuntimeContext(
                services=self.services,
                df=df,
                callbacks=callbacks,
            ),
        )
        artifacts, tool_calls, tool_names = self._collect_tool_stats(callbacks)
        response = AgentResponse(
            final_text=result.final_text or self._fallback_text(prompt, df),
            reasoning=result.reasoning,
            artifacts=artifacts,
            route=result.route if result.route in {"chat", "summary", "analysis"} else "analysis",
            tool_calls=max(tool_calls, result.tool_calls),
            tool_names=tool_names or result.tool_names,
            reasoning_steps=result.reasoning_steps,
        )
        if cache_allowed:
            self._cache_set(cache_key, response)
        return response

    def process_post_run_effects(
        self,
        *,
        user_id: int,
        session_id: str,
        user_memory_service: Any,
        session_store: Any,
    ) -> AgentPostRunReport:
        """Flush runtime memory side effects through the public adapter contract."""
        report = AgentPostRunReport()
        user_notes = self._clean_notes(self._user_memory_buffer)
        session_notes = self._clean_notes(self._session_memory_buffer)

        if user_notes:
            try:
                memory_llm = self._build_llm(
                    role="chat",
                    include_reasoning=False,
                    max_tokens_override=800,
                )
                user_memory_service.schedule_consolidation(
                    user_id,
                    user_notes,
                    memory_llm.invoke,
                )
                report.user_memory_notes = len(user_notes)
            except Exception:
                report.failed_steps.append("user_memory_consolidation")

        if session_notes:
            try:
                for note in session_notes:
                    session_store.append_session_memory(session_id, note)
                self._merge_session_notes(session_notes)
                report.session_memory_notes = len(session_notes)
            except Exception:
                report.failed_steps.append("session_memory_notes")

        if self.session_memory is not None:
            try:
                session_store.set_structured_memory(session_id, self.session_memory)
                report.structured_session_memory_persisted = True
            except Exception:
                report.failed_steps.append("structured_session_memory")

        self._user_memory_buffer.clear()
        self._session_memory_buffer.clear()
        return report

    def _build_llm(
        self,
        *,
        role: str,
        include_reasoning: bool,
        timeout_sec: int | None = None,
        max_tokens_override: int | None = None,
    ):
        enable_thinking = self.settings.llm_enable_thinking and include_reasoning
        temperature = (
            self.settings.llm_temperature_tool
            if role == "tool"
            else self.settings.llm_temperature_chat
        )
        max_tokens = max_tokens_override or self.settings.llm_max_tokens_default
        return make_reasoning_llm(
            provider=self.settings.llm_provider,
            model=self.settings.llm_model,
            base_url=self.settings.llm_base_url,
            api_key=self.settings.llm_api_key,
            enable_thinking=enable_thinking,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=self.settings.llm_streaming_force or self.settings.llm_streaming,
            timeout=float(timeout_sec or self.settings.backend_query_timeout_sec),
            top_p=self.settings.llm_top_p,
            top_k=self.settings.llm_top_k,
            num_ctx=self.settings.llm_num_ctx,
            presence_penalty=self.settings.llm_presence_penalty,
            chat_template_kwargs_enabled=self.settings.llm_chat_template_kwargs_enabled,
        )

    def generate_chat_title(
        self,
        *,
        dataset_name: str | None,
        user_queries: list[str],
        trace_context: dict[str, Any] | None = None,
    ) -> str | None:
        cleaned_queries = [str(query).strip() for query in user_queries if str(query).strip()]
        if not cleaned_queries:
            return None

        dataset_part = str(dataset_name or "").strip() or "не указан"
        query_lines = "\n".join(
            f"{idx}. {item}" for idx, item in enumerate(cleaned_queries[-8:], start=1)
        )
        messages = [
            SystemMessage(
                content=(
                    "Ты придумываешь названия чатов аналитики данных.\n"
                    "Верни только короткое название на русском языке.\n"
                    "Требования:\n"
                    "- Ровно 3 или 4 слова.\n"
                    "- Без кавычек, без пунктуации в конце, без пояснений.\n"
                    "- Название должно отражать датасет и суть пользовательских запросов."
                ),
            ),
            HumanMessage(
                content=(
                    f"Датасет: {dataset_part}\n"
                    "Запросы пользователя:\n"
                    f"{query_lines}\n\n"
                    "Сформируй название чата."
                ),
            ),
        ]
        runtime_config: dict[str, Any] = {}
        metadata = self._runtime_metadata(trace_context)
        if metadata:
            runtime_config["metadata"] = metadata

        try:
            response = self._build_llm(
                role="chat",
                include_reasoning=False,
                timeout_sec=max(8, min(20, self.settings.backend_query_timeout_sec)),
            ).invoke(messages, config=runtime_config or None)
            record_llm_usage_on_active_span(
                response,
                fallback_model=self.settings.llm_model,
                fallback_provider=self.settings.llm_provider,
            )
            generated = self._content_to_text(getattr(response, "content", ""))
        except Exception:
            generated = ""
        return self._normalize_title_candidate(generated)

    def _data_tools_disabled_response(
        self,
        *,
        df: pd.DataFrame | None,
        prompt: str,
        session_source: dict[str, Any],
    ) -> AgentResponse | None:
        if self._route_classifier.classify(prompt, has_data=df is not None) in {"chat", "summary"}:
            return None

        mode = detect_data_access_mode(
            has_dataframe=df is not None,
            session_source=session_source,
        )
        if mode is None:
            return None
        if has_enabled_data_tools(
            has_dataframe=df is not None,
            session_source=session_source,
            allowed_tool_keys=self.allowed_tool_keys,
        ):
            return None

        if mode == "db":
            text = (
                "Не могу выполнить анализ по подключенной базе данных или CSV в DuckDB: "
                "инструменты доступа к данным отключены. Включите минимум `sql_tool`."
            )
        else:
            text = (
                "Не могу выполнить анализ по датасету: инструменты работы с данными "
                "отключены. Включите минимум `pandas_tool` или `value_tool`."
            )
        return AgentResponse(
            final_text=text,
            reasoning=None,
            artifacts=[],
            route="analysis",
            tool_calls=0,
            tool_names=[],
        )

    @staticmethod
    def _collect_tool_stats(callbacks: list) -> tuple[list, int, list[str]]:
        collector = next(
            (cb for cb in callbacks if hasattr(cb, "artifacts") and hasattr(cb, "tool_calls")),
            None,
        )
        if collector is None:
            return [], 0, []
        seen: set[str] = set()
        names: list[str] = []
        for item in getattr(collector, "tool_names", []) or []:
            name = str(item).strip()
            if name and name not in seen:
                seen.add(name)
                names.append(name)
        artifacts = list(getattr(collector, "artifacts", []) or [])
        tool_calls = int(getattr(collector, "tool_calls", 0) or 0)
        return artifacts, tool_calls, names

    @staticmethod
    def _clean_notes(notes: list[str]) -> list[str]:
        return [text for text in (str(note).strip() for note in notes) if text]

    def _merge_session_notes(self, notes: list[str]) -> None:
        if self.session_memory is None:
            return
        existing = str(getattr(self.session_memory, "notes", "") or "").strip()
        merged = [existing] if existing else []
        merged.extend(notes)
        self.session_memory.notes = "\n".join(merged).strip()

    @staticmethod
    def _runtime_metadata(trace_context: dict[str, Any] | None) -> dict[str, Any]:
        if not trace_context:
            return {}
        metadata: dict[str, Any] = {}
        session_id = trace_context.get("session_id")
        if isinstance(session_id, str) and session_id:
            metadata["session_id"] = session_id
            metadata["thread_id"] = session_id
            metadata["conversation_id"] = session_id
        user_id = trace_context.get("user_id")
        if user_id is not None:
            metadata["user_id"] = str(user_id)
        username = trace_context.get("username")
        if isinstance(username, str) and username:
            metadata["username"] = username
        request_kind = trace_context.get("request_kind")
        if isinstance(request_kind, str) and request_kind:
            metadata["request_kind"] = request_kind
        return metadata

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text:
                        parts.append(text)
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts)
        return str(content or "")

    @classmethod
    def _normalize_title_candidate(cls, raw: str) -> str | None:
        text = str(raw or "").strip()
        if not text:
            return None
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
        first_line = first_line.strip("`\"'«».,:;!?-–— ")
        words = cls._title_words(first_line)
        if len(words) < 3:
            return None
        title = " ".join(words[:4]).strip()
        if not title:
            return None
        return f"{title[0].upper()}{title[1:]}" if len(title) > 1 else title.upper()

    @staticmethod
    def _title_words(text: str) -> list[str]:
        import re

        return re.findall(r"[A-Za-zА-Яа-яЁё0-9]+(?:-[A-Za-zА-Яа-яЁё0-9]+)?", text)

    def _query_cache_key(
        self,
        *,
        df: pd.DataFrame | None,
        prompt: str,
        history: list[dict[str, Any]],
        use_history: bool,
        include_reasoning: bool,
        selected_skill_ids: list[str],
    ) -> str:
        payload = {
            "model": self.settings.llm_model,
            "dataset": self._dataset_signature(df),
            "prompt": prompt[:600],
            "history": history[-8:] if use_history else [],
            "use_history": use_history,
            "include_reasoning": include_reasoning,
            "selected_skill_ids": selected_skill_ids,
            "analysis_depth": self.settings.agent_analysis_depth,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()

    @staticmethod
    def _dataset_signature(df: pd.DataFrame | None) -> str:
        if df is None:
            return "no-dataset"
        head = df.head(6).to_csv(index=False)
        columns = ",".join(str(c) for c in df.columns[:64])
        payload = f"{df.shape}|{columns}|{head}"
        return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()

    def _cache_get(self, key: str) -> AgentResponse | None:
        entry = self._query_cache.get(key)
        if entry is None:
            return None
        if time.time() - entry.created_at > max(1, self.settings.agent_cache_ttl_sec):
            self._query_cache.pop(key, None)
            return None
        self._query_cache.move_to_end(key)
        return copy.deepcopy(entry.response)

    def _cache_set(self, key: str, response: AgentResponse) -> None:
        self._query_cache[key] = QueryCacheEntry(time.time(), copy.deepcopy(response))
        self._query_cache.move_to_end(key)
        while len(self._query_cache) > max(8, self.settings.agent_cache_size):
            self._query_cache.popitem(last=False)

    @staticmethod
    def _fallback_text(prompt: str, df: pd.DataFrame | None) -> str:
        if df is not None:
            return "Запрос получен, но анализ не вернул финальный текст."
        if prompt.strip():
            return "Запрос получен, но не удалось сформировать содержательный ответ."
        return "Пустой запрос."
