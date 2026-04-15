from __future__ import annotations

import copy
import errno
import hashlib
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, ClassVar, Literal, TypedDict

import pandas as pd
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from backend.agent.callbacks import (
    AgentProgressCollector,
    LLMTextCollector,
    PhaseCollector,
    ToolCollector,
)
from backend.agent.llm_client import ReasoningChatOpenAI
from backend.agent.prompts import (
    chat_system_prompt,
    execution_agent_prompt,
    get_detailed_data_info,
)
from backend.artifacts.execution import artifact_type_label
from backend.auth.user_memory import UserMemory
from backend.core.config import DEPTH_PROFILES, Settings
from backend.core.llm_provider import get_provider_policy
from backend.data_access.db_runtime_service import DBRuntimeService, RuntimeDBConnectionConfig
from backend.integrations.anomaly_planfact import AnomalyPlanfactIntegrationService
from backend.integrations.forecast import ForecastIntegrationService
from backend.integrations.rag import RAGService
from backend.integrations.search import SearchIntegrationService
from backend.observability.phoenix import record_llm_usage_on_active_span
from backend.sessions.session_memory import SessionMemory
from backend.skills import SkillRegistry
from backend.tools.capabilities import build_runtime_capability_context
from backend.tools.context import ToolBuildContext
from backend.tools.impl.review_tool import ReviewTool as _ReviewTool
from backend.tools.policy import (
    detect_data_access_mode,
    has_enabled_data_tools,
    is_tool_allowed,
    normalize_allowed_tool_keys,
)
from backend.tools.registry import ToolRegistry
from backend.tools.sandbox_manager import SandboxManager

logger = logging.getLogger(__name__)


def _build_chat_data_context(df: pd.DataFrame | None, session_source: dict) -> str:
    """Compact data-context suffix for the chat LLM so it knows what sources are loaded."""
    parts: list[str] = []
    if df is not None:
        parts.append(f"Загружен датасет: {df.shape[0]} строк, {df.shape[1]} столбцов.")
    elif session_source.get("csv_loaded"):
        table_names = session_source.get("csv_table_names") or []
        tables_str = ", ".join(table_names) if table_names else "неизвестно"
        parts.append(f"Загружен CSV в DuckDB. Таблицы: {tables_str}.")
    source_label = str(session_source.get("source_label") or "").strip()
    source_type = str(session_source.get("source_type") or "").strip().lower()
    if source_type == "db_connection" and source_label:
        parts.append(f"Подключена база данных: {source_label}.")
    if not parts:
        return ""
    return "[КОНТЕКСТ ДАННЫХ]\n" + "\n".join(parts)


def is_chat_query(normalized_prompt: str, *, has_data: bool = False) -> bool:
    """Return True when the prompt looks like a chat/greeting, not an analysis request.

    Module-level so it can be imported by other layers (e.g. the API route layer for
    generating contextually appropriate fallback error messages) without duplicating
    the detection heuristics.  ``normalized_prompt`` must already be lowercased and
    stripped.
    """
    _CHAT_GREETINGS = (
        "привет", "здравствуй", "добрый день", "добрый вечер", "доброе утро",
        "hello", "hi", "hey", "хай", "ку", "приветствую",
        "спасибо", "благодарю", "thank", "пожалуйста",
        "пока", "до свидания", "bye", "good bye",
        "как дела", "как ты", "как поживаешь",
    )
    _CHAT_ABOUT_SELF = (
        "кто ты", "что ты умеешь", "расскажи о себе", "ты кто",
        "что можешь", "что умеешь", "помоги", "help",
        "что ты такое", "как тебя зовут", "твоё имя",
    )
    _DATA_MARKERS = (
        "загрузил", "загрузила", "загружено", "залил", "залила",
        "добавил", "добавила", "подключил", "подключила",
        "uploaded", "attached", "connected",
        "покажи", "покажи таблицы", "сколько строк", "какие колонки",
        "анализ", "analyze", "analyse",
    )
    prompt = normalized_prompt.strip()
    if not prompt:
        return True
    if any(m in prompt for m in _DATA_MARKERS):
        return False
    if any(prompt.startswith(g) or prompt == g for g in _CHAT_GREETINGS):
        return True
    if any(m in prompt for m in _CHAT_ABOUT_SELF):
        return True
    if len(prompt) < 4 and not has_data:
        return True
    return False


def _is_llm_transport_failure(exc: BaseException) -> bool:
    """True for timeouts and TCP/DNS failures (not 4xx/5xx API errors)."""
    visited: set[int] = set()

    def walk(err: BaseException | None) -> bool:
        if err is None:
            return False
        eid = id(err)
        if eid in visited:
            return False
        visited.add(eid)

        if isinstance(err, (TimeoutError, ConnectionError, BrokenPipeError)):
            return True
        if isinstance(err, OSError) and err.errno is not None:
            _transport_errno = {
                errno.ECONNREFUSED,
                errno.ENETUNREACH,
                errno.EHOSTUNREACH,
                errno.ENETDOWN,
                errno.EPIPE,
                errno.ECONNRESET,
                errno.ETIMEDOUT,
            }
            if hasattr(errno, "WSAENETUNREACH"):
                _transport_errno.add(int(errno.WSAENETUNREACH))
            if hasattr(errno, "WSAETIMEDOUT"):
                _transport_errno.add(int(errno.WSAETIMEDOUT))
            if err.errno in _transport_errno:
                return True

        try:
            import httpx

            if isinstance(
                err,
                (
                    httpx.ConnectError,
                    httpx.ConnectTimeout,
                    httpx.ReadTimeout,
                    httpx.WriteTimeout,
                    httpx.PoolTimeout,
                ),
            ):
                return True
        except ImportError:
            pass

        try:
            from openai import APIConnectionError, APITimeoutError

            if isinstance(err, (APIConnectionError, APITimeoutError)):
                return True
        except ImportError:
            pass

        if walk(err.__cause__):
            return True
        if err.__context__ is not err.__cause__ and walk(err.__context__):
            return True
        return False

    return walk(exc)


def _log_llm_invoke_failure(where: str, exc: BaseException, settings: Settings) -> None:
    if _is_llm_transport_failure(exc):
        logger.warning(
            "%s: LLM endpoint unreachable or timed out (%s). base_url=%s model=%s",
            where,
            exc,
            settings.llm_base_url,
            settings.llm_model,
        )
    else:
        logger.exception("%s failed", where)

def _build_tool_message_text(result: object) -> str:
    def _short(obj: object, limit: int = 1600) -> str:
        try:
            text = json.dumps(obj, ensure_ascii=False, default=str, indent=2)
        except Exception:
            text = str(obj)
        return text[:limit]

    def _preview_rows(table_obj: object, max_rows: int = 15) -> list[dict]:
        try:
            import pandas as pd

            if isinstance(table_obj, pd.DataFrame):
                return table_obj.head(max_rows).to_dict(orient="records")
        except Exception:
            pass

        if isinstance(table_obj, list):
            return table_obj[:max_rows]

        if isinstance(table_obj, dict):
            # sometimes already row-like or serialized table payload
            if "rows" in table_obj and isinstance(table_obj["rows"], list):
                return table_obj["rows"][:max_rows]
            return [table_obj]

        return []

    def _table_schema(obj: object) -> dict[str, str]:
        try:
            if isinstance(obj, pd.DataFrame):
                return {col: str(dtype) for col, dtype in obj.dtypes.items()}
        except Exception:
            pass
        return {}

    def _row_count(obj: object) -> int | None:
        try:
            if isinstance(obj, pd.DataFrame):
                return len(obj)
        except Exception:
            pass
        return None

    content_text = ""
    artifact = None

    if hasattr(result, "content") and hasattr(result, "artifact"):
        content_text = str(getattr(result, "content", "") or "")
        artifact = getattr(result, "artifact", None)
    elif isinstance(result, tuple):
        content_text = str(result[0] or "")
        artifact = result[1] if len(result) > 1 else None
    else:
        content_text = str(result)

    parts: list[str] = []
    if content_text.strip():
        parts.append(content_text.strip())

    if isinstance(artifact, dict):
        artifact_type = artifact.get("artifact_type")
        items = artifact.get("items")

        if artifact_type == "table" and isinstance(items, dict):
            previews = []
            for name, payload in items.items():
                schema = _table_schema(payload)
                total_rows = _row_count(payload)
                rows = _preview_rows(payload, max_rows=10)
                header = name
                if total_rows is not None:
                    header += f" — {total_rows} rows × {len(schema)} cols"
                previews.append({
                    "table": header,
                    "schema": schema,
                    f"sample_{min(10, len(rows))}_of_{total_rows or '?'}_rows": rows,
                })
            if previews:
                parts.append("TABLE_RESULT:\n" + _short(previews, limit=2000))

        elif artifact_type == "value" and isinstance(items, dict):
            parts.append("VALUE_RESULT:\n" + _short(items))

        elif artifact_type == "json" and isinstance(items, dict):
            parts.append("JSON_RESULT:\n" + _short(items))

        elif artifact_type == "plot" and isinstance(items, dict):
            plot_names = list(items.keys())[:5]
            parts.append("PLOT_RESULT:\n" + _short({"plot_names": plot_names}))

        else:
            parts.append("ARTIFACT_RESULT:\n" + _short(artifact))

    return "\n\n".join(p for p in parts if p).strip()


RECOVERY_TEXT_PREFIX = "Шаг анализа завершился с ограничением итераций модели"
GENERIC_ARTIFACT_SUMMARY_PREFIX = "Анализ выполнен, артефакты построены"

_LLM_UNAVAILABLE_USER_TEXT = (
    "Языковая модель сейчас недоступна: нет соединения с LLM-сервером или сработал таймаут. "
    "Проверьте, что Ollama (или другой провайдер) запущен и что "
    "LLM_MODEL_API_URL доступен из контейнера backend."
)

# DEPTH_PROFILES imported from backend.core.config — single source of truth.


@dataclass
class AgentResponse:
    final_text: str
    reasoning: str | None
    artifacts: list
    route: Literal["chat", "analysis", "summary"] = "chat"
    tool_calls: int = 0
    tool_names: list[str] = field(default_factory=list)
    llm_unreachable: bool = False
    reasoning_steps: list[str] = field(default_factory=list)


@dataclass
class QueryCacheEntry:
    created_at: float
    response: AgentResponse


class AgentGraphState(TypedDict, total=False):
    # Input
    df: pd.DataFrame | None
    prompt: str
    history: list[dict[str, Any]]
    use_history: bool
    include_reasoning: bool
    callbacks: list
    trace_context: dict[str, Any]
    session_source: dict[str, Any]
    selected_skill_ids: list[str]

    # Prepare → Agent handoff
    done: bool
    stop_reason: str
    step_index: int
    max_steps: int
    tools: list
    capability_context: dict[str, Any]
    llm_unreachable: bool
    sandbox: Any
    tool_db_runtime: Any  # RuntimeDBConnectionConfig | None — resolved once in dispatch

    # Internal — runner reference injected by run_query so static node methods
    # can access instance services (LLM clients, tool registry, settings) without
    # being bound to a specific AgentRunner instance at graph-compile time.
    _runner: Any

    # Output
    response: AgentResponse


class AgentRunner:
    # Shared compiled graph — built once for the class, reused across all instances.
    # Nodes are @staticmethods that receive the per-request AgentRunner via state["_runner"],
    # so the same compiled graph is valid for every request regardless of user settings.
    _compiled_graph: ClassVar[Any] = None
    _graph_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
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
    ) -> None:
        self.settings = settings
        self.enabled_analytical_skill_ids = (
            set(enabled_analytical_skill_ids) if enabled_analytical_skill_ids is not None else None
        )
        self.db_runtime_service = db_runtime_service
        self.search_service = search_service
        self.forecast_service = forecast_service
        self.anomaly_planfact_service = anomaly_planfact_service
        self.rag_service = rag_service
        self.allowed_tool_keys = normalize_allowed_tool_keys(allowed_tool_keys)
        self.user_memory: UserMemory = user_memory or UserMemory(profile="", notes="")
        self.session_memory: SessionMemory = session_memory or SessionMemory()
        self._user_memory_buffer: list[str] = []
        self._session_memory_buffer: list[str] = []
        self.skill_registry = skill_registry or SkillRegistry.from_path(self.settings.skills_dir)
        self.skill_registry.load()
        self._tool_registry = ToolRegistry.from_services(
            search_service=search_service,
            forecast_service=forecast_service,
            anomaly_planfact_service=anomaly_planfact_service,
            rag_service=rag_service,
            memory_note_callback=self._user_memory_buffer.append,
            session_note_callback=self._session_memory_buffer.append,
            skill_registry=self.skill_registry,
        )
        self._query_cache: OrderedDict[str, QueryCacheEntry] = OrderedDict()
        self._depth_profile = self._resolve_depth_profile()
        # Ensure the class-level graph is compiled exactly once (double-checked locking).
        if AgentRunner._compiled_graph is None:
            with AgentRunner._graph_lock:
                if AgentRunner._compiled_graph is None:
                    AgentRunner._compiled_graph = AgentRunner._build_query_graph()
        self._review_tool = _ReviewTool(
            llm_model=self.settings.llm_model,
            llm_base_url=self.settings.llm_base_url,
            llm_api_key=self.settings.llm_api_key,
        )

    def _resolve_depth_profile(self) -> dict[str, Any]:
        depth = self.settings.agent_analysis_depth
        return DEPTH_PROFILES.get(depth, DEPTH_PROFILES["light"])

    # ── Utility: LLM / data context ──────────────────────────────────────────

    @staticmethod
    def _db_session_prompt_block(
        *,
        session_source: dict[str, Any] | None,
        runtime: RuntimeDBConnectionConfig | None,
        df: pd.DataFrame | None,
    ) -> str:
        if runtime is None:
            return ""
        has_tabular = df is not None and len(df.columns) > 0
        lines = [
            "═══ Источник данных (сессия) ═══",
            "Тип: подключение к базе данных.",
            f"Имя подключения: {runtime.name}",
            f"Тип СУБД: {runtime.db_type}",
        ]
        if runtime.database:
            lines.append(f"База/каталог: {runtime.database}")
        configured_schema = runtime.options.get("schema")
        if isinstance(configured_schema, str) and configured_schema.strip():
            lines.append(f"Схема по умолчанию: {configured_schema.strip()}.")
        lines.append(f"Идентификатор подключения: {runtime.connection_id}")
        if isinstance(session_source, dict):
            label = str(session_source.get("source_label") or "").strip()
            if label:
                lines.append(f"Метка в интерфейсе: {label}")
        if not has_tabular:
            lines.append(
                "Переменная `df` пустая — НЕ используй `df` для получения данных. "
                "Для выборок из таблиц БД вызывай инструмент `sql_tool` с параметром `question` "
                "(формулировка на естественном языке). "
                "Для визуализации используй `plotly_tool` с `db.query_dataframe(sql)` внутри. "
                "Если пользователь спрашивает только о том, с какой базой данных ведётся работа, "
                "ответь по полям выше обычным текстом — для этого tool не обязателен."
            )
        return "\n".join(lines)

    def _build_llm(
        self,
        *,
        role: Literal["chat", "tool"],
        include_reasoning: bool,
        timeout_sec: int | None = None,
        max_tokens_override: int | None = None,
    ) -> ReasoningChatOpenAI:
        enable_thinking = (
            self.settings.llm_enable_thinking
            and include_reasoning
        )

        if enable_thinking:
            temperature = 1.0
            top_p = self.settings.llm_top_p
        else:
            temperature = (
                self.settings.llm_temperature_tool
                if role == "tool"
                else self.settings.llm_temperature_chat
            )
            top_p = 0.8

        presence_penalty = self.settings.llm_presence_penalty

        max_tokens = max_tokens_override or self.settings.llm_max_tokens_default
        if max_tokens_override is None and include_reasoning:
            max_tokens = self.settings.llm_max_tokens_reasoning

        extra_body: dict[str, Any] = {}
        if self.settings.llm_chat_template_kwargs_enabled:
            extra_body.update(
                get_provider_policy(self.settings.llm_provider)
                .build_extra_body(enable_thinking=enable_thinking)
            )
        if self.settings.llm_top_k > 0:
            extra_body["top_k"] = self.settings.llm_top_k
        if self.settings.llm_num_ctx > 0:
            extra_body["num_ctx"] = self.settings.llm_num_ctx

        kwargs: dict[str, Any] = {
            "model": self.settings.llm_model,
            "base_url": self.settings.llm_base_url,
            "api_key": self.settings.llm_api_key,
            "streaming": self.settings.llm_streaming_force or self.settings.llm_streaming,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "max_tokens": max_tokens,
            "timeout": timeout_sec or self.settings.backend_query_timeout_sec,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        return ReasoningChatOpenAI(**kwargs)

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

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        clean = str(text or "").strip()
        if len(clean) <= max_len:
            return clean
        return f"{clean[:max_len]}..."

    @staticmethod
    def _title_words(text: str) -> list[str]:
        return re.findall(r"[A-Za-zА-Яа-яЁё0-9]+(?:-[A-Za-zА-Яа-яЁё0-9]+)?", text)

    @staticmethod
    def _normalize_title_candidate(raw: str) -> str | None:
        text = str(raw or "").strip()
        if not text:
            return None
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
        if not first_line:
            return None
        first_line = first_line.strip("`\"'«»"".,:;!?-–— ")  # pylint: disable=implicit-str-concat
        words = AgentRunner._title_words(first_line)
        if len(words) < 3:
            return None
        compact = words[:4]
        title = " ".join(compact).strip()
        if not title:
            return None
        return f"{title[0].upper()}{title[1:]}" if len(title) > 1 else title.upper()

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

        truncated_queries = cleaned_queries[-8:]
        dataset_part = str(dataset_name or "").strip() or "не указан"
        query_lines = "\n".join(f"{idx}. {item}" for idx, item in enumerate(truncated_queries, start=1))

        prompt_messages = [
            SystemMessage(
                content=(
                    "Ты придумываешь названия чатов аналитики данных.\n"
                    "Верни только короткое название на русском языке.\n"
                    "Требования:\n"
                    "- Ровно 3 или 4 слова.\n"
                    "- Без кавычек, без пунктуации в конце, без пояснений.\n"
                    "- Название должно отражать датасет и суть пользовательских запросов."
                )
            ),
            HumanMessage(
                content=(
                    f"Датасет: {dataset_part}\n"
                    "Запросы пользователя:\n"
                    f"{query_lines}\n\n"
                    "Сформируй название чата."
                )
            ),
        ]
        runtime_config: dict[str, Any] = {}
        metadata = self._build_runtime_metadata(trace_context)
        if metadata:
            runtime_config["metadata"] = metadata

        try:
            llm = self._build_llm(
                role="chat",
                include_reasoning=False,
                timeout_sec=max(8, min(20, self.settings.backend_query_timeout_sec)),
            )
            response = llm.invoke(prompt_messages, config=runtime_config or None)
            record_llm_usage_on_active_span(
                response,
                fallback_model=self.settings.llm_model,
                fallback_provider=self.settings.llm_provider,
            )
            generated = self._content_to_text(getattr(response, "content", ""))
        except Exception:
            generated = ""

        return self._normalize_title_candidate(generated)

    # ── Utility: history / artifacts ─────────────────────────────────────────

    def _history_summary(self, older_history: list[dict[str, Any]]) -> str:
        if not older_history:
            return ""

        summary_rows: list[str] = []
        for item in older_history[-8:]:
            role = str(item.get("role", "assistant"))
            content = self._truncate(str(item.get("content", "")), 140)
            if not content:
                continue
            marker = "U" if role == "user" else "A"
            summary_rows.append(f"- {marker}: {content}")

        if not summary_rows:
            return ""

        max_chars = max(200, self.settings.agent_history_summary_chars)
        summary = "Краткая сводка предыдущего диалога:\n" + "\n".join(summary_rows)
        return self._truncate(summary, max_chars)

    def _artifact_table_to_text(self, data: Any, *, max_rows: int = 20, max_cols: int = 12) -> str:
        try:
            if isinstance(data, pd.Series):
                data = data.to_frame()

            if isinstance(data, pd.DataFrame):
                df = data.copy()
                if df.empty:
                    return "Пустая таблица."

                rows, cols = df.shape
                visible_cols = [str(c) for c in df.columns[:max_cols]]
                lines: list[str] = [
                    f"shape={rows}x{cols}",
                    f"columns={visible_cols}",
                ]

                if rows <= max_rows and cols <= max_cols:
                    try:
                        lines.append("full_table:")
                        lines.append(df.iloc[:max_rows, :max_cols].to_markdown(index=False))
                    except Exception:
                        lines.append("full_table:")
                        lines.append(str(df.iloc[:max_rows, :max_cols]))
                    return "\n".join(lines)

                preview_df = df.iloc[: min(8, len(df)), :max_cols]
                try:
                    lines.append("preview_rows:")
                    lines.append(preview_df.to_markdown(index=False))
                except Exception:
                    lines.append("preview_rows:")
                    lines.append(str(preview_df))

                numeric_cols = list(df.select_dtypes(include="number").columns[:max_cols])
                if numeric_cols:
                    try:
                        desc = df[numeric_cols].describe().transpose().round(4)
                        lines.append("numeric_describe:")
                        lines.append(desc.to_markdown())
                    except Exception:
                        pass

                return "\n".join(lines)

            if isinstance(data, dict):
                return json.dumps(data, ensure_ascii=False, default=str)[:1500]

            if isinstance(data, list):
                return json.dumps(data[:20], ensure_ascii=False, default=str)[:1500]

            return str(data)[:1500]
        except Exception as exc:
            return f"Не удалось сериализовать артефакт: {exc}"

    def _history_artifact_summary(self, history: list[dict[str, Any]]) -> str:
        if not history:
            return ""

        blocks: list[str] = []
        for item in history[-12:]:
            artifacts = item.get("artifacts")
            if not isinstance(artifacts, list) or not artifacts:
                continue

            for idx, artifact in enumerate(artifacts[:6], start=1):
                if not isinstance(artifact, dict):
                    continue

                artifact_type = str(artifact.get("type", "artifact")).strip() or "artifact"
                artifact_name = str(artifact.get("text", "")).strip() or f"{artifact_type}_{idx}"
                data = artifact.get("data")

                block_lines = [
                    f"Артефакт: {artifact_name}",
                    f"Тип: {artifact_type}",
                ]

                if artifact_type in {"table", "value"}:
                    block_lines.append(self._artifact_table_to_text(data))
                elif artifact_type == "plot":
                    if data is not None and not isinstance(data, str):
                        block_lines.append(self._artifact_table_to_text(data))
                    else:
                        block_lines.append(
                            "Построен график. Если данные недоступны, "
                            "ориентируйся на чат и связанные таблицы."
                        )
                elif artifact_type == "json":
                    if isinstance(data, dict):
                        answer = str(data.get("answer") or "").strip()
                        if answer:
                            block_lines.append(answer)
                        results = data.get("results")
                        if isinstance(results, list):
                            block_lines.append(f"Найдено результатов: {len(results)}.")
                    else:
                        block_lines.append(self._artifact_table_to_text(data))
                else:
                    block_lines.append(self._artifact_table_to_text(data))

                blocks.append("\n".join(block_lines))

        if not blocks:
            return ""

        return "Артефакты из истории:\n\n" + "\n\n---\n\n".join(blocks[:12])

    def _build_management_note(
        self,
        *,
        prompt: str,
        history: list[dict[str, Any]],
        include_reasoning: bool,
        callbacks: list,
        trace_context: dict[str, Any] | None = None,
    ) -> AgentResponse:
        llm = self._build_llm(
            role="chat",
            include_reasoning=include_reasoning,
            timeout_sec=min(30, self.settings.backend_query_timeout_sec),
        )

        chat_summary = self._history_summary(history)

        recent_rows: list[str] = []
        for item in history[-20:]:
            role = str(item.get("role", "assistant")).strip()
            content = self._truncate(str(item.get("content", "")).strip(), 700)
            if not content:
                continue
            marker = "Пользователь" if role == "user" else "Ассистент"
            recent_rows.append(f"{marker}: {content}")

        recent_chat_block = "\n".join(recent_rows).strip()
        artifact_block = self._history_artifact_summary(history)

        system_prompt = (
            "Ты готовишь управленческую записку по текущим данным и релевантной истории переписки.\n\n"
            "Формат ответа строго такой:\n"
            "УПРАВЛЕНЧЕСКАЯ ЗАПИСКА\n\n"
            "1. Цель анализа\n"
            "...\n"
            "2. Основные выводы\n"
            "...\n"
            "3. Рекомендации\n"
            "- действие — ответственный — KPI\n"
            "4. Заключение\n"
            "...\n"
            "5. Следующие шаги\n"
            "- что сделать — кто отвечает\n\n"
            "Правила:\n"
            "- Пиши по-русски.\n"
            "- Не выдумывай факты.\n"
            "- Опирайся только на чат и артефакты из входа.\n"
            "- Не добавляй даты, сроки, дедлайны и периоды в пункты 3 и 5.\n"
            "- В 'Рекомендации' пиши только практические рекомендации "
            "в формате 'действие — ответственный — KPI'.\n"
            "- В 'Заключение' отрази сильные стороны, зоны роста и цель на период.\n"
            "- В 'Следующие шаги' пиши только 'что сделать — кто отвечает', без дат и сроков.\n"
            "- Если таблица маленькая, используй её целиком как основание для выводов.\n"
            "- Не пиши markdown-таблицы, JSON, код и служебные комментарии.\n"
        )

        user_prompt = (
            f"Текущий запрос пользователя:\n{prompt.strip()}\n\n"
            f"{chat_summary or 'Краткой сводки чата нет.'}\n\n"
            f"Последние сообщения:\n{recent_chat_block or 'Нет доступных последних сообщений.'}\n\n"
            f"{artifact_block or 'Артефактов в истории нет.'}\n\n"
            "Сформируй управленческую записку."
        )

        runtime_config: dict[str, Any] = {"callbacks": callbacks}
        metadata = self._build_runtime_metadata(trace_context)
        if metadata:
            runtime_config["metadata"] = metadata

        try:
            response = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
                config=runtime_config,
            )
            record_llm_usage_on_active_span(
                response,
                fallback_model=self.settings.llm_model,
                fallback_provider=self.settings.llm_provider,
            )
            output_text = self._content_to_text(getattr(response, "content", ""))
            final_text = output_text.strip()
            reasoning = response.additional_kwargs.get("reasoning") or None

            if not final_text:
                final_text = (
                    "УПРАВЛЕНЧЕСКАЯ ЗАПИСКА\n\n"
                    "1. Цель анализа\nНедостаточно данных для формулировки цели.\n\n"
                    "2. Основные выводы\nНе удалось извлечь подтверждённые выводы.\n\n"
                    "3. Рекомендации\n- уточнить контекст запроса — пользователь — "
                    "наличие уточнённой постановки\n\n"
                    "4. Заключение\nСильные стороны: запрос на структурированный итог сформулирован. "
                    "Зоны роста: недостаточно данных. Цель на период: уточнить входные материалы.\n\n"
                    "5. Следующие шаги\n- уточнить, по какой части переписки нужен отчёт — пользователь"
                )

            return AgentResponse(
                final_text=final_text,
                reasoning=reasoning,
                artifacts=[],
                route="summary",
                tool_calls=0,
                tool_names=[],
            )
        except Exception as exc:
            return AgentResponse(
                final_text=(
                    "УПРАВЛЕНЧЕСКАЯ ЗАПИСКА\n\n"
                    "1. Цель анализа\nНе удалось сформировать.\n\n"
                    "2. Основные выводы\nОшибка генерации summary.\n\n"
                    "3. Рекомендации\n- повторить запрос — пользователь — получен корректный отчёт\n\n"
                    "4. Заключение\nСильные стороны: структура задана. Зоны роста: ошибка генерации.\n\n"
                    "5. Следующие шаги\n- повторить формирование управленческой записки — пользователь"
                ),
                reasoning=f"summary failed: {exc}",
                artifacts=[],
                route="summary",
                tool_calls=0,
                tool_names=[],
            )

    # ── Utility: message building ─────────────────────────────────────────────

    def _build_messages(
        self,
        prompt: str,
        history: list[dict[str, Any]],
        use_history: bool,
        system_prompt: str | None = None,
    ) -> list[BaseMessage]:
        messages: list[BaseMessage] = []
        system_parts: list[str] = []

        if system_prompt:
            system_parts.append(system_prompt)

        memory_block = self.user_memory.build_block()
        if memory_block:
            system_parts.append(memory_block)

        session_memory_block = self.session_memory.build_block()
        if session_memory_block:
            system_parts.append(session_memory_block)

        recent: list[dict[str, Any]] = []
        if use_history and history:
            max_msgs = max(0, self.settings.agent_history_max_messages)
            recent = history[-max_msgs:] if max_msgs > 0 else []
            older = history[:-max_msgs] if max_msgs > 0 else history

            summary = self._history_summary(older)
            if summary:
                system_parts.append(summary)

        if system_parts:
            messages.append(SystemMessage(content="\n\n".join(system_parts)))

        for i, item in enumerate(recent):
            role = item.get("role")
            content = str(item.get("content", "")).strip()
            artifacts = item.get("artifacts")
            if role != "user" and isinstance(artifacts, list) and artifacts:
                labels: list[str] = []
                for artifact in artifacts[:6]:
                    if not isinstance(artifact, dict):
                        continue
                    artifact_type = str(artifact.get("type", "artifact"))
                    artifact_text = str(artifact.get("text", "")).strip()
                    labels.append(
                        f"{artifact_type}:{artifact_text}" if artifact_text else artifact_type
                    )
                if labels:
                    labels_text = ", ".join(labels)
                    preceding_query = next(
                        (
                            self._truncate(str(recent[j].get("content", "")), 300)
                            for j in range(i - 1, -1, -1)
                            if recent[j].get("role") == "user"
                        ),
                        "",
                    )
                    artifact_ctx = f"Контекст предыдущих артефактов: {labels_text}"
                    if preceding_query:
                        artifact_ctx += f"\nЗапрос, породивший артефакты: {preceding_query}"
                    content = f"{content}\n\n{artifact_ctx}".strip()

            if not content:
                continue
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=prompt))
        return messages

    # ── Utility: cache ────────────────────────────────────────────────────────

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
                "content": self._truncate(str(item.get("content", "")), 220),
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
    ) -> str:
        payload = {
            "model": self.settings.llm_model,
            "dataset": self._dataset_signature(df),
            "prompt": self._truncate(prompt, 600),
            "history": self._history_cache_signature(history, use_history),
            "use_history": bool(use_history),
            "include_reasoning": bool(include_reasoning),
            "analysis_depth": str(self.settings.agent_analysis_depth or "light"),
            "selected_skill_ids": list(selected_skill_ids or []),
            "max_steps": self.settings.agent_max_steps,
            "step_timeout": self.settings.agent_step_timeout_sec,
            "inner_recursion_limit": self.settings.agent_inner_recursion_limit,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()

    def _cache_get(self, key: str) -> AgentResponse | None:
        if not self.settings.agent_cache_enabled:
            return None
        entry = self._query_cache.get(key)
        if entry is None:
            return None
        age_sec = time.time() - entry.created_at
        if age_sec > max(1, self.settings.agent_cache_ttl_sec):
            self._query_cache.pop(key, None)
            return None
        self._query_cache.move_to_end(key)
        return copy.deepcopy(entry.response)

    def _cache_set(self, key: str, response: AgentResponse) -> None:
        if not self.settings.agent_cache_enabled:
            return
        self._query_cache[key] = QueryCacheEntry(
            created_at=time.time(), response=copy.deepcopy(response)
        )
        self._query_cache.move_to_end(key)
        max_size = max(8, self.settings.agent_cache_size)
        while len(self._query_cache) > max_size:
            self._query_cache.popitem(last=False)

    # ── Utility: tools / policy ───────────────────────────────────────────────

    def _tool_allowed(self, tool_key: str) -> bool:
        return is_tool_allowed(tool_key, self.allowed_tool_keys)

    def _build_data_tools_disabled_response(
        self,
        df: pd.DataFrame | None,
        prompt: str,
        session_source: dict[str, Any] | None = None,
        trace_context: dict[str, Any] | None = None,
    ) -> AgentResponse | None:
        normalized_session_source = self._normalize_session_source_for_sql_mode(
            session_source, trace_context,
        )

        if self._quick_route(prompt) == "summary":
            return None
        _normalized = prompt.strip().lower()
        _chat_markers = (
            "привет", "здравствуй", "добрый", "hello", "hi ", "hey ",
            "спасибо", "thank you", "thanks", "кто ты", "что ты умеешь",
        )
        if len(_normalized) < 120 and any(m in _normalized for m in _chat_markers):
            return None

        mode = detect_data_access_mode(
            has_dataframe=df is not None,
            session_source=normalized_session_source,
        )
        if mode is None:
            return None

        if has_enabled_data_tools(
            has_dataframe=df is not None,
            session_source=normalized_session_source,
            allowed_tool_keys=self.allowed_tool_keys,
        ):
            return None

        if mode == "db":
            final_text = (
                "Не могу выполнить анализ по подключенной базе данных или CSV в DuckDB: "
                "инструменты доступа к данным отключены в настройках аккаунта. "
                "Включите как минимум `sql_tool`. Для построения графиков "
                "дополнительно можно включить `plotly_tool`."
            )
        else:
            final_text = (
                "Не могу выполнить анализ по датасету: "
                "инструменты работы с данными отключены в настройках аккаунта. "
                "Включите как минимум `pandas_tool` или `value_tool`. "
                "Для визуализаций дополнительно можно включить `plotly_tool`."
            )

        return AgentResponse(
            final_text=final_text,
            reasoning=None,
            artifacts=[],
            route="analysis",
            tool_calls=0,
            tool_names=[],
        )

    @staticmethod
    def _build_runtime_metadata(
        trace_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if not trace_context:
            return metadata

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

        for key in ("db_connection_id", "connection_id"):
            value = trace_context.get(key)
            if isinstance(value, str) and value.strip():
                metadata["db_connection_id"] = value.strip()
                metadata["data_source"] = "db_connection"
                break

        csv_session_id = trace_context.get("csv_session_id")
        if isinstance(csv_session_id, str) and csv_session_id.strip():
            metadata["csv_session_id"] = csv_session_id.strip()

        if bool(trace_context.get("csv_duckdb_loaded")) and "data_source" not in metadata:
            metadata["data_source"] = "csv_duckdb"

        return metadata

    @staticmethod
    def _extract_db_connection_id(
        session_source: dict[str, Any] | None,
        trace_context: dict[str, Any] | None,
    ) -> str | None:
        if isinstance(session_source, dict):
            source_type = str(session_source.get("source_type", "")).strip().lower()
            source_ref_id = session_source.get("source_ref_id")
            if source_type == "db_connection" and isinstance(source_ref_id, str) and source_ref_id.strip():
                return source_ref_id.strip()

        if not trace_context:
            return None

        for key in ("db_connection_id", "connection_id"):
            value = trace_context.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        for key in ("db_connection", "db_source", "source", "data_source"):
            nested = trace_context.get(key)
            if not isinstance(nested, dict):
                continue
            for nested_key in ("db_connection_id", "connection_id"):
                value = nested.get(nested_key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    def _resolve_tool_db_runtime_config(
        self,
        session_source: dict[str, Any] | None,
        trace_context: dict[str, Any] | None,
    ) -> RuntimeDBConnectionConfig | None:
        connection_id = self._extract_db_connection_id(session_source, trace_context)
        if not connection_id:
            return None
        if self.db_runtime_service is None:
            raise RuntimeError("DB runtime service is not configured.")

        user_id_raw = (trace_context or {}).get("user_id")
        try:
            user_id = int(user_id_raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "trace_context.user_id is required for DB tool runtime."
            ) from exc

        return self.db_runtime_service.get_runtime_config(
            user_id=user_id,
            connection_id=connection_id,
        )

    @staticmethod
    def _resolve_csv_runtime_state(
        session_source: dict[str, Any] | None,
        trace_context: dict[str, Any] | None,
    ) -> tuple[bool, str | None]:
        if isinstance(session_source, dict):
            direct_loaded = bool(session_source.get("csv_loaded"))
            direct_sid = session_source.get("csv_session_id")
            if direct_loaded and isinstance(direct_sid, str) and direct_sid.strip():
                return True, direct_sid.strip()

            source_type = str(session_source.get("source_type", "")).strip().lower()
            if source_type == "csv" and isinstance(direct_sid, str) and direct_sid.strip():
                return True, direct_sid.strip()

        if trace_context and bool(trace_context.get("csv_duckdb_loaded")):
            sid = trace_context.get("csv_session_id") or trace_context.get("session_id")
            if isinstance(sid, str) and sid.strip():
                return True, sid.strip()

        return False, None

    @staticmethod
    def _normalize_session_source_for_sql_mode(
        session_source: dict[str, Any] | None,
        trace_context: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not isinstance(session_source, dict):
            session_source = {}

        normalized = dict(session_source)

        csv_loaded, csv_session_id = AgentRunner._resolve_csv_runtime_state(
            session_source, trace_context,
        )
        if csv_loaded and csv_session_id:
            normalized["source_type"] = "db_connection"
            normalized["source_ref_id"] = csv_session_id
            normalized["source_label"] = str(
                normalized.get("source_label") or f"CSV DuckDB session {csv_session_id}"
            )
            normalized["source_mode"] = str(normalized.get("source_mode") or "read_only")
            normalized["csv_loaded"] = True
            normalized["csv_session_id"] = csv_session_id

        return normalized

    def _fallback_text(
        self,
        prompt: str,
        df: pd.DataFrame | None = None,
        stop_reason: str | None = None,
    ) -> str:
        if df is not None:
            if stop_reason == "max_steps_reached":
                return (
                    "Я выполнил несколько шагов анализа, но не получил надежный артефакт. "
                    "Уточните запрос или сузьте задачу."
                )
            return (
                "Запрос получен. Без подтвержденных артефактов возвращаю безопасный ответ. "
                "Могу продолжить анализ при повторном запросе."
            )

        if not prompt.strip():
            return "Я получил запрос, но не смог сформировать содержательный ответ."

        return (
            "Запрос получен. Сейчас недоступен расширенный аналитический режим, "
            "но я остаюсь на связи и могу продолжить сразу после повтора запроса."
        )

    # ── Execution system prompt ───────────────────────────────────────────────

    @staticmethod
    def _execution_runtime_section(
        source_mode: str,
        tool_list: str,
        today: str,
        tool_descriptions: str,
    ) -> list[str]:
        lines = [
            f"Сегодня: {today}.",
            f"Режим данных: `{source_mode}`.",
            f"Доступные tools в этом запуске: {tool_list}.",
        ]
        if tool_descriptions:
            lines += ["Описание доступных tools:", tool_descriptions]
        return lines

    def _matched_analytical_skills_hint(self, user_prompt: str | None) -> str:
        """Return a short bullet list of analytical skills whose triggers match the prompt.

        Honors the per-user enabled-skill filter. Used to steer planner_tool so
        the generated plan reflects the prescribed analytical algorithm.
        """
        if not user_prompt:
            return ""
        prompt_lower = str(user_prompt).lower()
        allow = self.enabled_analytical_skill_ids
        lines: list[str] = []
        for skill in self.skill_registry.list_skills():
            if skill.kind != "analytical":
                continue
            if allow is not None and skill.skill_id not in allow:
                continue
            for trigger in skill.triggers:
                if trigger and trigger in prompt_lower:
                    lines.append(f"- `{skill.skill_id}`: {skill.description}")
                    break
        return "\n".join(lines)

    def _build_execution_system_prompt(
        self,
        *,
        capability_context: dict[str, Any] | None = None,
        sandbox: Any | None = None,
        selected_skill_ids: list[str] | None = None,
        df: pd.DataFrame | None = None,
        session_source: dict[str, Any] | None = None,
        tool_db_runtime: Any | None = None,
        user_prompt: str | None = None,
    ) -> str:
        """Build the complete system prompt for the agent tool-calling loop.

        Single source of truth: base policy + runtime section + sandbox state +
        tool skills + analytical skills + user-selected skills + data context.
        """
        source_mode = str((capability_context or {}).get("source_mode", "")).strip() or "dataset"
        tool_descriptions = str((capability_context or {}).get("tool_descriptions", "")).strip()
        available_tools = [
            str(item).strip()
            for item in (capability_context or {}).get("available_tool_keys", [])
            if str(item).strip()
        ]
        tool_list = ", ".join(f"`{item}`" for item in available_tools) if available_tools else "нет"
        today = date.today().strftime("%Y-%m-%d")

        sections: list[str] = [execution_agent_prompt.strip()]
        sections.extend(self._execution_runtime_section(source_mode, tool_list, today, tool_descriptions))

        if sandbox:
            sandbox_block = sandbox.describe_for_prompt()
            if sandbox_block:
                sections.append(sandbox_block)

        # Tool skills: brief list with deferred full instructions via get_tool_instructions().
        tool_skills_block = self.skill_registry.build_tool_skills_brief_block(set(available_tools))
        if tool_skills_block:
            sections.append(tool_skills_block)

        # Analytical skills: filtered by user-enabled set; matching skills get full
        # instructions auto-expanded inline so the agent actually follows them.
        analytical_skills_block = self.skill_registry.build_analytical_skills_brief_block(
            enabled_skill_ids=self.enabled_analytical_skill_ids,
            user_prompt=user_prompt,
        )
        if analytical_skills_block:
            sections.append(analytical_skills_block)

        # User-selected skills (explicitly attached to this request).
        if selected_skill_ids:
            skills_block = self.skill_registry.build_prompt_block(selected_skill_ids)
            if skills_block:
                sections.append(skills_block)

        # Data context: dataset schema + DB connection info.
        data_context_parts: list[str] = []
        if df is not None:
            try:
                data_context_parts.append(
                    get_detailed_data_info(df, max_columns=self.settings.agent_prompt_max_columns)
                )
            except Exception:
                data_context_parts.append(
                    f"Датасет: {df.shape[0]} строк, {df.shape[1]} столбцов."
                )
        db_block = self._db_session_prompt_block(
            session_source=session_source,
            runtime=tool_db_runtime,
            df=df,
        )
        if db_block:
            data_context_parts.append(db_block)
        if data_context_parts:
            sections.append("\n\n".join(data_context_parts))

        return "\n\n".join(sections).strip()

    # ── Utility: artifact synthesis ───────────────────────────────────────────

    def _collect_tool_stats(self, callbacks: list) -> tuple[list, int, list[str]]:
        tool_collector = None
        for cb in callbacks:
            if isinstance(cb, ToolCollector):
                tool_collector = cb
                break

        artifacts = tool_collector.artifacts if tool_collector else []
        tool_calls = int(tool_collector.tool_calls) if tool_collector else 0
        tool_names: list[str] = []
        if tool_collector is not None:
            seen: set[str] = set()
            for item in tool_collector.tool_names:
                normalized = str(item).strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                tool_names.append(normalized)
        return artifacts, tool_calls, tool_names

    @staticmethod
    def _extract_value_payload(artifacts: list) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for artifact in artifacts:
            type_str = artifact_type_label(getattr(artifact, "artifact_type", ""))
            if type_str not in ("value", "scalar"):
                continue
            data = getattr(artifact, "data", None)
            if isinstance(data, dict):
                merged.update(data)
        return merged

    @staticmethod
    def _format_metric_value(value: Any) -> str:
        if isinstance(value, bool):
            return "да" if value else "нет"
        if isinstance(value, int):
            return f"{value:,}".replace(",", " ")
        if isinstance(value, float):
            text = f"{value:,.4f}".rstrip("0").rstrip(".")
            return text.replace(",", " ")
        return str(value)

    @staticmethod
    def _format_numeric_value(value: float) -> str:
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.4f}".rstrip("0").rstrip(".")

    @staticmethod
    def _extract_numeric_series_from_table(table_data: Any) -> list[pd.Series]:
        series_list: list[pd.Series] = []
        if isinstance(table_data, pd.Series):
            numeric = pd.to_numeric(table_data, errors="coerce").dropna()
            if not numeric.empty:
                series_list.append(numeric)
            return series_list

        if not isinstance(table_data, pd.DataFrame) or table_data.empty:
            return series_list

        if table_data.shape[1] == 1:
            only_col = table_data.iloc[:, 0]
            numeric = pd.to_numeric(only_col, errors="coerce").dropna()
            if not numeric.empty:
                series_list.append(numeric)
            return series_list

        first_col = table_data.iloc[:, 0]
        for col in table_data.columns[1:]:
            values = pd.to_numeric(table_data[col], errors="coerce")
            mask = values.notna()
            if not mask.any():
                continue
            candidate = pd.Series(values[mask].to_numpy(), index=first_col[mask].to_numpy())
            series_list.append(candidate)
        return series_list

    def _table_extreme_summary(self, prompt: str, artifacts: list) -> str:
        normalized = prompt.strip().lower()
        asks_max = any(
            token in normalized
            for token in ("больше всего", "наибол", "максим", "most", "highest", "max")
        )
        asks_min = any(
            token in normalized
            for token in ("меньше всего", "наим", "миним", "least", "lowest", "min")
        )
        if not asks_max and not asks_min:
            return ""

        candidate_series: list[pd.Series] = []
        for artifact in artifacts:
            if str(getattr(artifact, "artifact_type", "")).strip() != "table":
                continue
            candidate_series.extend(
                self._extract_numeric_series_from_table(getattr(artifact, "data", None))
            )

        if not candidate_series:
            return ""

        chosen = max(candidate_series, key=len)
        if chosen.empty:
            return ""

        if asks_min and not asks_max:
            label = chosen.idxmin()
            value = float(chosen.min())
            qualifier = "меньше всего"
        else:
            label = chosen.idxmax()
            value = float(chosen.max())
            qualifier = "больше всего"

        return f"По таблице {qualifier} значение у '{label}': {self._format_numeric_value(value)}."

    @staticmethod
    def _first_sentence(text: str, max_len: int = 260) -> str:
        clean = re.sub(r"\s+", " ", str(text or "")).strip()
        if not clean:
            return ""
        sentence = re.split(r"(?<=[.!?])\s+", clean, maxsplit=1)[0].strip()
        if not sentence:
            sentence = clean
        if len(sentence) > max_len:
            return f"{sentence[:max_len].rstrip()}..."
        return sentence

    def _artifact_method_lines(self, artifacts: list, max_items: int = 8) -> list[str]:
        lines: list[str] = []
        for artifact in artifacts[:max_items]:
            artifact_type = artifact_type_label(getattr(artifact, "artifact_type", "")) or "artifact"
            name = (
                str(getattr(artifact, "name", "") or getattr(artifact, "text", "")).strip()
                or artifact_type
            )
            data = getattr(artifact, "data", None)

            if artifact_type in ("value", "scalar") and isinstance(data, dict):
                metric_keys = [str(key) for key in data.keys()]
                preview = ", ".join(metric_keys[:5])
                suffix = ", ..." if len(metric_keys) > 5 else ""
                lines.append(
                    f"- `value` `{name}`: {len(metric_keys)} метрик"
                    + (f" ({preview}{suffix})" if preview else "")
                )
                continue

            if artifact_type in ("table", "dataframe", "sql_result"):
                if isinstance(data, pd.Series):
                    data = data.to_frame()
                if isinstance(data, pd.DataFrame):
                    rows, cols = data.shape
                    lines.append(f"- `table` `{name}`: {rows} строк, {cols} столбцов")
                else:
                    lines.append(f"- `table` `{name}`: табличный срез построен")
                continue

            if artifact_type == "plot":
                title_text = ""
                layout = getattr(data, "layout", None)
                if layout is not None:
                    raw_title = getattr(layout, "title", None)
                    if hasattr(raw_title, "text"):
                        raw_title = raw_title.text
                    if isinstance(raw_title, str):
                        title_text = raw_title.strip()
                if title_text:
                    lines.append(f"- `plot` `{name}`: график «{title_text}»")
                else:
                    lines.append(f"- `plot` `{name}`: визуализация построена")
                continue

            if artifact_type == "json":
                if isinstance(data, dict):
                    result_count = len(data.get("results") or [])
                    answer_preview = str(data.get("answer") or "").strip()[:80]
                    detail = f"{result_count} результатов" if result_count else ""
                    if answer_preview:
                        detail = (
                            f"{detail}; ответ: {answer_preview}"
                            if detail
                            else f"ответ: {answer_preview}"
                        )
                    lines.append(f"- `json` `{name}`: {detail or 'данные получены'}")
                else:
                    lines.append(f"- `json` `{name}`: JSON-данные получены")
                continue

            lines.append(f"- `{artifact_type}` `{name}`: артефакт сформирован")

        return lines

    def _value_observation_lines(
        self, value_payload: dict[str, Any], max_items: int = 6
    ) -> list[str]:
        if not value_payload:
            return []
        keys = sorted(value_payload.keys())
        return [
            f"- {key}: {self._format_metric_value(value_payload[key])}"
            for key in keys[:max_items]
        ]

    def _table_observation_lines(self, artifacts: list, max_items: int = 4) -> list[str]:
        lines: list[str] = []
        for artifact in artifacts:
            if str(getattr(artifact, "artifact_type", "")).strip() != "table":
                continue
            name = str(getattr(artifact, "text", "")).strip() or "table"
            series_list = self._extract_numeric_series_from_table(
                getattr(artifact, "data", None)
            )
            if not series_list:
                continue

            chosen = max(series_list, key=len)
            if chosen.empty:
                continue

            max_label = str(chosen.idxmax())
            max_value = self._format_numeric_value(float(chosen.max()))
            min_label = str(chosen.idxmin())
            min_value = self._format_numeric_value(float(chosen.min()))

            if max_label == min_label:
                lines.append(f"- `{name}`: значение {max_value} для `{max_label}`.")
            else:
                lines.append(
                    f"- `{name}`: максимум у `{max_label}` = {max_value}, "
                    f"минимум у `{min_label}` = {min_value}."
                )
            if len(lines) >= max_items:
                break
        return lines

    def _artifact_grounded_summary(
        self, prompt: str, artifacts: list, base_text: str | None = None
    ) -> str:
        normalized = prompt.strip().lower()
        value_payload = self._extract_value_payload(artifacts)
        asks_direct_answer = (
            prompt.strip().endswith("?")
            or any(token in normalized for token in ("в каком", "какой", "сколько", "кто", "где"))
        )
        table_count = sum(
            1 for artifact in artifacts
            if artifact_type_label(getattr(artifact, "artifact_type", "")) == "table"
        )
        plot_count = sum(
            1 for artifact in artifacts
            if artifact_type_label(getattr(artifact, "artifact_type", "")) == "plot"
        )
        value_count = sum(
            1 for artifact in artifacts
            if artifact_type_label(getattr(artifact, "artifact_type", "")) == "value"
        )

        direct_answer = self._table_extreme_summary(prompt, artifacts)
        if not direct_answer and value_payload:
            row_count = value_payload.get("row_count")
            column_count = value_payload.get("column_count")
            if asks_direct_answer and "сколько" in normalized:
                if isinstance(row_count, (int, float)) and ("строк" in normalized or "запис" in normalized):
                    direct_answer = f"В датасете {int(row_count)} строк."
                elif isinstance(column_count, (int, float)) and (
                    "столбц" in normalized or "колонк" in normalized
                ):
                    direct_answer = f"В датасете {int(column_count)} столбцов."

            if not direct_answer and (
                "датасет" in normalized or "данн" in normalized or "расскажи" in normalized
            ):
                if isinstance(row_count, (int, float)) and isinstance(column_count, (int, float)):
                    direct_answer = f"В датасете {int(row_count)} строк и {int(column_count)} столбцов."

        # Use full base_text as the main answer when available and not a plan/trace
        full_base_text = ""
        if base_text and not self._response_looks_like_plan_or_trace(base_text):
            full_base_text = base_text.strip()

        if not direct_answer and not full_base_text:
            direct_answer = "Ключевой вывод сформирован на основе полученных артефактов."

        method_lines = self._artifact_method_lines(artifacts)
        if not method_lines:
            method_lines = [
                f"- Получено артефактов: {len(artifacts)}",
                f"- По типам: table={table_count}, plot={plot_count}, value={value_count}",
            ]

        observation_lines: list[str] = []
        observation_lines.extend(self._value_observation_lines(value_payload, max_items=5))
        observation_lines.extend(self._table_observation_lines(artifacts, max_items=2))
        if not observation_lines:
            observation_lines = [
                f"- Построено артефактов: {len(artifacts)} "
                f"(table={table_count}, plot={plot_count}, value={value_count})."
            ]

        # Main answer: full base_text if available, otherwise direct_answer (e.g. table extreme)
        main_answer = full_base_text or direct_answer

        return (
            f"{main_answer}\n\n"
            "Что сделано:\n"
            + "\n".join(method_lines)
            + "\n\nКлючевые наблюдения:\n"
            + "\n".join(observation_lines)
        )

    @staticmethod
    def _response_too_generic(prompt: str, response_text: str) -> bool:
        text = response_text.strip().lower()
        if not text:
            return True
        generic_markers = ("анализ выполнен", "артефакт", "построены", "получены метрики")
        has_generic = any(marker in text for marker in generic_markers)
        asks_direct_answer = (
            prompt.strip().endswith("?")
            or any(token in prompt.lower() for token in ("в каком", "какой", "сколько", "кто"))
        )
        return asks_direct_answer and has_generic and len(text) < 180

    @staticmethod
    def _response_looks_like_plan_or_trace(response_text: str) -> bool:
        text = str(response_text or "").strip().lower()
        if not text:
            return True

        plan_prefixes = (
            "план анализа", "план решения", "план выполнения", "план:",
            "plan:", "корректировка плана", "интеграция внешней",
            "извлечение ключевых", "что хочет пользователь?",
            "какие инструменты использовать", "выполняю шаг",
            "проверяю результат шага", "доработка через повторный цикл",
            "рассуждение (chain of thought)",
        )
        if any(text.startswith(prefix) for prefix in plan_prefixes):
            return True

        trace_markers = (
            '"name": "value_tool"', '"name": "pandas_tool"', '"name": "sql_tool"',
            '"name": "plotly_tool"', '"name": "database_tool"', '"name": "search_tool"',
            '"artifact_type": "value"', '"artifact_type": "table"',
            "tool_result", "value_tool(", "pandas_tool(", "sql_tool(",
            "plotly_tool(", "database_tool(", "search_tool(", "forecast_tool(",
        )
        return any(marker in text for marker in trace_markers)

    @staticmethod
    def _latest_collected_text(callbacks: list) -> tuple[str, str | None]:
        for cb in callbacks:
            if not isinstance(cb, LLMTextCollector):
                continue
            if not cb.messages:
                continue
            latest = cb.messages[-1]
            text = str(latest.get("text", "")).strip()
            reasoning = str(latest.get("reasoning", "")).strip() or None
            return text, reasoning
        return "", None

    @staticmethod
    def _reset_text_collectors(callbacks: list) -> None:
        for cb in callbacks:
            if isinstance(cb, LLMTextCollector):
                cb.messages.clear()

    # ── Utility: event emission ───────────────────────────────────────────────

    @staticmethod
    def _collect_progress_collectors(callbacks: list) -> list[AgentProgressCollector]:
        return [cb for cb in callbacks if isinstance(cb, AgentProgressCollector)]

    @staticmethod
    def _emit_progress_event(
        callbacks: list,
        *,
        phase: str,
        title: str,
        details: str = "",
        step_index: int | None = None,
        max_steps: int | None = None,
    ) -> None:
        for collector in AgentRunner._collect_progress_collectors(callbacks):
            collector.add_event(
                phase=phase,
                title=title,
                details=details,
                step_index=step_index,
                max_steps=max_steps,
            )

    @staticmethod
    def _collect_phase_collectors(callbacks: list) -> list[PhaseCollector]:
        return [cb for cb in callbacks if isinstance(cb, PhaseCollector)]

    @staticmethod
    def _emit_phase_event(
        callbacks: list,
        *,
        phase: str,
        title: str,
        content: str = "",
        step_index: int | None = None,
        max_steps: int | None = None,
        status: str | None = None,
    ) -> None:
        for collector in AgentRunner._collect_phase_collectors(callbacks):
            collector.add_phase(
                phase=phase,
                title=title,
                content=content,
                step_index=step_index,
                max_steps=max_steps,
                status=status,
            )
            gt = getattr(collector, "graph_tracker", None)
            if gt is not None:
                si = step_index if isinstance(step_index, int) else 0
                if status == "streaming":
                    gt.phase_start(phase, si)
                elif status in ("done", "pass", "fail", "error"):
                    gt.phase_end(phase, si, status="done" if status in ("done", "pass") else "error")
                collector._graph_version += 1  # noqa: SLF001

    def _artifacts_recovery_text(self, artifacts: list) -> str:
        if not artifacts:
            return ""

        counts: dict[str, int] = {}
        labels: list[str] = []
        for artifact in artifacts[:8]:
            artifact_type = str(getattr(artifact, "artifact_type", "artifact")).strip() or "artifact"
            counts[artifact_type] = counts.get(artifact_type, 0) + 1
            label = str(getattr(artifact, "text", "")).strip()
            if label:
                labels.append(label)

        typed_counts = ", ".join(
            f"{name}: {value}" for name, value in sorted(counts.items(), key=lambda item: item[0])
        )
        if labels:
            labels_preview = ", ".join(labels[:4])
            if len(labels) > 4:
                labels_preview += ", ..."
            return (
                f"{RECOVERY_TEXT_PREFIX}, артефакты уже построены "
                f"({typed_counts}). Доступные артефакты: {labels_preview}."
            )

        return f"{RECOVERY_TEXT_PREFIX}, артефакты уже построены ({typed_counts})."

    def _artifacts_summary_text(self, artifacts: list) -> str:
        """Neutral summary when the loop ended normally but LLM returned empty text."""
        if not artifacts:
            return ""

        counts: dict[str, int] = {}
        labels: list[str] = []
        for artifact in artifacts[:8]:
            artifact_type = str(getattr(artifact, "artifact_type", "artifact")).strip() or "artifact"
            counts[artifact_type] = counts.get(artifact_type, 0) + 1
            label = str(getattr(artifact, "text", "")).strip()
            if label:
                labels.append(label)

        typed_counts = ", ".join(
            f"{name}: {value}" for name, value in sorted(counts.items(), key=lambda item: item[0])
        )
        if labels:
            labels_preview = ", ".join(labels[:4])
            if len(labels) > 4:
                labels_preview += ", ..."
            return f"Артефакты построены ({typed_counts}). Доступные артефакты: {labels_preview}."

        return f"Артефакты построены ({typed_counts})."

    # ── Single execution engine: direct tool-calling loop ────────────────────

    def _direct_tool_loop(
        self,
        *,
        prompt: str,
        history: list[dict[str, Any]],
        use_history: bool,
        include_reasoning: bool,
        tools: list,
        execution_system_prompt: str,
        callbacks: list,
        max_iterations: int,
        trace_context: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """The one and only tool-calling loop.

        Uses LLM native tool_use (bind_tools). No ReAct Thought/Action format.
        No conflict with <think> tags from reasoning models.
        """
        self._reset_text_collectors(callbacks)
        llm = self._build_llm(
            role="tool",
            include_reasoning=include_reasoning,
            timeout_sec=min(
                self.settings.agent_step_timeout_sec,
                self.settings.backend_query_timeout_sec,
            ),
        )
        bound_llm = llm.bind_tools(tools)

        messages: list[BaseMessage] = self._build_messages(
            prompt, history, use_history, system_prompt=execution_system_prompt,
        )

        tool_map = {
            str(getattr(t, "name", "")).strip(): t
            for t in tools
            if str(getattr(t, "name", "")).strip()
        }

        all_tool_names: list[str] = []
        total_tool_calls = 0
        final_text = ""
        reasoning = None
        reasoning_steps: list[str] = []
        _limit_reached = False

        runtime_config: dict[str, Any] = {"callbacks": callbacks}
        metadata = self._build_runtime_metadata(trace_context)
        if metadata:
            runtime_config["metadata"] = metadata

        for _iteration in range(max(1, max_iterations)):
            try:
                response = bound_llm.invoke(messages, config=runtime_config)
                record_llm_usage_on_active_span(
                    response,
                    fallback_model=self.settings.llm_model,
                    fallback_provider=self.settings.llm_provider,
                )
            except Exception as exc:
                if _is_llm_transport_failure(exc):
                    _log_llm_invoke_failure("direct_tool_loop LLM invoke", exc, self.settings)
                    artifacts, tc, tn = self._collect_tool_stats(callbacks)
                    return AgentResponse(
                        final_text=self._artifacts_recovery_text(artifacts) or _LLM_UNAVAILABLE_USER_TEXT,
                        reasoning=str(exc),
                        reasoning_steps=[],
                        artifacts=artifacts,
                        route="analysis",
                        tool_calls=total_tool_calls + tc,
                        tool_names=all_tool_names + tn,
                        llm_unreachable=True,
                    )
                raise

            step_r = response.additional_kwargs.get("reasoning") or None
            if step_r:
                reasoning_steps.append(step_r)
                if reasoning is None:
                    reasoning = step_r  # backward compat: первый шаг → reasoning поле

            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                final_text = self._content_to_text(getattr(response, "content", ""))
                # LLM produced only thinking (no visible text). Nudge it to act/respond.
                if not final_text and messages and isinstance(messages[-1], ToolMessage):
                    last_tool_call_name = ""
                    for msg in reversed(messages):
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            last_tool_call_name = msg.tool_calls[0].get("name", "")
                            break
                    if last_tool_call_name == "get_tool_instructions":
                        # Just received skill instructions — must now call the first tool.
                        messages.append(HumanMessage(
                            content=(
                                "Инструкции скила получены. "
                                "Следуй им: немедленно вызови первый аналитический инструмент "
                                "из полученных инструкций (pandas_tool, sql_tool, plotly_tool и т.п.). "
                                "НЕ вызывай get_tool_instructions снова. "
                                "Только вызов tool, без текста."
                            )
                        ))
                        continue
                    elif last_tool_call_name and _iteration < max(1, max_iterations) - 1:
                        # Analysis tools finished but LLM spent output budget on thinking.
                        # Nudge it to produce the final visible answer (allowed once).
                        messages.append(HumanMessage(
                            content=(
                                "Анализ завершён. Напиши финальный ответ пользователю: "
                                "кратко и конкретно, опираясь только на полученные результаты. "
                                "Без tool-вызовов, без пересказа плана — только выводы."
                            )
                        ))
                        continue
                break

            messages.append(response)

            for tc in tool_calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("args", {})
                tool_call_id = tc.get("id", "")
                total_tool_calls += 1
                if tool_name not in all_tool_names:
                    all_tool_names.append(tool_name)

                tool = tool_map.get(tool_name)
                if tool is None:
                    tool_message_text = f"Unknown tool: {tool_name}"
                else:
                    try:
                        tool_call_input = {
                            "name": tool_name,
                            "args": tool_args,
                            "id": tool_call_id,
                            "type": "tool_call",
                        }
                        result = tool.invoke(tool_call_input, config=runtime_config)
                        tool_message_text = _build_tool_message_text(result)
                    except Exception as tool_exc:
                        tool_message_text = f"Tool error: {tool_exc}"

                messages.append(ToolMessage(content=tool_message_text, tool_call_id=tool_call_id))
        else:
            # Max iterations reached — try to recover text from collector.
            _limit_reached = True
            text_collector = next(
                (cb for cb in callbacks if isinstance(cb, LLMTextCollector)), None
            )
            if text_collector and text_collector.messages:
                final_text = text_collector.messages[-1].get("text", "")

        artifacts, tc_count, tn_list = self._collect_tool_stats(callbacks)
        total_tool_calls += tc_count
        for name in tn_list:
            if name not in all_tool_names:
                all_tool_names.append(name)

        if not final_text and artifacts:
            if _limit_reached:
                final_text = self._artifacts_recovery_text(artifacts)
            else:
                final_text = self._artifacts_summary_text(artifacts)

        return AgentResponse(
            final_text=final_text.strip(),
            reasoning=reasoning,
            reasoning_steps=reasoning_steps,
            artifacts=artifacts,
            route="analysis",
            tool_calls=total_tool_calls,
            tool_names=all_tool_names,
        )

    # ── Graph ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_query_graph():
        """Compile the agent graph: dispatch → agent → finalize.

        dispatch: keyword pre-check for lightweight bypasses (chat/summary),
                  or builds tools/sandbox/capability context for analysis.
        agent: single direct tool-calling loop with skills and sandbox.
        finalize: synthesizes answer, rewrites if needed, runs quality review.

        Nodes are @staticmethods — they receive the per-request AgentRunner via
        state["_runner"] so this compiled graph is shared across all instances.
        """
        graph = StateGraph(AgentGraphState)

        graph.add_node("dispatch", AgentRunner._dispatch_node)
        graph.add_node("agent", AgentRunner._agent_node)
        graph.add_node("finalize", AgentRunner._finalize_node)

        graph.add_edge(START, "dispatch")
        graph.add_conditional_edges(
            "dispatch",
            lambda state: "finalize" if state.get("done") else "agent",
            {"agent": "agent", "finalize": "finalize"},
        )
        graph.add_edge("agent", "finalize")
        graph.add_edge("finalize", END)

        return graph.compile()

    # ── Routing helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _quick_route(
        prompt: str,
        *,
        has_data: bool = False,
    ) -> Literal["chat", "summary"] | None:
        """Lightweight keyword pre-check for bypass routes. No LLM call."""
        normalized = prompt.strip().lower()
        _summary_markers = (
            "управленческ", "итоги анализа",
            "резюмируй", "подведи итог", "executive summary", "сделай отчёт",
            "сводка по результатам", "ключевые выводы из чата",
        )
        if any(m in normalized for m in _summary_markers):
            return "summary"
        if AgentRunner._is_chat_message(normalized, has_data=has_data):
            return "chat"
        return None

    @staticmethod
    def _is_chat_message(normalized_prompt: str, *, has_data: bool = False) -> bool:
        """Detect greetings and simple chat messages that don't need tools.

        Delegates to the module-level ``is_chat_query`` so the same heuristics
        are used by the routing logic here and by the API error-message layer.
        """
        return is_chat_query(normalized_prompt, has_data=has_data)

    # ── Graph nodes ───────────────────────────────────────────────────────────

    @staticmethod
    def _dispatch_node(state: AgentGraphState) -> dict[str, Any]:
        """Dispatch node: keyword pre-check, then either bypass or build analysis context.

        Two lightweight keyword bypasses (no LLM call):
          chat    → plain chat response (greetings, self-description queries)
          summary → management note synthesized from conversation history

        All other requests fall through to context assembly: tools, sandbox,
        capability_context are built here and passed to _agent_node.
        Note: RagTool is a regular tool in the agent loop, not a bypass route.
        """
        runner: AgentRunner = state["_runner"]
        df = state.get("df")
        prompt = state.get("prompt", "")
        callbacks = state.get("callbacks", [])
        history = state.get("history", [])
        use_history = state.get("use_history", True)
        include_reasoning = state.get("include_reasoning", False)
        trace_context = state.get("trace_context") or {}
        session_source = state.get("session_source") or {}

        tool_db_runtime = runner._resolve_tool_db_runtime_config(session_source, trace_context)  # noqa: SLF001
        csv_loaded, csv_session_id = AgentRunner._resolve_csv_runtime_state(session_source, trace_context)
        has_data = bool(
            df is not None
            or tool_db_runtime is not None
            or (csv_loaded and str(csv_session_id or "").strip())
        )

        quick = AgentRunner._quick_route(prompt, has_data=has_data)

        # ── Chat bypass ──────────────────────────────────────────────────────
        if quick == "chat":
            data_suffix = _build_chat_data_context(df, session_source)
            try:
                response = runner.chat(
                    prompt=prompt,
                    history=history,
                    use_history=use_history,
                    include_reasoning=include_reasoning,
                    callbacks=callbacks,
                    trace_context=trace_context,
                    system_prompt_suffix=data_suffix,
                )
            except Exception:
                response = AgentResponse(
                    final_text=runner._fallback_text(prompt, df),  # noqa: SLF001
                    reasoning=None,
                    artifacts=[],
                    route="chat",
                )
            return {"response": response, "done": True, "stop_reason": "chat_route"}

        # ── Summary bypass ───────────────────────────────────────────────────
        if quick == "summary":
            runner._emit_phase_event(  # noqa: SLF001
                callbacks, phase="act", title="Формирование управленческой записки",
                content="", step_index=0, max_steps=1, status="streaming",
            )
            runner._emit_progress_event(  # noqa: SLF001
                callbacks, phase="act", title="Собираю управленческую записку",
                details="Анализирую релевантную историю переписки и артефакты.",
                step_index=0, max_steps=1,
            )
            response = runner._build_management_note(  # noqa: SLF001
                prompt=prompt,
                history=history,
                include_reasoning=include_reasoning,
                callbacks=callbacks,
                trace_context=trace_context,
            )
            runner._emit_phase_event(  # noqa: SLF001
                callbacks, phase="act", title="Формирование управленческой записки",
                content="Управленческая записка сформирована.",
                step_index=0, max_steps=1, status="done",
            )
            return {"response": response, "done": True, "stop_reason": "summary_route"}

        # ── Analysis: build tools, sandbox, capability context ───────────────
        csv_duckdb_mode = bool(csv_loaded and str(csv_session_id or "").strip())
        tool_df = None if csv_duckdb_mode else df

        session_id = trace_context.get("session_id", "default")
        sandbox = SandboxManager.get_instance().get_or_create(session_id)
        sandbox.ensure_storage_dir(Path(runner.settings.storage_dir) / session_id)
        if df is not None:
            source_label = str(trace_context.get("dataset_name", "") or "")
            sandbox.bind_dataframe(df, source_label=source_label, db_runtime_config=tool_db_runtime)

        _ctx = ToolBuildContext(
            settings=runner.settings,
            allowed_tool_keys=runner.allowed_tool_keys,
            df=tool_df,
            tool_db_runtime=tool_db_runtime,
            csv_loaded=csv_loaded,
            csv_session_id=csv_session_id,
            sandbox=sandbox,
        )
        tools = runner._tool_registry.build_tools(_ctx)  # noqa: SLF001
        tool_descriptions = runner._tool_registry.describe_available_tools(_ctx)  # noqa: SLF001

        # Inject tool descriptions into planner_tool, excluding itself to avoid
        # the planner recommending a recursive planner_tool call.
        _planner_descriptions = "\n".join(
            line for line in tool_descriptions.splitlines()
            if "planner_tool" not in line
        ).strip()
        for _tool in tools:
            if hasattr(_tool, "set_tool_descriptions"):
                _tool.set_tool_descriptions(_planner_descriptions)

        depth_inner_limit = runner._depth_profile.get("inner_recursion_limit")  # noqa: SLF001
        max_steps = max(
            1,
            depth_inner_limit if isinstance(depth_inner_limit, int)
            else runner.settings.agent_inner_recursion_limit,
        )

        # Infrastructure tools run outside the agent loop (planner is pre-executed,
        # review is called in _finalize_node). Exclude them from capability_context so
        # the agent does not see them as callable tools in the system prompt or tool schema.
        _HIDDEN_FROM_AGENT: frozenset[str] = frozenset({"planner_tool", "review_tool"})
        tool_keys = [
            str(getattr(tool, "name", "")).strip()
            for tool in tools
            if str(getattr(tool, "name", "")).strip()
            and str(getattr(tool, "name", "")).strip() not in _HIDDEN_FROM_AGENT
        ]
        csv_table_names = list((state.get("session_source") or {}).get("csv_table_names") or [])
        capability_context = build_runtime_capability_context(
            available_tool_keys=tool_keys,
            has_dataframe=tool_df is not None,
            has_db_source=(tool_db_runtime is not None) or csv_duckdb_mode,
            csv_table_names=csv_table_names or None,
        )
        # Filter descriptions to match the agent-visible tool list (no planner/review)
        _desc_lines = [
            line for line in tool_descriptions.splitlines()
            if not any(("`" + k + "`") in line for k in _HIDDEN_FROM_AGENT)
        ]
        capability_context["tool_descriptions"] = "\n".join(_desc_lines).strip()

        return {
            "max_steps": max_steps,
            "done": False,
            "stop_reason": "",
            "tools": tools,
            "step_index": 0,
            "sandbox": sandbox,
            "capability_context": capability_context,
            "llm_unreachable": False,
            "tool_db_runtime": tool_db_runtime,
        }

    @staticmethod
    def _agent_node(state: AgentGraphState) -> dict[str, Any]:
        """Agent node: single direct tool-calling loop with skills and sandbox."""
        runner: AgentRunner = state["_runner"]
        df = state.get("df")
        tools = state.get("tools", [])
        callbacks = state.get("callbacks", [])

        step_index = 1
        max_steps = int(state.get("max_steps", runner.settings.agent_inner_recursion_limit))

        tool_db_runtime = state.get("tool_db_runtime")
        sandbox = state.get("sandbox")

        execution_system_prompt = runner._build_execution_system_prompt(  # noqa: SLF001
            capability_context=state.get("capability_context"),
            sandbox=sandbox,
            selected_skill_ids=state.get("selected_skill_ids") or [],
            df=df,
            session_source=state.get("session_source"),
            tool_db_runtime=tool_db_runtime,
            user_prompt=state.get("prompt"),
        )

        # ── Pre-call planner_tool outside the iteration budget ───────────────
        # Invoke planner_tool before _direct_tool_loop so it does NOT consume
        # an iteration from max_steps. The plan is injected into the system
        # prompt so the LLM receives it at the start of the execution loop.
        # tool.invoke() still fires on_tool_start/on_tool_end callbacks, so
        # the UI planner block appears exactly as before.
        planner = next((t for t in tools if getattr(t, "name", "") == "planner_tool"), None)
        tools_for_loop = tools
        if planner is not None:
            planner_runtime_config: dict[str, Any] = {"callbacks": callbacks}
            if tc := state.get("trace_context"):
                planner_runtime_config["metadata"] = runner._build_runtime_metadata(tc)  # noqa: SLF001

            history = state.get("history", [])
            recent_history_snippet = "\n".join(
                f"{'Пользователь' if h.get('role') == 'user' else 'Ассистент'}: "
                f"{AgentRunner._truncate(str(h.get('content', '')), 200)}"
                for h in history[-4:]
                if h.get("content", "").strip()
            )

            # Hint planner about analytical skills whose triggers match the
            # prompt so the generated plan includes their prescribed steps.
            matched_skills_hint = runner._matched_analytical_skills_hint(state.get("prompt"))  # noqa: SLF001
            planner_context = recent_history_snippet
            if matched_skills_hint:
                planner_context = (
                    f"{planner_context}\n\n[Аналитические скилы, подходящие по триггерам — "
                    f"первым шагом плана ОБЯЗАТЕЛЬНО поставь get_tool_instructions(\"<skill_id>\") "
                    f"для каждого из них, затем опиши дальнейшие шаги]\n{matched_skills_hint}"
                ).strip()

            try:
                plan_result = planner.invoke(
                    {
                        "name": "planner_tool",
                        "args": {
                            "question": state.get("prompt", ""),
                            "context": planner_context,
                        },
                        "id": "pre_plan_0",
                        "type": "tool_call",
                    },
                    config=planner_runtime_config,
                )
                plan_text = str(plan_result).strip()
                if plan_text:
                    execution_system_prompt += f"\n\n## Предварительный план анализа\n{plan_text}"
            except Exception as _plan_exc:
                logger.warning("Pre-loop planner_tool failed: %s", _plan_exc)
            tools_for_loop = [t for t in tools if getattr(t, "name", "") != "planner_tool"]

        runner._emit_phase_event(  # noqa: SLF001
            callbacks, phase="act", title="Выполнение анализа",
            content="", step_index=step_index, max_steps=max_steps, status="streaming",
        )
        runner._emit_progress_event(  # noqa: SLF001
            callbacks, phase="act", title="Выполняю анализ",
            details="Подбираю инструмент и формирую вызов tool.",
            step_index=step_index, max_steps=max_steps,
        )

        tool_collector = next((cb for cb in callbacks if isinstance(cb, ToolCollector)), None)
        tool_events_offset = len(tool_collector.events) if tool_collector else 0

        started_at = time.perf_counter()
        try:
            response = runner._direct_tool_loop(  # noqa: SLF001
                prompt=state.get("prompt", ""),
                history=state.get("history", []),
                use_history=state.get("use_history", True),
                include_reasoning=state.get("include_reasoning", False),
                tools=tools_for_loop,
                execution_system_prompt=execution_system_prompt,
                callbacks=callbacks,
                max_iterations=max_steps,
                trace_context=state.get("trace_context"),
            )
        except Exception as exc:
            artifacts, tool_calls, tool_names = runner._collect_tool_stats(callbacks)  # noqa: SLF001
            response = AgentResponse(
                final_text=runner._artifacts_recovery_text(artifacts),  # noqa: SLF001
                reasoning=f"Agent step failed: {exc}",
                artifacts=artifacts,
                route="analysis",
                tool_calls=tool_calls,
                tool_names=tool_names,
            )

        elapsed_sec = time.perf_counter() - started_at
        if elapsed_sec > max(1, runner.settings.agent_step_timeout_sec):
            response.reasoning = (
                (response.reasoning or "")
                + f"\n\nStep timeout guard triggered ({int(elapsed_sec * 1000)} ms)."
            ).strip()

        tool_summary_lines: list[str] = []
        if tool_collector is not None:
            for ev in tool_collector.events[tool_events_offset:]:
                if ev.get("phase") == "start":
                    name = ev.get("tool_name", "")
                    inp = (ev.get("input_preview") or "").strip()
                    if inp:
                        tool_summary_lines.append(f"**{name}**: {inp[:400]}")
                elif ev.get("phase") == "end" and ev.get("code_preview"):
                    code = ev["code_preview"].strip()
                    tool_summary_lines.append(f"```sql\n{code[:800]}\n```")
        if not tool_summary_lines and response.tool_names:
            tool_summary_lines.append(f"Инструменты: {', '.join(response.tool_names)}")
        if response.artifacts:
            types = [artifact_type_label(getattr(a, "artifact_type", "")) for a in response.artifacts]
            tool_summary_lines.append(f"Артефакты: {', '.join(t for t in types if t)}")

        runner._emit_phase_event(  # noqa: SLF001
            callbacks, phase="act", title="Анализ завершён",
            content="\n".join(tool_summary_lines) if tool_summary_lines else "Шаг выполнен.",
            step_index=step_index, max_steps=max_steps, status="done",
        )

        return {"response": response, "step_index": step_index}

    @staticmethod
    def _finalize_node(state: AgentGraphState) -> dict[str, AgentResponse]:
        """Finalize node: rewrite generic text, run quality review, return response."""
        runner: AgentRunner = state["_runner"]
        callbacks = state.get("callbacks", [])
        step_index = int(state.get("step_index", 0))
        max_steps = int(state.get("max_steps", 1))

        runner._emit_phase_event(  # noqa: SLF001
            callbacks, phase="finalize", title="Финализация",
            content="", step_index=step_index, max_steps=max_steps, status="streaming",
        )
        runner._emit_progress_event(  # noqa: SLF001
            callbacks, phase="finalize", title="Формирую финальный ответ",
            details="Собираю выводы только по подтвержденным артефактам.",
            step_index=step_index, max_steps=max_steps,
        )

        response = state.get("response")
        prompt = state.get("prompt", "")
        df = state.get("df")
        stop_reason = state.get("stop_reason")

        # LLM unreachable before any response was produced.
        if state.get("llm_unreachable") and response is None:
            runner._emit_phase_event(  # noqa: SLF001
                callbacks, phase="finalize", title="Финализация",
                content=_LLM_UNAVAILABLE_USER_TEXT,
                step_index=step_index, max_steps=max_steps, status="done",
            )
            runner._emit_progress_event(  # noqa: SLF001
                callbacks, phase="finalize", title="LLM недоступна",
                details=_LLM_UNAVAILABLE_USER_TEXT, step_index=step_index, max_steps=max_steps,
            )
            return {
                "response": AgentResponse(
                    final_text=_LLM_UNAVAILABLE_USER_TEXT,
                    reasoning="LLM invoke failed",
                    artifacts=[],
                    route="analysis",
                    tool_calls=0,
                    tool_names=[],
                    llm_unreachable=True,
                )
            }

        # LLM became unreachable mid-execution — return partial response as-is.
        if response is not None and getattr(response, "llm_unreachable", False):
            text = (response.final_text or "").strip() or _LLM_UNAVAILABLE_USER_TEXT
            runner._emit_phase_event(  # noqa: SLF001
                callbacks, phase="finalize", title="Финализация",
                content=text, step_index=step_index, max_steps=max_steps, status="done",
            )
            runner._emit_progress_event(  # noqa: SLF001
                callbacks, phase="finalize", title="LLM недоступна",
                details=text, step_index=step_index, max_steps=max_steps,
            )
            return {"response": response}

        if response is None:
            runner._emit_phase_event(  # noqa: SLF001
                callbacks, phase="finalize", title="Финализация",
                content="Нет ответа от агента, формирую fallback.",
                step_index=step_index, max_steps=max_steps, status="done",
            )
            return {
                "response": AgentResponse(
                    final_text=runner._fallback_text(prompt, df, stop_reason=stop_reason),  # noqa: SLF001
                    reasoning="No response produced by graph.",
                    artifacts=[],
                    route="analysis",
                )
            }

        # Rewrite final text when it looks like a plan or is too generic.
        if response.artifacts:
            _used_search = any(t == "search_tool" for t in (response.tool_names or []))
            _plan_prefixes = (
                "план анализа", "план решения", "план (", "plan анализа",
                "план:", "plan:", "корректировка плана",
                "интеграция внешней", "извлечение ключевых",
            )
            _final_lower = response.final_text.strip().lower()
            _looks_like_plan = any(_final_lower.startswith(p) for p in _plan_prefixes)
            should_rewrite = (
                not response.final_text.strip()
                or response.final_text.strip().startswith(RECOVERY_TEXT_PREFIX)
                or response.final_text.strip().startswith(GENERIC_ARTIFACT_SUMMARY_PREFIX)
                or AgentRunner._response_too_generic(prompt, response.final_text)
                or prompt.strip().endswith("?")
                or _looks_like_plan
                or _used_search
            )
            if should_rewrite:
                grounded_summary = runner._artifact_grounded_summary(  # noqa: SLF001
                    prompt, response.artifacts, base_text=response.final_text,
                )
                if grounded_summary:
                    response.final_text = grounded_summary

        # Quality review for analytical responses.
        _analytical_tools = {
            "sql_tool", "plotly_tool", "pandas_tool", "value_tool",
            "search_tool", "forecast_tool", "database_tool",
        }
        _used_analytical = any(t in _analytical_tools for t in (response.tool_names or []))
        if _used_analytical or response.artifacts:
            try:
                review_raw = runner._review_tool._run(  # noqa: SLF001
                    question=prompt,
                    answer=response.final_text,
                    tool_calls_count=response.tool_calls,
                    artifact_count=len(response.artifacts),
                )
                _json_match = re.search(r'\{.*?\}', review_raw, re.DOTALL)
                review_result = json.loads(_json_match.group() if _json_match else review_raw)
                if not review_result.get("pass", True):
                    reason = review_result.get("reason") or str(review_result.get("issues", ""))
                    response.reasoning = (
                        f"{response.reasoning or ''}\n\nReview failed: {reason}"
                    ).strip()
            except Exception:
                pass

        if not response.final_text.strip():
            response.final_text = runner._fallback_text(prompt, df, stop_reason=stop_reason)  # noqa: SLF001

        runner._emit_phase_event(  # noqa: SLF001
            callbacks, phase="finalize", title="Финализация",
            content="Ответ сформирован.",
            step_index=step_index, max_steps=max_steps, status="done",
        )
        return {"response": response}

    # ── Public API ────────────────────────────────────────────────────────────

    def chat(
        self,
        prompt: str,
        history: list[dict[str, Any]],
        use_history: bool,
        include_reasoning: bool,
        callbacks: list,
        trace_context: dict[str, Any] | None = None,
        system_prompt_suffix: str = "",
    ) -> AgentResponse:
        llm = self._build_llm(role="chat", include_reasoning=include_reasoning)
        full_system_prompt = (
            chat_system_prompt + "\n\n" + system_prompt_suffix.strip()
            if system_prompt_suffix.strip()
            else chat_system_prompt
        )
        prompt_messages = self._build_messages(
            prompt, history, use_history, system_prompt=full_system_prompt
        )
        runtime_config: dict[str, Any] = {"callbacks": callbacks}
        metadata = self._build_runtime_metadata(trace_context)
        if metadata:
            runtime_config["metadata"] = metadata

        text_collector = None
        for cb in callbacks:
            if isinstance(cb, LLMTextCollector):
                text_collector = cb
                break

        response = llm.invoke(prompt_messages, config=runtime_config)
        record_llm_usage_on_active_span(
            response,
            fallback_model=self.settings.llm_model,
            fallback_provider=self.settings.llm_provider,
        )
        output_text = self._content_to_text(getattr(response, "content", ""))

        final_text = ""
        reasoning = None
        if text_collector and text_collector.messages:
            final_text = text_collector.messages[-1].get("text", "")
            reasoning = text_collector.messages[-1].get("reasoning") or None
        if not final_text:
            final_text = output_text
        if reasoning is None:
            reasoning = response.additional_kwargs.get("reasoning") or None

        if not final_text.strip():
            final_text = "Я получил запрос, но не смог сформировать содержательный ответ."

        return AgentResponse(
            final_text=final_text,
            reasoning=reasoning,
            artifacts=[],
            route="chat",
            tool_calls=0,
            tool_names=[],
        )

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
        # Cache only for persistent queries (/query endpoint).
        # /evaluate (persist=False) and /stream both bypass the cache:
        # evaluate is designed for preview without side-effects, and stream is real-time.
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

        data_tools_disabled = self._build_data_tools_disabled_response(
            df, prompt, session_source=session_source, trace_context=trace_context,
        )
        if data_tools_disabled is not None:
            if cache_allowed:
                self._cache_set(cache_key, data_tools_disabled)
            return data_tools_disabled

        # Graph: prepare → agent → finalize (3 supersteps max).
        try:
            result = AgentRunner._compiled_graph.invoke(
                {
                    "_runner": self,
                    "df": df,
                    "prompt": prompt,
                    "history": history,
                    "use_history": use_history,
                    "include_reasoning": include_reasoning,
                    "callbacks": callbacks,
                    "trace_context": trace_context or {},
                    "session_source": session_source or {},
                    "selected_skill_ids": resolved_skill_ids,
                },
                config={"recursion_limit": 20},
            )
        except Exception:
            logger.exception("graph.invoke failed for prompt=%r", prompt[:60])
            fallback = AgentResponse(
                final_text=self._fallback_text(prompt, df),
                reasoning=None,
                artifacts=[],
                route="analysis",
            )
            if cache_allowed:
                self._cache_set(cache_key, fallback)
            return fallback

        response = result.get("response")
        if not isinstance(response, AgentResponse):
            response = AgentResponse(
                final_text=self._fallback_text(prompt, df),
                reasoning=None,
                artifacts=[],
                route="analysis",
            )

        if cache_allowed:
            self._cache_set(cache_key, response)
        return response

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
