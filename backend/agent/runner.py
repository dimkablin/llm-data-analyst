from __future__ import annotations

import copy
import errno
import hashlib
import json
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal, TypedDict

import pandas as pd
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from backend.agent.callbacks import (
    AgentProgressCollector,
    LLMTextCollector,
    PhaseCollector,
    PhaseTokenStreamHandler,
    TokenStreamCallbackHandler,
    ToolCollector,
    extract_thinking,
)
from backend.agent.llm_client import ThinkingAwareChatOpenAI
from backend.agent.pandas_agent import (
    create_pandas_dataframe_agent,
    extract_agent_output_text,
    normalize_agent_messages,
)
from backend.agent.prompts import (
    chat_system_prompt,
    execution_agent_prompt,
    get_detailed_data_info,
)
from backend.artifacts.execution import artifact_type_label
from backend.auth.user_memory import UserMemory
from backend.core.config import Settings
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
from backend.tools.policy import (
    detect_data_access_mode,
    has_enabled_data_tools,
    is_tool_allowed,
    normalize_allowed_tool_keys,
    supports_artifact_optional_output,
)
from backend.tools.registry import ToolRegistry
from backend.tools.sandbox_manager import SandboxManager

logger = logging.getLogger(__name__)


def _is_llm_transport_failure(exc: BaseException) -> bool:
    """Timeouts and TCP/DNS failures (operational); not 4xx/5xx from the API."""
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


RECOVERY_TEXT_PREFIX = "Шаг анализа завершился с ограничением итераций модели"
GENERIC_ARTIFACT_SUMMARY_PREFIX = "Анализ выполнен, артефакты построены"

_LLM_UNAVAILABLE_USER_TEXT = (
    "Языковая модель сейчас недоступна: нет соединения с LLM-сервером или сработал таймаут. "
    "Проверьте, что Ollama (или другой провайдер) запущен и что LLM_MODEL_API_URL доступен из контейнера backend."  # noqa: E501
)

DEPTH_PROFILES: dict[str, dict[str, Any]] = {
    "light": {
        # Верхний лимит внешних итераций ReAct (модель останавливается раньше при успехе / оценке).
        "max_steps_cap": 3,
        "evaluate_enabled": False,
        "inner_recursion_limit": 4,
        "step_timeout_sec": 90,
        "think_instruction": (
            "Стиль: лёгкий и быстрый. Максимум 1-2 шага.\n"
            "Один tool call = один шаг. Для простых запросов — 1 шаг.\n"
            "Остановись мысленно на минимально достаточной цепочке инструментов.\n"
            "Предпочитай value_tool для коротких метрик, pandas_tool для компактных таблиц.\n"
            "Не подменяй внешние search-задачи value-артефактом, если search_tool доступен.\n"
            "Внешний лимит итераций высокий — не обязан исчерпывать его; заверши, когда вопрос закрыт или данных мало."  # noqa: E501
        ),
    },
    "medium": {
        "max_steps_cap": 8,
        "evaluate_enabled": True,
        "think_instruction": (
            "Стиль: сбалансированный.\n"
            "Планируй столько шагов, сколько нужно для уверенного ответа; избегай лишних повторов.\n"
            "Фильтры, агрегации и графики — по мере необходимости, не «для галочки».\n"
            "Если после пары шагов результат уже ясен — не размножай шаги; при нехватке данных явно зафиксируй ограничение."  # noqa: E501
        ),
    },
    "deep": {
        "max_steps_cap": 15,
        "evaluate_enabled": True,
        "think_instruction": (
            "Стиль: глубокий.\n"
            "Допускается длинная цепочка шагов, если это оправдано запросом.\n"
            "Корреляции, распределения, тренды, проверка выбросов — когда уместно.\n"
            "Всё равно завершай, когда анализ исчерпан или дальше нет смысла без новых данных."
        ),
    },
}


@dataclass
class AgentResponse:
    final_text: str
    reasoning: str | None
    artifacts: list
    route: Literal["chat", "analysis", "rag", "summary"] = "chat"
    tool_calls: int = 0
    tool_names: list[str] = field(default_factory=list)
    llm_unreachable: bool = False


@dataclass
class QueryCacheEntry:
    created_at: float
    response: AgentResponse


class AgentGraphState(TypedDict, total=False):
    df: pd.DataFrame | None
    prompt: str
    history: list[dict[str, Any]]
    use_history: bool
    include_reasoning: bool
    callbacks: list
    trace_context: dict[str, Any]
    session_source: dict[str, Any]
    selected_skill_ids: list[str]
    route: Literal["rag", "summary"] | None

    plan: str
    step_index: int
    max_steps: int
    done: bool
    eval_passed: bool
    eval_reason: str
    refinement_feedback: str
    stop_reason: str
    tools: list
    capability_context: dict[str, Any]
    llm_unreachable: bool
    sandbox: Any
    graph_tracker: Any

    response: AgentResponse


class AgentRunner:
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
    ) -> None:
        self.settings = settings
        self.db_runtime_service = db_runtime_service
        self.search_service = search_service
        self.forecast_service = forecast_service
        self.anomaly_planfact_service = anomaly_planfact_service
        self.rag_service = rag_service
        self.allowed_tool_keys = normalize_allowed_tool_keys(allowed_tool_keys)
        self.user_memory: UserMemory = user_memory or UserMemory(profile="", notes="")
        self.session_memory: SessionMemory = session_memory or SessionMemory()
        # Buffers for notes appended by memory tools during this request cycle.
        self._user_memory_buffer: list[str] = []
        self._session_memory_buffer: list[str] = []
        self.skill_registry = skill_registry or SkillRegistry.from_path(self.settings.skills_dir)
        self.skill_registry.load()
        self._tool_registry = ToolRegistry.from_services(
            search_service=search_service,
            forecast_service=forecast_service,
            anomaly_planfact_service=anomaly_planfact_service,
            memory_note_callback=self._user_memory_buffer.append,
            session_note_callback=self._session_memory_buffer.append,
            skill_registry=self.skill_registry,
        )
        self._query_cache: OrderedDict[str, QueryCacheEntry] = OrderedDict()
        self._depth_profile = self._resolve_depth_profile()
        self._graph = self._build_query_graph()

    def _resolve_depth_profile(self) -> dict[str, Any]:
        depth = self.settings.agent_analysis_depth
        return DEPTH_PROFILES.get(depth, DEPTH_PROFILES["light"])

    def _depth_max_steps_cap(self) -> int:
        """Upper bound on outer ReAct iterations for the current analysis depth."""
        depth = str(self.settings.agent_analysis_depth or "light").strip().lower()
        profile = DEPTH_PROFILES.get(depth, DEPTH_PROFILES["light"])
        cap = profile.get("max_steps_cap")
        return max(2, int(cap)) if isinstance(cap, int) else 20

    def _effective_outer_max_steps(self, *, prompt: str) -> int:
        """min(user setting, depth cap); insight-style prompts get +2 within cap."""
        depth_cap = self._depth_max_steps_cap()
        user_cap = max(2, int(self.settings.agent_max_steps))
        max_steps = min(user_cap, depth_cap)
        return max_steps


    @staticmethod
    def _has_confirmed_analysis_output(response: AgentResponse | None) -> bool:
        if response is None:
            return False
        if response.artifacts:
            return True
        return (
            response.tool_calls > 0
            and bool(str(response.final_text or "").strip())
            and supports_artifact_optional_output(response.tool_names)
        )

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
        role: Literal["chat", "plan", "tool", "evaluate"],
        include_reasoning: bool,
        timeout_sec: int | None = None,
        max_tokens_override: int | None = None,
    ) -> ThinkingAwareChatOpenAI:
        enable_thinking = self.settings.llm_enable_thinking and include_reasoning
        if role in ("evaluate", "plan", "tool"):
            enable_thinking = False

        # Qwen3.5: thinking -> temp=1.0, top_p=0.95; non-thinking -> configured, top_p=0.8
        if role == "evaluate":
            temperature = 0.1
            top_p = 0.8
        elif enable_thinking:
            temperature = 1.0
            top_p = self.settings.llm_top_p
        else:
            temperature = (
                self.settings.llm_temperature_tool
                if role == "tool"
                else self.settings.llm_temperature_chat
            )
            top_p = 0.8

        # Disable presence_penalty for tool and evaluate roles: high values
        # can suppress tool-calling tokens on some models (e.g. Qwen3).
        presence_penalty = 0.0 if role in ("evaluate", "tool") else self.settings.llm_presence_penalty

        max_tokens = max_tokens_override or self.settings.llm_max_tokens_default
        if max_tokens_override is None and include_reasoning:
            max_tokens = self.settings.llm_max_tokens_reasoning
        if role == "evaluate" and max_tokens_override is None:
            max_tokens = self.settings.agent_evaluate_max_tokens

        extra_body: dict[str, Any] = {}
        # Skip chat_template_kwargs for tool role: it can interfere with
        # tool-calling reliability on some models (e.g. Qwen3 via Ollama).
        if self.settings.llm_chat_template_kwargs_enabled and role != "tool":
            extra_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        if self.settings.llm_top_k > 0:
            extra_body["top_k"] = self.settings.llm_top_k
        if self.settings.llm_num_ctx > 0:
            extra_body["num_ctx"] = self.settings.llm_num_ctx

        kwargs: dict[str, Any] = {
            "model": self.settings.llm_model,
            "base_url": self.settings.llm_base_url,
            "api_key": self.settings.llm_api_key,
            "streaming": self.settings.llm_streaming_force
            or self.settings.llm_streaming,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "max_tokens": max_tokens,
            "timeout": timeout_sec or self.settings.backend_query_timeout_sec,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        return ThinkingAwareChatOpenAI(**kwargs)

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
        first_line = first_line.strip("`\"'«»"".,:;!?-–— ")
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

        normalized = self._normalize_title_candidate(generated)
        if normalized:
            return normalized
        return None

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
                            "Построен график. Если отдельные данные графика недоступны, ориентируйся на чат и связанные таблицы."  # noqa: E501
                        )
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
            "- В 'Основные выводы' можно дать несколько пунктов, если в чате несколько важных тем.\n"
            "- В 'Рекомендации' пиши только практические рекомендации в формате 'действие — ответственный — KPI'.\n"  # noqa: E501
            "- В 'Заключение' обязательно отрази сильные стороны, зоны роста и цель на период.\n"
            "- В 'Следующие шаги' пиши только 'что сделать — кто отвечает', без дат и сроков.\n"
            "- Если таблица маленькая, используй её целиком как основание для выводов.\n"
            "- Если таблица большая, опирайся на preview, shape, columns и describe.\n"
            "- Если график есть, но его содержимое неполное, опирайся на чат и связанные таблицы.\n"
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
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ],
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
                    "3. Рекомендации\n- уточнить контекст запроса — пользователь — наличие уточнённой постановки\n\n"  # noqa: E501
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
                    "3. Рекомендации\n- повторить запрос в более узкой формулировке — пользователь — получен корректный отчёт\n\n"  # noqa: E501
                    "4. Заключение\nСильные стороны: структура отчёта задана. "
                    "Зоны роста: произошла ошибка генерации. Цель на период: успешно сформировать итоговый отчёт.\n\n"  # noqa: E501
                    "5. Следующие шаги\n- повторить формирование управленческой записки — пользователь"
                ),
                reasoning=f"summary failed: {exc}",
                artifacts=[],
                route="summary",
                tool_calls=0,
                tool_names=[],
            )

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

            for item in recent:
                role = item.get("role")
                content = str(item.get("content", "")).strip()
                artifacts = item.get("artifacts")
                if isinstance(artifacts, list) and artifacts:
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
                        content = (
                            f"{content}\n\nКонтекст предыдущих артефактов: {labels_text}"
                        ).strip()

                if not content:
                    continue
                if role == "user":
                    messages.append(HumanMessage(content=content))
                else:
                    messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=prompt))
        return messages

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
        normalized: list[dict[str, Any]] = [
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
            session_source,
            trace_context,
        )

        # Don't intercept non-analytical messages (greetings, chat, summary).
        if self._quick_route(prompt, has_rag=False) == "summary":
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

        direct_keys = ("db_connection_id", "connection_id")
        for key in direct_keys:
            value = trace_context.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        for key in ("db_connection", "db_source", "source", "data_source"):
            nested = trace_context.get(key)
            if not isinstance(nested, dict):
                continue
            for nested_key in direct_keys:
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
            session_source,
            trace_context,
        )
        if csv_loaded and csv_session_id:
            normalized["source_type"] = "db_connection"
            normalized["source_ref_id"] = csv_session_id
            normalized["source_label"] = str(
                normalized.get("source_label") or f"CSV DuckDB session {csv_session_id}"
            )
            normalized["source_mode"] = str(
                normalized.get("source_mode") or "read_only"
            )
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
                    "Я выполнил несколько шагов анализа, но не получил надежный артефакт для финального вывода. "  # noqa: E501
                    "Уточните запрос или сузьте задачу (например, один график или одну метрику)."
                )
            if stop_reason == "eval_failed":
                return (
                    "Я получил промежуточный результат, но оценка не подтвердила, что выводы полностью опираются "  # noqa: E501
                    "на артефакты. Повторите запрос в более узкой формулировке."
                )
            if stop_reason == "act_recursion_limit":
                return (
                    "Я несколько раз подряд не смог корректно завершить tool-шаг в отведенный лимит. "
                    "Повторите запрос, лучше в более узкой формулировке."
                )
            return (
                "Запрос получен. Без подтвержденных артефактов я возвращаю безопасный ответ. "
                "Могу продолжить анализ при повторном запросе."
            )

        if not prompt.strip():
            return "Я получил запрос, но не смог сформировать содержательный ответ."

        return (
            "Запрос получен. Сейчас недоступен расширенный аналитический режим, "
            "но я остаюсь на связи и могу продолжить сразу после повтора запроса."
        )

    _THINK_SYSTEM_PROMPT_BASE = (
        "Ты — планировщик аналитического агента. Составь КРАТКИЙ план действий.\n\n"
        "## Формат плана\n"
        "Для каждого шага укажи: номер, tool, что получить.\n"
        "Пример: 1. `pandas_tool` → показать первые строки таблицы.\n\n"
        "## Правила\n"
        "- Используй ТОЛЬКО инструменты из [ДОСТУПНЫЕ ИНСТРУМЕНТЫ].\n"
        "- Минимум шагов: не добавляй лишние.\n"
        "- Для простых запросов (показать данные, структура) → 1 шаг.\n"
        "- Для графиков → обязательно `plotly_tool`.\n"
        "- Не путай `value_tool` (метрики из df) с `search_tool` (веб-поиск).\n"
        "- Если в контексте написано «Источники данных: НЕ прикреплены» "
        "и запрос требует табличных данных — НЕ планируй tool-вызовы для анализа данных. "
        "Ответь пользователю, что нужно загрузить CSV или подключить БД.\n"
    )

    def _think_system_prompt(
        self,
        capability_context: dict[str, Any] | None = None,
        selected_skill_ids: list[str] | None = None,
        tool_descriptions: str = "",
    ) -> str:
        depth_instruction = self._depth_profile.get("think_instruction", "")
        depth_label = self.settings.agent_analysis_depth.upper()
        today = date.today().strftime("%Y-%m-%d")
        prompt = (
            self._THINK_SYSTEM_PROMPT_BASE
            + f"\n[СЕГОДНЯ: {today}]\n"
            + f"[УРОВЕНЬ АНАЛИЗА: {depth_label}]\n{depth_instruction}\n"
        )
        if tool_descriptions:
            prompt += f"\n[ДОСТУПНЫЕ ИНСТРУМЕНТЫ]\n{tool_descriptions}\n"
        capability_block = str((capability_context or {}).get("prompt_block", "")).strip()
        if capability_block:
            prompt += f"\n{capability_block}\n"
        memory_block = self.user_memory.build_block()
        if memory_block:
            prompt += f"\n{memory_block}\n"
        session_memory_block = self.session_memory.build_block()
        if session_memory_block:
            prompt += f"\n{session_memory_block}\n"
        skills_block = self.skill_registry.build_prompt_block(selected_skill_ids)
        if skills_block:
            prompt += f"\n{skills_block}\n"
        return prompt

    _EVALUATE_PROMPT_TEMPLATE = (
        "Вопрос пользователя: {question}\n"
        "План выполнения: {plan}\n"
        "Полученные артефакты: {artifact_summary}\n"
        "Доступные возможности runtime: {capability_summary}\n"
        "Использованные инструменты (включая неудачные): {used_tools}\n"
        "Текст ответа: {response_text}\n\n"
        "Ответь строго в формате JSON:\n"
        '{{"pass": true/false, "reason": "краткое обоснование"}}\n\n'
        "Критерии оценки (pass=true если выполнено хотя бы одно):\n"
        "1. Ответ прямо и по существу отвечает на вопрос пользователя.\n"
        "2. Для простых вопросов (метаданные, структура, количество строк/столбцов) "
        "достаточно корректного текстового ответа без артефактов.\n"
        "3. Для аналитических вопросов ответ подкреплён артефактами (table/plot/value) "
        "или содержит конкретные числа/факты из данных.\n"
        "4. Если capability недоступен или вернул ошибку — честное объяснение ограничения допустимо.\n"
        "5. Если план предполагал конкретные инструменты — проверь, что они были вызваны.\n"
        "pass=false только если: ответ пустой, полностью нерелевантен вопросу, "
        "содержит только технические ошибки, или LLM придумала данные не вызвав инструмент.\n"
        "Только JSON, без пояснений."
    )

    @staticmethod
    def _execution_runtime_section(
        source_mode: str,
        tool_list: str,
        today: str,
        tool_descriptions: str,
    ) -> list[str]:
        """Runtime context: date, data mode, available tools."""
        lines = [
            f"Сегодня: {today}.",
            f"Режим данных: `{source_mode}`.",
            f"Доступные tools в этом запуске: {tool_list}.",
        ]
        if tool_descriptions:
            lines += ["Описание доступных tools:", tool_descriptions]
        return lines

    @staticmethod
    def _execution_step_section(step_index: int, max_steps: int, depth: str) -> str:
        """Current step counter and analysis depth."""
        return f"Текущий шаг: {step_index}/{max_steps}. Уровень анализа: {depth}."

    @staticmethod
    def _execution_plan_section(plan: str) -> str | None:
        """Plan orientation from the THINK phase."""
        return f"Ориентир плана: {plan.strip()}" if plan.strip() else None

    @staticmethod
    def _execution_refinement_section(feedback: str) -> str | None:
        """Feedback from the previous failed attempt."""
        return (
            f"Исправь предыдущую неудачную попытку: {feedback.strip()}"
            if feedback.strip()
            else None
        )

    def _build_execution_system_prompt(
        self,
        *,
        user_prompt: str,
        plan: str,
        refinement_feedback: str,
        step_index: int,
        max_steps: int,
        capability_context: dict[str, Any] | None = None,
        sandbox: Any | None = None,
    ) -> str:
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
        # When ReAct is active, prevent double-think conflict with reasoning models.
        if self.settings.agent_react_enabled:
            sections.append(
                "ВАЖНО: НЕ используй теги <think>. "
                "Все рассуждения пиши напрямую в текст ответа или в поле Thought."
            )
        sections.extend(
            self._execution_runtime_section(source_mode, tool_list, today, tool_descriptions)
        )
        sections.append(
            self._execution_step_section(step_index, max_steps, self.settings.agent_analysis_depth)
        )
        sections.extend(
            opt
            for opt in (
                self._execution_plan_section(plan),
                self._execution_refinement_section(refinement_feedback),
            )
            if opt
        )

        # Sandbox context: available variables + session notebook.
        if sandbox:
            sandbox_block = sandbox.describe_for_prompt()
            if sandbox_block:
                sections.append(sandbox_block)

        # Brief tool descriptions — LLM must call get_tool_instructions(tool_name)
        # before first use of each tool to get full instructions and examples.
        tool_skills_block = self.skill_registry.build_tool_skills_brief_block(set(available_tools))
        if tool_skills_block:
            sections.append(tool_skills_block)

        return "\n".join(sections).strip()

    def _collect_text_and_reasoning(
        self,
        *,
        callbacks: list,
        response_payload: Any,
    ) -> tuple[str, str | None]:
        text_collector = None
        for cb in callbacks:
            if isinstance(cb, LLMTextCollector):
                text_collector = cb
                break

        output_text = extract_agent_output_text(response_payload)
        final_text = ""
        reasoning = None
        if text_collector and text_collector.messages:
            final_text = text_collector.messages[-1].get("text", "")
            reasoning = text_collector.messages[-1].get("reasoning") or None

        if not final_text:
            final_text = output_text
        if reasoning is None:
            reasoning = extract_thinking(output_text) or None
        return final_text.strip(), reasoning

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

        chosen = max(candidate_series, key=lambda item: len(item))
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

        formatted_value = self._format_numeric_value(value)
        label_text = str(label)
        return f"По таблице {qualifier} значение у '{label_text}': {formatted_value}."

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

            lines.append(f"- `{artifact_type}` `{name}`: артефакт сформирован")

        return lines

    def _value_observation_lines(
        self, value_payload: dict[str, Any], max_items: int = 6
    ) -> list[str]:
        if not value_payload:
            return []

        keys = sorted(value_payload.keys())
        lines: list[str] = [
            f"- {key}: {self._format_metric_value(value_payload[key])}"
            for key in keys[:max_items]
        ]
        return lines

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

            chosen = max(series_list, key=lambda item: len(item))
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
                    f"- `{name}`: максимум у `{max_label}` = {max_value}, минимум у `{min_label}` = {min_value}."  # noqa: E501
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
            1
            for artifact in artifacts
            if artifact_type_label(getattr(artifact, "artifact_type", "")) == "table"
        )
        plot_count = sum(
            1
            for artifact in artifacts
            if artifact_type_label(getattr(artifact, "artifact_type", "")) == "plot"
        )
        value_count = sum(
            1
            for artifact in artifacts
            if artifact_type_label(getattr(artifact, "artifact_type", "")) == "value"
        )

        direct_answer = self._table_extreme_summary(prompt, artifacts)
        if not direct_answer and value_payload:
            row_count = value_payload.get("row_count")
            column_count = value_payload.get("column_count")
            if asks_direct_answer and "сколько" in normalized:
                if isinstance(row_count, (int, float)) and (
                    "строк" in normalized or "запис" in normalized
                ):
                    direct_answer = f"В датасете {int(row_count)} строк."
                elif isinstance(column_count, (int, float)) and (
                    "столбц" in normalized or "колонк" in normalized
                ):
                    direct_answer = f"В датасете {int(column_count)} столбцов."

            if not direct_answer and (
                "датасет" in normalized or "данн" in normalized or "расскажи" in normalized
            ):
                if isinstance(row_count, (int, float)) and isinstance(column_count, (int, float)):
                    direct_answer = (
                        f"В датасете {int(row_count)} строк и {int(column_count)} столбцов."
                    )

        if not direct_answer and not self._response_looks_like_plan_or_trace(base_text):
            candidate = self._first_sentence(str(base_text or ""))
            if candidate:
                direct_answer = candidate

        if not direct_answer:
            direct_answer = "Ключевой вывод сформирован на основе полученных артефактов."

        method_lines = self._artifact_method_lines(artifacts)
        if not method_lines:
            method_lines = [
                f"- Получено артефактов: {len(artifacts)}",
                f"- По типам: table={table_count}, plot={plot_count}, value={value_count}",
            ]

        observation_lines: list[str] = []
        observation_lines.extend(
            self._value_observation_lines(value_payload, max_items=5)
        )
        observation_lines.extend(
            self._table_observation_lines(artifacts, max_items=2)
        )
        if not observation_lines:
            observation_lines = [
                f"- Построено артефактов: {len(artifacts)} (table={table_count}, plot={plot_count}, value={value_count})."  # noqa: E501
            ]

        conclusion = (
            "Итог: ответ сформирован по подтвержденным артефактам; при необходимости могу расширить анализ "
            "дополнительными срезами или детализацией по конкретным группам."
        )

        return (
            f"{direct_answer}\n\n"
            "Что сделано:\n"
            + "\n".join(method_lines)
            + "\n\n"
            + "Ключевые наблюдения:\n"
            + "\n".join(observation_lines)
            + "\n\n"
            + "Итог:\n"
            + conclusion
        )

    @staticmethod
    def _response_too_generic(prompt: str, response_text: str) -> bool:
        text = response_text.strip().lower()
        if not text:
            return True

        generic_markers = (
            "анализ выполнен",
            "артефакт",
            "построены",
            "получены метрики",
        )
        has_generic = any(marker in text for marker in generic_markers)
        asks_direct_answer = (
            prompt.strip().endswith("?")
            or any(token in prompt.lower() for token in ("в каком", "какой", "сколько", "кто"))
        )
        if asks_direct_answer and has_generic and len(text) < 180:
            return True
        return False

    @staticmethod
    def _response_looks_like_plan_or_trace(response_text: str) -> bool:
        text = str(response_text or "").strip().lower()
        if not text:
            return True

        plan_prefixes = (
            "план анализа",
            "план решения",
            "план выполнения",
            "план:",
            "plan:",
            "корректировка плана",
            "интеграция внешней",
            "извлечение ключевых",
            "что хочет пользователь?",
            "какие инструменты использовать",
            "выполняю шаг",
            "проверяю результат шага",
            "доработка через повторный цикл",
            "рассуждение (chain of thought)",
        )
        if any(text.startswith(prefix) for prefix in plan_prefixes):
            return True

        trace_markers = (
            '"name": "value_tool"',
            '"name": "pandas_tool"',
            '"name": "memory_tool"',
            '"name": "sql_tool"',
            '"name": "plotly_tool"',
            '"name": "database_tool"',
            '"name": "search_tool"',
            '"name": "forecast_tool"',
            '"artifact_type": "value"',
            '"artifact_type": "table"',
            "tool_result",
            "value_tool(",
            "pandas_tool(",
            "memory_tool(",
            "sql_tool(",
            "plotly_tool(",
            "database_tool(",
            "search_tool(",
            "forecast_tool(",
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

    @staticmethod
    def _response_has_artifact_type(
        response: AgentResponse | None,
        artifact_type: str,
    ) -> bool:
        if response is None:
            return False
        expected = str(artifact_type or "").strip().lower()
        if not expected:
            return False
        # Map legacy type names to ExecArtifactType values
        _LEGACY_ALIASES = {"table": "dataframe", "value": "scalar"}
        expected_exec = _LEGACY_ALIASES.get(expected, expected)
        for artifact in response.artifacts:
            current = artifact_type_label(getattr(artifact, "artifact_type", ""))
            if current == expected or current == expected_exec:
                return True
        return False

    @staticmethod
    def _collect_progress_collectors(callbacks: list) -> list[AgentProgressCollector]:
        result: list[AgentProgressCollector] = [
            cb for cb in callbacks if isinstance(cb, AgentProgressCollector)
        ]
        return result

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
            # Update graph tracker.
            gt = getattr(collector, "graph_tracker", None)
            if gt is not None:
                si = step_index if isinstance(step_index, int) else 0
                if status == "streaming":
                    gt.phase_start(phase, si)
                elif status in ("done", "pass", "fail", "error"):
                    gt.phase_end(phase, si, status="done" if status in ("done", "pass") else "error")
                collector._graph_version += 1  # noqa: SLF001

    @staticmethod
    def _silent_callbacks(callbacks: list) -> list:
        """Return callbacks without TokenStreamCallbackHandler so LLM
        output doesn't leak into the chat stream.  Keeps
        PhaseTokenStreamHandler so tokens stream to the activity panel."""
        return [cb for cb in callbacks if not isinstance(cb, TokenStreamCallbackHandler)]

    @staticmethod
    def _mute_all_stream_callbacks(callbacks: list) -> list:
        """Remove both chat and activity token handlers (for evaluate calls
        whose JSON output is not useful to stream)."""
        return [
            cb for cb in callbacks
            if not isinstance(cb, (TokenStreamCallbackHandler, PhaseTokenStreamHandler))
        ]

    @staticmethod
    def _emit_rag_stream_chunk(callbacks: list, chunk: str) -> None:
        if not chunk:
            return
        for cb in callbacks:
            if not isinstance(cb, TokenStreamCallbackHandler):
                continue
            try:
                cb.on_llm_new_token(chunk)
            except Exception:
                continue

    def _stream_rag_answer(
        self,
        *,
        prompt: str,
        callbacks: list,
    ) -> str:
        if self.rag_service is None:
            return ""

        parts: list[str] = []
        for chunk in self.rag_service.stream_search(
            query=prompt,
            include_references=False,
        ):
            parts.append(chunk)
            self._emit_rag_stream_chunk(callbacks, chunk)

        final_text = "".join(parts).strip()
        if final_text:
            return final_text

        result = self.rag_service.search(
            query=prompt,
            include_references=False,
        )
        return self.rag_service.format_for_user(result)

    @staticmethod
    def _artifacts_recovery_text(artifacts: list) -> str:
        if not artifacts:
            return ""

        counts: dict[str, int] = {}
        labels: list[str] = []
        for artifact in artifacts[:8]:
            artifact_type = str(getattr(artifact, "artifact_type", "artifact")).strip()
            artifact_type = artifact_type or "artifact"
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
                "Шаг анализа завершился с ограничением итераций модели, но артефакты уже построены "
                f"({typed_counts}). Доступные артефакты: {labels_preview}."
            )

        return (
            "Шаг анализа завершился с ограничением итераций модели, но артефакты уже построены "
            f"({typed_counts})."
        )

    def _analysis_step(
        self,
        *,
        df: pd.DataFrame,
        prompt: str,
        history: list[dict[str, Any]],
        use_history: bool,
        include_reasoning: bool,
        callbacks: list,
        trace_context: dict[str, Any] | None,
        tools: list,
        session_source: dict[str, Any] | None = None,
        tool_db_runtime: RuntimeDBConnectionConfig | None = None,
        selected_skill_ids: list[str] | None = None,
        execution_system_prompt: str | None = None,
    ) -> AgentResponse:
        self._reset_text_collectors(callbacks)
        llm = self._build_llm(
            role="tool",
            include_reasoning=include_reasoning,
            timeout_sec=min(
                self.settings.agent_step_timeout_sec,
                self.settings.backend_query_timeout_sec,
            ),
        )

        db_suffix = self._db_session_prompt_block(
            session_source=session_source,
            runtime=tool_db_runtime,
            df=df,
        )

        skills_block = self.skill_registry.build_prompt_block(selected_skill_ids)
        act_prefix = (execution_system_prompt or execution_agent_prompt).strip()
        if skills_block:
            act_prefix = f"{act_prefix}\n\n{skills_block}"

        depth_inner_limit = self._depth_profile.get("inner_recursion_limit")
        effective_inner_limit = (
            depth_inner_limit
            if isinstance(depth_inner_limit, int)
            else self.settings.agent_inner_recursion_limit
        )

        # Build messages first so that memory/history-summary SystemMessage
        # content can be merged into act_prefix before the agent is created.
        # This ensures create_agent receives a single unified system prompt
        # instead of two separate ones, which models like qwen reject.
        prompt_messages = self._build_messages(prompt, history, use_history)
        extra_system = next(
            (m.content for m in prompt_messages if isinstance(m, SystemMessage)), None
        )
        if extra_system:
            act_prefix = f"{act_prefix}\n\n{extra_system}".strip()
        agent_messages = [m for m in prompt_messages if not isinstance(m, SystemMessage)]

        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df.copy(),
            tools=tools,
            verbose=False,
            return_intermediate_steps=True,
            max_iterations=max(1, effective_inner_limit),
            max_execution_time=float(
                self._depth_profile.get("step_timeout_sec", self.settings.agent_step_timeout_sec)
            ),
            prefix=act_prefix,
            suffix=db_suffix or None,
            include_df_in_prompt=True,
            number_of_head_rows=min(2, max(1, self.settings.agent_prompt_head_rows)),
            data_info_max_columns=min(10, max(6, self.settings.agent_prompt_max_columns)),
        )
        # Use depth-profile override if available, else global setting.
        # Minimum 4 to allow at least: LLM call → tool → LLM response.
        recursion_limit = max(
            4,
            min(24, effective_inner_limit * 2 + 2),
        )
        runtime_config: dict[str, Any] = {
            "callbacks": callbacks,
            "recursion_limit": recursion_limit,
        }
        metadata = self._build_runtime_metadata(trace_context)
        if metadata:
            runtime_config["metadata"] = metadata

        try:
            response_payload = agent.invoke(
                {"messages": normalize_agent_messages(agent_messages)},
                config=runtime_config,
            )
            final_text, reasoning = self._collect_text_and_reasoning(
                callbacks=callbacks,
                response_payload=response_payload,
            )
        except Exception as exc:
            final_text, collector_reasoning = self._latest_collected_text(callbacks)
            artifacts, tool_calls, tool_names = self._collect_tool_stats(callbacks)
            if _is_llm_transport_failure(exc):
                _log_llm_invoke_failure("analysis ACT agent.invoke", exc, self.settings)
                if not final_text and artifacts:
                    final_text = self._artifacts_recovery_text(artifacts)
                return AgentResponse(
                    final_text=final_text.strip() or _LLM_UNAVAILABLE_USER_TEXT,
                    reasoning=str(exc),
                    artifacts=artifacts,
                    route="analysis",
                    tool_calls=tool_calls,
                    tool_names=tool_names,
                    llm_unreachable=True,
                )
            reasoning_parts: list[str] = []
            if collector_reasoning:
                reasoning_parts.append(collector_reasoning)
            if artifacts:
                reasoning_parts.append(
                    "Итерационный лимит шага достигнут после получения артефактов; "
                    "используется артефакт-ориентированный ответ."
                )
            else:
                reasoning_parts.append(f"ACT step failed: {exc}")
            reasoning = "\n\n".join(reasoning_parts).strip()
            if not final_text and artifacts:
                final_text = self._artifacts_recovery_text(artifacts)
            return AgentResponse(
                final_text=final_text.strip(),
                reasoning=reasoning,
                artifacts=artifacts,
                route="analysis",
                tool_calls=tool_calls,
                tool_names=tool_names,
            )

        artifacts, tool_calls, tool_names = self._collect_tool_stats(callbacks)
        if not final_text and artifacts:
            final_text = self._artifacts_recovery_text(artifacts)

        return AgentResponse(
            final_text=final_text,
            reasoning=reasoning,
            artifacts=artifacts,
            route="analysis",
            tool_calls=tool_calls,
            tool_names=tool_names,
        )

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
        """Direct tool-calling loop using bind_tools — no ReAct Thought/Action format.

        The LLM decides which tools to call via its native tool_use protocol.
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

        runtime_config: dict[str, Any] = {"callbacks": callbacks}
        metadata = self._build_runtime_metadata(trace_context)
        if metadata:
            runtime_config["metadata"] = metadata

        for iteration in range(max(1, max_iterations)):
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
                        artifacts=artifacts,
                        route="analysis",
                        tool_calls=total_tool_calls + tc,
                        tool_names=all_tool_names + tn,
                        llm_unreachable=True,
                    )
                raise

            # Extract reasoning from ThinkingAwareChatOpenAI
            if reasoning is None:
                reasoning = response.additional_kwargs.get("reasoning") or None

            # Check for tool calls
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                # No tool calls — LLM is done. Extract final text.
                final_text = self._content_to_text(getattr(response, "content", ""))
                break

            # Append the AI message with tool calls to context
            messages.append(response)

            # Execute each tool call
            for tc in tool_calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("args", {})
                tool_call_id = tc.get("id", "")
                total_tool_calls += 1
                if tool_name not in all_tool_names:
                    all_tool_names.append(tool_name)

                tool = tool_map.get(tool_name)
                if tool is None:
                    result_text = f"Unknown tool: {tool_name}"
                else:
                    try:
                        result = tool.invoke(tool_args, config=runtime_config)
                        # BaseTool returns (content, artifact) tuple or string
                        if isinstance(result, tuple):
                            result_text = str(result[0]) if result[0] else str(result[1])
                        else:
                            result_text = str(result)
                    except Exception as tool_exc:
                        result_text = f"Tool error: {tool_exc}"

                messages.append(ToolMessage(
                    content=result_text,
                    tool_call_id=tool_call_id,
                ))
        else:
            # Loop exhausted without LLM stopping — extract what we have
            text_collector = next(
                (cb for cb in callbacks if isinstance(cb, LLMTextCollector)), None
            )
            if text_collector and text_collector.messages:
                final_text = text_collector.messages[-1].get("text", "")

        # Collect artifacts from ToolCollector
        artifacts, tc_count, tn_list = self._collect_tool_stats(callbacks)
        total_tool_calls += tc_count
        for name in tn_list:
            if name not in all_tool_names:
                all_tool_names.append(name)

        if not final_text and artifacts:
            final_text = self._artifacts_recovery_text(artifacts)

        return AgentResponse(
            final_text=final_text.strip(),
            reasoning=reasoning,
            artifacts=artifacts,
            route="analysis",
            tool_calls=total_tool_calls,
            tool_names=all_tool_names,
        )

    def _build_query_graph(self):
        graph = StateGraph(AgentGraphState)

        graph.add_node("rag", self._rag_node)
        graph.add_node("summary", self._summary_node)
        graph.add_node("think", self._think_node)
        graph.add_node("act", self._act_node)
        graph.add_node("finalize", self._finalize_node)

        graph.add_edge(START, "think")
        graph.add_edge("rag", "finalize")
        graph.add_edge("summary", "finalize")

        graph.add_conditional_edges(
            "think",
            self._think_edge,
            {"act": "act", "rag": "rag", "summary": "summary", "finalize": "finalize"},
        )
        # Agent runs once, then goes directly to finalize (no evaluate/decide loop).
        graph.add_edge("act", "finalize")

        graph.add_edge("finalize", END)
        return graph.compile()

    @staticmethod
    def _think_edge(
        state: AgentGraphState,
    ) -> Literal["act", "rag", "summary", "finalize"]:
        route = state.get("route")
        if route == "chat":
            return "finalize"
        if route == "rag":
            return "rag"
        if route == "summary":
            return "summary"
        return "finalize" if state.get("llm_unreachable") else "act"

    @staticmethod
    def _act_edge(state: AgentGraphState) -> Literal["evaluate", "finalize"]:
        response = state.get("response")
        if response is not None and getattr(response, "llm_unreachable", False):
            return "finalize"
        return "evaluate"

    def _rag_node(self, state: AgentGraphState) -> dict[str, Any]:
        prompt = state.get("prompt", "")
        callbacks = state.get("callbacks", [])
        df = state.get("df")

        self._emit_phase_event(
            callbacks,
            phase="act",
            title="Поиск по базе знаний",
            content="",
            step_index=0,
            max_steps=1,
            status="streaming",
        )
        self._emit_progress_event(
            callbacks,
            phase="act",
            title="Ищу ответ в базе знаний",
            details="Отправляю запрос в RAG сервис.",
            step_index=0,
            max_steps=1,
        )

        if not self._tool_allowed("rag_tool"):
            response = AgentResponse(
                final_text=(
                    "Не могу выполнить поиск по базе знаний: RAG интеграция "
                    "отключена в настройках аккаунта."
                ),
                reasoning="RAG route selected but rag_tool is disabled for the user.",
                artifacts=[],
                route="rag",
                tool_calls=0,
                tool_names=[],
            )
            self._emit_phase_event(
                callbacks,
                phase="act",
                title="Поиск по базе знаний",
                content="RAG отключен для текущего пользователя.",
                step_index=0,
                max_steps=1,
                status="fail",
            )
            return {
                "response": response,
                "done": True,
                "stop_reason": "rag_disabled",
            }

        if self.rag_service is None or not self.rag_service.is_enabled:
            response = AgentResponse(
                final_text=self._fallback_text(prompt, df),
                reasoning="RAG route selected but RAG service is not configured.",
                artifacts=[],
                route="rag",
                tool_calls=0,
                tool_names=[],
            )
            self._emit_phase_event(
                callbacks,
                phase="act",
                title="Поиск по базе знаний",
                content="RAG сервис не настроен.",
                step_index=0,
                max_steps=1,
                status="fail",
            )
            return {
                "response": response,
                "done": True,
                "stop_reason": "rag_unavailable",
            }

        try:
            final_text = self._stream_rag_answer(
                prompt=prompt,
                callbacks=callbacks,
            )
            if not final_text.strip():
                final_text = "RAG вернул пустой ответ."

            response = AgentResponse(
                final_text=final_text,
                reasoning=None,
                artifacts=[],
                route="rag",
                tool_calls=0,
                tool_names=[],
            )

            self._emit_phase_event(
                callbacks,
                phase="act",
                title="Поиск по базе знаний",
                content="RAG-ответ получен.",
                step_index=0,
                max_steps=1,
                status="done",
            )
            self._emit_progress_event(
                callbacks,
                phase="act",
                title="Ответ из базы знаний готов",
                details="Передаю результат в финализацию.",
                step_index=0,
                max_steps=1,
            )
            return {
                "response": response,
                "done": True,
                "stop_reason": "rag_ready",
            }
        except Exception as exc:
            self._emit_phase_event(
                callbacks,
                phase="act",
                title="Поиск по базе знаний",
                content=f"Ошибка RAG: {exc}",
                step_index=0,
                max_steps=1,
                status="fail",
            )
            response = AgentResponse(
                final_text=self._fallback_text(prompt, df),
                reasoning=f"RAG route failed: {exc}",
                artifacts=[],
                route="rag",
                tool_calls=0,
                tool_names=[],
            )
            return {
                "response": response,
                "done": True,
                "stop_reason": "rag_failed",
            }

    def _summary_node(self, state: AgentGraphState) -> dict[str, AgentResponse]:
        callbacks = state.get("callbacks", [])

        self._emit_phase_event(
            callbacks,
            phase="act",
            title="Формирование управленческой записки",
            content="",
            step_index=0,
            max_steps=1,
            status="streaming",
        )
        self._emit_progress_event(
            callbacks,
            phase="act",
            title="Собираю управленческую записку",
            details="Анализирую релевантную историю переписки и артефакты.",
            step_index=0,
            max_steps=1,
        )

        response = self._build_management_note(
            prompt=state.get("prompt", ""),
            history=state.get("history", []),
            include_reasoning=state.get("include_reasoning", False),
            callbacks=callbacks,
            trace_context=state.get("trace_context"),
        )

        self._emit_phase_event(
            callbacks,
            phase="act",
            title="Формирование управленческой записки",
            content="Управленческая записка сформирована.",
            step_index=0,
            max_steps=1,
            status="done",
        )

        return {
            "response": response,
            "done": True,
            "stop_reason": "summary_ready",
        }

    @staticmethod
    def _quick_route(
        prompt: str,
        *,
        has_rag: bool,
        has_data: bool = False,
    ) -> Literal["chat", "rag", "summary"] | None:
        """Lightweight keyword pre-check for chat/rag/summary. No LLM call."""
        normalized = prompt.strip().lower()
        _summary_markers = (
            "управленческ", "итоги анализа",
            "резюмируй", "подведи итог", "executive summary", "сделай отчёт",
            "сводка по результатам", "ключевые выводы из чата",
        )
        if any(m in normalized for m in _summary_markers):
            return "summary"
        if has_rag:
            _rag_markers = (
                "rag ", " rag", "документац", "база знаний", "в базе знаний",
                "knowledge base", "в документации", "из документации",
            )
            if any(m in normalized for m in _rag_markers):
                return "rag"
        if AgentRunner._is_chat_message(normalized, has_data=has_data):
            return "chat"
        return None

    @staticmethod
    def _is_chat_message(normalized_prompt: str, *, has_data: bool = False) -> bool:
        """Detect greetings and simple chat messages that don't need tools."""
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
        prompt = normalized_prompt.strip()
        if not prompt:
            return True
        # Short greetings
        if any(prompt.startswith(g) or prompt == g for g in _CHAT_GREETINGS):
            return True
        # Questions about the assistant
        if any(m in prompt for m in _CHAT_ABOUT_SELF):
            return True
        # Very short messages without data context are likely chat
        if len(prompt) < 12 and not has_data:
            return True
        return False

    def _think_node(self, state: AgentGraphState) -> dict[str, Any]:
        """Router node — lightweight keyword routing, tool building, NO LLM call.

        Routes to: chat, rag, summary, or agent (analysis).
        Planning is delegated to planner_tool when the LLM decides it is needed.
        """
        df = state.get("df")
        prompt = state.get("prompt", "")
        callbacks = state.get("callbacks", [])

        # ── Fast keyword routing (no LLM call) ──────────────────────────
        has_rag = self.rag_service is not None and self.rag_service.is_enabled
        tool_db_runtime = self._resolve_tool_db_runtime_config(
            state.get("session_source"),
            state.get("trace_context"),
        )
        csv_loaded, csv_session_id = self._resolve_csv_runtime_state(
            state.get("session_source"),
            state.get("trace_context"),
        )
        has_data = bool(
            df is not None
            or tool_db_runtime is not None
            or (csv_loaded and str(csv_session_id or "").strip())
        )

        quick = self._quick_route(prompt, has_rag=has_rag, has_data=has_data)
        if quick == "chat":
            return {"route": "chat", "done": False}
        if quick is not None:
            return {"route": quick, "done": False}

        # ── Build tools and context for the agent node ───────────────────
        csv_duckdb_mode = bool(csv_loaded and str(csv_session_id or "").strip())
        tool_df = None if csv_duckdb_mode else df

        trace_ctx = state.get("trace_context") or {}
        session_id = trace_ctx.get("session_id", "default")
        sandbox = SandboxManager.get_instance().get_or_create(session_id)
        sandbox.ensure_storage_dir(Path(self.settings.storage_dir) / session_id)
        if df is not None:
            source_label = str(trace_ctx.get("dataset_name", "") or "")
            sandbox.bind_dataframe(df, source_label=source_label, db_runtime_config=tool_db_runtime)

        _ctx = ToolBuildContext(
            settings=self.settings,
            allowed_tool_keys=self.allowed_tool_keys,
            df=tool_df,
            tool_db_runtime=tool_db_runtime,
            csv_loaded=csv_loaded,
            csv_session_id=csv_session_id,
            sandbox=sandbox,
        )
        tools: list = self._tool_registry.build_tools(_ctx)
        tool_descriptions = self._tool_registry.describe_available_tools(_ctx)

        max_steps = self._effective_outer_max_steps(prompt=prompt)

        tool_keys = [
            str(getattr(tool, "name", "")).strip()
            for tool in tools
            if str(getattr(tool, "name", "")).strip()
        ]
        capability_context = build_runtime_capability_context(
            available_tool_keys=tool_keys,
            has_dataframe=tool_df is not None,
            has_db_source=(tool_db_runtime is not None) or csv_duckdb_mode,
        )
        capability_context["tool_descriptions"] = tool_descriptions

        return {
            "plan": "",
            "max_steps": max_steps,
            "done": False,
            "eval_passed": False,
            "eval_reason": "",
            "stop_reason": "",
            "llm_unreachable": False,
            "tools": tools,
            "step_index": 0,
            "sandbox": sandbox,
            "capability_context": capability_context,
        }

    def _act_node(self, state: AgentGraphState) -> dict[str, Any]:
        df = state.get("df")
        tools = state.get("tools", [])
        if df is None and not tools:
            # No data and no tools — delegate to the plain chat LLM.
            prompt = state.get("prompt", "")
            try:
                response = self.chat(
                    prompt=prompt,
                    history=state.get("history", []),
                    use_history=state.get("use_history", True),
                    include_reasoning=state.get("include_reasoning", False),
                    callbacks=state.get("callbacks", []),
                    trace_context=state.get("trace_context"),
                )
            except Exception:
                response = AgentResponse(
                    final_text=self._fallback_text(prompt, df),
                    reasoning=None,
                    artifacts=[],
                    route="chat",
                )
            return {"response": response, "done": True, "stop_reason": "chat_route"}

        step_index = int(state.get("step_index", 0)) + 1
        plan = state.get("plan", "")
        refinement_feedback = state.get("refinement_feedback", "")
        callbacks = state.get("callbacks", [])
        max_steps = int(state.get("max_steps", self.settings.agent_max_steps))
        tool_df = df if df is not None else pd.DataFrame()
        tool_db_runtime = self._resolve_tool_db_runtime_config(
            state.get("session_source"),
            state.get("trace_context"),
        )

        sandbox = state.get("sandbox")
        execution_system_prompt = self._build_execution_system_prompt(
            user_prompt=state.get("prompt", ""),
            plan=plan,
            refinement_feedback=refinement_feedback,
            step_index=step_index,
            max_steps=max_steps,
            capability_context=state.get("capability_context"),
            sandbox=sandbox,
        )

        # Add explicit data context so the LLM knows what dataset/DB is attached.
        # This mirrors what the old _think_node LLM plan call used to expose.
        data_context_parts: list[str] = []
        if df is not None:
            try:
                data_context_parts.append(
                    get_detailed_data_info(
                        df, max_columns=self.settings.agent_prompt_max_columns
                    )
                )
            except Exception:
                data_context_parts.append(
                    f"Датасет: {df.shape[0]} строк, {df.shape[1]} столбцов."
                )
        db_block = self._db_session_prompt_block(
            session_source=state.get("session_source"),
            runtime=tool_db_runtime,
            df=df,
        )
        if db_block:
            data_context_parts.append(db_block)
        if data_context_parts:
            execution_system_prompt = (
                execution_system_prompt + "\n\n" + "\n\n".join(data_context_parts)
            )

        self._emit_phase_event(
            callbacks,
            phase="act",
            title=f"Выполнение шага {step_index}",
            content="",
            step_index=step_index,
            max_steps=max_steps,
            status="streaming",
        )
        self._emit_progress_event(
            callbacks,
            phase="act",
            title=f"Выполняю шаг {step_index}",
            details="Подбираю инструмент и формирую вызов tool.",
            step_index=step_index,
            max_steps=max_steps,
        )

        tool_collector = next((cb for cb in callbacks if isinstance(cb, ToolCollector)), None)
        tool_events_offset = len(tool_collector.events) if tool_collector else 0

        started_at = time.perf_counter()
        try:
            if self.settings.agent_react_enabled:
                # ReAct mode: use LangChain pandas agent (Thought/Action/Observation)
                response = self._analysis_step(
                    df=tool_df,
                    prompt=state.get("prompt", ""),
                    history=state.get("history", []),
                    use_history=state.get("use_history", True),
                    include_reasoning=state.get("include_reasoning", False),
                    callbacks=callbacks,
                    trace_context=state.get("trace_context"),
                    tools=tools,
                    session_source=state.get("session_source"),
                    tool_db_runtime=tool_db_runtime,
                    selected_skill_ids=state.get("selected_skill_ids"),
                    execution_system_prompt=execution_system_prompt,
                )
            else:
                # Direct tool-calling mode (default): bind_tools loop
                depth_inner_limit = self._depth_profile.get("inner_recursion_limit")
                effective_inner_limit = (
                    depth_inner_limit
                    if isinstance(depth_inner_limit, int)
                    else self.settings.agent_inner_recursion_limit
                )
                response = self._direct_tool_loop(
                    prompt=state.get("prompt", ""),
                    history=state.get("history", []),
                    use_history=state.get("use_history", True),
                    include_reasoning=state.get("include_reasoning", False),
                    tools=tools,
                    execution_system_prompt=execution_system_prompt,
                    callbacks=callbacks,
                    max_iterations=max(1, effective_inner_limit),
                    trace_context=state.get("trace_context"),
                )
        except Exception as exc:
            artifacts, tool_calls, tool_names = self._collect_tool_stats(callbacks)
            response = AgentResponse(
                final_text=self._artifacts_recovery_text(artifacts),
                reasoning=f"ACT step failed: {exc}",
                artifacts=artifacts,
                route="analysis",
                tool_calls=tool_calls,
                tool_names=tool_names,
            )

        elapsed_sec = time.perf_counter() - started_at
        if elapsed_sec > max(1, self.settings.agent_step_timeout_sec):
            response.reasoning = (
                (response.reasoning or "")
                + "\n\n"
                + f"Step timeout guard triggered ({int(elapsed_sec * 1000)} ms)."
            ).strip()

        tool_summary_lines: list[str] = []
        if tool_collector is not None:
            step_tool_events = tool_collector.events[tool_events_offset:]
            for ev in step_tool_events:
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

        self._emit_phase_event(
            callbacks,
            phase="act",
            title=f"Шаг {step_index} завершён",
            content="\n".join(tool_summary_lines) if tool_summary_lines else "Шаг выполнен.",
            step_index=step_index,
            max_steps=max_steps,
            status="done",
        )

        return {
            "response": response,
            "step_index": step_index,
        }

    def _evaluate_node(self, state: AgentGraphState) -> dict[str, Any]:
        response = state.get("response")
        callbacks = state.get("callbacks", [])
        prompt = state.get("prompt", "")
        step_index = int(state.get("step_index", 0))
        max_steps = int(state.get("max_steps", self.settings.agent_max_steps))
        capability_context = state.get("capability_context") or {}

        if response is None:
            self._emit_phase_event(
                callbacks,
                phase="evaluate",
                title="Оценка результата",
                content="Нет ответа после шага ACT.",
                step_index=step_index,
                max_steps=max_steps,
                status="fail",
            )
            return {
                "eval_passed": False,
                "eval_reason": "Нет ответа после шага ACT.",
            }

        has_confirmed_output = self._has_confirmed_analysis_output(response)
        if has_confirmed_output and (
            self._response_too_generic(prompt, response.final_text)
            or self._response_looks_like_plan_or_trace(response.final_text)
        ):
            grounded_summary = self._artifact_grounded_summary(
                prompt,
                response.artifacts,
                base_text=response.final_text,
            )
            if grounded_summary:
                response.final_text = grounded_summary

        if not response.final_text.strip():
            # If confirmed artifacts exist, treat empty text as acceptable — finalize will generate summary.
            if has_confirmed_output:
                response.final_text = self._artifact_grounded_summary(
                    prompt, response.artifacts, base_text=response.final_text,
                ) or "Результат получен."
            else:
                reason = "Пустой финальный ответ"
                self._emit_phase_event(
                    callbacks, phase="evaluate", title="Оценка результата",
                    content=reason, step_index=step_index, max_steps=max_steps, status="fail",
                )
                return {"eval_passed": False, "eval_reason": reason}

        # If no tools were called at all and plan expected tool usage, retry.
        if response.tool_calls == 0 and not has_confirmed_output:
            plan = state.get("plan", "")
            if plan and any(t in plan for t in ("_tool", "tool_")):
                reason = "План предполагал вызов инструмента, но ни один не был вызван. ОБЯЗАТЕЛЬНО вызови инструмент из плана."  # noqa: E501
                self._emit_phase_event(
                    callbacks, phase="evaluate", title="Оценка результата",
                    content=reason, step_index=step_index, max_steps=max_steps, status="fail",
                )
                return {"eval_passed": False, "eval_reason": reason}

        # Structural check: visualization prompt requires a plot artifact.
        # Runs regardless of agent_evaluate_enabled so the agent retries if needed.
        _plot_keywords = ("график", "диаграмм", "plot", "chart", "визуализ", "гистограмм", "scatter")
        _prompt_lower = prompt.strip().lower()
        _needs_plot = any(kw in _prompt_lower for kw in _plot_keywords)
        _has_plot = self._response_has_artifact_type(response, "plot")
        _plot_available = (
            "plotly_tool" in (capability_context.get("available_tool_keys") or [])
            or self._tool_allowed("plotly_tool")
        )
        if _needs_plot and not _has_plot and _plot_available and step_index < max_steps:
            reason = "Запрос требует plot-артефакт, но получена только таблица или значение. Вызови `plotly_tool`."  # noqa: E501
            self._emit_phase_event(
                callbacks, phase="evaluate", title="Оценка результата",
                content=reason, step_index=step_index, max_steps=max_steps, status="fail",
            )
            return {"eval_passed": False, "eval_reason": reason}

        depth_eval = self._depth_profile.get("evaluate_enabled", True)
        if not self.settings.agent_evaluate_enabled or not depth_eval:
            self._emit_phase_event(
                callbacks, phase="evaluate", title="Оценка результата",
                content=f"LLM-оценка отключена (уровень: {self.settings.agent_analysis_depth}).",
                step_index=step_index, max_steps=max_steps, status="pass",
            )
            return {"eval_passed": True, "eval_reason": "LLM evaluate disabled, pre-check passed"}

        artifact_lines = self._artifact_method_lines(response.artifacts, max_items=6)
        artifact_summary = "\n".join(artifact_lines) if artifact_lines else "нет артефактов"

        used_tools_text = ", ".join(sorted(set(response.tool_names))) if response.tool_names else "нет"
        evaluate_prompt = self._EVALUATE_PROMPT_TEMPLATE.format(
            question=prompt.strip(),
            plan=self._truncate(state.get("plan", "нет"), 400),
            artifact_summary=artifact_summary,
            capability_summary=str(capability_context.get("prompt_block", "нет")),
            used_tools=used_tools_text,
            response_text=self._truncate(response.final_text, 1200),
        )

        self._emit_phase_event(
            callbacks, phase="evaluate", title="Оценка результата",
            content="", step_index=step_index, max_steps=max_steps, status="streaming",
        )

        try:
            llm = self._build_llm(
                role="evaluate",
                include_reasoning=False,
                timeout_sec=min(15, self.settings.agent_step_timeout_sec),
            )
            muted_cbs = self._mute_all_stream_callbacks(callbacks)
            runtime_config: dict[str, Any] = {"callbacks": muted_cbs}
            metadata = self._build_runtime_metadata(state.get("trace_context"))
            if metadata:
                runtime_config["metadata"] = metadata

            llm_response = llm.invoke(
                [SystemMessage(content="Ты оцениваешь, насколько ответ агента соответствует запросу пользователя. Отвечай только JSON."),  # noqa: E501
                 HumanMessage(content=evaluate_prompt)],
                config=runtime_config,
            )
            record_llm_usage_on_active_span(
                llm_response,
                fallback_model=self.settings.llm_model,
                fallback_provider=self.settings.llm_provider,
            )
            raw_text = self._content_to_text(getattr(llm_response, "content", "")).strip()

            json_match = re.search(r"\{[^}]+\}", raw_text)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                parsed = json.loads(raw_text)

            passed = bool(parsed.get("pass", False))
            reason = str(parsed.get("reason", "")).strip() or ("ok" if passed else "fail")
        except Exception as exc:
            passed = True
            reason = f"Ошибка LLM-оценки ({exc}), принимаем результат."

        display_content = f"{'✅ Принято' if passed else '❌ Не принято'}: {reason}"

        self._emit_phase_event(
            callbacks, phase="evaluate", title="Оценка результата",
            content=display_content, step_index=step_index, max_steps=max_steps, status="done",
        )
        self._emit_progress_event(
            callbacks, phase="observe", title="Проверяю результат шага",
            details=display_content, step_index=step_index, max_steps=max_steps,
        )
        return {"eval_passed": passed, "eval_reason": reason}

    def _decide_node(self, state: AgentGraphState) -> dict[str, Any]:
        step_index = int(state.get("step_index", 0))
        max_steps = int(state.get("max_steps", self.settings.agent_max_steps))
        response = state.get("response")
        eval_passed = bool(state.get("eval_passed", False))
        eval_reason = state.get("eval_reason", "")
        callbacks = state.get("callbacks", [])

        if response is None:
            return {"done": True, "stop_reason": "empty_response"}

        if getattr(response, "llm_unreachable", False):
            return {"done": True, "stop_reason": "llm_unreachable"}

        has_confirmed_output = self._has_confirmed_analysis_output(response)

        # Fast exit: if we have confirmed artifacts, stop even without text
        # (finalize_node will generate summary text from artifacts).
        if has_confirmed_output and eval_passed:
            self._emit_progress_event(
                callbacks, phase="decide",
                title="Завершаю цикл ReAct",
                details="Артефакты получены, перехожу к финализации.",
                step_index=step_index, max_steps=max_steps,
            )
            return {"done": True, "stop_reason": "ready"}

        if eval_passed and response.final_text.strip():
            self._emit_progress_event(
                callbacks, phase="decide",
                title="Завершаю цикл ReAct",
                details="Оценка пройдена, перехожу к финализации ответа.",
                step_index=step_index, max_steps=max_steps,
            )
            return {"done": True, "stop_reason": "ready"}

        if step_index >= max_steps:
            self._emit_progress_event(
                callbacks, phase="decide",
                title="Достигнут лимит шагов",
                details="Останавливаю цикл и формирую лучший доступный итог.",
                step_index=step_index, max_steps=max_steps,
            )
            return {
                "done": True,
                "stop_reason": "max_steps_reached" if has_confirmed_output else "eval_failed",
            }

        reasoning_lower = str(response.reasoning or "").lower()
        if "recursion limit" in reasoning_lower and not has_confirmed_output:
            self._emit_progress_event(
                callbacks, phase="decide",
                title="Остановка по лимиту рекурсии",
                details="Не удалось получить подтвержденный артефакт до лимита.",
                step_index=step_index, max_steps=max_steps,
            )
            return {"done": True, "stop_reason": "act_recursion_limit"}

        # If no tools were called for 2+ consecutive retry steps, LLM is stuck — stop looping.
        if step_index > 2 and response.tool_calls == 0:
            self._emit_progress_event(
                callbacks, phase="decide",
                title="Остановка: инструменты не вызваны",
                details="LLM не вызвала ни одного инструмента повторно, retry бессмысленен.",
                step_index=step_index, max_steps=max_steps,
            )
            return {"done": True, "stop_reason": "no_tool_calls"}

        feedback = eval_reason or "Результат не прошёл оценку."
        self._emit_progress_event(
            callbacks, phase="decide",
            title="Доработка через повторный цикл",
            details=f"Причина: {feedback}",
            step_index=step_index, max_steps=max_steps,
        )
        return {
            "done": False,
            "refinement_feedback": feedback,
            "stop_reason": "retry",
        }

    @staticmethod
    def _decide_edge(state: AgentGraphState) -> Literal["think", "finalize"]:
        return "finalize" if state.get("done", False) else "think"

    def _finalize_node(self, state: AgentGraphState) -> dict[str, AgentResponse]:
        callbacks = state.get("callbacks", [])
        step_index = int(state.get("step_index", 0))
        max_steps = int(state.get("max_steps", self.settings.agent_max_steps))

        self._emit_phase_event(
            callbacks, phase="finalize", title="Финализация",
            content="", step_index=step_index, max_steps=max_steps, status="streaming",
        )
        self._emit_progress_event(
            callbacks, phase="finalize",
            title="Формирую финальный ответ",
            details="Собираю выводы только по подтвержденным артефактам.",
            step_index=step_index, max_steps=max_steps,
        )

        response = state.get("response")
        route = state.get("route", "chat")
        prompt = state.get("prompt", "")
        df = state.get("df")
        stop_reason = state.get("stop_reason")

        if state.get("llm_unreachable") and response is None:
            self._emit_phase_event(
                callbacks,
                phase="finalize",
                title="Финализация",
                content=_LLM_UNAVAILABLE_USER_TEXT,
                step_index=step_index,
                max_steps=max_steps,
                status="done",
            )
            self._emit_progress_event(
                callbacks,
                phase="finalize",
                title="LLM недоступна",
                details=_LLM_UNAVAILABLE_USER_TEXT,
                step_index=step_index,
                max_steps=max_steps,
            )
            return {
                "response": AgentResponse(
                    final_text=_LLM_UNAVAILABLE_USER_TEXT,
                    reasoning="think/plan: LLM invoke failed",
                    artifacts=[],
                    route=route,
                    tool_calls=0,
                    tool_names=[],
                    llm_unreachable=True,
                )
            }

        if response is not None and getattr(response, "llm_unreachable", False):
            text = (response.final_text or "").strip() or _LLM_UNAVAILABLE_USER_TEXT
            self._emit_phase_event(
                callbacks,
                phase="finalize",
                title="Финализация",
                content=text,
                step_index=step_index,
                max_steps=max_steps,
                status="done",
            )
            self._emit_progress_event(
                callbacks,
                phase="finalize",
                title="LLM недоступна",
                details=text,
                step_index=step_index,
                max_steps=max_steps,
            )
            return {"response": response}

        if response is None and route == "chat":
            # Chat route: bypass graph, call LLM directly.
            try:
                chat_response = self.chat(
                    prompt=prompt,
                    history=state.get("history", []),
                    use_history=state.get("use_history", True),
                    include_reasoning=state.get("include_reasoning", False),
                    callbacks=callbacks,
                    trace_context=state.get("trace_context"),
                )
            except Exception:
                chat_response = AgentResponse(
                    final_text=self._fallback_text(prompt, df),
                    reasoning=None,
                    artifacts=[],
                    route="chat",
                )
            self._emit_phase_event(
                callbacks, phase="finalize", title="Финализация",
                content="Ответ сформирован.",
                step_index=step_index, max_steps=max_steps, status="done",
            )
            return {"response": chat_response}

        if response is None:
            self._emit_phase_event(
                callbacks, phase="finalize", title="Финализация",
                content="Нет ответа от агента, формирую fallback.",
                step_index=step_index, max_steps=max_steps, status="done",
            )
            return {
                "response": AgentResponse(
                    final_text=self._fallback_text(prompt, df, stop_reason=stop_reason),
                    reasoning="No response produced by graph.",
                    artifacts=[],
                    route=route,
                )
            }

        has_confirmed_output = self._has_confirmed_analysis_output(response)

        _search_tools = {"search_tool"}
        _used_search = any(t in (response.tool_names or []) for t in _search_tools)

        if route == "analysis":
            if response.artifacts:
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
                    or self._response_too_generic(prompt, response.final_text)
                    or prompt.strip().endswith("?")
                    or _looks_like_plan
                    or _used_search  # always synthesize when search was used
                )
                if should_rewrite:
                    grounded_summary = self._artifact_grounded_summary(
                        prompt,
                        response.artifacts,
                        base_text=response.final_text,
                    )
                    if grounded_summary:
                        response.final_text = grounded_summary

            if not has_confirmed_output:
                prior_reasoning = (response.reasoning or "").strip()
                prior_tool_calls = response.tool_calls
                prior_tool_names = response.tool_names
                try:
                    response = self.chat(
                        prompt=prompt,
                        history=state.get("history", []),
                        use_history=state.get("use_history", True),
                        include_reasoning=state.get("include_reasoning", False),
                        callbacks=callbacks,
                        trace_context=state.get("trace_context"),
                    )
                    response.route = "analysis"
                    response.tool_calls = prior_tool_calls
                    response.tool_names = prior_tool_names
                    if prior_reasoning:
                        response.reasoning = (
                            f"{prior_reasoning}\n\nFallback: no confirmed artifacts, answered via chat."
                        )
                except Exception:
                    fallback_text = self._fallback_text(prompt, df, stop_reason=stop_reason)
                    reason_suffix = f"Tool-required policy enforced: {stop_reason or 'missing artifacts'}."
                    reasoning = f"{prior_reasoning}\n\n{reason_suffix}".strip()
                    response = AgentResponse(
                        final_text=fallback_text,
                        reasoning=reasoning,
                        artifacts=[],
                        route="analysis",
                        tool_calls=prior_tool_calls,
                        tool_names=prior_tool_names,
                    )
            elif not bool(state.get("eval_passed", False)):
                reasoning = (response.reasoning or "").strip()
                eval_reason = state.get("eval_reason", "")
                output_note = (
                    "но подтвержденный результат уже получен."
                    if not response.artifacts
                    else "но подтвержденные артефакты уже построены."
                )
                response.reasoning = (
                    f"{reasoning}\n\nFinalize note: оценка не пройдена ({eval_reason}), "
                    f"{output_note}"
                ).strip()

        if not response.final_text.strip():
            response.final_text = self._fallback_text(prompt, df, stop_reason=stop_reason)

        self._emit_phase_event(
            callbacks, phase="finalize", title="Финализация",
            content="Ответ сформирован.",
            step_index=step_index, max_steps=max_steps, status="done",
        )
        return {"response": response}

    def chat(
        self,
        prompt: str,
        history: list[dict[str, Any]],
        use_history: bool,
        include_reasoning: bool,
        callbacks: list,
        trace_context: dict[str, Any] | None = None,
    ) -> AgentResponse:
        llm = self._build_llm(role="chat", include_reasoning=include_reasoning)
        prompt_messages = self._build_messages(prompt, history, use_history, system_prompt=chat_system_prompt)
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
        cache_allowed = self.settings.agent_cache_enabled and request_kind != "stream"

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
            df,
            prompt,
            session_source=session_source,
            trace_context=trace_context,
        )
        if data_tools_disabled is not None:
            if cache_allowed:
                self._cache_set(cache_key, data_tools_disabled)
            return data_tools_disabled

        # Simplified graph: think → act → finalize (no evaluate/decide loop).
        # 3 supersteps max. Keep a safe margin.
        _outer_recursion_limit = 20
        try:
            result = self._graph.invoke(
                {
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
                config={"recursion_limit": _outer_recursion_limit},
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
            # Warmup is best-effort; backend should stay available even if model is cold.
            return
