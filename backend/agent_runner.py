from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

import pandas as pd
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agent.pandas_agent import (
    create_pandas_dataframe_agent,
    extract_agent_output_text,
    normalize_agent_messages,
)
from agent.prompts import agent_prompt, get_detailed_data_info as _get_data_info
from agent.tools import DBTool, PandasTool, PlotlyTool, ValueTool
from backend.callbacks import (
    AgentProgressCollector,
    LLMTextCollector,
    PhaseCollector,
    PhaseTokenStreamHandler,
    TokenStreamCallbackHandler,
    ToolCollector,
    extract_thinking,
    strip_thinking,
)
from backend.config import Settings
from backend.db_runtime_service import DBRuntimeService, RuntimeDBConnectionConfig
from backend.observability import record_llm_usage_on_active_span


ANALYTICAL_HINTS = (
    "таблиц",
    "таблица",
    "график",
    "графика",
    "графики",
    "диаграмм",
    "распредел",
    "корреляц",
    "средн",
    "медиан",
    "сумм",
    "посчитай",
    "агрег",
    "pivot",
    "hist",
    "scatter",
    "plot",
    "выживаем",
    "статист",
    "dataset",
    "датасет",
    "данных",
)
INSIGHT_HINTS = (
    "инсайт",
    "insight",
    "вывод",
    "observations",
    "наблюден",
    "паттерн",
    "гипотез",
    "что интересного",
    "что можно сказать",
    "комплекс",
    "полный анализ",
)

CHAT_HINTS_RE = re.compile(
    r"^(привет|здравствуй|здравствуйте|добрый|как дела|что нового|кто ты|помоги|hello|hi)\b",
    re.IGNORECASE,
)
RECOVERY_TEXT_PREFIX = "Шаг анализа завершился с ограничением итераций модели"
GENERIC_ARTIFACT_SUMMARY_PREFIX = "Анализ выполнен, артефакты построены"

DEPTH_PROFILES: dict[str, dict[str, Any]] = {
    "light": {
        "max_steps": 2,
        "inner_recursion_limit": 3,
        "evaluate_enabled": False,
        "think_instruction": (
            "Составь КРАТКИЙ план (1-2 шага).\n"
            "Используй минимум инструментов.\n"
            "Предпочитай value_tool для одиночных метрик, "
            "pandas_tool для простых таблиц."
        ),
    },
    "medium": {
        "max_steps": 4,
        "inner_recursion_limit": 5,
        "evaluate_enabled": True,
        "think_instruction": (
            "Составь план анализа (2-4 шага).\n"
            "Применяй фильтры и агрегации по необходимости.\n"
            "Используй графики, если запрос подразумевает визуализацию."
        ),
    },
    "deep": {
        "max_steps": 8,
        "inner_recursion_limit": 8,
        "evaluate_enabled": True,
        "think_instruction": (
            "Составь ДЕТАЛЬНЫЙ план (4-8 шагов).\n"
            "Проведи глубокий анализ: корреляции, распределения, тренды.\n"
            "Используй комбинацию инструментов для полной картины.\n"
            "Добавь валидацию данных и проверку выбросов."
        ),
    },
}


@dataclass
class AgentResponse:
    final_text: str
    reasoning: str | None
    artifacts: list
    route: Literal["chat", "analysis"] = "chat"
    tool_calls: int = 0
    tool_names: list[str] = field(default_factory=list)


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
    route: Literal["chat", "analysis"]

    plan: str
    step_index: int
    max_steps: int
    done: bool
    eval_passed: bool
    eval_reason: str
    refinement_feedback: str
    stop_reason: str
    tools: list

    response: AgentResponse


class AgentRunner:
    def __init__(
        self,
        settings: Settings,
        db_runtime_service: DBRuntimeService | None = None,
    ) -> None:
        self.settings = settings
        self.db_runtime_service = db_runtime_service
        self._query_cache: OrderedDict[str, QueryCacheEntry] = OrderedDict()
        self._depth_profile = self._resolve_depth_profile()
        self._graph = self._build_query_graph()

    def _resolve_depth_profile(self) -> dict[str, Any]:
        depth = self.settings.agent_analysis_depth
        return DEPTH_PROFILES.get(depth, DEPTH_PROFILES["light"])

    def _build_llm(
        self,
        *,
        role: Literal["chat", "plan", "tool", "evaluate"],
        include_reasoning: bool,
        timeout_sec: int | None = None,
        max_tokens_override: int | None = None,
    ) -> ChatOpenAI:
        enable_thinking = self.settings.llm_enable_thinking and (
            include_reasoning or role == "plan"
        )
        if role == "evaluate":
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

        presence_penalty = 0.0 if role == "evaluate" else self.settings.llm_presence_penalty

        max_tokens = max_tokens_override or self.settings.llm_max_tokens_default
        if max_tokens_override is None and include_reasoning:
            max_tokens = self.settings.llm_max_tokens_reasoning
        if role == "evaluate" and max_tokens_override is None:
            max_tokens = self.settings.agent_evaluate_max_tokens

        extra_body: dict[str, Any] = {}
        if self.settings.llm_chat_template_kwargs_enabled:
            extra_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        if self.settings.llm_top_k > 0:
            extra_body["top_k"] = self.settings.llm_top_k

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

        return ChatOpenAI(**kwargs)

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
        first_line = first_line.strip("`\"'«»“”.,:;!?-–— ")
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

    def _build_messages(
        self, prompt: str, history: list[dict[str, Any]], use_history: bool
    ) -> list[BaseMessage]:
        messages: list[BaseMessage] = []

        if use_history and history:
            max_msgs = max(0, self.settings.agent_history_max_messages)
            recent = history[-max_msgs:] if max_msgs > 0 else []
            older = history[:-max_msgs] if max_msgs > 0 else history

            summary = self._history_summary(older)
            if summary:
                messages.append(SystemMessage(content=summary))

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
        normalized: list[dict[str, Any]] = []
        for item in recent:
            normalized.append(
                {
                    "role": str(item.get("role", "assistant")),
                    "content": self._truncate(str(item.get("content", "")), 220),
                }
            )

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
    ) -> str:
        payload = {
            "model": self.settings.llm_model,
            "dataset": self._dataset_signature(df),
            "prompt": self._truncate(prompt, 600),
            "history": self._history_cache_signature(history, use_history),
            "use_history": bool(use_history),
            "include_reasoning": bool(include_reasoning),
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

    def _route_intent(
        self,
        df: pd.DataFrame | None,
        prompt: str,
        session_source: dict[str, Any] | None = None,
    ) -> Literal["chat", "analysis"]:
        has_db_source = False
        if isinstance(session_source, dict):
            source_type = str(session_source.get("source_type", "")).strip().lower()
            has_db_source = source_type == "db_connection"

        if df is None and not has_db_source:
            return "chat"

        normalized = prompt.strip().lower()
        if not normalized:
            return "chat"

        has_analytics_hint = any(hint in normalized for hint in ANALYTICAL_HINTS)
        has_chat_hint = CHAT_HINTS_RE.search(normalized) is not None

        if has_chat_hint and not has_analytics_hint:
            return "chat"
        if has_analytics_hint:
            return "analysis"

        # При наличии датасета все содержательные запросы считаем аналитическими.
        return "analysis"

    @staticmethod
    def _is_insight_request(prompt: str) -> bool:
        normalized = prompt.strip().lower()
        if not normalized:
            return False
        return any(token in normalized for token in INSIGHT_HINTS)

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
    def _safe_dataset_fallback(df: pd.DataFrame, prompt: str) -> str | None:
        normalized = prompt.strip().lower()
        if not normalized:
            return None

        if "столбц" in normalized or "колонк" in normalized:
            columns = list(map(str, df.columns))
            preview = ", ".join(columns[:20])
            suffix = "" if len(columns) <= 20 else ", ..."
            return f"В датасете {len(columns)} столбцов. Первые: {preview}{suffix}."

        if (
            "сколько" in normalized
            and any(token in normalized for token in ("строк", "запис", "пассажир"))
        ):
            return f"В датасете {len(df)} строк."

        if "пропуск" in normalized or "missing" in normalized:
            missing = df.isna().sum().sort_values(ascending=False)
            top = [(str(k), int(v)) for k, v in missing.items() if int(v) > 0][:8]
            if not top:
                return "В датасете нет пропусков."
            parts = ", ".join(f"{name}: {count}" for name, count in top)
            return f"Топ пропусков по столбцам: {parts}."

        return None

    def _fallback_text(
        self,
        prompt: str,
        df: pd.DataFrame | None = None,
        stop_reason: str | None = None,
    ) -> str:
        normalized = prompt.strip().lower()
        if CHAT_HINTS_RE.search(normalized):
            if "как дела" in normalized:
                return "Все в порядке, спасибо. Готов продолжать анализ данных и отвечать на вопросы."
            return "Привет. Я на связи и готов помочь с анализом данных."

        if df is not None:
            dataset_fallback = self._safe_dataset_fallback(df, prompt)
            if dataset_fallback:
                return dataset_fallback

            if stop_reason == "max_steps_reached":
                return (
                    "Я выполнил несколько шагов анализа, но не получил надежный артефакт для финального вывода. "
                    "Уточните запрос или сузьте задачу (например, один график или одну метрику)."
                )
            if stop_reason == "eval_failed":
                return (
                    "Я получил промежуточный результат, но оценка не подтвердила, что выводы полностью опираются "
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
        "Ты аналитик данных. Тебе дан вопрос пользователя и описание датасета.\n"
        "Сформулируй план анализа:\n"
        "1) Какой тип задачи (метрика / таблица / график / комплексный)?\n"
        "2) Какие инструменты вызвать и в каком порядке?\n"
        "3) Какие проверки/фильтры применить?\n"
        "4) Как сформулировать финальный ответ?\n"
        "Будь конкретен: укажи названия столбцов, фильтры, агрегации.\n"
        "Доступные инструменты:\n"
        "- `plotly_tool` — построение графиков (bar, scatter, histogram и т.д.)\n"
        "- `pandas_tool` — табличные срезы, группировки, pivot-таблицы\n"
        "- `value_tool` — одиночные метрики (count, mean, median, etc.)\n"
    )

    def _think_system_prompt(self) -> str:
        depth_instruction = self._depth_profile.get("think_instruction", "")
        depth_label = self.settings.agent_analysis_depth.upper()
        return (
            self._THINK_SYSTEM_PROMPT_BASE
            + f"\n[УРОВЕНЬ АНАЛИЗА: {depth_label}]\n{depth_instruction}\n"
        )

    _EVALUATE_PROMPT_TEMPLATE = (
        "Вопрос пользователя: {question}\n"
        "Полученные артефакты: {artifact_summary}\n"
        "Текст ответа: {response_text}\n\n"
        "Ответь строго в формате JSON:\n"
        '{{"pass": true/false, "reason": "краткое обоснование"}}\n\n'
        "Критерии оценки:\n"
        "- Ответ прямо отвечает на вопрос пользователя?\n"
        "- Все утверждения подкреплены артефактами (таблица/график/метрика)?\n"
        "- Нет выдуманных данных?\n"
        "Только JSON, без пояснений."
    )

    def _compose_step_prompt(
        self,
        *,
        user_prompt: str,
        plan: str,
        refinement_feedback: str,
        step_index: int,
        max_steps: int,
    ) -> str:
        blocks = [
            user_prompt.strip(),
            "",
            "[ROLE: PLAN]",
            plan.strip(),
            "",
            "[ROLE: ACT]",
            (
                "Следуй плану из секции PLAN. "
                "Используй tool-вызовы для получения данных. "
                "Передавай в tool только чистый Python-код (без markdown-блоков и без ```)."
            ),
            "",
            "[ROLE: CONTRACT]",
            (
                "tool_result должен строго соответствовать JSON-схеме: "
                '{"schema_version":"1.0","artifact_type":"<plot|table|value>","items":{...}}'
            ),
            "",
            "[ROLE: STEP]",
            f"Текущий шаг: {step_index}/{max_steps}. Уровень анализа: {self.settings.agent_analysis_depth}.",
        ]

        if refinement_feedback.strip():
            blocks.extend([
                "",
                "[ROLE: REFINEMENT]",
                f"Предыдущая попытка не прошла оценку: {refinement_feedback.strip()}",
                "Скорректируй подход с учётом этой обратной связи.",
            ])

        blocks.extend(
            [
                "",
                "[ROLE: FINALIZE]",
                "Финальный текст только по фактам из tool-результата. Без выдуманных значений.",
            ]
        )

        return "\n".join(blocks).strip()

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
            final_text = strip_thinking(output_text)
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
            if str(getattr(artifact, "artifact_type", "")).strip() != "value":
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
            artifact_type = str(getattr(artifact, "artifact_type", "")).strip() or "artifact"
            name = str(getattr(artifact, "text", "")).strip() or artifact_type
            data = getattr(artifact, "data", None)

            if artifact_type == "value" and isinstance(data, dict):
                metric_keys = [str(key) for key in data.keys()]
                preview = ", ".join(metric_keys[:5])
                suffix = ", ..." if len(metric_keys) > 5 else ""
                lines.append(
                    f"- `value` `{name}`: {len(metric_keys)} метрик"
                    + (f" ({preview}{suffix})" if preview else "")
                )
                continue

            if artifact_type == "table":
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
                        raw_title = getattr(raw_title, "text")
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
        lines: list[str] = []
        for key in keys[:max_items]:
            lines.append(f"- {key}: {self._format_metric_value(value_payload[key])}")
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
                    f"- `{name}`: максимум у `{max_label}` = {max_value}, минимум у `{min_label}` = {min_value}."
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
        is_insight = self._is_insight_request(prompt)

        table_count = sum(
            1
            for artifact in artifacts
            if str(getattr(artifact, "artifact_type", "")).strip() == "table"
        )
        plot_count = sum(
            1
            for artifact in artifacts
            if str(getattr(artifact, "artifact_type", "")).strip() == "plot"
        )
        value_count = sum(
            1
            for artifact in artifacts
            if str(getattr(artifact, "artifact_type", "")).strip() == "value"
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

        if not direct_answer:
            candidate = self._first_sentence(str(base_text or ""))
            if candidate:
                direct_answer = candidate

        if not direct_answer:
            direct_answer = (
                "Ниже структурированный вывод по построенным артефактам."
                if is_insight
                else "Ключевой вывод сформирован на основе полученных артефактов."
            )

        method_lines = self._artifact_method_lines(artifacts)
        if not method_lines:
            method_lines = [
                f"- Получено артефактов: {len(artifacts)}",
                f"- По типам: table={table_count}, plot={plot_count}, value={value_count}",
            ]

        observation_lines: list[str] = []
        observation_lines.extend(
            self._value_observation_lines(value_payload, max_items=8 if is_insight else 5)
        )
        observation_lines.extend(
            self._table_observation_lines(artifacts, max_items=4 if is_insight else 2)
        )
        if not observation_lines:
            observation_lines = [
                f"- Построено артефактов: {len(artifacts)} (table={table_count}, plot={plot_count}, value={value_count})."
            ]

        if is_insight:
            conclusion = (
                "Итог: наблюдения основаны на построенных таблицах, графиках и метриках. "
                "Чтобы углубить анализ, стоит проверить причинно-следственные гипотезы на отдельных сегментах "
                "и дополнительно валидировать устойчивость найденных закономерностей."
            )
        else:
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
    def _collect_progress_collectors(callbacks: list) -> list[AgentProgressCollector]:
        result: list[AgentProgressCollector] = []
        for cb in callbacks:
            if isinstance(cb, AgentProgressCollector):
                result.append(cb)
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
    ) -> AgentResponse:
        llm = self._build_llm(
            role="tool",
            include_reasoning=include_reasoning,
            timeout_sec=min(
                self.settings.agent_step_timeout_sec,
                self.settings.backend_query_timeout_sec,
            ),
        )

        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df.copy(),
            tools=tools,
            verbose=False,
            return_intermediate_steps=True,
            max_iterations=max(1, min(
                self.settings.agent_inner_recursion_limit,
                int(self._depth_profile.get("inner_recursion_limit", self.settings.agent_inner_recursion_limit)),
            )),
            max_execution_time=float(self.settings.agent_step_timeout_sec),
            prefix=agent_prompt,
            include_df_in_prompt=True,
            number_of_head_rows=max(1, self.settings.agent_prompt_head_rows),
            data_info_max_columns=max(6, self.settings.agent_prompt_max_columns),
        )

        prompt_messages = self._build_messages(prompt, history, use_history)
        runtime_config: dict[str, Any] = {
            "callbacks": callbacks,
            "recursion_limit": max(8, min(24, int(self._depth_profile.get(
                "inner_recursion_limit", self.settings.agent_inner_recursion_limit
            )))),
        }
        metadata = self._build_runtime_metadata(trace_context)
        if metadata:
            runtime_config["metadata"] = metadata

        try:
            response_payload = agent.invoke(
                {"messages": normalize_agent_messages(prompt_messages)},
                config=runtime_config,
            )
            final_text, reasoning = self._collect_text_and_reasoning(
                callbacks=callbacks,
                response_payload=response_payload,
            )
        except Exception as exc:
            final_text, collector_reasoning = self._latest_collected_text(callbacks)
            artifacts, tool_calls, tool_names = self._collect_tool_stats(callbacks)
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

    def _build_query_graph(self):
        graph = StateGraph(AgentGraphState)

        graph.add_node("route", self._route_node)
        graph.add_node("chat", self._chat_node)
        graph.add_node("think", self._think_node)
        graph.add_node("act", self._act_node)
        graph.add_node("evaluate", self._evaluate_node)
        graph.add_node("decide", self._decide_node)
        graph.add_node("finalize", self._finalize_node)

        graph.add_edge(START, "route")
        graph.add_conditional_edges(
            "route",
            self._route_edge,
            {"chat": "chat", "analysis": "think"},
        )
        graph.add_edge("chat", "finalize")

        graph.add_edge("think", "act")
        graph.add_edge("act", "evaluate")
        graph.add_edge("evaluate", "decide")
        graph.add_conditional_edges(
            "decide",
            self._decide_edge,
            {"think": "think", "finalize": "finalize"},
        )

        graph.add_edge("finalize", END)
        return graph.compile()

    def _route_node(
        self, state: AgentGraphState
    ) -> dict[str, Literal["chat", "analysis"]]:
        route = self._route_intent(
            state.get("df"),
            state.get("prompt", ""),
            state.get("session_source"),
        )
        return {"route": route}

    @staticmethod
    def _route_edge(state: AgentGraphState) -> Literal["chat", "analysis"]:
        return state.get("route", "chat")

    def _chat_node(self, state: AgentGraphState) -> dict[str, AgentResponse]:
        prompt = state.get("prompt", "")
        df = state.get("df")
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
        return {
            "response": response,
            "done": True,
            "stop_reason": "chat_route",
        }

    def _think_node(self, state: AgentGraphState) -> dict[str, Any]:
        df = state.get("df")
        prompt = state.get("prompt", "")
        callbacks = state.get("callbacks", [])
        refinement_feedback = state.get("refinement_feedback", "")
        tool_db_runtime = self._resolve_tool_db_runtime_config(
            state.get("session_source"),
            state.get("trace_context")
        )

        tools: list = []
        tool_df = df if df is not None else pd.DataFrame()
        if df is not None:
            tools = [
                PlotlyTool(
                    tool_df,
                    execution_timeout_sec=self.settings.tool_exec_timeout_sec,
                    tool_cache_size=self.settings.tool_cache_size,
                    db_runtime_config=tool_db_runtime,
                ),
                PandasTool(
                    tool_df,
                    execution_timeout_sec=self.settings.tool_exec_timeout_sec,
                    tool_cache_size=self.settings.tool_cache_size,
                    db_runtime_config=tool_db_runtime,
                ),
                ValueTool(
                    tool_df,
                    execution_timeout_sec=self.settings.tool_exec_timeout_sec,
                    tool_cache_size=self.settings.tool_cache_size,
                    db_runtime_config=tool_db_runtime,
                ),
            ]
        if tool_db_runtime is not None:
            tools.append(
                DBTool(
                    tool_df,
                    execution_timeout_sec=self.settings.tool_exec_timeout_sec,
                    tool_cache_size=max(8, self.settings.tool_cache_size // 2),
                    db_runtime_config=tool_db_runtime,
                )
            )

        depth_max = int(self._depth_profile.get("max_steps", self.settings.agent_max_steps))
        max_steps = max(2, min(depth_max, self.settings.agent_max_steps))
        if self._is_insight_request(prompt):
            max_steps = max(max_steps, min(10, max_steps + 2))

        data_context = ""
        if df is not None:
            try:
                data_context = _get_data_info(
                    df, max_columns=max(6, self.settings.agent_prompt_max_columns)
                )
            except Exception:
                data_context = f"Датасет: {df.shape[0]} строк, {df.shape[1]} столбцов."

        user_block = f"Вопрос пользователя: {prompt.strip()}"
        if refinement_feedback.strip():
            user_block += (
                f"\n\nПредыдущая попытка не прошла оценку: "
                f"{refinement_feedback.strip()}\n"
                "Скорректируй план с учётом этой обратной связи."
            )

        think_messages = [
            SystemMessage(content=self._think_system_prompt()),
            HumanMessage(content=f"{data_context}\n\n{user_block}"),
        ]

        llm = self._build_llm(
            role="plan",
            include_reasoning=True,
            timeout_sec=min(
                self.settings.agent_step_timeout_sec,
                self.settings.backend_query_timeout_sec,
            ),
        )

        current_step = int(state.get("step_index", 0))
        self._emit_phase_event(
            callbacks,
            phase="think",
            title="Рассуждение (Chain of Thought)",
            content="",
            step_index=current_step,
            max_steps=max_steps,
            status="streaming",
        )

        silent_cbs = self._silent_callbacks(callbacks)
        runtime_config: dict[str, Any] = {"callbacks": silent_cbs}
        metadata = self._build_runtime_metadata(state.get("trace_context"))
        if metadata:
            runtime_config["metadata"] = metadata

        plan = ""
        try:
            response = llm.invoke(think_messages, config=runtime_config)
            record_llm_usage_on_active_span(
                response,
                fallback_model=self.settings.llm_model,
                fallback_provider=self.settings.llm_provider,
            )
            raw_content = self._content_to_text(getattr(response, "content", ""))
            plan = strip_thinking(raw_content).strip()
            reasoning = extract_thinking(raw_content)
            if not plan:
                plan = reasoning or "Анализировать данные по запросу пользователя."
        except Exception:
            plan = "Ошибка при планировании. Выполнить прямой анализ данных."

        self._emit_phase_event(
            callbacks,
            phase="think",
            title="Рассуждение (Chain of Thought)",
            content=plan,
            step_index=current_step,
            max_steps=max_steps,
            status="done",
        )
        self._emit_progress_event(
            callbacks,
            phase="plan",
            title="Планирую решение",
            details=plan,
            step_index=0,
            max_steps=max_steps,
        )

        first_run = not state.get("tools")
        result: dict[str, Any] = {
            "plan": plan,
            "max_steps": max_steps,
            "done": False,
            "eval_passed": False,
            "eval_reason": "",
            "stop_reason": "",
        }
        if first_run:
            result["tools"] = tools
            result["step_index"] = 0
        return result

    def _act_node(self, state: AgentGraphState) -> dict[str, Any]:
        df = state.get("df")
        tools = state.get("tools", [])
        if df is None and not tools:
            return self._chat_node(state)

        step_index = int(state.get("step_index", 0)) + 1
        plan = state.get("plan", "")
        refinement_feedback = state.get("refinement_feedback", "")
        callbacks = state.get("callbacks", [])
        max_steps = int(state.get("max_steps", self.settings.agent_max_steps))
        tool_df = df if df is not None else pd.DataFrame()

        step_prompt = self._compose_step_prompt(
            user_prompt=state.get("prompt", ""),
            plan=plan,
            refinement_feedback=refinement_feedback,
            step_index=step_index,
            max_steps=max_steps,
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

        started_at = time.perf_counter()
        try:
            response = self._analysis_step(
                df=tool_df,
                prompt=step_prompt,
                history=state.get("history", []),
                use_history=state.get("use_history", True),
                include_reasoning=state.get("include_reasoning", False),
                callbacks=callbacks,
                trace_context=state.get("trace_context"),
                tools=tools,
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
        if response.tool_names:
            tool_summary_lines.append(f"Инструменты: {', '.join(response.tool_names)}")
        if response.artifacts:
            types = [
                str(getattr(a, "artifact_type", "")).strip() for a in response.artifacts
            ]
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

        if not response.final_text.strip():
            reason = "Пустой финальный ответ"
            self._emit_phase_event(
                callbacks, phase="evaluate", title="Оценка результата",
                content=reason, step_index=step_index, max_steps=max_steps, status="fail",
            )
            return {"eval_passed": False, "eval_reason": reason}

        if response.tool_calls <= 0 or not response.artifacts:
            reason = "Нет подтвержденных tool-вызовов с артефактами"
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

        evaluate_prompt = self._EVALUATE_PROMPT_TEMPLATE.format(
            question=prompt.strip(),
            artifact_summary=artifact_summary,
            response_text=self._truncate(response.final_text, 600),
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
                [SystemMessage(content="Ты оцениваешь качество аналитических ответов. Отвечай только JSON."),
                 HumanMessage(content=evaluate_prompt)],
                config=runtime_config,
            )
            record_llm_usage_on_active_span(
                llm_response,
                fallback_model=self.settings.llm_model,
                fallback_provider=self.settings.llm_provider,
            )
            raw_text = self._content_to_text(getattr(llm_response, "content", ""))
            raw_text = strip_thinking(raw_text).strip()

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

        if eval_passed and response.artifacts and response.final_text.strip():
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
                "stop_reason": "max_steps_reached" if response.artifacts else "eval_failed",
            }

        reasoning_lower = str(response.reasoning or "").lower()
        if "recursion limit" in reasoning_lower and not response.artifacts:
            self._emit_progress_event(
                callbacks, phase="decide",
                title="Остановка по лимиту рекурсии",
                details="Не удалось получить подтвержденный артефакт до лимита.",
                step_index=step_index, max_steps=max_steps,
            )
            return {"done": True, "stop_reason": "act_recursion_limit"}

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

        if route == "analysis":
            if response.artifacts:
                should_rewrite = (
                    not response.final_text.strip()
                    or response.final_text.strip().startswith(RECOVERY_TEXT_PREFIX)
                    or response.final_text.strip().startswith(GENERIC_ARTIFACT_SUMMARY_PREFIX)
                    or self._response_too_generic(prompt, response.final_text)
                    or self._is_insight_request(prompt)
                    or prompt.strip().endswith("?")
                    or len(response.final_text.strip()) < 260
                )
                if should_rewrite:
                    grounded_summary = self._artifact_grounded_summary(
                        prompt,
                        response.artifacts,
                        base_text=response.final_text,
                    )
                    if grounded_summary:
                        response.final_text = grounded_summary

            if response.tool_calls <= 0 or not response.artifacts:
                fallback_text = self._fallback_text(prompt, df, stop_reason=stop_reason)
                reasoning = (response.reasoning or "").strip()
                reason_suffix = f"Tool-required policy enforced: {stop_reason or 'missing artifacts'}."
                reasoning = f"{reasoning}\n\n{reason_suffix}".strip()
                response = AgentResponse(
                    final_text=fallback_text,
                    reasoning=reasoning,
                    artifacts=[],
                    route="analysis",
                    tool_calls=response.tool_calls,
                    tool_names=response.tool_names,
                )
            elif not bool(state.get("eval_passed", False)):
                reasoning = (response.reasoning or "").strip()
                eval_reason = state.get("eval_reason", "")
                response.reasoning = (
                    f"{reasoning}\n\nFinalize note: оценка не пройдена ({eval_reason}), "
                    "но подтвержденные артефакты уже построены."
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
        prompt_messages = self._build_messages(prompt, history, use_history)
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
            final_text = strip_thinking(output_text)
        if reasoning is None:
            reasoning = extract_thinking(output_text) or None

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
    ) -> AgentResponse:
        request_kind = str((trace_context or {}).get("request_kind", "")).strip().lower()
        cache_allowed = self.settings.agent_cache_enabled and request_kind != "stream"

        cache_key = self._query_cache_key(
            df=df,
            prompt=prompt,
            history=history,
            use_history=use_history,
            include_reasoning=include_reasoning,
        )
        if cache_allowed:
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached

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
                }
            )
        except Exception:
            fallback = AgentResponse(
                final_text=self._fallback_text(prompt, df),
                reasoning=None,
                artifacts=[],
                route=self._route_intent(df, prompt),
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
                route=self._route_intent(df, prompt),
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
