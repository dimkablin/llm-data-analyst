from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

import pandas as pd
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from backend.data_access.dataframe_utils import numeric_summary_rows
from pydantic import BaseModel, ConfigDict, Field

from backend.agent.dataset_profiles import build_dataset_profile_block
from backend.agent.prompts import execution_agent_prompt, get_detailed_data_info
from backend.agent.services.runtime_context import build_rag_session_prompt_block
from backend.auth.user_memory import UserMemory
from backend.core.config import Settings
from backend.sessions.session_memory import SessionMemory
from backend.skills import SkillRegistry
from backend.tools.instructions import get_default_tool_instruction_registry


class MessageBuildRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt: str
    history: list[dict[str, Any]] = Field(default_factory=list)
    use_history: bool
    settings: Settings
    user_memory: UserMemory
    session_memory: SessionMemory
    system_prompt: str | None = None
    enable_thinking: bool = True


class ExecutionSystemPromptRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    settings: Settings
    skill_registry: SkillRegistry
    enabled_analytical_skill_ids: set[str] | None = None
    capability_context: dict[str, Any] | None = None
    sandbox: Any | None = None
    selected_skill_ids: list[str] | None = None
    df: Any | None = None
    session_source: dict[str, Any] | None = None
    tool_db_runtime: Any | None = None


def _text_message(prefix: str, body: str) -> HumanMessage:
    return HumanMessage(content=f"{prefix}:\n{body.strip()}")


def truncate(text: str, max_len: int) -> str:
    clean = str(text or "").strip()
    if len(clean) <= max_len:
        return clean
    return f"{clean[:max_len]}..."


def history_summary(
    older_history: list[dict[str, Any]],
    *,
    max_chars: int,
) -> str:
    if not older_history:
        return ""

    summary_rows: list[str] = []
    for item in older_history[-8:]:
        role = str(item.get("role", "assistant"))
        content = truncate(str(item.get("content", "")), 140)
        if not content:
            continue
        marker = "U" if role == "user" else "A"
        summary_rows.append(f"- {marker}: {content}")

    if not summary_rows:
        return ""

    summary = "Краткая сводка предыдущего диалога:\n" + "\n".join(summary_rows)
    return truncate(summary, max(200, max_chars))


def artifact_table_to_text(data: Any, *, max_rows: int = 20, max_cols: int = 12) -> str:
    try:
        if isinstance(data, pd.Series):
            data = data.to_frame()

        if isinstance(data, pd.DataFrame):
            df = data.copy()
            if df.empty:
                return "Пустая таблица."

            rows, cols = df.shape
            visible_cols = [str(column) for column in df.columns[:max_cols]]
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
                    summary = pd.DataFrame(numeric_summary_rows(df))
                    if not summary.empty:
                        lines.append("numeric_summary_rows_appended:")
                        lines.append(summary.iloc[:, :max_cols].to_markdown(index=False))
                except Exception:
                    pass
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


def history_has_substantive_artifacts(history: list[dict[str, Any]]) -> bool:
    for item in history:
        artifacts = item.get("artifacts")
        if isinstance(artifacts, list) and artifacts:
            return True
    return False


def history_artifact_summary(history: list[dict[str, Any]]) -> str:
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
                block_lines.append(artifact_table_to_text(data))
            elif artifact_type == "plot":
                if data is not None and not isinstance(data, str):
                    block_lines.append(artifact_table_to_text(data))
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
                    block_lines.append(artifact_table_to_text(data))
            else:
                block_lines.append(artifact_table_to_text(data))

            blocks.append("\n".join(block_lines))

    if not blocks:
        return ""

    return "Артефакты из истории:\n\n" + "\n\n---\n\n".join(blocks[:12])


def build_memory_messages(
    *,
    user_memory: UserMemory,
    session_memory: SessionMemory,
    include_context_summary: bool = True,
) -> list[BaseMessage]:
    system_parts: list[str] = []
    memory_block = user_memory.build_block()
    if memory_block:
        system_parts.append(memory_block)

    session_memory_block = session_memory.build_block(
        include_context_summary=include_context_summary,
    )
    if session_memory_block:
        system_parts.append(session_memory_block)
    return [SystemMessage(content="\n\n".join(system_parts))] if system_parts else []


def build_history_messages(request: MessageBuildRequest) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    history = list(request.history or [])
    skip_count = max(
        0,
        int(getattr(request.session_memory, "compacted_message_count", 0) or 0),
    )
    if skip_count:
        history = history[skip_count:]

    recent: list[dict[str, Any]] = []
    if request.use_history and history:
        max_msgs = max(0, request.settings.agent_history_max_messages)
        recent = history[-max_msgs:] if max_msgs > 0 else []
        older = history[:-max_msgs] if max_msgs > 0 else history

        summary = history_summary(
            older,
            max_chars=request.settings.agent_history_summary_chars,
        )
        if summary:
            messages.append(SystemMessage(content=summary))

    for index, item in enumerate(recent):
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
                        truncate(str(recent[j].get("content", "")), 300)
                        for j in range(index - 1, -1, -1)
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
    return messages


def build_messages(request: MessageBuildRequest) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    system_prompt = str(request.system_prompt or "").strip()
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.extend(
        build_memory_messages(
            user_memory=request.user_memory,
            session_memory=request.session_memory,
            include_context_summary=request.use_history,
        )
    )
    messages.extend(build_history_messages(request))
    messages.append(HumanMessage(content=str(request.prompt or "")))
    return messages


def tool_data_flow_policy_block() -> str:
    return (
        "═══ Политика доступа к данным ═══\n"
        "Основной технический pipeline:\n"
        "- `sql_tool` материализует исходные таблицы/БД/DuckDB в dataframe artifact "
        "и кладет результат в одном session sandbox под именем `artifact_name`.\n"
        "- `pandas_tool` обрабатывает только уже существующие dataframe-переменные "
        "из session sandbox и возвращает table artifact; он не строит графики и не читает БД/файлы.\n"
        "- `plotly_tool` строит chart artifact только из уже существующих dataframe-переменных "
        "из session sandbox; подготовку таблиц делай до него через `sql_tool` или `pandas_tool`.\n"
        "- Successful tool outputs become sandbox variables. "
        "Failed tool calls do not create sandbox variables; do not reference names from failed steps.\n"
        "- Use exact variable names from successful tool outputs or from the sandbox context. "
        "Do not inspect Python namespace with `globals()`, `locals()`, or `__import__`.\n"
        "- Для первичного получения табличных данных из БД используй только `sql_tool`.\n"
        "- `database_tool` используй только для структуры БД: таблицы, схемы, колонки, preview.\n"
        "- `pandas_tool` и `plotly_tool` не должны ходить в БД напрямую.\n"
        "- Эти инструменты работают только с dataframe-переменными, уже существующими в sandbox.\n"
        "- Если нужный датафрейм ещё не создан, сначала вызови `sql_tool`.\n"
        "- Если нужный датафрейм уже есть в sandbox, используй его напрямую "
        "по имени и не делай повторный SQL."
    )


def db_session_prompt_block(
    *,
    session_source: dict[str, Any] | None,
    runtime: Any | None,
    df: pd.DataFrame | None,
) -> str:
    if runtime is None:
        return ""

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
        schema_name = configured_schema.strip()
        lines.append(f"Схема подключения: `{schema_name}` — все SQL и каталог только в этой схеме.")

    lines.append(
        "Для первичного получения табличных данных из БД используй только `sql_tool`. "
        "Всегда передавай оба аргумента: `question` и `artifact_name`."
    )
    lines.append(
        "`question` — это естественно-языковой запрос о том, какие данные нужно получить из БД. "
        "`artifact_name` — имя результата в snake_case."
    )
    lines.append(
        "После выполнения `sql_tool` результат будет доступен в sandbox по имени `artifact_name`, "
        "и его можно напрямую использовать в `pandas_tool` и `plotly_tool`."
    )
    lines.append(
        "Сложный аналитический запрос → `get_tool_instructions(\"general_analytics\")` "
        "и следуй алгоритму (схема → SQL → визуализация)."
    )

    return "\n".join(lines)


def execution_runtime_section(
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


def _build_runtime_context_messages(request: ExecutionSystemPromptRequest) -> list[BaseMessage]:
    if not request.sandbox:
        return []
    sandbox_block = request.sandbox.describe_for_prompt()
    return [_text_message("SANDBOX_CONTEXT", sandbox_block)] if sandbox_block else []


def _build_skill_context_messages(request: ExecutionSystemPromptRequest) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    analytical_skills_block = request.skill_registry.build_analytical_skills_brief_block(
        enabled_skill_ids=request.enabled_analytical_skill_ids,
    )
    if analytical_skills_block:
        messages.append(_text_message("SKILL_CATALOG_CONTEXT", analytical_skills_block))

    selected_skill_ids = request.selected_skill_ids or []
    allowed = request.enabled_analytical_skill_ids
    filtered_selected_skill_ids = (
        [skill_id for skill_id in selected_skill_ids if skill_id in allowed]
        if allowed is not None
        else selected_skill_ids
    )
    if filtered_selected_skill_ids:
        skills_block = request.skill_registry.build_prompt_block(filtered_selected_skill_ids)
        if skills_block:
            messages.append(_text_message("SKILL_CONTEXT", skills_block))
    return messages


def _build_data_context_messages(request: ExecutionSystemPromptRequest) -> list[BaseMessage]:
    data_context_parts: list[str] = []
    catalog_prompt = str((request.session_source or {}).get("data_catalog_prompt") or "").strip()
    semantic_prompt = str((request.session_source or {}).get("semantic_context_prompt") or "").strip()
    df = request.df if isinstance(request.df, pd.DataFrame) else None
    if catalog_prompt:
        data_context_parts.append(catalog_prompt)
    elif df is not None:
        try:
            data_context_parts.append(
                get_detailed_data_info(df, max_columns=request.settings.agent_prompt_max_columns)
            )
        except Exception:
            data_context_parts.append(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    if semantic_prompt:
        data_context_parts.append(semantic_prompt)

    db_block = db_session_prompt_block(
        session_source=request.session_source,
        runtime=request.tool_db_runtime,
        df=df,
    )
    if db_block:
        data_context_parts.append(db_block)
    rag_block = build_rag_session_prompt_block(request.session_source)
    if rag_block:
        data_context_parts.append(rag_block)

    dataset_label = str((request.session_source or {}).get("source_label") or "").strip()
    db_name = ""
    db_type = ""
    db_schema = ""
    if request.tool_db_runtime is not None:
        db_name = str(getattr(request.tool_db_runtime, "name", "") or "").strip()
        db_type = str(getattr(request.tool_db_runtime, "db_type", "") or "").strip()
        configured = request.tool_db_runtime.options.get("schema")
        if isinstance(configured, str) and configured.strip():
            db_schema = configured.strip()
    profile_block = build_dataset_profile_block(
        df,
        dataset_name=dataset_label,
        session_source=request.session_source,
        db_name=db_name,
        db_type=db_type,
        db_schema=db_schema,
    )
    if profile_block:
        data_context_parts.append(profile_block)

    if not data_context_parts:
        return []
    return [_text_message("DATA_CONTEXT", "\n\n".join(data_context_parts))]


def build_execution_context_messages(request: ExecutionSystemPromptRequest) -> list[BaseMessage]:
    return [
        *_build_runtime_context_messages(request),
        *_build_skill_context_messages(request),
        *_build_data_context_messages(request),
    ]


def build_execution_system_prompt(request: ExecutionSystemPromptRequest) -> str:
    source_mode = str((request.capability_context or {}).get("source_mode", "")).strip() or "dataset"
    tool_descriptions = str((request.capability_context or {}).get("tool_descriptions", "")).strip()
    available_tools = [
        str(item).strip()
        for item in (request.capability_context or {}).get("available_tool_keys", [])
        if str(item).strip()
    ]
    tool_list = ", ".join(f"`{item}`" for item in available_tools) if available_tools else "нет"
    today = date.today().strftime("%Y-%m-%d")

    sections: list[str] = [execution_agent_prompt.strip()]
    sections.extend(execution_runtime_section(source_mode, tool_list, today, tool_descriptions))

    sections.append(tool_data_flow_policy_block())

    tool_skills_block = get_default_tool_instruction_registry().build_brief_block(
        set(available_tools)
    )
    if tool_skills_block:
        sections.append(tool_skills_block)

    return "\n\n".join(sections).strip()



def polish_management_note_markdown(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return cleaned

    if not cleaned.lstrip().startswith("#"):
        cleaned = re.sub(
            r"^УПРАВЛЕНЧЕСКАЯ\s+ЗАПИСКА\s*$",
            "## УПРАВЛЕНЧЕСКАЯ ЗАПИСКА",
            cleaned,
            count=1,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if "## УПРАВЛЕНЧЕСКАЯ" not in cleaned.upper():
            cleaned = f"## УПРАВЛЕНЧЕСКАЯ ЗАПИСКА\n\n{cleaned}"

    cleaned = re.sub(
        r"^(\d+\.\s+[^\n]+)$",
        lambda match: (
            match.group(1)
            if match.group(1).startswith("**")
            else f"**{match.group(1).strip()}**"
        ),
        cleaned,
        flags=re.MULTILINE,
    )

    def _bold_metrics(line: str) -> str:
        if line.lstrip().startswith(("#", "-", "*", ">")):
            pass
        line = re.sub(
            r"(?<!\*\*)(\d[\d\s.,]*\s*(?:млн|млрд|тыс\.?)?\s*(?:руб\.?|₽|%|п\.п\.|п\.п|пп))(?!\*\*)",
            r"**\1**",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(
            r"(?<!\*\*)\b([A-ZА-Я]{2,6}\d{0,3})\b(?!\*\*)",
            r"**\1**",
            line,
        )
        return line

    cleaned = "\n".join(_bold_metrics(line) for line in cleaned.splitlines())
    cleaned = re.sub(r"\*\*\*\*+", "**", cleaned)
    return cleaned.strip()
