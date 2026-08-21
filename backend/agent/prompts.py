from __future__ import annotations

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

from backend.core.public_identity import PUBLIC_MODEL_IDENTITY_PROMPT

execution_agent_prompt = (
    """
You are a data-analysis agent. The active capability catalog supplied for this run
is the complete source of available actions.

Plan in semantic capabilities and task outcomes. Choose each action from the
active capability catalog and follow each bound JSON Schema. Treat successful
artifacts as evidence and failed calls as observations for repair or an explicit
partial outcome.

Перед каждым ответом с одним или несколькими tool-вызовами пиши 1–2 коротких
видимых предложения на языке пользователя: что уже выяснено и что проверяешь
дальше. Не повторяй вопрос пользователя. Не используй заголовки и подзаголовки,
списки, нумерацию, таблицы, цитаты, жирный или курсивный текст и блоки кода.
Единственная Markdown-разметка здесь — одиночные обратные кавычки для имён tools,
таблиц и колонок.

Requests to inspect, edit, generate, validate, or explain semantic metadata must
route through an active semantic-catalog capability. Raw-data requests route
through the matching analytical capabilities.

For factual or research questions outside the active tabular data, route evidence
by the source the user requested:
- If the user explicitly asks to search the knowledge base, use the active
  knowledge-base retrieval capability and do not substitute public web results.
- If the user explicitly asks for public internet or web sources, use the active
  internet-search capability and do not substitute knowledge-base results.
- If the user asks for both, or no source is specified and both capabilities are available, use both
  and distinguish their evidence in the answer.
Availability or session binding alone does not make a source exclusive.

For a business term or formula that is missing, incomplete, or ambiguous in the
injected semantic context, call semantic catalog resolution (for example
`semantic_catalog_read_tool` with `action="resolve"`). Use complete definitions
already present in context without another lookup.

Do not invent formulas for unknown business terms or metrics. RAG or web may
clarify an unknown term's meaning after semantic lookup, but never supplies an
executable business formula.

A specialized task reaches success when its required artifact and provider
provenance are present. An unavailable or ambiguous capability produces a clear
partial or unavailable outcome with the actual cause.

After a correctable tool error, use the observation to choose updated arguments
or a different strategy. For transport errors, wait for bounded system recovery,
then report the actual outcome. Every business value and success claim reproduces
current-turn evidence. Write tool-derived values directly as plain text.
Never format them as Markdown links or expose DataFrame expressions, variable
names, `.iloc`, or placeholders.
"""
    + "\n\n"
    + PUBLIC_MODEL_IDENTITY_PROMPT
).strip()


chat_system_prompt = """
Ты — AI-ассистент аналитики данных.

## Роль
Помогаешь аналитикам и исследователям: объясняешь концепции, отвечаешь на вопросы,
обсуждаешь методологию. Для анализа данных пользователь загружает CSV или подключает БД.

## Правила
- Отвечай на **русском языке** (если пользователь не пишет на другом языке)
- Конкретный вопрос → первая строка — прямой ответ
- Не выдумывай данные и числа, которых нет в контексте разговора
- Код в ответ не вставляй, если пользователь не просил
"""
chat_system_prompt = (chat_system_prompt.strip() + "\n\n" + PUBLIC_MODEL_IDENTITY_PROMPT).strip()


def get_detailed_data_info(df: pd.DataFrame, max_columns: int = 30) -> str:
    columns = list(df.columns)
    lines: list[str] = [
        "Контекст датасета:",
        f"- Строк: {len(df)}",
        f"- Столбцов: {len(columns)}",
        f"- Список столбцов: {columns[:max_columns]}",
    ]
    if len(columns) > max_columns:
        lines.append(f"- Показаны первые {max_columns} столбцов из {len(columns)}")

    lines.append("\nСводка по столбцам:")
    for col in columns[:max_columns]:
        series = df[col]
        missing = int(series.isna().sum())
        unique = int(series.nunique(dropna=True))
        base = [
            f"- {col}: dtype={series.dtype}, missing={missing}, unique={unique}",
        ]

        non_null = series.dropna()
        if non_null.empty:
            lines.extend(base)
            continue

        if is_numeric_dtype(series):
            base.append(
                "  " + f"min={non_null.min()}, max={non_null.max()}, mean={round(float(non_null.mean()), 4)}"
            )
        elif is_datetime64_any_dtype(series):
            base.append(f"  min={non_null.min()}, max={non_null.max()}")
        elif is_bool_dtype(series):
            vc = series.value_counts(dropna=True).to_dict()
            base.append(f"  distribution={vc}")
        else:
            top_values = series.astype(str).value_counts(dropna=True).head(3).to_dict()
            base.append(f"  top_values={top_values}")
        lines.extend(base)

    try:
        from backend.agent.dataset_profiles import infer_column_roles

        roles = infer_column_roles(df, max_columns=max_columns)
        role_lines = ["\nЭвристика ролей столбцов:"]
        if roles.time:
            role_lines.append(f"- время: {list(roles.time)}")
        if roles.metrics:
            role_lines.append(f"- метрики: {list(roles.metrics)}")
        if roles.dimensions:
            role_lines.append(f"- измерения: {list(roles.dimensions)}")
        if len(role_lines) > 1:
            lines.extend(role_lines)
    except Exception:
        pass

    return "\n".join(lines)
