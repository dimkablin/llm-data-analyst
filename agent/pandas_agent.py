from __future__ import annotations

from typing import Any

import pandas as pd
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from agent.prompts import get_detailed_data_info


def _message_to_role(message: BaseMessage) -> str:
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    if isinstance(message, SystemMessage):
        return "system"
    return "assistant"


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
    return str(content)


def normalize_agent_messages(raw_input: Any) -> list[dict[str, str]]:
    if isinstance(raw_input, str):
        return [{"role": "user", "content": raw_input}]

    if isinstance(raw_input, list):
        normalized: list[dict[str, str]] = []
        for item in raw_input:
            if isinstance(item, BaseMessage):
                text = _content_to_text(item.content)
                if text:
                    normalized.append(
                        {
                            "role": _message_to_role(item),
                            "content": text,
                        }
                    )
            elif isinstance(item, dict):
                role = str(item.get("role", "user"))
                text = _content_to_text(item.get("content", ""))
                if text:
                    normalized.append({"role": role, "content": text})
            else:
                text = _content_to_text(item)
                if text:
                    normalized.append({"role": "user", "content": text})
        return normalized

    return [{"role": "user", "content": _content_to_text(raw_input)}]


def extract_agent_output_text(result: Any) -> str:
    if not isinstance(result, dict):
        return ""

    direct_output = result.get("output")
    if isinstance(direct_output, str) and direct_output.strip():
        return direct_output

    messages = result.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                text = _content_to_text(message.content)
                if text:
                    return text
            if isinstance(message, dict) and message.get("role") == "assistant":
                text = _content_to_text(message.get("content", ""))
                if text:
                    return text

    final_output = result.get("final_output")
    if isinstance(final_output, str):
        return final_output

    return ""


def get_prompt(
    df: pd.DataFrame,
    *,
    prefix: str | None = None,
    suffix: str = "",
    include_df_in_prompt: bool | None = True,
    number_of_head_rows: int = 5,
    data_info_max_columns: int = 30,
) -> str:
    """
    Формирует system prompt для ReAct-агента с учетом структуры DataFrame.
    """
    prompt_parts: list[str] = []
    if prefix:
        prompt_parts.append(prefix)

    prompt_parts.append(get_detailed_data_info(df, max_columns=data_info_max_columns))

    if include_df_in_prompt:
        df_head = str(df.head(number_of_head_rows).to_markdown())
        prompt_parts.append(f"\nПервые строки данных:\n{df_head}")

    if suffix:
        prompt_parts.append(suffix)

    return "\n\n".join(part for part in prompt_parts if part)


def create_pandas_dataframe_agent(
    llm: Any,
    df: pd.DataFrame,
    callback_manager: Any | None = None,
    prefix: str | None = None,
    suffix: str | None = None,
    verbose: bool = False,
    return_intermediate_steps: bool = False,
    max_iterations: int | None = 15,
    max_execution_time: float | None = None,
    agent_executor_kwargs: dict[str, object] | None = None,
    include_df_in_prompt: bool | None = True,
    number_of_head_rows: int = 5,
    data_info_max_columns: int = 30,
    tools: list[BaseTool] = (),
) -> Any:
    """
    Создаёт ReAct-агента на базе LangChain create_agent.

    `max_iterations` передаётся как `recursion_limit` в конфиг графа
    вызывающим кодом (agent_runner.py). Остальные legacy-параметры
    игнорируются для обратной совместимости сигнатуры.
    """
    del callback_manager, verbose, return_intermediate_steps
    del max_execution_time, agent_executor_kwargs, max_iterations

    system_prompt = get_prompt(
        df,
        prefix=prefix,
        suffix=suffix or "",
        include_df_in_prompt=include_df_in_prompt,
        number_of_head_rows=number_of_head_rows,
        data_info_max_columns=data_info_max_columns,
    )

    return create_agent(
        model=llm,
        tools=list(tools),
        system_prompt=system_prompt,
    )
