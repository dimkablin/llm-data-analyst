import json
import re

import numpy as np
import pandas as pd

from backend.agent.prompts import value_tool_prompt
from backend.data_access.db_runtime_service import RuntimeDBConnectionConfig
from backend.tools.impl.base_tool import BaseExecTool


class ValueTool(BaseExecTool):
    """
    Инструмент для вычисления числовых, строковых и булевых значений.

    Attributes:
        name (str): Имя инструмента.
        artifact_name (str): Ключ результата.
        human_name (str): Человеко-понятное имя.
        description (str): Описание инструмента.
        _locals (dict): Локальные переменные для exec.
        allowed_artifact_types (tuple): Разрешённые типы артефактов.
        allowed_libs (set[str]): Разрешённые библиотеки.
    """

    name: str = "value_tool"
    artifact_name: str = "value"
    human_name: str = "значений"
    description: str = value_tool_prompt
    allowed_artifact_types: tuple = (float, int, str, bool, np.generic)
    allowed_libs: set[str] = {"pandas", "numpy"}
    max_string_value_len: int = 160

    def __init__(
        self,
        df: pd.DataFrame,
        execution_timeout_sec: float = 25.0,
        tool_cache_size: int = 48,
        db_runtime_config: "RuntimeDBConnectionConfig | None" = None,
        sandbox: object | None = None,
        llm_base_url: str | None = None,
        llm_model: str | None = None,
        llm_api_key: str | None = None,
        llm_enable_thinking: bool = False,
        llm_chat_template_kwargs_enabled: bool = True,
        code_fix_max_retries: int = 3,
    ) -> None:
        super().__init__(
            df,
            execution_timeout_sec=execution_timeout_sec,
            include_plotly=False,
            tool_cache_size=tool_cache_size,
            db_runtime_config=db_runtime_config,
            sandbox=sandbox,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_enable_thinking=llm_enable_thinking,
            llm_chat_template_kwargs_enabled=llm_chat_template_kwargs_enabled,
            code_fix_max_retries=code_fix_max_retries,
        )

    def validate_tool_result(self, tool_result: dict[str, object]) -> tuple[bool, str]:
        """
        Проверяет корректность результата инструмента.

        Args:
            tool_result (dict[str, object]): Результат выполнения кода.

        Returns:
            tuple[bool, str]: Флаг валидности и сообщение.
        """
        invalid_keys: list[str] = []
        for key, value in tool_result.items():
            if not isinstance(value, (float, int, str, bool, np.generic)):
                invalid_keys.append(key)
        if invalid_keys:
            invalid_types = ", ".join(
                f"{key}:{type(tool_result[key]).__name__}" for key in invalid_keys
            )
            return (
                False,
                "Неверный тип значений для value_tool. "
                f"Ожидались скаляры (float/int/str/bool), получено: {invalid_types}",
            )
        invalid_strings: list[str] = []
        for key, value in tool_result.items():
            if not isinstance(value, str):
                continue
            text = value.strip()
            sentence_marks = text.count(".") + text.count("!") + text.count("?")
            if (
                len(text) > self.max_string_value_len
                or "\n" in text
                or sentence_marks > 2
            ):
                invalid_strings.append(key)
        if invalid_strings:
            return (
                False,
                "value_tool предназначен только для коротких value-like результатов. "
                f"Слишком длинные или объяснительные строковые значения: {', '.join(invalid_strings)}. "
                "Для развернутого объяснения верни обычный текстовый ответ без value artifact.",
            )
        return True, ""

    def _normalize_scalar(self, value: object) -> object:
        if isinstance(value, np.generic):
            value = value.item()

        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            rounded = round(float(value), 4)
            if rounded == -0.0:
                rounded = 0.0
            return rounded
        if isinstance(value, str):
            return value.strip()
        return value

    @staticmethod
    def _metric_name(*parts: str) -> str:
        joined = "_".join(part for part in parts if part).strip().lower()
        if not joined:
            return "metric"
        cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", joined)
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        return cleaned or "metric"

    def _flatten_metric_mapping(
        self,
        *,
        prefix: str,
        value: dict[object, object],
        out: dict[str, object],
        depth: int = 0,
    ) -> None:
        if depth > 2:
            out[self._metric_name(prefix, "json")] = json.dumps(
                value, ensure_ascii=False, default=str
            )[:260]
            return
        for child_key, child_value in value.items():
            base_name = self._metric_name(prefix, str(child_key))
            if isinstance(child_value, dict):
                self._flatten_metric_mapping(
                    prefix=base_name,
                    value=child_value,
                    out=out,
                    depth=depth + 1,
                )
                continue
            if isinstance(child_value, (list, tuple, set)):
                sequence = list(child_value)
                scalar_seq = all(
                    isinstance(item, (float, int, str, bool, np.generic))
                    for item in sequence
                )
                if scalar_seq and len(sequence) <= 8:
                    for idx, item in enumerate(sequence, start=1):
                        out[self._metric_name(base_name, str(idx))] = self._normalize_scalar(
                            item
                        )
                else:
                    out[self._metric_name(base_name, "size")] = len(sequence)
                continue
            out[base_name] = self._normalize_scalar(child_value)

    def post_process_tool_result(self, tool_result: dict[str, object]) -> dict[str, object]:
        base = super().post_process_tool_result(tool_result)
        processed: dict[str, object] = {}
        for name, value in base.items():
            if isinstance(value, dict):
                self._flatten_metric_mapping(prefix=name, value=value, out=processed)
                continue
            if isinstance(value, (list, tuple, set)):
                sequence = list(value)
                scalar_seq = all(
                    isinstance(item, (float, int, str, bool, np.generic))
                    for item in sequence
                )
                if scalar_seq and len(sequence) <= 8:
                    for idx, item in enumerate(sequence, start=1):
                        processed[self._metric_name(name, str(idx))] = self._normalize_scalar(
                            item
                        )
                else:
                    processed[self._metric_name(name, "size")] = len(sequence)
                continue
            processed[name] = self._normalize_scalar(value)
        return processed


