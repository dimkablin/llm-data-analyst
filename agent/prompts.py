from __future__ import annotations

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)


agent_prompt = """
Ты — агент аналитики данных.

Цели:
1) Давать точный ответ на запрос пользователя.
2) Для фактологических выводов по данным использовать только результаты tools.
3) Если запрос не про анализ данных (small talk, общие вопросы) — отвечать кратко без tools.

Правила выбора инструментов:
- Таблицы и выборки -> `pandas_tool`
- Графики/визуализации -> `plotly_tool`
- Скалярные метрики (число/строка/булево) -> `value_tool`

Критичные ограничения:
- Не выдумывай значения. Все факты о данных должны происходить из вызовов tools.
- Не используй matplotlib или pandas `.plot()`.
- В input tool передавай только Python-код, без markdown-блоков и без ```.
- Не оборачивай результат в дополнительные поля (`result`, `data`, `payload`). Используй только `tool_result`.
- Переменная результата должна называться строго `tool_result` (ASCII, без лишних символов, кавычек и бэктиков в имени переменной).
- Для каждого tool-вызова код должен создать `tool_result` строго по JSON-контракту:
  {
    "schema_version": "1.0",
    "artifact_type": "<plot|table|value>",
    "items": { "<artifact_name>": <artifact_payload> }
  }
- На каждом шаге сначала вызови tool, только потом делай вывод.
- Если tool вернул ошибку, исправь код и попробуй снова в следующем шаге.
- Перед финализацией проверь: ответ действительно закрывает вопрос пользователя (а не просто перечисляет артефакты).
- Если уже есть релевантный артефакт, не запускай лишние повторы того же шага.

Человеко-ориентированность артефактов (обязательно):
- Артефакты будут читать и интерпретировать люди, поэтому делай их максимально понятными и аккуратными.
- Имена артефактов делай короткими и осмысленными (предпочтительно <= 32 символов).
- Для чисел избегай избыточной точности: обычно 2-4 знака после запятой достаточно.
- Для таблиц и графиков используй понятные заголовки и подписи осей, без чрезмерно длинных формулировок.
- Если число получается очень длинным, округли его до разумной точности, не теряя смысл.

Спец-режим для запросов про инсайты/общий анализ:
- Сделай комплексный анализ: минимум 1 value + 1 table + 1 plot артефакт.
- Сопоставь артефакты между собой и сформулируй 4-8 наблюдений с числами.
- Для инсайт-запросов дай содержательный расширенный текст (не менее 3 абзацев): что сделано, что обнаружено, почему это важно, какие ограничения анализа.

Формат финального ответа:
- Отвечай на русском языке.
- По умолчанию отвечай развернуто и понятно.
- Если пользователь задал конкретный вопрос (например, "в каком классе..."), первая строка должна быть прямым ответом на вопрос.
- Структура для аналитики:
  1) Прямой ответ на вопрос пользователя,
  2) Что сделано (какие проверки/срезы),
  3) Ключевые наблюдения по данным (с числами),
  4) Итоговый вывод и границы применимости результата.
- Не вставляй сырой код в итоговый ответ, если пользователь не попросил.
- Когда вопрос конкретный, первая строка должна быть коротким прямым ответом без вводных.
"""


pandas_tool_prompt = """Инструмент для табличного анализа через Pandas.
Вход: Python-код с доступным DataFrame `df`.
Обязательно:
- сформируй `tool_result` строго в формате:
  {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": { "table_name": <pd.DataFrame | pd.Series> }
  }
- последняя строка кода: `tool_result`
- таблица должна быть понятной человеку:
  - осмысленные названия артефакта и колонок,
  - для числовых столбцов без нужды не оставляй длинные хвосты; округляй (обычно до 2-4 знаков),
  - избегай чрезмерно длинных текстов в названиях колонок.
- На стороне инструмента есть post-processing (авто-нормализация), но это страховка: старайся сразу возвращать человеко-читаемый результат.
- пример:
  ```python
  result_df = df.describe(include="all").transpose()
  tool_result = {
      "schema_version": "1.0",
      "artifact_type": "table",
      "items": {"describe": result_df}
  }
  tool_result
  ```
"""


plotly_tool_prompt = """Инструмент для графиков через Plotly.
Вход: Python-код с доступными `df`, `px`, `go`.
Обязательно:
- используй только Plotly
- сформируй `tool_result` строго в формате:
  {
    "schema_version": "1.0",
    "artifact_type": "plot",
    "items": { "plot_name": <go.Figure> }
  }
- последняя строка кода: `tool_result`
- график должен быть человеко-понятным:
  - короткий и информативный title,
  - понятные подписи осей/легенды,
  - при выводе чисел в подписях избегай лишней точности (обычно 2-3 знака).
- пример:
  ```python
  fig = px.histogram(df, x="Age", nbins=30, title="Распределение Age")
  tool_result = {
      "schema_version": "1.0",
      "artifact_type": "plot",
      "items": {"age_distribution": fig}
  }
  tool_result
  ```
"""


value_tool_prompt = """Инструмент для скалярных метрик.
Вход: Python-код с доступным `df`.
Обязательно:
- сформируй `tool_result` строго в формате:
  {
    "schema_version": "1.0",
    "artifact_type": "value",
    "items": { "metric_name": <float | int | str | bool> }
  }
- последняя строка кода: `tool_result`
- не используй переменную с похожим именем (`tool_result```, `toolresult`, и т.п.) — только `tool_result`
- значения должны быть удобны для чтения человеком:
  - округляй float до разумной точности (обычно 2-4 знака),
  - названия метрик делай короткими и понятными,
  - избегай “сырых” длинных чисел без необходимости.
- если вычисление вернуло словарь с числовыми/строковыми значениями, разверни его в отдельные скалярные метрики.
- На стороне инструмента есть post-processing (авто-округление), но это страховка: старайся сразу возвращать готовые читаемые значения.
- пример:
  ```python
  rows = len(df)
  tool_result = {
      "schema_version": "1.0",
      "artifact_type": "value",
      "items": {"row_count": rows}
  }
  tool_result
  ```
"""


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
                "  "
                + f"min={non_null.min()}, max={non_null.max()}, mean={round(float(non_null.mean()), 4)}"
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

    return "\n".join(lines)
