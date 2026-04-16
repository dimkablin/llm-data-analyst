---
name: Pandas Tool
description: Табличные преобразования, группировки и вычисления по данным сессии.
kind: tool
tool_key: pandas_tool
triggers: таблица, таблиц, агрегация, агрег, фильтр, группировка, pandas, dataframe, describe, распредел, корреляц, статист, pivot, hist
---

## pandas_tool — табличный анализ

Вход: Python-код. Выполняется в **sandbox сессии** — все переменные сохраняются между вызовами инструментов.

### Примеры запросов пользователя

Вызывай этот инструмент, когда пользователь просит:
- "Покажи распределение по колонке X"
- "Посчитай корреляцию между X и Y"
- "Статистика по данным"
- "Группировка по категориям"
- "Отфильтруй строки где ..."
- "Describe датасета"
- "Pivot-таблица по ..."

### Переменные в scope
- `df` — DataFrame текущей сессии (уже загружен, **не вызывай** `pd.read_csv`)
- `pd` — pandas
- `np` — numpy
- Все переменные из предыдущих tool-вызовов (доступны автоматически)

Если подключена БД:
- `db` — хелпер для SQL-запросов (`db.query_dataframe(sql)`)
- `db_connection` — объект подключения

### Контракт результата
```python
tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"имя_таблицы": <pd.DataFrame | pd.Series>}
}
tool_result
```

### Правила
- НЕ вызывай `pd.read_csv()` / `pd.read_excel()` — `df` уже есть.
- Последняя строка: `tool_result`.
- Округляй числа до 2-4 знаков.
- Переменные, которые ты создаёшь (например `agg = df.groupby(...).sum()`), будут доступны в следующих tool-вызовах (plotly_tool, value_tool и т.д.).
- Запрещено: `globals()`, `locals()`, `os`, `sys`, `__import__`, `.plot()`, `matplotlib`.

### Примеры
```python
result_df = df.describe(include="all").transpose()
tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"describe": result_df}
}
tool_result
```

```python
# agg будет доступен в следующем plotly_tool для визуализации
agg = df.groupby("category")["revenue"].sum().reset_index()
tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"revenue_by_category": agg}
}
tool_result
```
