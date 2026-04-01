---
name: Pandas Tool
description: Табличные преобразования, группировки и вычисления по данным сессии.
kind: tool
tool_key: pandas_tool
triggers: таблица, агрегация, фильтр, группировка, pandas, dataframe, describe
---

## pandas_tool — табличный анализ

Вход: Python-код; `df` уже загружен.

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
top5 = df.head(5)
tool_result = {
    "schema_version": "1.0",
    "artifact_type": "table",
    "items": {"table": top5}
}
tool_result
```
