---
name: Value Tool
description: Быстрые скалярные метрики и компактные числовые результаты.
kind: tool
tool_key: value_tool
triggers: метрика, среднее, количество, сумма, максимум, минимум, count, mean, sum
---

## value_tool — скалярные метрики

Вход: Python-код; `df` уже загружен.

### Контракт результата
```python
tool_result = {
    "schema_version": "1.0",
    "artifact_type": "value",
    "items": {"metric_name": <float | int | str | bool>}
}
tool_result
```

### Правила
- Только для коротких скалярных значений (не длинные тексты).
- Округляй float до 2-4 знаков.
- Последняя строка: `tool_result`.
- НЕ вызывай `pd.read_csv()`.
- Запрещено: `os`, `sys`, `globals()`.

### Пример
```python
rows = len(df)
cols = len(df.columns)
tool_result = {
    "schema_version": "1.0",
    "artifact_type": "value",
    "items": {"row_count": rows, "col_count": cols}
}
tool_result
```
