---
name: Value Tool
description: Быстрые скалярные метрики и компактные числовые результаты.
kind: tool
tool_key: value_tool
triggers: метрика, среднее, средн, количество, сумма, сумм, максимум, минимум, count, mean, sum, медиан, посчитай
---

## value_tool — скалярные метрики

Вход: Python-код. Выполняется в **sandbox сессии** — все переменные сохраняются между вызовами инструментов.

### Переменные в scope
- `df` — DataFrame текущей сессии (уже загружен, **не вызывай** `pd.read_csv`)
- `pd` — pandas
- `np` — numpy
- Все переменные из предыдущих tool-вызовов (доступны автоматически)

Если подключена БД:
- `db` — хелпер для SQL-запросов (`db.query_dataframe(sql)`)

### Примеры запросов пользователя

Вызывай этот инструмент, когда пользователь просит:
- "Сколько строк в таблице?"
- "Какое среднее значение revenue?"
- "Посчитай медиану"
- "Максимум / минимум по колонке"
- "Суммарная выручка"
- "Count уникальных клиентов"

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
- Можешь использовать переменные из предыдущих tool-вызовов (например `agg` из `pandas_tool`).
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
