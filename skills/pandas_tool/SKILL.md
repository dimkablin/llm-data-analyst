---
name: Pandas Tool
description: Табличные преобразования, группировки и вычисления по данным сессии.
kind: tool
tool_key: pandas_tool
triggers: таблица, таблиц, агрегация, агрег, фильтр, группировка, pandas, dataframe, describe, распредел, корреляц, статист, pivot, hist
---

## pandas_tool — табличный анализ

Вход: Python-код. Выполняется в **sandbox сессии** — все переменные сохраняются между вызовами инструментов.

### API
```python
# Нет фиксированного API — свободный Python-код в sandbox
# Доступно всегда: df, pd, np
# При подключённой БД: db.query_dataframe(sql: str) -> pd.DataFrame, db_connection
```

### Scope
- `df` — DataFrame текущей сессии (уже загружен, **не вызывай** `pd.read_csv`)
- `pd`, `np` — всегда доступны
- `db`, `db_connection` — при подключённой БД
- Все переменные из предыдущих tool-вызовов доступны по имени

### Final result protocol
Последнее выражение в коде должно быть `tool_result`. Sandbox захватывает только последнее выражение — print или присвоение последней строкой дадут пустой результат.

Для табличных данных `tool_result` содержит `artifact_type: "table"` и `items` с DataFrame'ами.
Структуру смотри в DETAILS.md — вызови `get_tool_instructions('pandas_tool', details=True)`.

### Rules
- НЕ вызывай `pd.read_csv()` / `pd.read_excel()` — `df` уже есть
- Последняя строка: `tool_result`
- Округляй числа до 2-4 знаков
- Переменные, созданные здесь (например `agg`), доступны в следующих tool-вызовах
- Запрещено: `globals()`, `locals()`, `os`, `sys`, `__import__`, `.plot()`, `matplotlib`
