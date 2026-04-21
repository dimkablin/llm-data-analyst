---
name: SQL Tool
description: Аналитические запросы к БД и CSV в DuckDB. Возвращает табличный артефакт.
kind: tool
tool_key: sql_tool
triggers: sql, база данных, таблица, таблиц, запрос, query, database, db, выборка, джойн, join, агрегация, агрег, посчитай, сумм, средн, медиан, pivot, dataset, датасет, данных
---

## sql_tool — SQL queries

Entry: one `question` argument in natural language. Tool generates safe SELECT, returns table artifact.

### API
```python
sql_tool(question: str) -> table_artifact
```

### When to use
- Aggregations with GROUP BY, JOIN, subqueries, window functions
- Working with CSV loaded into DuckDB session
- When `database_tool` is too simple (no aggregation capability)

Prefer `database_tool` for: list tables, describe columns, preview rows.

### Question quality
- ✅ `"Average age by Age column in table titanic"`
- ✅ `"Top-5 categories by revenue sum in table sales"`
- ❌ `"Show data"` — unclear
- ❌ `"Analyze"` — too abstract

### Final result protocol
This tool returns a table artifact directly — no code execution, no `tool_result` needed.
The result variable name is stated in the tool response. Use that exact name in subsequent tools — do not invent `sql_dataset` or `data`.
For DB sessions: prefer `db.query_dataframe(sql)` inside `plotly_tool` directly (one call instead of two).

### Limits
- Read-only: INSERT / UPDATE / DELETE / DROP blocked
- Max 200 rows in result
