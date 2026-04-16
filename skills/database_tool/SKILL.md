---
name: Database Tool
description: Быстрый просмотр структуры БД — таблицы, колонки, превью строк, схемы. Без генерации SQL.
kind: tool
tool_key: database_tool
triggers: таблицы, покажи таблицы, структура бд, колонки, схема, превью, первые строки, список таблиц, describe, list tables, show tables, какие таблицы, перечисли таблицы
---

## Database Tool — structure inspection

Light tool for structural DB operations. **Does not generate SQL** — calls catalog directly, fast.

### API
```python
database_tool(action: Literal["list_tables","describe_table","preview","list_schemas"],
              table: str | None = None, db_schema: str | None = None, limit: int = 10)
# table required for describe_table/preview; db_schema from list_schemas; limit max 50
```

### Mandatory order
⚠️ Never call `preview` or `describe_table` with a guessed table name.

1. Schema unknown → `list_schemas` → `list_tables(db_schema=<result>)`
2. Schema known → `list_tables` directly
3. Only after getting real table names → `preview` or `describe_table`

If `list_tables` returns empty → call `list_schemas` first, then retry.

### Final result protocol
This tool returns results directly to the agent — no code execution, no `tool_result` needed.
Use `describe_table` or `preview` results as context; pass table names to `sql_tool` for further querying.

### When NOT to use
- Complex queries (JOIN, GROUP BY, subqueries) → `sql_tool`
- Aggregation, filtering, computation → `sql_tool`
- CSV data → `pandas_tool`
