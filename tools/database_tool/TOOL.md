---
id: database_tool
name: Database Tool
kind: tool
tool_key: database_tool
description: "Lightweight database catalog inspection: list tables, describe columns, preview rows, and list schemas."
enabled_by_default: true
triggers:
  - database structure
  - list tables
  - show tables
  - describe table
  - schema
  - preview rows
---

## Purpose

Use `database_tool` for database structure inspection when no LLM-generated SQL is needed.

### API

```json
{"mode": "list_tables"}
{"mode": "describe_table", "table_name": "orders"}
{"mode": "preview", "table_name": "orders", "limit": 10}
```

### Final result protocol

Return structured catalog or preview data. For analytical extraction use `sql_tool`.

### Runtime rules

- Do not use for complex joins or calculations.
- Use `sql_tool` when the result should become a named analytical artifact.
