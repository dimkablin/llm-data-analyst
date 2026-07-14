---
id: data_catalog_tool
name: Data Catalog Tool
kind: tool
tool_key: data_catalog_tool
description: Inspect session sources, tables, columns, and duplicate table-name mappings before analytical tool calls.
enabled_by_default: true
triggers:
  - catalog
  - data catalog
  - sources
  - columns
  - table names
---

## Purpose

Use `data_catalog_tool` to disambiguate available files, tables, columns, and source aliases.

### API

```json
{"mode": "list_sources"}
{"mode": "list_tables"}
{"mode": "search_columns", "query": "amount"}
```

### Final result protocol

Return structured JSON describing sources and table metadata. Do not create analytical artifacts.

### Runtime rules

- Use before SQL when table names are ambiguous.
- Use when multiple uploaded files contain similarly named tables.
