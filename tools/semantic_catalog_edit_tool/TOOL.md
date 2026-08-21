---
id: semantic_catalog_edit_tool
name: Semantic Catalog Edit Tool
kind: tool
tool_key: semantic_catalog_edit_tool
description: "Edit the current semantic layer: descriptions, aliases, metrics, relationships, and terms."
enabled_by_default: true
triggers:
  - add metric
  - edit metric
  - semantic layer
  - relationship
  - synonym
---

## Purpose

Use `semantic_catalog_edit_tool` to apply confirmed semantic-layer changes for the current session source.

### API

```json
{"action": "patch_column", "object_id": "column:sales.amount", "column_patch": {"description": "Revenue amount", "semantic_role": "metric_candidate"}}
{"action": "create_metric", "metric": {"key": "revenue", "name": "Revenue", "type": "simple", "base_table": "sales", "expr": "amount", "agg": "sum"}}
{"action": "create_relationship", "relationship": {"from_table": "orders", "from_column": "customer_id", "to_table": "customers", "to_column": "customer_id", "cardinality": "many_to_one"}}
```

### Runtime rules

- Use this tool, not `sql_tool`, `pandas_tool`, or `plotly_tool`, when the user asks to change semantic metadata such as metric definitions, aliases, relationships, terms, descriptions, facts, or dimensions.
- Ask the user before applying uncertain business assumptions.
- Prefer column/table patches for facts and dimensions; direct fact/dimension CRUD is intentionally not exposed yet.
- Run `semantic_catalog_read_tool` with `validate` after edits.
