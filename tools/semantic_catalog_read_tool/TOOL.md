---
id: semantic_catalog_read_tool
name: Semantic Catalog Read Tool
kind: tool
tool_key: semantic_catalog_read_tool
description: "Read semantic metadata that is missing, incomplete, or ambiguous in the injected context. It resolves metric contracts, terms, relationships, catalog status, objects, and validation; it does not calculate data."
enabled_by_default: true
---

## Purpose

Use `semantic_catalog_read_tool` to inspect semantic metadata that is not already complete in the injected semantic context.

### API

```json
{"action": "status"}
{"action": "get_catalog"}
{"action": "search", "query": "revenue", "top_k": 8}
{"action": "resolve", "query": "revenue by month", "top_k": 8}
{"action": "list_metrics"}
{"action": "list_relationships"}
{"action": "list_terms"}
{"action": "validate"}
```

### Runtime rules

- `search` returns `matched` plus ranked, typed `candidates`; treat `matched=true`
  as a successful catalog hit even when the candidate is a term rather than a metric.
- Use complete term definitions, confirmed metric keys, and declared relationships already present
  in the injected context without retrieving them again. Call `search` or `resolve` only for
  missing, incomplete, or ambiguous metadata, or when the user explicitly asks to inspect the catalog.
- If `matched=false`, RAG or web may clarify an unknown term's meaning but cannot supply its
  executable formula; ask the user for the formula before calculating.
- A confirmed metric key is a complete executable contract, including derived metric dependencies.
  Pass that key once to `sql_tool` mode `semantic_query`; do not separately resolve dependencies,
  inspect raw tables, or reimplement the formula.
- Follow `execution_mode` returned by `resolve`. For `execute_sql`, use the returned metric
  contracts, note, and relationships; do not send those metrics together as `semantic_query`.
- A known term-definition question can be answered from the injected context without a tool call.
  Semantic metadata is not a charting or SQL result.
- When a question asks how datasets or metrics are joined, use a relationship contract already in
  context; otherwise inspect `search` results and `list_relationships`. Never infer a join from
  similar column names.
- Read before edit when object ids are unknown.
- Use `validate` after semantic edits.
- This tool is read-only and may run in parallel with other read-only tools.
