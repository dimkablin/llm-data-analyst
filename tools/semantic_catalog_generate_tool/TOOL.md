---
id: semantic_catalog_generate_tool
name: Semantic Catalog Generate Tool
kind: tool
tool_key: semantic_catalog_generate_tool
description: "Generate and apply a semantic layer draft for the current DB or CSV/XLSX source."
enabled_by_default: true
triggers:
  - generate semantic layer
  - build semantic layer
  - infer metrics
  - analyze dataset semantics
---

## Purpose

Use `semantic_catalog_generate_tool` when the user asks the agent to build or improve the semantic layer from the current dataset.

### API

```json
{"apply": false, "sample_rows": 5, "max_tables": 20}
{"apply": true, "sample_rows": 5, "max_tables": 20}
```

### Runtime rules

- By default this tool does not mutate the catalog. Call with `apply=true` only after user confirmation.
- If column meaning is ambiguous, ask the user first instead of guessing.
- Run `semantic_catalog_read_tool` with `validate` after generation.
