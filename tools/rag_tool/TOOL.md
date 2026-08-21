---
id: rag_tool
name: RAG Tool
kind: tool
tool_key: rag_tool
description: Query the configured indexed knowledge base for a synthesized grounded answer with references. Use only entities explicitly named in that answer. For exhaustive inventories, use complementary queries unless the result explicitly establishes completeness; never complete a list from model memory.
enabled_by_default: true
triggers:
  - knowledge base
  - policy
  - process
  - documentation
  - regulation
  - source reference
---

## Purpose

Use `rag_tool` to answer questions from indexed documents, policies, procedures, and knowledge-base content.

### API

```json
{"query": "Natural-language retrieval query"}
```

### Final result protocol

Return JSON/text with retrieved content and source references.

### Runtime rules

- Use for document-grounded knowledge retrieval.
- Do not use for calculations over session dataframe artifacts.
- When retrieved context conflicts with session data, explain the source boundary.
- For a requested list, include only items and attributes explicitly present in
  the returned knowledge-base answer. An incomplete result produces a grounded partial
  list, never completion from model memory.
- For an exhaustive inventory, run complementary queries using vocabulary from
  the returned answer unless it explicitly establishes completeness.

- If retrieval returns no relevant context, the tool output must state that context is missing and request clarification; it must not return inferred or invented definitions.
