---
id: rag_tool
name: RAG Tool
kind: tool
tool_key: rag_tool
description: Perform semantic retrieval of relevant passages from the configured indexed knowledge base with source references; limited to content available in that index.
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
