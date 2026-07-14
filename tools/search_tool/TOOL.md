---
id: search_tool
name: Search Tool
kind: tool
tool_key: search_tool
description: Search the web or fetch web pages for current external facts with source references.
enabled_by_default: true
triggers:
  - search
  - web
  - internet
  - latest
  - current
  - news
  - sources
---

## Purpose

Use `search_tool` when the answer depends on external current information not present in the session.

### API

Input is Python code with helper `search`.

```python
search.search("query", max_results=5)
search.search_result("query", artifact_name="web_results")
search.fetch(["https://example.com"])
```

### Final result protocol

Last line must be `tool_result`.

### Runtime rules

- Use search results for current facts.
- Fetch pages when snippets are insufficient.
- Base answers on returned sources, not model memory.
