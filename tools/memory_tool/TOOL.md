---
id: memory_tool
name: Memory Tool
kind: tool
tool_key: memory_tool
description: Save stable long-term facts about the user when they are explicitly useful for future sessions.
enabled_by_default: true
triggers: []
---

## Purpose

Use `memory_tool` only for durable user preferences or stable facts.

### API

```json
{"text": "Stable memory item"}
```

### Final result protocol

Return a short confirmation.

### Runtime rules

- Do not save transient analytical results.
- Do not save secrets.
