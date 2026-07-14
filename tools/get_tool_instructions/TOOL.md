---
id: get_tool_instructions
name: Get Tool Instructions
kind: tool
tool_key: get_tool_instructions
description: Load full markdown instructions for an enabled tool or analytical skill on demand.
enabled_by_default: true
triggers: []
---

## Purpose

Use `get_tool_instructions` before using an unfamiliar, complex, or domain-specific tool/skill.

### API

```json
{"skill_id": "sql_tool", "details": false}
{"skill_id": "general_analytics", "details": true}
```

### Final result protocol

Return markdown instructions. `details=false` returns core instructions; `details=true` returns `DETAILS.md` when available.

### Runtime rules

- Do not call the same id with the same details flag repeatedly.
- Load details after a tool failure or before complex scenarios.
