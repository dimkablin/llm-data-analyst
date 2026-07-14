---
id: session_note_tool
name: Session Note Tool
kind: tool
tool_key: session_note_tool
description: Save compact session-level notes that help later turns continue the same analysis.
enabled_by_default: true
triggers: []
---

## Purpose

Use `session_note_tool` to preserve concise analysis context inside the current chat session.

### API

```json
{"text": "Session note"}
```

### Final result protocol

Return a short confirmation.

### Runtime rules

- Save only facts useful for later turns in the same session.
- Do not replace artifacts or final answers with notes.
