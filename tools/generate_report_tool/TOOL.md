---
id: generate_report_tool
name: Generate Report Tool
kind: tool
tool_key: generate_report_tool
description: Export the current persisted session chat history and artifacts as a downloadable DOCX report.
enabled_by_default: true
triggers:
  - report
  - docx
  - export report
  - word report
  - download report
---

## Purpose

Use `generate_report_tool` only when the user explicitly asks for a report file/export.

### API

```json
{"title": "Optional report title"}
```

### Final result protocol

Return JSON with status, message, download URL, and file name.

### Runtime rules

- Do not perform new analysis.
- Use existing persisted session history and artifacts.
- If the session has no artifacts, return a clear failure status.
