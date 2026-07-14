---
id: generate_summary_tool
name: Generate Summary Tool
kind: tool
tool_key: generate_summary_tool
description: Summarize current session history, notes, and artifact summaries without computing new facts.
enabled_by_default: true
triggers:
  - summary
  - summarize
  - recap
  - executive summary
  - key takeaways
---

## Purpose

Use `generate_summary_tool` when the user asks to summarize already produced session work.

### API

```json
{"focus": "Optional focus", "max_history_items": 12}
```

### Final result protocol

Return JSON containing status, message, summary markdown, history items used, and artifact count.

### Runtime rules

- Do not use for new calculations.
- Use analytical tools first when the answer requires new facts.
- Keep the summary grounded in chat history, session notes, and artifacts.
