---
id: update_plan
name: Update Plan
kind: tool
tool_key: update_plan
description: Create or replace the main agent's concise checklist before multi-step analytical work, then update the complete checklist whenever progress or new evidence changes the remaining route. Do not use it for a simple one-tool or informational request.
enabled_by_default: true
triggers: []
---

## Purpose

For a request with multiple deliverables or expected analytical tool calls, use
`update_plan` as the only tool call before the first analytical tool.

### API

Send the complete current checklist on every update. Each item has `step` and
`status`: `pending`, `in_progress`, or `completed`. You may add, remove, reorder,
or replace remaining steps when evidence changes the route.

### Runtime rules

- Keep the checklist concise and at most one item `in_progress`.
- Update the checklist immediately after a meaningful step completes or evidence changes the route.
- Do not use it for a single straightforward action or an informational answer.
- This tool records state only; it does not execute analysis or call other tools.
