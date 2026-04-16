---
name: Root Cause Investigation
description: Systematic metric change investigation — z-score validation, dimension drill-down, top contributors, hypothesis testing with chi-square, waterfall.
triggers: root cause, first cause, why dropped, why grew, what changed, drill down, segment contribution, metric change, investigation, decomposition, первопричина, почему упало, почему выросло, что изменилось, расследование
---

## Root Cause Investigation

Use when you need to understand WHY a metric changed. Validates the change is real (z-score), finds guilty segments via drill-down, tests hypotheses.

### Algorithm (4 steps)
1. **Change validation** → `pandas_tool`: z-score vs baseline. If |Z| < 1 → stop, change is noise.
2. **Dimension drill-down** → `pandas_tool`: delta contribution per segment, up to 3 categorical dims.
3. **Top contributors** → `pandas_tool`: rank all segments across dims, identify main culprit.
4. **Hypothesis testing + waterfall** → `pandas_tool` + `plotly_tool`: chi-square mix shift, volume change, per-unit quality. Waterfall chart for top dimension.

### Rules
- Start with z-score: |Z| < 1 → change is normal noise, don't hunt root causes
- Mix shift — ALWAYS validate with chi-square (p < 0.05)
- If one segment contributes > 80% → explicitly state "Main culprit: [segment]"
- Three hypothesis types: mix shift, volume, quality (metric per unit)
- If fewer than two periods in data → ask user to clarify what to compare
