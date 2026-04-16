---
name: Insight Synthesis
description: Transform analysis results into structured business insights using the So What / Why / Now What framework with impact prioritization.
triggers: insights, conclusions, analysis summary, what does this mean, business conclusion, results, recommendations, executive summary, what to do next, interpretation, инсайты, выводы, резюме анализа, рекомендации, интерпретация
---

## Insight Synthesis

Final step after completing analysis. Structures observations via So What / Why / Now What, prioritizes by impact×confidence, produces executive summary.

### Algorithm (3 steps)
1. **Key metrics** → `value_tool`: total/mean/median per numeric column; skip constants (CV ≤ 0.01); flag mean/median divergence > 20%.
2. **Programmatic insight extraction** → `pandas_tool`: auto-detect missing > 30%, extreme outliers (3×IQR, > 1%), dominant category (> 70%). Append manual insights from prior session steps (root_cause, ab_test, etc.). Sort by priority, keep top 5.
3. **Priority chart** → `plotly_tool`: horizontal bar, colored by Critical / Important / FYI.

### Rules
- Generate insights programmatically from real data — do NOT summarize from memory
- After auto_insights, manually append findings from prior session tool calls
- Maximum 5 insights — prioritize by impact × confidence
- Each insight: **So What** (numbers) + **Why** (hypothesis) + **Now What** (concrete action)
- Recommendations must be actionable with owner + timeline, not "conduct further analysis"
