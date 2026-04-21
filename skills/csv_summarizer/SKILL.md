---
name: CSV Summarizer
description: Fast automatic dataset overview — types, missing values, statistics, top values, and basic visualizations.
triggers: overview, summary, describe dataset, show structure, what's in the file, initial analysis, csv summarizer, обзор, резюме, опиши датасет, покажи структуру, что в файле, первичный анализ
---

## CSV Summarizer — quick dataset overview

Use when a file was just uploaded or a quick initial analysis is needed (not deep EDA).

### Algorithm (5 steps)
1. **Size + sample** → `pandas_tool`: shape, memory MB, first 5 rows.
2. **Schema + types** → `pandas_tool`: dtype, null_pct, unique count, `likely_type` (ID / binary / low_cardinality / high_cardinality / datetime-like / numeric).
3. **Descriptive statistics** → `pandas_tool`: `df.describe(include="all")`.
4. **Top categorical values** → `pandas_tool`: top-3 per column; if > 5 cat columns, select top-5 by entropy.
5. **Visualizations** → `plotly_tool`: missing values bar chart + numeric histograms (up to 4 non-ID columns).

### Rules
- ALWAYS start with Step 1 — agent needs scale context before proceeding
- Dataset > 500 columns → show only Steps 1–2, then ask which blocks are needed
- Columns with unique == nrows → mark as `ID`, exclude from histograms
- Object columns with `likely_type == "datetime-like"` → warn to cast to datetime
