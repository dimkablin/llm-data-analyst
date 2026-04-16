---
name: Data Quality Audit
description: Comprehensive data checks — duplicates, missing values, outliers, type mismatches, referential integrity.
triggers: data quality, duplicates, deduplication, missing values, outliers, data anomalies, data check, dq, audit, integrity, качество данных, дубли, пропуски, выбросы, проверка данных, целостность
---

## Data Quality Audit

Systematic dataset quality checks before analysis or when data issues are suspected.

### Algorithm (4 steps)
1. **DQ report per column** → `pandas_tool`: null_pct, unique count, outliers (3×IQR), numeric-as-object detection. Severity: `critical` / `warning` / `ok`.
2. **Duplicates** → `pandas_tool`: full-row dedup + key-based dedup on auto-detected ID candidates.
3. **Cross-column validation** → `pandas_tool`: date ordering (end < start), negative values in positive-only columns.
4. **Issue visualization** → `plotly_tool`: missing % by column + outlier counts.

### Rules
- Severity: `critical` → blocks analysis; `warning` → needs attention
- ALWAYS run Step 3 (cross-column) — often catches critical errors in dates and amounts
- Duplicates > 5% of rows → stop and warn the user before proceeding
- DO NOT fix data — diagnostics only; let the user decide
- Conclude explicitly: "Data is ready for analysis" or "Cleaning required"
- Object columns with > 80% numeric values → suggest `pd.to_numeric` conversion
