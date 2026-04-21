---
name: Auto EDA
description: Systematic exploratory data analysis — distributions, correlations, outliers, type anomalies.
triggers: eda, exploratory analysis, data exploration, correlation, distribution, outliers, profiling, full analysis, разведочный анализ, исследование данных
---

## Auto EDA

Deep initial dataset analysis: distributions, correlations, outliers, type anomalies.

### Algorithm (5 steps)
1. **Numeric statistics** → `pandas_tool`: mean, median, std, skew, IQR outliers per numeric column.
2. **Correlation matrix** → `plotly_tool`: Pearson heatmap (`px.imshow`).
3. **Numeric distributions** → `plotly_tool`: box + histogram per column, top-6 by coefficient of variation.
4. **Categorical columns** → `pandas_tool`: cardinality, top value %, null %.
5. **Automated observations** → `pandas_tool`: flag missing > 10%, high correlation (|r| > 0.7), |skew| > 2, IQR outliers > 5%. Return as `observations_df` table.

### Rules
- Sample to 50k rows for Steps 1–3 when `len(df) > 50_000`
- Exclude ID-like columns (unique == nrows) from numeric analysis
- Step 3: prioritize columns by coefficient of variation, not column order
- Step 5: return computed `observations_df` table, not a text list
