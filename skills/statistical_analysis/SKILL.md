---
name: Statistical Analysis
description: Hypothesis testing, regression, ANOVA, correlations — full statistical toolkit with result interpretation.
triggers: statistics, hypothesis, regression, anova, correlation, pearson, spearman, t-test, chi-square, normality, linear dependence, statistical test, статистика, гипотеза, регрессия, корреляция, нормальность
---

## Statistical Analysis

Rigorous hypothesis testing, dependency modeling, and group comparison.

### Algorithm (order matters)
1. **Normality** → `pandas_tool`: Shapiro-Wilk (n ≤ 5000) or D'Agostino (n > 5000) per numeric column. Result determines parametric vs non-parametric branch.
2. **Correlation** → `pandas_tool`: Pearson + Spearman for all numeric pairs, ranked by |r|.
3. **One-Way ANOVA** → `pandas_tool`: if categorical + numeric columns present (min 5 rows per group).
4. **Chi-Square** → `pandas_tool`: categorical column pairs (min expected cell frequency ≥ 5).
5. **Linear regression** → `pandas_tool` + `plotly_tool`: auto-select highest-|r| pair; scatter + OLS line; residuals Q-Q plot.

### Rules
- Normality test first — determines which subsequent tests apply
- Regression: only for the pair with highest absolute correlation
- ALWAYS visualize regression residuals (Q-Q plot)
- R² < 0.3 → weak model, warn the user
- p > 0.05 = "no evidence of an effect", not "no effect"
