---
name: A/B Test Analysis
description: Statistically rigorous A/B test analysis — SRM check, auto-selected test (proportion z-test / Welch t-test), power analysis (Cohen's h/d), Bonferroni correction, guardrail metrics.
triggers: a/b test, ab test, ab тест, control group, test group, statistical significance, conversion, experiment, hypothesis test, significance, srm, sample ratio mismatch, контрольная группа, тестовая группа, конверсия, эксперимент
---

## A/B Test Analysis

Checks SRM, computes statistical significance and power, gives SHIP / DO_NOT_SHIP / INCONCLUSIVE recommendation.

### Algorithm (3 steps)
1. **SRM + base metrics** → `pandas_tool`: chi-square split check, conversion rate per group with 95% CIs.
2. **Significance + power** → `pandas_tool`: auto-select test (binary → proportion z-test; continuous → Welch t-test), Cohen's h/d, achieved power. Run Bonferroni if multiple numeric columns.
3. **Visualization** → `plotly_tool`: conversion rate bars with CI + relative uplift bar.

### Recommendation logic
- `SHIP` ✅ requires: no SRM + p < 0.05 + power > 0.70 + diff > 0
- `DO_NOT_SHIP` ❌: SRM detected OR (p < 0.05 AND diff < 0)
- `INCONCLUSIVE` ⚠️: power < 0.70 OR p ≥ 0.05

### Rules
- Check SRM first — if `srm_detected=True`, results are unreliable
- Run Step 2.5 (Bonferroni) when there are multiple numeric columns
- Warn if n < 100 in either group — small sample
- SRM assumes 50/50 split; adjust `expected_ratio` if intended split differs
