---
name: Plan/Fact Variance Analysis
description: SQL-first analysis only for active sources containing the required planfact_* tables; not for generic plan-versus-fact metrics.
enabled_by_default: true
triggers: planfact, planfact_by_cfo_period, planfact_by_cfo_article_period, planfact_plan_long, planfact_fact_monthly
---

## Plan/Fact Variance Analysis

Use this skill when the active source is `planfact` or the user asks to analyze Plan/Fact variance.

Required tables:

- `planfact_by_cfo_period`
- `planfact_by_cfo_article_period`
- `planfact_plan_long`
- `planfact_fact_monthly`

Do not use this workflow for an arbitrary source merely because the request
mentions plan, fact, or variance; the required `planfact_*` tables must exist.

### Algorithm

1. Use `sql_tool` first. Start with `catalog_tables`, then inspect the Plan/Fact table schemas.
2. For Plan/Fact comparisons, restrict both plan and fact to periods present in `planfact_fact_monthly`.
3. Compute numeric variance in SQL: `variance_amount = fact_amount - plan_amount`.
4. Rank drivers by `ABS(variance_amount)`, not by raw amount alone.
5. Separate `fact_amount <> 0 AND plan_amount = 0` from `plan_amount <> 0 AND fact_amount = 0`.
6. After numeric SQL, analyze descriptive fields: `service_content`, `plan_counterparty`, `fact_counterparty`, `fact_contract`.
7. Use text fields to explain reasons, not as join keys.
8. When the user names a CFO and article, resolve them in
   `planfact_by_cfo_article_period` before calculating anything:
   - match the CFO by normalized text;
   - prefer `article_key` for a supplied numeric article code;
   - use `article`, `plan_article`, and `fact_article` only to confirm the label;
   - if the exact pair returns no row, query candidate values and retry with the
     value that exists in the table. Never silently switch to another CFO or article.

For a request such as “explain the variance and give details for CFO Центр
сопровождения клиентов, article 11020406 | Заказная разработка ПО”, start with
this exact-pair filter (after confirming the column names):

```sql
WITH fact_periods AS (
  SELECT DISTINCT period FROM planfact_fact_monthly
)
SELECT cfo, article_key, MAX(article) AS article,
       SUM(plan_amount) AS plan_amount,
       SUM(fact_amount) AS fact_amount,
       SUM(fact_amount - plan_amount) AS variance_amount
FROM planfact_by_cfo_article_period
WHERE period IN (SELECT period FROM fact_periods)
  AND LOWER(TRIM(cfo)) = LOWER(TRIM('Центр сопровождения клиентов'))
  AND article_key = '11020406'
GROUP BY cfo, article_key
```

If it returns a row, reuse both resolved values unchanged in every following
detail query. If it returns no row, list candidate CFOs and article keys first;
do not replace the pair with the globally largest variance.

### Rules

- Never use `MAX(period)` as the default Plan/Fact comparison period rule.
- For plan-only analysis, all available plan periods may be used.
- Keep SQL read-only.
- Do not invent table names; inspect runtime tables first.
- Keep both CFO and article predicates in every detail query. Do not answer a
  named-pair question from the global top-variance row.
- Every article analysis must include a separate detail section covering
  `service_content`, `plan_counterparty`, `fact_counterparty`, and
  `fact_contract`. Query these fields for the same CFO, article, and fact
  periods; if a field is empty, explicitly say that the source has no value
  instead of omitting or inventing it.

See `DETAILS.md` for query patterns.
