# Plan/Fact Variance Analysis Details

Use `sql_tool` for the numeric pass before prose.

Example driver query:

```sql
WITH fact_periods AS (
  SELECT DISTINCT period FROM planfact_fact_monthly
)
SELECT
  cfo,
  article,
  service_content,
  plan_counterparty,
  fact_counterparty,
  fact_contract,
  SUM(plan_amount) AS plan_amount,
  SUM(fact_amount) AS fact_amount,
  SUM(fact_amount - plan_amount) AS variance_amount
FROM planfact_by_cfo_article_period
WHERE period IN (SELECT period FROM fact_periods)
GROUP BY
  cfo,
  article,
  service_content,
  plan_counterparty,
  fact_counterparty,
  fact_contract
ORDER BY ABS(SUM(fact_amount - plan_amount)) DESC;
```

Use `fact_amount <> 0 AND plan_amount = 0` for fact without plan, and
`plan_amount <> 0 AND fact_amount = 0` for plan without fact.

## Named CFO + article case

For a question such as:

> Объясни превышение и дай детализацию по ЦФО Центр сопровождения клиентов,
> статья 11020406 | Заказная разработка ПО.

Do not start from the global top drivers. First resolve the requested pair.
Prefer the numeric article code because the displayed article label may come
from either plan or fact after matching.

```sql
SELECT DISTINCT
  cfo,
  article_key,
  article,
  plan_article,
  fact_article
FROM planfact_by_cfo_article_period
WHERE LOWER(TRIM(cfo)) = LOWER(TRIM('Центр сопровождения клиентов'))
  AND article_key = '11020406';
```

If this returns no rows, inspect candidates instead of guessing:

```sql
SELECT DISTINCT
  cfo,
  article_key,
  article,
  plan_article,
  fact_article
FROM planfact_by_cfo_article_period
WHERE cfo ILIKE '%Центр сопровождения клиентов%'
  AND (
    article_key LIKE '11020406%'
    OR article ILIKE '%Заказная разработка ПО%'
    OR plan_article ILIKE '%Заказная разработка ПО%'
    OR fact_article ILIKE '%Заказная разработка ПО%'
  )
ORDER BY cfo, article_key, article;
```

After resolving the pair, calculate its numbers while keeping both filters:

```sql
WITH fact_periods AS (
  SELECT DISTINCT period FROM planfact_fact_monthly
)
SELECT
  cfo,
  article_key,
  COALESCE(
    MAX(NULLIF(plan_article, '')),
    MAX(NULLIF(fact_article, '')),
    MAX(article)
  ) AS article,
  SUM(plan_amount) AS plan_amount,
  SUM(fact_amount) AS fact_amount,
  SUM(fact_amount - plan_amount) AS variance_amount,
  CASE
    WHEN SUM(plan_amount) <> 0
    THEN 100.0 * SUM(fact_amount - plan_amount) / SUM(plan_amount)
  END AS variance_pct
FROM planfact_by_cfo_article_period
WHERE period IN (SELECT period FROM fact_periods)
  AND cfo = 'Центр сопровождения клиентов'
  AND article_key = '11020406'
GROUP BY cfo, article_key;
```

Then fetch the reason evidence for the same pair:

```sql
WITH fact_periods AS (
  SELECT DISTINCT period FROM planfact_fact_monthly
)
SELECT
  period,
  service_content,
  plan_counterparty,
  fact_counterparty,
  fact_contract,
  plan_amount,
  fact_amount,
  variance_amount
FROM planfact_by_cfo_article_period
WHERE period IN (SELECT period FROM fact_periods)
  AND cfo = 'Центр сопровождения клиентов'
  AND article_key = '11020406'
ORDER BY period;
```

Answer in this order:

1. Plan, fact, absolute variance, and variance percent.
2. Periods contributing to the variance.
3. Service content, counterparties, and contracts found in the selected rows.
4. Evidence-backed explanation. Label any interpretation not stated in the
   text fields as a hypothesis, not a fact.
