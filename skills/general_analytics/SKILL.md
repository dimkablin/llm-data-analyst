---
name: General Analytics
description: Default workflow for any domain-neutral tabular analysis of CSV/XLSX, DuckDB, session artifacts, and connected databases.
enabled_by_default: true
triggers: проанализируй, анализ данных, analytics, analyze, исследуй, посчитай, сравни, динамика, топ, рейтинг, тренд, отчет, отчёт, покажи, выведи, метрика, dashboard, дашборд, sql, база, таблица, dataset, датасет
---

## General Analytics

Use this skill as the default workflow for domain-neutral analysis of tabular
data from CSV/XLSX uploads, DuckDB/session artifacts, and connected databases.
It describes how to discover data, compute evidence, create artifacts, and
write a final answer without relying on customer-specific schemas or canned
conclusions.

### Algorithm
1. **Source context** -> inspect available tables, files, or session artifacts.
   For connected databases, call `database_tool` when table names or schemas are
   unknown; for known session tables, start with a small `sql_tool` query.
2. **Question-to-schema mapping** -> map the user request to real columns only.
   If a requested dimension or metric is ambiguous, inspect distinct values or
   schema metadata before calculating.
3. **Data extraction** -> use `sql_tool` for filtering, joins, top-N, grouping,
   windowing, and reusable result tables. Give important query outputs stable
   artifact names so later tools can reference them.
4. **Computation** -> use `pandas_tool` for derived metrics, rankings,
   descriptive statistics, contribution analysis, validation checks, and
   compact analytical tables.
5. **Visualization** -> use `plotly_tool` for trends, comparisons,
   distributions, and contribution views unless the user explicitly asks for a
   text-only answer.
6. **Answer metrics check** -> before the final answer, ensure every metric,
   date, percentage, comparison, min/max, ranking, delta, or statistic needed
   in prose exists in tool output. If not, create one compact `answer_metrics`
   table with `sql_tool` or `pandas_tool`.
7. **Final synthesis** -> cite the tool-derived numbers, interpret the result,
   explain caveats, and list concrete follow-up checks when uncertainty remains.
8. **Specialized workflows** -> if the request matches a dedicated analytical
   skill, call `get_tool_instructions(skill_id)` first and follow that skill's
   stricter contract.

### Rules
- Do not answer with numbers before successful tool output exists.
- Do not output placeholders (`<value>`, `X/Y/Z`, `...`) or say exact values
  must be extracted later. Compute the metric first or omit that claim.
- A file name is not a filter value; verify real values with schema inspection
  or distinct-value queries.
- Empty SQL results are evidence to refine filters or ask for clarification, not
  permission to invent a result.
- Multi-step requests need multiple tool calls when each step depends on a
  different calculation or artifact.
- Percent, rate, and ratio columns are not additive weights. Use weighted
  formulas only when a valid denominator or absolute measure is present.
- Preview/head/sample output only proves what appears in the sample; do not
  conclude that absent values do not exist without a complete check.
- Keep final answers structured as: conclusion, key numbers, interpretation,
  artifacts, and next checks.
- A chart description without numbers is insufficient; a table dump without
  interpretation is also insufficient.
