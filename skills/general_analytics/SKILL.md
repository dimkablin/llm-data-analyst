---
name: General Analytics
description: Default workflow for any domain-neutral tabular analysis of CSV/XLSX, DuckDB, session artifacts, and connected databases.
enabled_by_default: true
---

## General Analytics

Use this skill as the default workflow for domain-neutral analysis of tabular
data from CSV/XLSX uploads, DuckDB/session artifacts, and connected databases.
It describes how to discover data, compute evidence, create artifacts, and
write a final answer without relying on customer-specific schemas or canned
conclusions.


### Algorithm
These are decision checkpoints, not mandatory separate tool calls. Use the
fewest calls that produce complete evidence and the required artifact.
1. **Plan** -> for a new multi-step task or uncertain source/schema mapping,
   form a brief working plan in the main agent reasoning and execute its first
   necessary tool. Simple one-tool requests proceed directly. Revise the plan
   only when tool evidence changes the route. The plan is not an answer.
2. **Source context** -> inspect available tables, files, or session artifacts.
   For connected databases, call `database_tool` when table names or schemas are
   unknown; when they are known, skip discovery and query the final requested grain.
3. **Question-to-schema mapping** -> map the user request to real columns only.
   If requested dimensions, metric variants, or scenario labels in one table are
   ambiguous, inspect them together in one compact grouped/distinct query.
   Copy observed column names verbatim; never invent convenience total columns.
4. **Data readiness** -> confirm the row grain and analysis period; inspect required
   types, nulls, duplicates, and invalid values; choose and report treatments that
   preserve the metric's meaning. Reuse readiness evidence already present in context;
   when preparation changes the data, publish the result as an analysis-ready artifact.
5. **Data extraction** -> use `sql_tool` for filtering, joins, top-N, grouping,
   windowing, and one complete answer-ready table at the final requested grain.
   Give important query outputs stable artifact names so later tools can reference them.
   For top-N over repeated rows, aggregate to the final period, slice, and item
   grain before applying window ranks or limits.
6. **Computation** -> use `pandas_tool` only for a derived transformation that
   the final SQL result does not already contain; keep it to one complete
   transformation when possible.
7. **Visualization** -> add a chart when requested or when it materially improves
   a comparison or trend. Explicit chart bans win.
8. **Answer metrics check** -> before the final answer, ensure every metric,
   date, percentage, comparison, min/max, ranking, delta, or statistic needed
   in prose exists in tool output. Reuse an existing final evidence table; create
   one compact `answer_metrics` table only for a claim that is still missing.
9. **Final synthesis** -> write the tool-derived numbers directly as plain
   text, interpret the result, explain caveats, and list concrete follow-up
   checks when uncertainty remains. Once evidence and any required chart exist,
   answer immediately instead of consuming unused tool iterations.
10. **Specialized workflows** -> if the request matches a dedicated analytical
   skill, call `get_tool_instructions(skill_id)` first and follow that skill's
   instructions.

### Rules
- Do not answer with numbers before successful tool output exists.
- For a named business metric that is not a directly observed field, use its
  semantic definition or an explicit user formula; otherwise ask for the formula.
- Do not output placeholders (`<value>`, `X/Y/Z`, `...`) or say exact values
  must be extracted later. Compute the metric first or omit that claim.
- Never turn tool-derived values into Markdown links or expose DataFrame
  expressions, variable names, or `.iloc` in the final answer.
- A file name is not a filter value; verify real values with schema inspection
  or distinct-value queries.
- Use typed range predicates for date and timestamp columns; never apply
  string-pattern operators such as `LIKE` to typed temporal values.
- When the requested period is coarser, use an explicit bucket (for example `date_trunc`) in
  SELECT/GROUP BY or a verified period column; grouping by the raw date remains raw-date grain.
- Empty SQL results require a compact check of exact labels and types followed
  by a corrected full extraction. Do not calculate from the empty artifact or
  switch to an unrelated preview.
- Never use preview, head, sample, or limited rows as the analysis dataset. If a
  result is truncated, aggregate to the final requested grain in SQL and rerun
  until the complete requested period and slices fit.
- If the final grain still exceeds a row cap, extract complete non-overlapping partitions and combine them before calculation or visualization.
- Before any SUM, ranking, or forecast input, inspect categorical scenario and
  member columns. Never combine mutually exclusive scenario rows or a roll-up
  with its components; select one scenario and either total or members.
- Multi-step requests need multiple tool calls when each step depends on a
  different calculation or artifact.
- Prefer SQL for grouped first/last values, window rankings, and independent
  dimension slices. In pandas, use named aggregation or rename explicit
  columns; never replace the complete column list without verifying its shape.
- Printed output is available to the next decision as diagnostic context, but
  final claims require a published table or value artifact.
- Do not repeat an equivalent successful call that returned an empty candidate.
  Change the filter, source labels, or grain.
- Do not repeat any equivalent successful call. Reuse its artifact; once evidence
  covers the request, stop calling tools and synthesize the answer.
- Percent, rate, and ratio columns are not additive weights. Use weighted
  formulas only when a valid denominator or absolute measure is present.
- Preserve units through every transformation. Apply scaling only when the
  conversion is explicit, and label percentage change separately from
  percentage-point change.
- For PostgreSQL wide measures, reshape exact observed columns with
  `CROSS JOIN LATERAL (VALUES ...)`; PostgreSQL has no `UNPIVOT` syntax, and
  independent theme, branch, or channel slices stay separate.
- A same-named column is not automatically the requested dimension or a join
  key. Its observed values must match the requested entity and overlap across
  sources. On mismatch, inspect the remaining schema candidates instead of
  continuing the calculation or declaring the dimension absent.
- Before any join, verify key uniqueness and include every shared entity key
  plus time; date alone duplicates repeated entities. For cross-table labels,
  materialize and verify a distinct-value key mapping
  before `COALESCE`, filling missing values, or excluding unmatched rows.
- Compare or rank magnitudes only when their units and scale match.
- Correlation, temporal coincidence, and seasonal co-movement are observational.
  Report unmatched joins, exceptions, and plausible confounders; do not use
  causal wording or chart titles without evidence from a causal design.
- For forecasts, report provider warnings and uncertainty intervals, compare
  plan values with those intervals, and distinguish point gaps from statistically
  meaningful deviations.
- For recommendations, enumerate every field requested by the user and state a
  concrete intervention for each item. Keep the intervention distinct from the problem and KPI.
- Make chart labels, units, periods, and titles match the plotted values; omit
  redundant or misleading charts.
- Keep final answers structured as: conclusion, key numbers, interpretation,
  artifacts, and next checks.
- A chart description without numbers is insufficient; a table dump without
  interpretation is also insufficient.
