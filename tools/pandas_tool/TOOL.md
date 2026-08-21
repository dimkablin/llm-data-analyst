---
id: pandas_tool
name: Pandas Tool
kind: tool
tool_key: pandas_tool
description: Execute one dataframe transformation over an existing dataframe artifact and return one canonical table envelope. Use a current-turn sandbox variable directly and omit input_artifacts. Only for a durable artifact from history pass input_artifacts as alias to the stable artifact_id shown in history, never an artifact name. Keep dependent filtering, reshaping, derived columns, and ranking in that call when they use the same inputs; data acquisition and visualization are separate top-level actions. Use only the exact variable and output columns reported by the latest successful observation; never retype artifact rows as Python literals or branch on absent columns; when separate scenario columns already exist, calculate from them directly instead of pivoting the discriminator again; before groupby sum or mean use df.groupby(keys)[numeric_measure_columns], never the whole mixed-type frame; convert object/string dates with pd.to_datetime before .dt access; publish every output under a unique descriptive item key, never generic result or table; assign the final dataframe to tool_result.
enabled_by_default: true
triggers:
  - pandas
  - dataframe
  - aggregation
  - filter
  - pivot
  - statistics
  - table transform
---

## Purpose

Use `pandas_tool` for one dataframe transformation over an existing named sandbox artifact.

### API

Input is Python code executed in the session sandbox. Begin with the exact dataframe
variable and columns reported by the preceding successful tool observation. Available
common libraries are already available as `pd` and `np`; named artifact variables are
available directly. Source acquisition, dataframe transformation, and charting are
separate top-level actions.

For a dataframe named by the latest successful tool observation, use that sandbox variable
directly and omit `input_artifacts`. Only for an artifact listed in durable history, pass
`input_artifacts={"source": "artifact_id"}` and reference `source` in the code. The value
must be the stable ID shown in history, never the artifact name.

### Final result protocol

Complete the code with this table envelope:

```python
tool_result = {"schema_version": "1.0", "artifact_type": "table", "items": {"name": result_df}}
tool_result
```

Use a unique descriptive item key for each output. Do not overwrite generic
`result` or `table` names when more than one artifact exists in the turn.

For inspection or diagnostics, return findings as a compact table artifact through `tool_result`.
Narrative interpretation belongs in the final response and cites that table.

### Runtime contract

- Operate on named sandbox dataframe artifacts with pandas, numpy, re, or datetime.
- Do not use Pandas for grouping, deltas, shares, ranking, sorting, or relabeling that
  can be returned by the final SQL query; one complete SQL table is the smaller path.
- Copy exact variable names, column names, dtypes, and row counts from tool observations.
- Produce unique output column labels and a non-empty candidate set for reductions.
- Submit one transformation and wait for its result. A repair attempt starts from the returned error and observed dataframe schema; never submit alternative repairs in the same assistant turn.
