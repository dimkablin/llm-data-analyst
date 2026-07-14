## database_tool details

Use for lightweight database catalog operations. This tool does not generate analytical SQL.

### Actions

| Action | Purpose |
|---|---|
| `list_schemas` | list database schemas |
| `list_tables` | list available tables |
| `describe_table` | inspect columns for a known table |
| `preview` | preview rows from a known table |

### Required workflow

Do not call `preview` or `describe_table` with guessed table names.

1. If schema is unknown, call `list_schemas`.
2. Call `list_tables`.
3. Use only table names returned by `list_tables` for `preview` or `describe_table`.

### When to use sql_tool instead

Use `sql_tool` for analytical queries with joins, filters, grouping, aggregations, or subqueries.

### Examples

```json
{"action": "list_tables"}
{"action": "describe_table", "table": "orders"}
{"action": "preview", "table": "orders", "limit": 5}
{"action": "list_schemas"}
```
