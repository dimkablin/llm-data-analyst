---
name: DuckDB Large File Analysis
description: SQL analysis of large CSV and Parquet files via DuckDB — no in-memory loading, supports JOINs across multiple files.
triggers: duckdb, large file, large csv, parquet, sql on file, multiple files, join files, out of memory, large dataset, gb file, большой файл, несколько файлов, большой датасет
---

## DuckDB Large File Analysis

Use when pandas can't handle the file size, or when you need SQL queries on CSV/Parquet without loading into memory.

In this project `sql_tool` uses DuckDB under the hood — it can read CSV and Parquet files directly.

### Finding the file path → sql_tool

If the user just uploaded a file and didn't specify the path — ask explicitly. If the path is partially known — use glob:

```sql
-- Search by pattern in the uploads directory
SELECT * FROM read_csv_auto('/uploads/*.csv') LIMIT 5;

-- Check if a specific file is accessible
SELECT COUNT(*) FROM read_csv_auto('/path/to/file.csv');
```

### Exploring file schema → sql_tool

```sql
-- View structure and first rows
DESCRIBE SELECT * FROM read_csv_auto('/path/to/file.csv');

SELECT * FROM read_csv_auto('/path/to/file.csv') LIMIT 10;
```

### Handling encodings and delimiters → sql_tool

```sql
-- Non-standard delimiter or encoding
SELECT * FROM read_csv(
    '/path/to/file.csv',
    encoding='utf-8',        -- or 'cp1251' for Windows Cyrillic
    delim=';',               -- semicolon delimiter
    header=true,
    auto_detect=true
) LIMIT 10;
```

### Reading multiple files via glob → sql_tool

```sql
-- All CSVs from a directory
SELECT * FROM read_csv_auto('/data/monthly/*.csv') LIMIT 10;

-- Multiple Parquet files
SELECT year, SUM(sales)
FROM read_parquet('/data/sales/*.parquet')
GROUP BY year ORDER BY year;
```

### Aggregation without loading into memory → sql_tool

```sql
-- Group and aggregate directly from file
SELECT
    category,
    COUNT(*) AS count,
    SUM(revenue) AS total_revenue,
    AVG(revenue) AS avg_revenue,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY revenue) AS median_revenue
FROM read_csv_auto('/path/to/file.csv')
GROUP BY category
ORDER BY total_revenue DESC
LIMIT 20;
```

### JOIN across multiple files → sql_tool

```sql
-- Join two CSVs on a key
SELECT
    a.user_id,
    a.event_date,
    b.segment,
    b.region,
    a.revenue
FROM read_csv_auto('/path/to/events.csv') a
LEFT JOIN read_csv_auto('/path/to/users.csv') b
    ON a.user_id = b.user_id
WHERE a.event_date >= '2024-01-01'
LIMIT 1000;
```

### Temporary tables for complex queries → sql_tool

```sql
-- Create a temp table for reuse across queries
CREATE OR REPLACE TABLE tmp_filtered AS
SELECT * FROM read_csv_auto('/path/to/file.csv')
WHERE status = 'active' AND amount > 0;

SELECT
    DATE_TRUNC('month', CAST(created_at AS DATE)) AS month,
    COUNT(*) AS events,
    SUM(amount) AS total
FROM tmp_filtered
GROUP BY 1
ORDER BY 1;
```

### Parquet file analysis → sql_tool

```sql
-- Parquet works the same — just use read_parquet instead of read_csv_auto
SELECT
    year,
    region,
    SUM(sales) AS total_sales
FROM read_parquet('/path/to/data.parquet')
GROUP BY year, region
ORDER BY year, total_sales DESC;
```

### Passing SQL results to pandas_tool

After running a query via sql_tool, the result is automatically available as `df` in the next pandas_tool:

```python
# In pandas_tool after sql_tool:
# df already contains the SQL query result
df.dtypes      # check types
df.describe()  # statistics
```

### Rules

- ALWAYS use `read_csv_auto` — it auto-detects types, delimiter, and encoding
- ALWAYS add `LIMIT` when initially exploring files > 1 GB
- DO NOT guess the file path — if the user hasn't specified it, ask explicitly
- For Cyrillic files try `auto_detect=true` first; if characters are garbled, explicitly set `encoding='cp1251'`
- `PERCENTILE_CONT` and `PERCENTILE_DISC` are DuckDB-specific, not standard SQL
- After aggregating via sql_tool, the result is available as `df` in pandas_tool for further processing
