---
name: DuckDB Large File Analysis
description: SQL analysis of large CSV and Parquet files via DuckDB — no in-memory loading, supports JOINs across multiple files.
triggers: duckdb, large file, large csv, parquet, sql on file, multiple files, join files, out of memory, large dataset, gb file, большой файл, несколько файлов, большой датасет
---

## DuckDB Large File Analysis

SQL analysis of large CSV/Parquet files via DuckDB — no in-memory loading.
In this project `sql_tool` uses DuckDB — queries run directly on files.

### Key functions (pass as question to `sql_tool`)
```sql
read_csv_auto('/path/file.csv')           -- auto-detects types, delimiter, encoding
read_parquet('/path/*.parquet')           -- glob patterns work
read_csv('/path', encoding='cp1251', delim=';')  -- explicit options
```

### Algorithm
1. Locate file path — ask user if unknown; glob `/uploads/*.csv` if partially known
2. Explore schema: `DESCRIBE SELECT * FROM read_csv_auto(...)` + `LIMIT 10`
3. Aggregate/filter/join directly on files without loading into memory
4. After `sql_tool` → result is available as `df` in `pandas_tool`

### Rules
- ALWAYS use `read_csv_auto` — auto-detects everything
- ALWAYS add `LIMIT` when initially exploring files > 1 GB
- Never guess file paths — ask explicitly if unknown
- Cyrillic encoding: try auto_detect first; if garbled → `encoding='cp1251'`
- `PERCENTILE_CONT` is DuckDB-specific, not standard SQL
