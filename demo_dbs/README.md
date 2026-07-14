# Demo databases

Run:

```powershell
docker compose -f docker-compose.demo-dbs.yaml up -d
```

Reset and reload from CSV:

```powershell
docker compose -f docker-compose.demo-dbs.yaml down -v
docker compose -f docker-compose.demo-dbs.yaml up -d
```

## Postgres

- Type: `postgresql`
- Host in docker app: `demo_postgres`
- Port in docker app: `5432`
- Host from local backend: `localhost`
- Port from local backend: `55432`
- Database: `demo_analytics`
- Username: `demo`
- Password: `demo`
- Schema: `demo`
- Tables: `manual_plan`, `uh_receipts_2026_03`

## ClickHouse

- Type: `clickhouse`
- Host in docker app: `demo_clickhouse`
- HTTP port in docker app: `8123`
- Host from local backend: `localhost`
- HTTP port from local backend: `18123`
- Native port from local tools: `19000`
- Database: `demo`
- Username: `demo`
- Password: `demo`
- Tables: `manual_plan`, `uh_receipts_2026_03`
