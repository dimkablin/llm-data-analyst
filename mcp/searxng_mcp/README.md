# SearXNG MCP

This directory owns the full search stack:

- `searxng` — metasearch engine;
- `searxng-mcp` — MCP search tools;
- `search-service` — the FastAPI search/fetch bridge used by the backend.

## GitLab dev deploy

Push в ветку `dev` запускает один atomic build+deploy job на runner с тегом
`deploy-shell`. Runtime хранится в `/opt/deploy/searxng-mcp-dev`.

При первом deploy ключ генерируется на dev-хосте. Опциональная masked GitLab CI
variable `SEARXNG_MCP_API_KEY` позволяет задать его явно. Следующие deploy
используют сохранённый secret-файл. Backend из основного dev Compose подключается
по `http://searxng-mcp:8811/mcp` через сеть `llm-data-analyst-dev_default`.

Start every MCP service from `mcp/`:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

The MCP endpoint is `http://127.0.0.1:8811/mcp` by default and requires:

```text
Authorization: Bearer <SEARXNG_MCP_API_KEY>
```

Chronos and SearXNG use different Docker secrets.
