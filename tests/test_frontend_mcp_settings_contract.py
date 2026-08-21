from pathlib import Path


def test_frontend_exposes_mcp_settings_contract() -> None:
    api = Path("frontend/src/app/lib/backend-api.ts").read_text(encoding="utf-8")
    types = Path("frontend/src/app/lib/backend-types.ts").read_text(encoding="utf-8")
    settings = Path(
        "frontend/src/app/components/account/ToolAccessSection.tsx"
    ).read_text(encoding="utf-8")

    assert "MCPServerAvailability" in types
    assert "AdminMCPServerConfig" in types
    assert "listMcpServers" in api
    assert "updateMcpServerEnabled" in api
    assert "listAdminMcpServers" in api
    assert "upsertAdminMcpServer" in api
    assert "deleteAdminMcpServer" in api
    assert "testAdminMcpServer" in api
    assert "McpServersCard" in settings
    assert "MCP" in settings
    assert "updateMcpServerEnabled" in settings
    assert "upsertAdminMcpServer" in settings
    assert 'aria-label="Проверить подключение"' in settings
