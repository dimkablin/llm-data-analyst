from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_each_mcp_server_has_independent_dev_pipeline() -> None:
    for server in ("chronos_mcp", "searxng_mcp"):
        root = ROOT / "mcp" / server
        pipeline = (root / ".gitlab-ci.yml").read_text(encoding="utf-8")
        deploy = (root / "scripts" / "ci" / "deploy.sh").read_text(encoding="utf-8")

        assert '$CI_COMMIT_BRANCH == "dev"' in pipeline
        assert "deploy-shell" in pipeline
        assert "up -d --no-build" in deploy
        assert "exec -T" in deploy
        assert "openssl rand -hex 32" in deploy


def test_mcp_deploy_compose_uses_ci_built_image_tags() -> None:
    chronos = (ROOT / "mcp" / "chronos_mcp" / "docker-compose.deploy.yml").read_text(
        encoding="utf-8"
    )
    searxng = (ROOT / "mcp" / "searxng_mcp" / "docker-compose.deploy.yml").read_text(
        encoding="utf-8"
    )

    assert "chronos-mcp:${IMAGE_TAG" in chronos
    assert "searxng-mcp:${IMAGE_TAG" in searxng
    assert "CHRONOS_MCP_API_KEY" in chronos
    assert "searxng_mcp_api_key" in searxng
    assert "ports:" not in chronos
    assert "ports:" not in searxng
