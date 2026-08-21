from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEPLOY_COMPOSE = ROOT / "docker-compose.deploy.yaml"
DEPLOY_SCRIPT = ROOT / "scripts" / "ci" / "deploy.sh"


def test_deploy_runtime_defaults_use_local_stack_services() -> None:
    content = DEPLOY_COMPOSE.read_text(encoding="utf-8")

    assert (
        "- PHOENIX_COLLECTOR_ENDPOINT=${PHOENIX_COLLECTOR_ENDPOINT:-http://phoenix:6006/v1/traces}"
        in content
    )
    assert "- BACKEND_URL=${BACKEND_URL:-http://backend:8000}" in content
    assert "- PHOENIX_URL=${PHOENIX_URL:-http://phoenix:6006}" in content


def test_deploy_runtime_defaults_do_not_point_frontend_to_work_server() -> None:
    content = DEPLOY_COMPOSE.read_text(encoding="utf-8")

    assert "http://10.9.168.20:8605" not in content
    assert "http://10.9.168.20:8607" not in content


def test_deploy_compose_uses_ci_built_images() -> None:
    content = DEPLOY_COMPOSE.read_text(encoding="utf-8")

    assert "image: llm-data-analyst-backend:${IMAGE_TAG:-latest}" in content
    assert "image: llm-data-analyst-frontend:${IMAGE_TAG:-latest}" in content


def test_deploy_updates_compose_and_verifies_frontend_commit() -> None:
    content = DEPLOY_SCRIPT.read_text(encoding="utf-8")

    assert 'cp docker-compose.deploy.yaml "$compose_file"' in content
    assert '"${CI_COMMIT_SHA:-local}" /usr/share/nginx/html/assets' in content
