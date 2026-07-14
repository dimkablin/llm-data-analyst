from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEPLOY_COMPOSE = ROOT / "docker-compose.deploy.yaml"


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
