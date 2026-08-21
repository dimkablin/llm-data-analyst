from importlib import import_module

from fastapi import HTTPException


def test_api_app_imports_with_route_wiring() -> None:
    app_module = import_module("backend.api.app")

    assert app_module.app is not None


def test_planfact_routes_are_registered() -> None:
    app_module = import_module("backend.api.app")
    paths = {route.path for route in app_module.app.routes}

    assert {
        "/sessions/{session_id}/source/planfact/detect",
        "/sessions/{session_id}/source/planfact/confirm",
        "/sessions/{session_id}/source/planfact/config",
    } <= paths


def test_planfact_error_handler_preserves_http_status() -> None:
    data_routes = import_module("backend.api.routes.data")
    error = HTTPException(status_code=413, detail="Dataset exceeds size limit")

    assert data_routes._handle_planfact_error(error) is error
