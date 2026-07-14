from importlib import import_module


def test_api_app_imports_with_route_wiring() -> None:
    app_module = import_module("backend.api.app")

    assert app_module.app is not None
