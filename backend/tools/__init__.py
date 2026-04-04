from __future__ import annotations

from importlib import import_module

__all__ = [
    "ARTIFACT_OPTIONAL_TOOL_KEYS",
    "DATAFRAME_BASE_TOOL_KEYS",
    "DB_BASE_TOOL_KEYS",
    "KNOWN_TOOL_KEYS",
    "ToolBuildContext",
    "ToolCatalogSpec",
    "ToolRegistry",
    "build_runtime_capability_context",
    "build_tool_catalog",
    "detect_data_access_mode",
    "effective_enabled_tool_keys",
    "has_enabled_data_tools",
    "is_tool_allowed",
    "normalize_allowed_tool_keys",
    "normalize_tool_names",
    "required_data_tool_keys",
    "supports_artifact_optional_output",
]


def __getattr__(name: str):
    if name == "build_runtime_capability_context":
        module = import_module("backend.tools.capabilities")
        return getattr(module, name)
    if name in {"KNOWN_TOOL_KEYS", "ToolCatalogSpec", "build_tool_catalog"}:
        module = import_module("backend.tools.catalog")
        return getattr(module, name)
    if name == "ToolBuildContext":
        module = import_module("backend.tools.context")
        return getattr(module, name)
    if name in {
        "DATAFRAME_BASE_TOOL_KEYS",
        "DB_BASE_TOOL_KEYS",
        "ARTIFACT_OPTIONAL_TOOL_KEYS",
        "detect_data_access_mode",
        "effective_enabled_tool_keys",
        "has_enabled_data_tools",
        "is_tool_allowed",
        "normalize_allowed_tool_keys",
        "normalize_tool_names",
        "required_data_tool_keys",
        "supports_artifact_optional_output",
    }:
        module = import_module("backend.tools.policy")
        return getattr(module, name)
    if name == "ToolRegistry":
        module = import_module("backend.tools.registry")
        return getattr(module, name)
    raise AttributeError(f"module 'backend.tools' has no attribute {name!r}")
