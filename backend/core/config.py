from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

from backend.core.llm_provider import get_provider_policy

_log = logging.getLogger(__name__)

# Single source of truth for analysis depth limits.
# inner_recursion_limit — max tool-call iterations inside one agent step.
DEPTH_PROFILES: dict[str, dict[str, int]] = {
    "light":  {"inner_recursion_limit": 32},
    "medium": {"inner_recursion_limit": 64},
    "deep":   {"inner_recursion_limit": 120},
}
# Maximum agent_max_steps allowed per depth (mirrors inner_recursion_limit).
DEPTH_MAX_STEPS: dict[str, int] = {
    k: v["inner_recursion_limit"] for k, v in DEPTH_PROFILES.items()
}


def _clear_missing_tls_env_path(name: str, *, expect_dir: bool = False) -> None:
    raw_value = os.getenv(name)
    if raw_value is None:
        return
    path = raw_value.strip()
    if not path:
        os.environ.pop(name, None)
        return

    exists = os.path.isdir(path) if expect_dir else os.path.isfile(path)
    if exists:
        return

    _log.warning(
        "Ignoring %s because path does not exist: %s",
        name,
        path,
    )
    os.environ.pop(name, None)


def _sanitize_tls_env() -> None:
    _clear_missing_tls_env_path("SSL_CERT_FILE")
    _clear_missing_tls_env_path("REQUESTS_CA_BUNDLE")
    _clear_missing_tls_env_path("CURL_CA_BUNDLE")
    _clear_missing_tls_env_path("SSL_CERT_DIR", expect_dir=True)


_sanitize_tls_env()


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_docker_runtime() -> bool:
    return os.path.exists("/.dockerenv")


def _normalize_llm_base_url(raw_url: str) -> str:
    if not raw_url:
        return raw_url
    if not _is_docker_runtime():
        return raw_url

    parsed = urlparse(raw_url)
    host = (parsed.hostname or "").lower()
    if host not in {"localhost", "127.0.0.1", "::1"}:
        return raw_url

    userinfo = ""
    if parsed.username:
        userinfo = parsed.username
        if parsed.password:
            userinfo += f":{parsed.password}"
        userinfo += "@"

    port = f":{parsed.port}" if parsed.port else ""
    new_netloc = f"{userinfo}host.docker.internal{port}"
    return urlunparse(parsed._replace(netloc=new_netloc))


def _normalize_llm_provider(raw_provider: str | None) -> str:
    normalized = str(raw_provider or "").strip().lower()
    if normalized == "vllm":
        return "vllm"
    return "ollama"


def _resolve_llm_provider() -> str:
    return _normalize_llm_provider(os.getenv("LLM_PROVIDER", "ollama"))


def _resolve_llm_model(provider: str) -> str:
    default = "qwen3-next:80b-cloud" if provider == "vllm" else "qwen3-coder-next:cloud"
    return (os.getenv("LLM_MODEL_NAME") or "").strip() or default


def _resolve_llm_base_url(provider: str) -> str:
    default = (
        "http://localhost:8001/v1" if provider == "vllm" else "http://localhost:11434/v1"
    )
    raw_url = (os.getenv("LLM_MODEL_API_URL") or "").strip() or default
    return _normalize_llm_base_url(raw_url)


def _resolve_llm_api_key(provider: str) -> str:
    default = "EMPTY" if provider == "vllm" else "ollama"
    return (os.getenv("LLM_API_KEY") or "").strip() or default


def _default_chat_template_kwargs_enabled(provider: str) -> bool:
    return get_provider_policy(provider).thinking_control_mode != "none"


# ── HARDCODED LLM CONFIG ─────────────────────────────────────────────

# _LLM_PROVIDER = "vllm"
# _LLM_MODEL = "Qwen3.5-35B-A3B-FP8"
# _LLM_BASE_URL = "http://10.9.88.17:8123/v1"
# _LLM_API_KEY = "sk-fake-for-vllm"
#
# _LLM_CHAT_TEMPLATE_KWARGS_ENABLED = True


# ── LLM CONFIG (from env with sane defaults) ───────────────────────────
_LLM_PROVIDER = _resolve_llm_provider()
_LLM_MODEL = _resolve_llm_model(_LLM_PROVIDER)
_LLM_BASE_URL = _resolve_llm_base_url(_LLM_PROVIDER)
_LLM_API_KEY = _resolve_llm_api_key(_LLM_PROVIDER)
_LLM_CHAT_TEMPLATE_KWARGS_ENABLED = _get_bool(
    "LLM_CHAT_TEMPLATE_KWARGS_ENABLED",
    _default_chat_template_kwargs_enabled(_LLM_PROVIDER),
)


@dataclass(frozen=True)
class Settings:
    llm_provider: str = _LLM_PROVIDER
    llm_model: str = _LLM_MODEL
    llm_base_url: str = _LLM_BASE_URL
    llm_api_key: str = _LLM_API_KEY
    llm_chat_template_kwargs_enabled: bool = _LLM_CHAT_TEMPLATE_KWARGS_ENABLED
    llm_temperature_chat: float = float(os.getenv("LLM_TEMPERATURE_CHAT", "0.7"))
    llm_temperature_tool: float = float(os.getenv("LLM_TEMPERATURE_TOOL", "0.5"))
    llm_top_p: float = float(os.getenv("LLM_TOP_P", "0.95"))
    llm_top_k: int = int(os.getenv("LLM_TOP_K", "20"))
    llm_presence_penalty: float = float(os.getenv("LLM_PRESENCE_PENALTY", "1.5"))
    llm_max_tokens_default: int = int(os.getenv("LLM_MAX_TOKENS_DEFAULT", "2048"))
    llm_max_tokens_reasoning: int = int(os.getenv("LLM_MAX_TOKENS_REASONING", "4096"))
    llm_enable_thinking: bool = _get_bool("LLM_ENABLE_THINKING", False)
    llm_show_think: bool = _get_bool("LLM_SHOW_THINK", True)
    llm_num_ctx: int = int(os.getenv("LLM_NUM_CTX", "32768"))
    max_context_per: float = float(os.getenv("MAX_CONTEXT_PER", "0.8"))
    llm_streaming: bool = _get_bool("LLM_STREAMING", True)
    llm_warmup_enabled: bool = _get_bool("LLM_WARMUP_ENABLED", True)
    llm_warmup_timeout_sec: int = int(os.getenv("LLM_WARMUP_TIMEOUT_SEC", "12"))
    tool_exec_timeout_sec: int = int(os.getenv("TOOL_EXEC_TIMEOUT_SEC", "25"))
    tool_cache_size: int = int(os.getenv("TOOL_CACHE_SIZE", "48"))
    backend_query_timeout_sec: int = int(os.getenv("BACKEND_QUERY_TIMEOUT_SEC", "180"))
    agent_max_steps: int = int(os.getenv("AGENT_MAX_STEPS", "32"))
    agent_step_timeout_sec: int = int(os.getenv("AGENT_STEP_TIMEOUT_SEC", "45"))
    agent_history_max_messages: int = int(os.getenv("AGENT_HISTORY_MAX_MESSAGES", "8"))
    agent_history_summary_chars: int = int(
        os.getenv("AGENT_HISTORY_SUMMARY_CHARS", "700")
    )
    agent_inner_recursion_limit: int = int(
        os.getenv("AGENT_INNER_RECURSION_LIMIT", "32")
    )
    max_tools_per_cycle: int = max(2, int(os.getenv("MAX_TOOLS_PER_CYCLE", "4")))
    agent_prompt_max_columns: int = int(os.getenv("AGENT_PROMPT_MAX_COLUMNS", "16"))
    agent_prompt_head_rows: int = int(os.getenv("AGENT_PROMPT_HEAD_ROWS", "3"))
    agent_cache_enabled: bool = _get_bool("AGENT_CACHE_ENABLED", True)
    observation_mask_enabled: bool = _get_bool("OBSERVATION_MASK_ENABLED", True)
    agent_cache_size: int = int(os.getenv("AGENT_CACHE_SIZE", "128"))
    agent_cache_ttl_sec: int = int(os.getenv("AGENT_CACHE_TTL_SEC", "900"))
    agent_analysis_depth: str = (
        os.getenv("AGENT_ANALYSIS_DEPTH", "light").strip().lower()
    )
    skills_dir: str = os.getenv("AGENT_SKILLS_DIR", "./skills")

    session_ttl_days: int = int(os.getenv("BACKEND_SESSION_TTL_DAYS", "7"))
    max_dataset_mb: int = int(os.getenv("BACKEND_MAX_DATASET_MB", "100"))
    storage_dir: str = os.getenv("BACKEND_STORAGE_DIR", "./backend_storage")
    auth_db_path: str = os.getenv("AUTH_DB_PATH", "./backend_storage/app.db")
    auth_token_ttl_days: int = int(os.getenv("AUTH_TOKEN_TTL_DAYS", "30"))
    auth_default_admin_username: str = os.getenv(
        "AUTH_DEFAULT_ADMIN_USERNAME", "admin"
    )
    auth_default_admin_password: str = os.getenv(
        "AUTH_DEFAULT_ADMIN_PASSWORD", "admin"
    )
    db_connections_encryption_key: str = os.getenv(
        "DB_CONNECTIONS_ENCRYPTION_KEY", ""
    ).strip()
    db_connections_encryption_key_current: str = os.getenv(
        "DB_CONNECTIONS_ENCRYPTION_KEY_CURRENT", ""
    ).strip()
    db_connections_encryption_keys_old: str = os.getenv(
        "DB_CONNECTIONS_ENCRYPTION_KEYS_OLD", ""
    ).strip()
    db_connections_allow_private_hosts: bool = _get_bool(
        "DB_CONNECTIONS_ALLOW_PRIVATE_HOSTS", False
    )
    db_connections_test_timeout_sec: int = int(
        os.getenv("DB_CONNECTIONS_TEST_TIMEOUT_SEC", "8")
    )

    rag_url: str = os.getenv("RAG_URL", "http://10.9.168.20:9621").strip()
    rag_timeout_sec: int = int(os.getenv("RAG_TIMEOUT_SEC", "45"))
    rag_verify_ssl: bool = _get_bool("RAG_VERIFY_SSL", False)
    rag_query_mode: str = os.getenv("RAG_QUERY_MODE", "hybrid").strip().lower()
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "5"))

    csv_session_ttl_sec: int = int(os.getenv("CSV_SESSION_TTL_SEC", "7200"))

    semantic_layer_enabled: bool = _get_bool("SEMANTIC_LAYER_ENABLED", True)
    semantic_profile_max_tables: int = int(os.getenv("SEMANTIC_PROFILE_MAX_TABLES", "100"))
    semantic_profile_max_columns_per_table: int = int(
        os.getenv("SEMANTIC_PROFILE_MAX_COLUMNS_PER_TABLE", "200")
    )
    semantic_profile_sample_rows: int = int(os.getenv("SEMANTIC_PROFILE_SAMPLE_ROWS", "1000"))
    semantic_profile_top_values: int = int(os.getenv("SEMANTIC_PROFILE_TOP_VALUES", "20"))
    semantic_profile_timeout_sec: int = int(os.getenv("SEMANTIC_PROFILE_TIMEOUT_SEC", "60"))
    semantic_catalog_store: str = os.getenv("SEMANTIC_CATALOG_STORE", "file").strip().lower()
    semantic_catalog_postgres_dsn: str = os.getenv("SEMANTIC_CATALOG_POSTGRES_DSN", "").strip()
    semantic_qdrant_url: str = os.getenv("SEMANTIC_QDRANT_URL", "http://qdrant:6333").strip()
    semantic_qdrant_api_key: str = os.getenv("SEMANTIC_QDRANT_API_KEY", "").strip()
    semantic_qdrant_collection: str = os.getenv(
        "SEMANTIC_QDRANT_COLLECTION",
        "semantic_catalog_chunks",
    ).strip()
    semantic_vector_enabled: bool = _get_bool("SEMANTIC_VECTOR_ENABLED", False)
    semantic_embedding_provider: str = os.getenv(
        "SEMANTIC_EMBEDDING_PROVIDER", "local"
    ).strip().lower()
    semantic_qdrant_timeout_sec: int = int(os.getenv("SEMANTIC_QDRANT_TIMEOUT_SEC", "10"))
    semantic_top_k: int = int(os.getenv("SEMANTIC_TOP_K", "8"))
    semantic_embedding_base_url: str = os.getenv("SEMANTIC_EMBEDDING_BASE_URL", "").strip()
    semantic_embedding_api_key: str = os.getenv("SEMANTIC_EMBEDDING_API_KEY", "").strip()
    semantic_embedding_model: str = os.getenv(
        "SEMANTIC_EMBEDDING_MODEL",
        "text-embedding-3-small",
    ).strip()
    semantic_embedding_dim: int = int(os.getenv("SEMANTIC_EMBEDDING_DIM", "1536"))
    semantic_embedding_timeout_sec: int = int(os.getenv("SEMANTIC_EMBEDDING_TIMEOUT_SEC", "30"))
    semantic_embedding_batch_size: int = int(os.getenv("SEMANTIC_EMBEDDING_BATCH_SIZE", "64"))

    openproject_base_url: str = os.getenv("OPENPROJECT_BASE_URL", "").strip()
    openproject_host_header: str = os.getenv("OPENPROJECT_HOST_HEADER", "").strip()
    openproject_api_key: str = os.getenv("OPENPROJECT_API_KEY", "").strip()
    openproject_project: str = os.getenv("OPENPROJECT_PROJECT", "").strip()
    openproject_days: int = int(os.getenv("OPENPROJECT_DAYS", "90"))
    openproject_max_items: int = int(os.getenv("OPENPROJECT_MAX_ITEMS", "0"))
    openproject_pg_host: str = os.getenv(
        "OPENPROJECT_PG_HOST", os.getenv("POSTGRES_HOST", "localhost")
    ).strip()
    openproject_pg_port: int = int(os.getenv("OPENPROJECT_PG_PORT", os.getenv("POSTGRES_PORT", "5432")))
    openproject_pg_database: str = os.getenv(
        "OPENPROJECT_PG_DATABASE", os.getenv("POSTGRES_DB", "llm_data_analyst")
    ).strip()
    openproject_pg_username: str = os.getenv(
        "OPENPROJECT_PG_USERNAME", os.getenv("POSTGRES_USER", "postgres")
    ).strip()
    openproject_pg_password: str = os.getenv(
        "OPENPROJECT_PG_PASSWORD", os.getenv("POSTGRES_PASSWORD", "")
    ).strip()
    openproject_pg_schema: str = os.getenv("OPENPROJECT_PG_SCHEMA", "openproject").strip()
    openproject_pg_sslmode: str = os.getenv("OPENPROJECT_PG_SSLMODE", "prefer").strip()

    backend_public_api_url: str = os.getenv("BACKEND_PUBLIC_API_URL", "http://10.9.168.20:8605").strip()
    predict_backend_url: str = os.getenv("PREDICT_BACKEND_URL", "http://10.9.168.20:8802").strip()
    predict_theme: str = os.getenv("PREDICT_THEME", "dark").strip()

    cors_allow_origins: str = os.getenv("BACKEND_CORS_ALLOW_ORIGINS", "*")
    phoenix_enabled: bool = _get_bool("PHOENIX_ENABLED", True)
    phoenix_project_name: str = os.getenv("PHOENIX_PROJECT_NAME", "llm-data-analyst")
    phoenix_host: str = os.getenv("PHOENIX_HOST", "localhost")
    phoenix_ui_port: int = int(os.getenv("PHOENIX_UI_PORT", "8607"))
    phoenix_host_root_path: str = os.getenv("PHOENIX_HOST_ROOT_PATH", "/phoenix")
    phoenix_collector_endpoint: str = (
        os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
        or f"http://{os.getenv('PHOENIX_HOST', 'localhost')}:6006/v1/traces"
    )
    phoenix_auto_instrument: bool = _get_bool("PHOENIX_AUTO_INSTRUMENT", True)


settings = Settings()

if settings.auth_default_admin_password == "admin":
    _log.warning(
        "AUTH_DEFAULT_ADMIN_PASSWORD is 'admin'. "
        "Change it before production deployment!"
    )
