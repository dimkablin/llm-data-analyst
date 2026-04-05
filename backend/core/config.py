from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

_log = logging.getLogger(__name__)


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
    return provider != "vllm"


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
    llm_num_ctx: int = int(os.getenv("LLM_NUM_CTX", "32768"))
    llm_streaming: bool = _get_bool("LLM_STREAMING", True)
    llm_streaming_force: bool = _get_bool("LLM_STREAMING_FORCE", True)
    llm_warmup_enabled: bool = _get_bool("LLM_WARMUP_ENABLED", True)
    llm_warmup_timeout_sec: int = int(os.getenv("LLM_WARMUP_TIMEOUT_SEC", "12"))
    tool_exec_timeout_sec: int = int(os.getenv("TOOL_EXEC_TIMEOUT_SEC", "25"))
    tool_cache_size: int = int(os.getenv("TOOL_CACHE_SIZE", "48"))
    backend_query_timeout_sec: int = int(os.getenv("BACKEND_QUERY_TIMEOUT_SEC", "180"))
    agent_max_steps: int = int(os.getenv("AGENT_MAX_STEPS", "20"))
    agent_step_timeout_sec: int = int(os.getenv("AGENT_STEP_TIMEOUT_SEC", "45"))
    agent_history_max_messages: int = int(os.getenv("AGENT_HISTORY_MAX_MESSAGES", "8"))
    agent_history_summary_chars: int = int(
        os.getenv("AGENT_HISTORY_SUMMARY_CHARS", "700")
    )
    agent_inner_recursion_limit: int = int(
        os.getenv("AGENT_INNER_RECURSION_LIMIT", "14")
    )
    agent_prompt_max_columns: int = int(os.getenv("AGENT_PROMPT_MAX_COLUMNS", "16"))
    agent_prompt_head_rows: int = int(os.getenv("AGENT_PROMPT_HEAD_ROWS", "3"))
    agent_cache_enabled: bool = _get_bool("AGENT_CACHE_ENABLED", True)
    agent_cache_size: int = int(os.getenv("AGENT_CACHE_SIZE", "128"))
    agent_cache_ttl_sec: int = int(os.getenv("AGENT_CACHE_TTL_SEC", "900"))
    agent_analysis_depth: str = (
        os.getenv("AGENT_ANALYSIS_DEPTH", "light").strip().lower()
    )
    llm_no_think_prefix: str = os.getenv("LLM_NO_THINK_PREFIX", "/no_think").strip()
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


