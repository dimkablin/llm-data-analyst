from __future__ import annotations

import importlib
import json
import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen
from uuid import uuid4

import pandas as pd
import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
TRUE_VALUES = {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True, slots=True)
class E2ELlmConfig:
    provider: str
    base_url: str
    model: str
    api_key: str
    timeout_sec: int

    @classmethod
    def from_env(cls) -> E2ELlmConfig:
        provider = _env("E2E_LLM_PROVIDER", "LLM_PROVIDER", default="ollama").lower()
        base_url = _env(
            "E2E_LLM_BASE_URL",
            "LLM_MODEL_API_URL",
            default="http://localhost:11434",
        )
        model = _env("E2E_LLM_MODEL", "LLM_MODEL_NAME", default="qwen3:14b")
        api_key = _env(
            "E2E_LLM_API_KEY",
            "LLM_API_KEY",
            default="ollama" if provider == "ollama" else "EMPTY",
        )
        timeout_sec = int(_env("E2E_LLM_TIMEOUT_SEC", default="180"))
        return cls(
            provider="vllm" if provider == "vllm" else "ollama",
            base_url=base_url.rstrip("/"),
            model=model,
            api_key=api_key,
            timeout_sec=max(30, timeout_sec),
        )

    @property
    def ollama_tags_url(self) -> str:
        return f"{self.base_url.removesuffix('/v1')}/api/tags"

    @property
    def openai_models_url(self) -> str:
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/models"
        return f"{self.base_url}/v1/models"

    def healthcheck(self) -> tuple[bool, str]:
        url = self.ollama_tags_url if self.provider == "ollama" else self.openai_models_url
        try:
            payload = _get_json(url=url, api_key=self.api_key, timeout_sec=2.5)
        except (OSError, URLError, TimeoutError) as exc:
            return False, f"{url} is not reachable: {exc}"
        except Exception as exc:
            return False, f"{url} healthcheck failed: {exc}"

        if self.provider == "ollama":
            models = {
                str(item.get("name") or item.get("model") or "").strip()
                for item in payload.get("models", [])
                if isinstance(item, dict)
            }
            if models and self.model not in models:
                return False, (
                    f"Ollama is reachable, but model {self.model!r} is not listed. "
                    f"Available models: {sorted(models)!r}"
                )
        return True, ""


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return default


def _is_enabled() -> bool:
    return any(
        os.getenv(name, "").strip().lower() in TRUE_VALUES
        for name in ("E2E_LLM_TESTS", "LIVE_LLM_TESTS")
    )


def _get_json(*, url: str, api_key: str, timeout_sec: float) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    if api_key and api_key not in {"EMPTY", "ollama"}:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout_sec) as response:  # noqa: S310
        raw = response.read().decode("utf-8", errors="replace")
    return json.loads(raw or "{}")


@pytest.fixture(scope="session")
def live_llm_config() -> E2ELlmConfig:
    load_dotenv(ROOT / ".env", override=False)
    if not _is_enabled():
        pytest.skip("Set E2E_LLM_TESTS=1 to run local LLM e2e tests.")

    config = E2ELlmConfig.from_env()
    ok, reason = config.healthcheck()
    if not ok:
        pytest.skip(reason)
    return config


@pytest.fixture
def live_agent_settings(live_llm_config: E2ELlmConfig):
    import backend.core.config as config_module

    config_module = importlib.reload(config_module)
    runtime_dir = ROOT / ".e2e_runtime" / uuid4().hex
    runtime_dir.mkdir(parents=True, exist_ok=True)
    settings = replace(
        config_module.Settings(),
        llm_provider=live_llm_config.provider,
        llm_model=live_llm_config.model,
        llm_base_url=live_llm_config.base_url,
        llm_api_key=live_llm_config.api_key,
        llm_temperature_chat=0.0,
        llm_temperature_tool=0.0,
        llm_enable_thinking=False,
        llm_streaming=False,
        llm_streaming_force=False,
        llm_warmup_enabled=False,
        agent_cache_enabled=False,
        backend_query_timeout_sec=live_llm_config.timeout_sec,
        agent_step_timeout_sec=min(90, live_llm_config.timeout_sec),
        agent_inner_recursion_limit=10,
        agent_analysis_depth="light",
        tool_exec_timeout_sec=30,
        storage_dir=str(runtime_dir / "storage"),
        skills_dir=str(ROOT / "skills"),
    )
    try:
        yield settings
    finally:
        shutil.rmtree(runtime_dir, ignore_errors=True)


@pytest.fixture
def live_agent_runner(live_agent_settings):
    from backend.agent_graph.adapter import AgentGraphQueryRunner
    from backend.sessions.session_memory import StructuredSessionMemory

    return AgentGraphQueryRunner(
        live_agent_settings,
        allowed_tool_keys={"planner_tool", "pandas_tool", "value_tool"},
        session_memory=StructuredSessionMemory(),
    )


@pytest.fixture
def invoke_live_agent(live_agent_runner):
    from backend.agent.callbacks import ToolCollector

    def _invoke(
        prompt: str,
        *,
        df: pd.DataFrame | None = None,
        session_source: dict[str, Any] | None = None,
        selected_skill_ids: list[str] | None = None,
    ):
        source = dict(session_source or {})
        collector = ToolCollector(source_context=source)
        response = live_agent_runner.run_query(
            df,
            prompt,
            history=[],
            use_history=False,
            include_reasoning=False,
            callbacks=[collector],
            trace_context={
                "request_kind": "live_e2e",
                "session_id": "live-e2e",
                "dataset_name": source.get("source_label") or "e2e.csv",
            },
            session_source=source,
            selected_skill_ids=selected_skill_ids or [],
        )
        if _looks_unavailable(response):
            pytest.skip(f"LLM endpoint became unavailable during e2e run: {response.reasoning}")
        return response, collector, live_agent_runner

    return _invoke


def _looks_unavailable(response: Any) -> bool:
    text = str(getattr(response, "final_text", "") or "")
    reasoning = str(getattr(response, "reasoning", "") or "")
    return bool(
        getattr(response, "llm_unreachable", False)
        or text == "Language model is unavailable."
        or "LLM invoke failed" in reasoning
        or "connection" in reasoning.lower()
        or "timed out" in reasoning.lower()
    )
