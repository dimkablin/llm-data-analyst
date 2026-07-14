from backend.core.public_identity import (
    PUBLIC_ASSISTANT_MODEL,
    display_model_name,
    runtime_model_payload,
)


def test_runtime_model_payload_redacts_for_regular_user():
    payload = runtime_model_payload(
        is_admin=False,
        provider="vllm",
        model="qwen3.5-35b-a3b-fp8",
        base_url="http://localhost:8000/v1",
        max_context_tokens=32768,
        context_window_source="settings",
    )
    assert payload["provider"] != "vllm"
    assert payload["model"] != "qwen3.5-35b-a3b-fp8"
    assert payload["base_url"] == ""
    assert payload["max_context_tokens"] == 32768
    assert payload["context_window_source"] == "settings"


def test_runtime_model_payload_keeps_real_values_for_admin():
    payload = runtime_model_payload(
        is_admin=True,
        provider="vllm",
        model="qwen3.5-35b-a3b-fp8",
        base_url="http://localhost:8000/v1",
        max_context_tokens=131072,
        context_window_source="settings",
    )
    assert payload == {
        "provider": "vllm",
        "model": "qwen3.5-35b-a3b-fp8",
        "base_url": "http://localhost:8000/v1",
        "max_context_tokens": 131072,
        "context_window_source": "settings",
    }


def test_display_model_name_for_user():
    assert display_model_name(is_admin=False, model="qwen3.5-35b-a3b-fp8") == PUBLIC_ASSISTANT_MODEL
