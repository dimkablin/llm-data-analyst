from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OBSOLETE_ENV_KEYS = {
    "LLM_NO_THINK_PREFIX",
    "AGENT_EVALUATE_ENABLED",
    "AGENT_EVALUATE_MAX_TOKENS",
    "LLM_TEMPERATURE",
    "LLM_MAX_ITERATIONS",
    "LLM_MAX_EXECUTION_TIME",
    "LLM_STREAMING_FORCE",
}
FILES_TO_SCAN = [
    ".env.example",
    "k8s/configmap.yaml",
]


def test_obsolete_env_keys_are_removed_from_examples_and_legacy_ui() -> None:
    for relative_path in FILES_TO_SCAN:
        content = (ROOT / relative_path).read_text(encoding="utf-8")
        for key in OBSOLETE_ENV_KEYS:
            pattern = rf"(?<![A-Z0-9_]){re.escape(key)}(?![A-Z0-9_])"
            assert not re.search(pattern, content), f"{key} remains in {relative_path}"
