from __future__ import annotations

import pandas as pd

from backend.agent.services.artifacts import (
    ArtifactRehydrationRequest,
    rehydrate_artifacts_from_sandbox,
)


class _Sandbox:
    def get_user_scope(self) -> dict[str, object]:
        return {"old_table": pd.DataFrame({"value": [1]})}


class _SandboxManager:
    def get(self, session_id: str) -> _Sandbox:
        assert session_id == "session-1"
        return _Sandbox()


def test_rehydrate_skips_sandbox_when_turn_has_no_tool_signal(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.agent.services.artifacts.SandboxManager.get_instance",
        lambda: _SandboxManager(),
    )

    artifacts = rehydrate_artifacts_from_sandbox(
        ArtifactRehydrationRequest(
            artifacts=[],
            session_id="session-1",
            tool_names=[],
        )
    )

    assert artifacts == []


def test_rehydrate_recovers_sandbox_frames_when_tool_ran(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.agent.services.artifacts.SandboxManager.get_instance",
        lambda: _SandboxManager(),
    )

    artifacts = rehydrate_artifacts_from_sandbox(
        ArtifactRehydrationRequest(
            artifacts=[],
            session_id="session-1",
            tool_names=["sql_tool"],
        )
    )

    assert [artifact.name for artifact in artifacts] == ["old_table"]
