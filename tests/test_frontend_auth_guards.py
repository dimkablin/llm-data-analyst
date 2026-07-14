from pathlib import Path


def test_workspace_redirects_before_reading_user_admin_flag() -> None:
    source = Path("frontend/src/app/pages/Workspace.tsx").read_text(encoding="utf-8")

    guard_index = source.index('return <Navigate to="/auth" replace />;')
    admin_read_index = source.index("const modelLabel")

    assert guard_index < admin_read_index


def test_chat_agent_hydrates_persisted_context_usage() -> None:
    source = Path("frontend/src/app/hooks/useChatAgent.ts").read_text(encoding="utf-8")

    hydrate_start = source.index("const hydrate = useCallback((")
    hydrate_end = source.index("const clearError = useCallback", hydrate_start)
    hydrate_source = source[hydrate_start:hydrate_end]

    assert "setContextUsage(session.context_usage ?? null);" in hydrate_source


def test_chat_agent_ignores_background_context_usage_events() -> None:
    source = Path("frontend/src/app/hooks/useChatAgent.ts").read_text(encoding="utf-8")

    handler_start = source.index("onContextUsage: (snapshot) => {")
    handler_end = source.index("},", handler_start)
    handler_source = source[handler_start:handler_end]

    assert "capturedSessionId === displayedSessionIdRef.current" in handler_source


def test_context_usage_indicator_is_accent_circle_without_visible_numbers() -> None:
    source = Path("frontend/src/app/components/workspace/ContextUsageRing.tsx").read_text(
        encoding="utf-8"
    )

    assert "buildContextUsageTooltipDetails" in source
    assert "rounded-full" in source
    assert "<svg" in source
    assert "strokeDasharray" in source
    assert "strokeDashoffset" in source
    assert "details.percentLine" in source
    assert "bottom-full right-0" in source
    assert "left-1/2 z-50 mb-2 w-[210px] -translate-x-1/2" not in source
    assert "bg-background/90" not in source
    assert "ring-border/60" not in source
    assert "shadow-sm" not in source
    assert "details.remainingLine" not in source
    assert "details.fragmentsLine" not in source
    assert "displayValue" not in source
    assert "tabular-nums" not in source
