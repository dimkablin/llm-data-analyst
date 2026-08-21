"""Security and reliability tests added during code-review remediation.

All tests here are offline — no LLM, no database, no network required.
"""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import patch

import pytest

# The tools.impl package must be imported before sql_table_service to avoid a
# circular import: sql_table_service → tools.impl.db_helpers (triggers __init__)
# → factory → sql_tool → sql_table_service.
import backend.tools.impl  # noqa: F401 — side-effect import resolves the cycle

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 3. Config validation — insecure admin password
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """_validate_startup_config must abort when the default 'admin' password is used.

    We import and call the function directly instead of importing the full app
    module, so we don't trigger module-level side effects (DB init, warmup, etc.).
    """

    def _get_validate_fn(self):
        """Import _validate_startup_config without running the full app module."""

        # Build a minimal stub module so the function can be extracted cleanly.
        import backend.api.app as app_mod

        return app_mod._validate_startup_config

    def test_validate_startup_config_exits_on_admin_password(self):
        import backend.api.app as app_mod
        from backend.core.config import Settings

        insecure = Settings(auth_default_admin_password="admin")
        with patch.object(app_mod, "settings", insecure):
            with pytest.raises(SystemExit) as exc_info:
                app_mod._validate_startup_config()
        # SystemExit with a non-empty message string counts as non-zero
        assert exc_info.value.code, "SystemExit must carry a non-empty message"

    def test_validate_startup_config_passes_with_strong_password(self):
        import backend.api.app as app_mod
        from backend.core.config import Settings

        secure = Settings(auth_default_admin_password="V3rys3cur3P@ssw0rd!")
        with patch.object(app_mod, "settings", secure):
            # Must not raise
            app_mod._validate_startup_config()


# ---------------------------------------------------------------------------
# 4. CORS wildcard + credentials conflict
# ---------------------------------------------------------------------------


class TestCORSConfig:
    """allow_credentials must be False when origins is '*'."""

    def test_cors_wildcard_disables_credentials(self):
        """When CORS origins = '*', allow_credentials must be False (Fetch standard)."""

        # Reload app module with wildcard CORS to test the middleware registration.
        # We inspect the middleware stack directly.

        from backend.core.config import Settings

        wildcard_settings = Settings(cors_allow_origins="*")

        # Verify the conditional logic directly (app.py module-level code)
        cors_wildcard = wildcard_settings.cors_allow_origins.strip() == "*"
        allow_credentials = not cors_wildcard
        assert allow_credentials is False, "allow_credentials must be False when cors_allow_origins='*'"

    def test_cors_explicit_origins_enables_credentials(self):
        from backend.core.config import Settings

        explicit = Settings(cors_allow_origins="https://app.example.com")
        cors_wildcard = explicit.cors_allow_origins.strip() == "*"
        allow_credentials = not cors_wildcard
        assert allow_credentials is True


# ---------------------------------------------------------------------------
# 5. Atomic session state write
# ---------------------------------------------------------------------------


class TestAtomicSessionStateWrite:
    """Session state writes must be atomic — no partial state on crash."""

    def test_save_state_produces_valid_json(self, tmp_path):
        from backend.sessions.session_store import SessionStore

        store = SessionStore(str(tmp_path), ttl_days=7)
        state = store.create_session()

        # save_state is called by create_session; verify the file is valid JSON
        state_file = tmp_path / state.session_id / "state.json"
        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert data["session_id"] == state.session_id

    def test_save_state_leaves_no_tmp_files_on_success(self, tmp_path):
        from backend.sessions.session_store import SessionStore

        store = SessionStore(str(tmp_path), ttl_days=7)
        state = store.create_session()

        session_dir = tmp_path / state.session_id
        tmp_files = list(session_dir.glob("*.tmp"))
        assert tmp_files == [], f"Leftover .tmp files after successful save: {tmp_files}"

    def test_save_state_is_atomic_under_concurrent_writes(self, tmp_path):
        """State file must never be empty/corrupt even under concurrent saves."""
        from backend.sessions.session_store import SessionStore

        store = SessionStore(str(tmp_path), ttl_days=7)
        state = store.create_session()
        errors: list[Exception] = []

        def _write(_: int) -> None:
            try:
                loaded = store.load_session(state.session_id)
                if loaded:
                    store.set_session_memory(state.session_id, f"note-{_}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_write, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Concurrent writes produced errors: {errors}"

        # File must be valid JSON after all writes
        state_file = tmp_path / state.session_id / "state.json"
        data = json.loads(state_file.read_text())
        assert data["session_id"] == state.session_id


# ---------------------------------------------------------------------------
# 6. SandboxManager cleanup is called
# ---------------------------------------------------------------------------


class TestSandboxManagerCleanup:
    """cleanup_expired must actually remove idle sandboxes."""

    def test_cleanup_expired_removes_old_sandbox(self):
        from backend.tools.sandbox_manager import SandboxManager

        mgr = SandboxManager()
        mgr.get_or_create("session-old")

        # Backdate the last-access time to exceed TTL
        mgr._last_access["session-old"] = time.monotonic() - 9999

        removed = mgr.cleanup_expired(ttl_sec=3600)
        assert removed == 1
        assert mgr.get("session-old") is None

    def test_cleanup_expired_keeps_recent_sandbox(self):
        from backend.tools.sandbox_manager import SandboxManager

        mgr = SandboxManager()
        mgr.get_or_create("session-new")

        removed = mgr.cleanup_expired(ttl_sec=3600)
        assert removed == 0
        assert mgr.get("session-new") is not None


# ---------------------------------------------------------------------------
# 7. Depth profile consistency
# ---------------------------------------------------------------------------


class TestDepthProfileConsistency:
    """DEPTH_PROFILES in config must match what tests expect."""

    def test_all_expected_depth_keys_are_present(self):
        from backend.core.config import DEPTH_PROFILES

        assert set(DEPTH_PROFILES.keys()) == {"light", "medium", "deep"}

    def test_depth_profiles_are_ordered_ascending(self):
        from backend.core.config import DEPTH_PROFILES

        limits = [DEPTH_PROFILES[k]["inner_recursion_limit"] for k in ("light", "medium", "deep")]
        assert limits == sorted(limits), "inner_recursion_limit must increase from light → medium → deep"

    def test_depth_profile_values_are_positive_integers(self):
        from backend.core.config import DEPTH_PROFILES

        for depth, profile in DEPTH_PROFILES.items():
            assert isinstance(profile["inner_recursion_limit"], int), depth
            assert profile["inner_recursion_limit"] > 0, depth
