from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.api.models import UserSettingsResponse, UserSettingsUpdateRequest
from backend.auth import AuthDB


class UserThinkSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = str(Path(self.tmpdir) / "app.db")
        self.auth_db = AuthDB(self.db_path, token_ttl_days=30)
        self.user = self.auth_db.create_user("alice_think", "secret", is_admin=False)

    def tearDown(self) -> None:
        del self.auth_db
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_new_settings_have_correct_defaults(self) -> None:
        settings = self.auth_db.get_user_settings(self.user.id)
        self.assertTrue(settings.llm_streaming)
        self.assertTrue(settings.show_thinking)
        self.assertTrue(settings.show_think_planning)
        self.assertTrue(settings.show_think_tool)
        self.assertTrue(settings.show_think_final)
        self.assertTrue(settings.show_rag_errors)
        self.assertFalse(settings.anomaly_check_enabled)
        self.assertFalse(settings.always_use_analysis_plan)

    def test_update_and_persist_think_settings(self) -> None:
        updated = self.auth_db.update_user_settings(
            self.user.id,
            llm_streaming=False,
            show_thinking=True,
            show_think_planning=True,
            show_think_tool=False,
            show_think_final=False,
        )
        self.assertFalse(updated.llm_streaming)
        self.assertTrue(updated.show_thinking)
        self.assertTrue(updated.show_think_planning)
        self.assertFalse(updated.show_think_tool)
        self.assertFalse(updated.show_think_final)

        reloaded = self.auth_db.get_user_settings(self.user.id)
        self.assertFalse(reloaded.llm_streaming)
        self.assertFalse(reloaded.show_think_tool)

    def test_update_and_persist_rag_error_visibility(self) -> None:
        updated = self.auth_db.update_user_settings(self.user.id, show_rag_errors=False)
        self.assertFalse(updated.show_rag_errors)
        self.assertFalse(self.auth_db.get_user_settings(self.user.id).show_rag_errors)

    def test_update_and_persist_anomaly_check(self) -> None:
        updated = self.auth_db.update_user_settings(
            self.user.id,
            anomaly_check_enabled=True,
        )
        self.assertTrue(updated.anomaly_check_enabled)
        self.assertTrue(self.auth_db.get_user_settings(self.user.id).anomaly_check_enabled)

    def test_update_and_persist_always_use_analysis_plan(self) -> None:
        updated = self.auth_db.update_user_settings(
            self.user.id,
            always_use_analysis_plan=True,
        )

        self.assertTrue(updated.always_use_analysis_plan)
        self.assertTrue(self.auth_db.get_user_settings(self.user.id).always_use_analysis_plan)

    def test_anomaly_column_migration_tolerates_concurrent_worker(self) -> None:
        with self.auth_db._connect() as conn:
            columns = [
                {"name": row["name"]}
                for row in conn.execute("PRAGMA table_info(user_settings)").fetchall()
                if row["name"] != "anomaly_check_enabled"
            ]

        class ConcurrentConnection:
            def execute(self, sql: str):
                if sql.startswith("PRAGMA"):
                    return self
                if "ADD COLUMN anomaly_check_enabled" not in sql:
                    return self
                raise sqlite3.OperationalError("duplicate column name: anomaly_check_enabled")

            def fetchall(self):
                return columns

        AuthDB._ensure_user_settings_columns(ConcurrentConnection())

    def test_partial_update_preserves_other_fields(self) -> None:
        self.auth_db.update_user_settings(self.user.id, show_think_final=False)
        settings = self.auth_db.get_user_settings(self.user.id)
        self.assertTrue(settings.llm_streaming)
        self.assertTrue(settings.show_thinking)
        self.assertFalse(settings.show_think_final)

    def test_analysis_mode_uses_deep_contract_with_demo_legacy_alias(self) -> None:
        updated = self.auth_db.update_user_settings(self.user.id, analysis_mode="demo")

        self.assertEqual(updated.analysis_mode, "deep")
        self.assertEqual(updated.analysis_depth, "deep")

        reloaded = self.auth_db.get_user_settings(self.user.id)
        self.assertEqual(reloaded.analysis_mode, "deep")
        self.assertEqual(reloaded.analysis_depth, "deep")


def test_user_settings_api_exposes_deep_mode_and_accepts_demo_alias_input() -> None:
    UserSettingsResponse(analysis_mode="deep")
    UserSettingsUpdateRequest(analysis_mode="demo")
    assert UserSettingsResponse(agent_react_enabled=True).agent_react_enabled is True
    assert UserSettingsResponse().show_rag_errors is True
    assert UserSettingsResponse().anomaly_check_enabled is False
    assert UserSettingsResponse().always_use_analysis_plan is False
    assert (
        UserSettingsUpdateRequest(agent_react_enabled=True).agent_react_enabled is True
    )
    assert UserSettingsUpdateRequest(show_rag_errors=False).show_rag_errors is False
    assert UserSettingsUpdateRequest(always_use_analysis_plan=True).always_use_analysis_plan is True

    with pytest.raises(ValidationError):
        UserSettingsResponse(analysis_mode="demo")
