from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import MethodType
from unittest.mock import patch

from backend.agent import runtime_llm
from backend.api.models import UserSettingsResponse, UserSettingsUpdateRequest
from backend.api.services.query_execution import QueryExecutionService
from backend.auth import AuthDB
from backend.auth.user_settings_defaults import (
    UserSettingsDefaults,
    user_settings_defaults_from_runtime,
)
from backend.core.config import Settings


class UserSettingsDefaultsContractTests(unittest.TestCase):
    def test_new_user_settings_use_injected_runtime_defaults(self) -> None:
        tmpdir = tempfile.mkdtemp()
        auth_db = None
        defaults = UserSettingsDefaults(
            analysis_depth="medium",
            llm_temperature_chat=0.2,
            llm_temperature_tool=0.1,
            llm_max_tokens_default=1234,
            llm_max_tokens_reasoning=2345,
            backend_query_timeout_sec=99,
            agent_max_steps=17,
            agent_step_timeout_sec=33,
            agent_inner_recursion_limit=18,
            llm_streaming=False,
        )
        try:
            auth_db = AuthDB(
                str(Path(tmpdir) / "app.db"),
                token_ttl_days=30,
                user_settings_defaults=defaults,
            )
            user = auth_db.create_user("defaults_user", "secret", is_admin=False)

            settings = auth_db.get_user_settings(user.id)

            self.assertEqual(settings.analysis_depth, "medium")
            self.assertEqual(settings.llm_temperature_chat, 0.2)
            self.assertEqual(settings.llm_temperature_tool, 0.1)
            self.assertEqual(settings.llm_max_tokens_default, 1234)
            self.assertEqual(settings.llm_max_tokens_reasoning, 2345)
            self.assertEqual(settings.backend_query_timeout_sec, 99)
            self.assertEqual(settings.agent_max_steps, 17)
            self.assertEqual(settings.agent_step_timeout_sec, 33)
            self.assertEqual(settings.agent_inner_recursion_limit, 18)
            self.assertFalse(settings.llm_streaming)
        finally:
            if auth_db is not None:
                del auth_db
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_user_settings_preserve_zero_temperatures(self) -> None:
        tmpdir = tempfile.mkdtemp()
        auth_db = None
        try:
            auth_db = AuthDB(str(Path(tmpdir) / "app.db"), token_ttl_days=30)
            user = auth_db.create_user("zero_temperature", "secret", is_admin=False)

            auth_db.update_user_settings(
                user.id,
                llm_temperature_chat=0.0,
                llm_temperature_tool=0.0,
            )
            settings = auth_db.get_user_settings(user.id)

            self.assertEqual(settings.llm_temperature_chat, 0.0)
            self.assertEqual(settings.llm_temperature_tool, 0.0)
        finally:
            if auth_db is not None:
                del auth_db
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_default_admin_creation_ignores_concurrent_insert(self) -> None:
        tmpdir = tempfile.mkdtemp()
        auth_db = None
        try:
            auth_db = AuthDB(str(Path(tmpdir) / "app.db"), token_ttl_days=30)
            auth_db.create_user("admin_race", "secret", is_admin=True)
            real_lookup = auth_db.get_user_by_username
            calls = {"count": 0}

            def racy_lookup(self: AuthDB, username: str):
                calls["count"] += 1
                if calls["count"] == 1:
                    return None
                return real_lookup(username)

            auth_db.get_user_by_username = MethodType(racy_lookup, auth_db)

            auth_db.ensure_default_admin("admin_race", "secret")

            self.assertEqual(calls["count"], 2)
        finally:
            if auth_db is not None:
                del auth_db
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_runtime_defaults_are_capped_by_analysis_depth(self) -> None:
        defaults = user_settings_defaults_from_runtime(
            Settings(
                agent_analysis_depth="light",
                agent_max_steps=80,
                agent_inner_recursion_limit=56,
            )
        )

        self.assertEqual(defaults.agent_max_steps, 32)
        self.assertEqual(defaults.agent_inner_recursion_limit, 32)

    def test_settings_code_defaults_follow_light_profile(self) -> None:
        settings = Settings()

        self.assertEqual(settings.agent_max_steps, 32)
        self.assertEqual(settings.agent_inner_recursion_limit, 32)

    def test_settings_include_max_tools_per_cycle_default(self) -> None:
        settings = Settings()

        self.assertGreaterEqual(settings.max_tools_per_cycle, 2)

    def test_runtime_llm_respects_user_streaming_setting(self) -> None:
        captured: dict[str, object] = {}

        def fake_make_reasoning_llm(**kwargs: object) -> object:
            captured.update(kwargs)
            return object()

        with patch(
            "backend.agent.runtime_llm.make_reasoning_llm",
            side_effect=fake_make_reasoning_llm,
        ):
            runtime_llm.build_runtime_llm(
                Settings(llm_streaming=False),
                role="chat",
                include_reasoning=False,
            )

        self.assertFalse(captured["streaming"])

    def test_runtime_llm_uses_tool_temperature_with_thinking_enabled(self) -> None:
        captured: dict[str, object] = {}

        def fake_make_reasoning_llm(**kwargs: object) -> object:
            captured.update(kwargs)
            return object()

        with patch(
            "backend.agent.runtime_llm.make_reasoning_llm",
            side_effect=fake_make_reasoning_llm,
        ):
            runtime_llm.build_runtime_llm(
                Settings(
                    llm_enable_thinking=True,
                    llm_temperature_tool=0.0,
                ),
                role="tool",
                include_reasoning=True,
            )

        self.assertTrue(captured["enable_thinking"])
        self.assertEqual(captured["temperature"], 0.0)

    def test_runtime_llm_keeps_chat_thinking_temperature_at_one(self) -> None:
        captured: dict[str, object] = {}

        def fake_make_reasoning_llm(**kwargs: object) -> object:
            captured.update(kwargs)
            return object()

        with patch(
            "backend.agent.runtime_llm.make_reasoning_llm",
            side_effect=fake_make_reasoning_llm,
        ):
            runtime_llm.build_runtime_llm(
                Settings(
                    llm_enable_thinking=True,
                    llm_temperature_chat=0.0,
                ),
                role="chat",
                include_reasoning=True,
            )

        self.assertEqual(captured["temperature"], 1.0)

    def test_react_setting_round_trips_through_api_models(self) -> None:
        response = UserSettingsResponse(agent_react_enabled=True)
        update = UserSettingsUpdateRequest(agent_react_enabled=True)

        self.assertTrue(response.agent_react_enabled)
        self.assertTrue(update.agent_react_enabled)

    def test_runtime_settings_respect_user_numeric_settings(self) -> None:
        deps = type(
            "Deps",
            (),
            {
                "settings": Settings(
                    agent_max_steps=80,
                    agent_inner_recursion_limit=56,
                    backend_query_timeout_sec=180,
                ),
                "auth_db": type("Auth", (), {})(),
            },
        )()
        service = QueryExecutionService.model_construct(dependencies=deps)
        user_settings = UserSettingsDefaults(
            analysis_mode="fast",
            analysis_depth="light",
            llm_temperature_chat=0.11,
            llm_temperature_tool=0.12,
            llm_max_tokens_default=777,
            llm_max_tokens_reasoning=888,
            backend_query_timeout_sec=44,
            agent_max_steps=7,
            agent_step_timeout_sec=22,
            agent_inner_recursion_limit=9,
            llm_streaming=False,
            always_use_analysis_plan=True,
        ).to_user_settings()
        service.dependencies.auth_db.get_user_settings = MethodType(
            lambda _self, _uid: user_settings,
            deps.auth_db,
        )

        runtime = service._effective_runtime_settings(1)

        self.assertEqual(runtime.agent_max_steps, 7)
        self.assertEqual(runtime.agent_inner_recursion_limit, 9)
        self.assertEqual(runtime.llm_max_tokens_default, 777)
        self.assertEqual(runtime.backend_query_timeout_sec, 44)
        self.assertFalse(runtime.llm_streaming)
        self.assertTrue(runtime.always_use_analysis_plan)
