from __future__ import annotations

from backend.api.routes import admin_skills as admin_skills_route
from backend.api.routes import skills as skills_route
from backend.auth.auth_db import AuthUser
from backend.skills.models import Skill


class _FakeAuthDB:
    def __init__(self, user_settings: dict[str, bool] | None = None) -> None:
        self._user_settings = user_settings or {}

    def list_user_skill_settings(self, user_id: int) -> dict[str, bool]:
        assert user_id == 7
        return dict(self._user_settings)

    def set_user_skill_enabled(self, user_id: int, skill_id: str, enabled: bool) -> None:
        assert user_id == 7
        self._user_settings[skill_id] = enabled


class _FakeSkillRegistry:
    def __init__(self, skills: list[Skill]) -> None:
        self._skills = {skill.skill_id: skill for skill in skills}

    def list_skills(self) -> list[Skill]:
        return list(self._skills.values())

    def get(self, skill_id: str) -> Skill:
        return self._skills[skill_id]


class _FakeOverrideStore:
    def get_override(self, skill_id: str) -> None:
        return None


def _disabled_skill() -> Skill:
    return Skill(
        skill_id="disabled_by_default",
        name="Disabled By Default",
        description="Contract test skill.",
        core_markdown="## Contract",
        details_markdown=None,
        source_path="skills/disabled_by_default/SKILL.md",
        kind="analytical",
        enabled_by_default=False,
    )


def _user() -> AuthUser:
    return AuthUser(id=7, username="analyst", is_admin=True, created_at="now")


def test_skills_route_uses_skill_enabled_by_default_when_user_has_no_override() -> None:
    skills_route.setup(
        skill_registry=_FakeSkillRegistry([_disabled_skill()]),  # type: ignore[arg-type]
        auth_db=_FakeAuthDB(),
    )

    [payload] = skills_route.list_skills(_user())

    assert payload.skill_id == "disabled_by_default"
    assert payload.enabled_by_default is False
    assert payload.enabled_for_user is False


def test_skills_route_user_override_wins_over_skill_default() -> None:
    skills_route.setup(
        skill_registry=_FakeSkillRegistry([_disabled_skill()]),  # type: ignore[arg-type]
        auth_db=_FakeAuthDB({"disabled_by_default": True}),
    )

    [payload] = skills_route.list_skills(_user())

    assert payload.enabled_by_default is False
    assert payload.enabled_for_user is True


def test_admin_skills_route_exposes_enabled_by_default() -> None:
    admin_skills_route.setup(
        skill_registry=_FakeSkillRegistry([_disabled_skill()]),  # type: ignore[arg-type]
        auth_db=_FakeAuthDB(),  # type: ignore[arg-type]
        override_store=_FakeOverrideStore(),  # type: ignore[arg-type]
    )

    [payload] = admin_skills_route.admin_list_skills(_user())

    assert payload.skill_id == "disabled_by_default"
    assert payload.enabled_by_default is False
