from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.api.deps import get_current_user
from backend.api.models import SkillResponse
from backend.auth.auth_db import AuthDB, AuthUser
from backend.skills.registry import SkillRegistry

router = APIRouter(tags=["Навыки"])

_skill_registry: SkillRegistry = None  # type: ignore
_auth_db: AuthDB = None  # type: ignore


def setup(skill_registry: SkillRegistry, auth_db: AuthDB) -> None:
    global _skill_registry, _auth_db
    _skill_registry = skill_registry
    _auth_db = auth_db


class SkillEnabledUpdateRequest(BaseModel):
    enabled: bool


@router.get("/skills", response_model=list[SkillResponse])
def list_skills(current_user: Annotated[AuthUser, Depends(get_current_user)]) -> list[SkillResponse]:
    user_settings = _auth_db.list_user_skill_settings(current_user.id)
    return [
        SkillResponse(
            skill_id=skill.skill_id,
            name=skill.name,
            description=skill.description,
            triggers=list(skill.triggers),
            source_path=skill.source_path,
            enabled_by_default=skill.enabled_by_default,
            enabled_for_user=user_settings.get(skill.skill_id, skill.enabled_by_default),
        )
        for skill in _skill_registry.list_skills()
        if skill.kind == "analytical"
    ]


@router.patch("/skills/{skill_id}", response_model=SkillResponse)
def update_skill_enabled(
    skill_id: str,
    payload: SkillEnabledUpdateRequest,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> SkillResponse:
    try:
        skill = _skill_registry.get(skill_id)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_id}' not found") from None
    if skill.kind != "analytical":
        raise HTTPException(status_code=404, detail=f"Skill '{skill_id}' not found")

    _auth_db.set_user_skill_enabled(current_user.id, skill_id, payload.enabled)

    return SkillResponse(
        skill_id=skill.skill_id,
        name=skill.name,
        description=skill.description,
        triggers=list(skill.triggers),
        source_path=skill.source_path,
        enabled_by_default=skill.enabled_by_default,
        enabled_for_user=payload.enabled,
    )
