from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from backend.api.deps import get_current_user
from backend.api.models import SkillResponse
from backend.auth.auth_db import AuthUser

router = APIRouter(tags=["Навыки"])

_runner = None  # type: ignore


def setup(runner) -> None:
    global _runner
    _runner = runner


@router.get("/skills", response_model=list[SkillResponse])
def list_skills(current_user: Annotated[AuthUser, Depends(get_current_user)]) -> list[SkillResponse]:
    _ = current_user
    return [
        SkillResponse(
            skill_id=skill.skill_id,
            name=skill.name,
            description=skill.description,
            triggers=list(skill.triggers),
            source_path=skill.source_path,
        )
        for skill in _runner.skill_registry.list_skills()
    ]
