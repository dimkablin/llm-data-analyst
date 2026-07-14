from __future__ import annotations

import io
import logging
import re
import zipfile
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from backend.api.deps import get_admin_user
from backend.api.models import AdminSkillResponse, AdminSkillUpdateRequest, MessageResponse
from backend.auth.auth_db import AuthDB, AuthUser
from backend.skills.override_store import SkillOverrideStore
from backend.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Администрирование"])

_skill_registry: SkillRegistry = None  # type: ignore
_auth_db: AuthDB = None
_override_store: SkillOverrideStore = None


def setup(
    skill_registry: SkillRegistry,
    auth_db: AuthDB,
    override_store: SkillOverrideStore,
) -> None:
    global _skill_registry, _auth_db, _override_store
    _skill_registry = skill_registry
    _auth_db = auth_db
    _override_store = override_store


def _skill_to_admin_response(skill) -> AdminSkillResponse:
    overridden = skill.metadata.get("overridden", False) if hasattr(skill, "metadata") else False
    override = _override_store.get_override(skill.skill_id) if overridden else None
    return AdminSkillResponse(
        skill_id=skill.skill_id,
        name=skill.name,
        description=skill.description,
        triggers=list(skill.triggers),
        source_path=skill.source_path,
        enabled_by_default=skill.enabled_by_default,
        kind=skill.kind,
        tool_key=skill.tool_key,
        core_markdown=skill.core_markdown,
        details_markdown=skill.details_markdown,
        is_overridden=overridden,
        updated_by=override.updated_by if override else None,
        updated_at=override.updated_at if override else None,
    )


@router.get("/admin/skills", response_model=list[AdminSkillResponse])
def admin_list_skills(
    _: Annotated[AuthUser, Depends(get_admin_user)],
) -> list[AdminSkillResponse]:
    return [
        _skill_to_admin_response(skill)
        for skill in _skill_registry.list_skills()
    ]


@router.get("/admin/skills/{skill_id}", response_model=AdminSkillResponse)
def admin_get_skill(
    skill_id: str,
    _: Annotated[AuthUser, Depends(get_admin_user)],
) -> AdminSkillResponse:
    try:
        skill = _skill_registry.get(skill_id)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_id}' not found") from None
    return _skill_to_admin_response(skill)


@router.put("/admin/skills/{skill_id}", response_model=AdminSkillResponse)
def admin_update_skill(
    skill_id: str,
    payload: AdminSkillUpdateRequest,
    current_admin: Annotated[AuthUser, Depends(get_admin_user)],
) -> AdminSkillResponse:
    try:
        skill = _skill_registry.get(skill_id)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_id}' not found") from None

    core = payload.core_markdown if payload.core_markdown is not None else skill.core_markdown
    _validate_skill_markdown(skill_id, skill.kind, core)

    _override_store.save_override(
        skill_id,
        name=payload.name,
        description=payload.description,
        triggers=list(payload.triggers) if payload.triggers is not None else None,
        core_markdown=payload.core_markdown,
        details_markdown=payload.details_markdown,
        user_id=current_admin.id,
    )

    updated = _skill_registry.reload_skill(skill_id)
    if updated is None:
        raise HTTPException(status_code=500, detail="Failed to reload skill after update")
    return _skill_to_admin_response(updated)


@router.delete("/admin/skills/{skill_id}/override", response_model=AdminSkillResponse)
def admin_delete_skill_override(
    skill_id: str,
    _: Annotated[AuthUser, Depends(get_admin_user)],
) -> AdminSkillResponse:
    try:
        _skill_registry.get(skill_id)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_id}' not found") from None

    deleted = _override_store.delete_override(skill_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="No override found for this skill")

    updated = _skill_registry.reload_skill(skill_id)
    if updated is None:
        raise HTTPException(status_code=500, detail="Failed to reload skill after reset")
    return _skill_to_admin_response(updated)


@router.post("/admin/skills/reload", response_model=MessageResponse)
def admin_reload_skills(
    _: Annotated[AuthUser, Depends(get_admin_user)],
) -> MessageResponse:
    count_before = len(_skill_registry.list_skills())
    _skill_registry.reload()
    count_after = len(_skill_registry.list_skills())
    logger.info("Admin reloaded skills: %d before → %d after", count_before, count_after)
    return MessageResponse(message=f"Skills reloaded: {count_before} → {count_after}")


@router.get("/admin/skills/export/zip")
def admin_export_skills_zip(
    _: Annotated[AuthUser, Depends(get_admin_user)],
) -> Response:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for skill in _skill_registry.list_skills():
            safe_dir = f"skills/{skill.skill_id}"
            zf.writestr(f"{safe_dir}/SKILL.md", skill.core_markdown)
            if skill.details_markdown:
                zf.writestr(f"{safe_dir}/DETAILS.md", skill.details_markdown)

    buf.seek(0)
    return Response(
        content=buf.read(),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=skills-export.zip"},
    )


def _validate_skill_markdown(skill_id: str, kind: str, core_markdown: str) -> None:
    if kind == "tool":
        if "### API" not in core_markdown:
            raise HTTPException(
                status_code=422,
                detail=f"Skill '{skill_id}': tool skill must contain '### API' section",
            )
        if "### Final result protocol" not in core_markdown:
            raise HTTPException(
                status_code=422,
                detail=f"Skill '{skill_id}': tool skill must contain '### Final result protocol' section",
            )
    else:
        if not re.search(r"^### (Algorithm|Алгоритм)", core_markdown, re.MULTILINE):
            raise HTTPException(
                status_code=422,
                detail=f"Skill '{skill_id}': analytical skill must contain '### Algorithm' section",
            )
        if not re.search(r"^### (Rules|Правила)", core_markdown, re.MULTILINE):
            raise HTTPException(
                status_code=422,
                detail=f"Skill '{skill_id}': analytical skill must contain '### Rules' section",
            )

    if len(core_markdown.encode("utf-8")) > 8 * 1024:
        raise HTTPException(
            status_code=422,
            detail=f"Skill '{skill_id}': core markdown exceeds 8KB limit",
        )
