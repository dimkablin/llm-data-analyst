from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from backend.api.deps import get_current_user
from backend.api.models import (
    DBConnectionCreateRequest,
    DBConnectionResponse,
    DBConnectionSchemaResponse,
    DBConnectionTableResponse,
    DBConnectionTestResponse,
    DBConnectionUpdateRequest,
    MessageResponse,
)
from backend.auth.auth_db import AuthDB, AuthUser

router = APIRouter(tags=["Подключения к БД"])

# Singletons set during app startup
_auth_db: AuthDB = None  # type: ignore
_db_connections_service = None  # type: ignore
_db_runtime_service = None  # type: ignore


def setup(auth_db: AuthDB, db_connections_service, db_runtime_service) -> None:
    global _auth_db, _db_connections_service, _db_runtime_service
    _auth_db = auth_db
    _db_connections_service = db_connections_service
    _db_runtime_service = db_runtime_service


def _to_db_connection_response(connection) -> DBConnectionResponse:
    return DBConnectionResponse(
        id=connection.id,
        name=connection.name,
        db_type=connection.db_type,
        host=connection.host,
        port=connection.port,
        database=connection.database,
        username=connection.username,
        options_json=connection.options_json,
        password_present=connection.password_present,
        last_test_at=connection.last_test_at,
        last_test_ok=connection.last_test_ok,
        last_error=connection.last_error,
        created_at=connection.created_at,
        updated_at=connection.updated_at,
    )


@router.get("/db-connections", response_model=list[DBConnectionResponse])
def list_db_connections(
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> list[DBConnectionResponse]:
    rows = _db_connections_service.list_connections(current_user.id)
    return [_to_db_connection_response(row) for row in rows]


@router.post("/db-connections", response_model=DBConnectionResponse, status_code=201)
def create_db_connection(
    payload: DBConnectionCreateRequest,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> DBConnectionResponse:
    created = _db_connections_service.create_connection(
        current_user.id,
        name=payload.name,
        db_type=payload.db_type,
        host=payload.host,
        port=payload.port,
        database=payload.database,
        username=payload.username,
        password=payload.password,
        options_json=payload.options_json,
    )
    return _to_db_connection_response(created)


@router.get("/db-connections/{connection_id}", response_model=DBConnectionResponse)
def get_db_connection(
    connection_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> DBConnectionResponse:
    row = _db_connections_service.get_connection(current_user.id, connection_id)
    return _to_db_connection_response(row)


@router.patch("/db-connections/{connection_id}", response_model=DBConnectionResponse)
def update_db_connection(
    connection_id: str,
    payload: DBConnectionUpdateRequest,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> DBConnectionResponse:
    updated = _db_connections_service.update_connection(
        current_user.id,
        connection_id,
        name=payload.name,
        db_type=payload.db_type,
        host=payload.host,
        port=payload.port,
        database=payload.database,
        username=payload.username,
        password=payload.password,
        clear_password=payload.clear_password,
        options_json=payload.options_json,
        options_json_set="options_json" in payload.model_fields_set,
    )
    return _to_db_connection_response(updated)


@router.delete("/db-connections/{connection_id}", response_model=MessageResponse)
def delete_db_connection(
    connection_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> MessageResponse:
    _db_connections_service.delete_connection(current_user.id, connection_id)
    return MessageResponse(message="DB connection deleted")


@router.post("/db-connections/{connection_id}/test", response_model=DBConnectionTestResponse)
def test_db_connection(
    connection_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> DBConnectionTestResponse:
    tested = _db_connections_service.test_connection(current_user.id, connection_id)
    if tested.last_test_at is None or tested.last_test_ok is None:
        raise HTTPException(
            status_code=500,
            detail="Connection test state was not persisted.",
        )
    return DBConnectionTestResponse(
        ok=tested.last_test_ok,
        checked_at=tested.last_test_at,
        last_test_at=tested.last_test_at,
        last_test_ok=tested.last_test_ok,
        error=tested.last_error,
    )


@router.get("/db-connections/{connection_id}/schemas", response_model=list[DBConnectionSchemaResponse])
def list_db_connection_schemas(
    connection_id: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> list[DBConnectionSchemaResponse]:
    try:
        items = _db_runtime_service.list_schemas(
            user_id=current_user.id,
            connection_id=connection_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return [
        DBConnectionSchemaResponse(
            name=item.name,
            display_name=item.display_name,
        )
        for item in items
    ]


@router.get("/db-connections/{connection_id}/tables", response_model=list[DBConnectionTableResponse])
def list_db_connection_tables(
    connection_id: str,
    schema: str,
    current_user: Annotated[AuthUser, Depends(get_current_user)],
) -> list[DBConnectionTableResponse]:
    try:
        items = _db_runtime_service.list_tables(
            user_id=current_user.id,
            connection_id=connection_id,
            schema=schema,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return [
        DBConnectionTableResponse(
            schema=item.schema,
            name=item.name,
            table_type=item.table_type,
            qualified_name=item.qualified_name,
        )
        for item in items
    ]


