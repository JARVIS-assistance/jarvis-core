from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from ..db_connection import DBClient
from .common import now_iso

TODO_COLUMNS = (
    "id",
    "user_id",
    "title",
    "description",
    "status",
    "priority",
    "due_at",
    "remind_at",
    "timezone",
    "calendar_provider",
    "calendar_id",
    "calendar_event_id",
    "calendar_sync_status",
    "chat_id",
    "source_message_id",
    "metadata",
    "created_at",
    "updated_at",
    "completed_at",
)

MUTABLE_COLUMNS = {
    "title",
    "description",
    "status",
    "priority",
    "due_at",
    "remind_at",
    "timezone",
    "calendar_provider",
    "calendar_id",
    "calendar_event_id",
    "calendar_sync_status",
    "metadata",
}


def _placeholder(db: DBClient) -> str:
    return "%s" if db.backend == "postgres" else "?"


def _json_dumps(value: Any) -> str:
    return json.dumps(value if isinstance(value, dict) else {}, ensure_ascii=False)


def _decode_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _row_to_todo(row: Any) -> dict[str, Any]:
    return {
        "id": str(row[0]),
        "user_id": str(row[1]),
        "title": row[2],
        "description": row[3],
        "status": row[4],
        "priority": row[5],
        "due_at": row[6],
        "remind_at": row[7],
        "timezone": row[8],
        "calendar_provider": row[9],
        "calendar_id": row[10],
        "calendar_event_id": row[11],
        "calendar_sync_status": row[12],
        "chat_id": str(row[13]) if row[13] is not None else None,
        "source_message_id": str(row[14]) if row[14] is not None else None,
        "metadata": _decode_metadata(row[15]),
        "created_at": row[16],
        "updated_at": row[17],
        "completed_at": row[18],
    }


def _select_by_id(db: DBClient, *, user_id: str, todo_id: str) -> dict[str, Any] | None:
    marker = _placeholder(db)
    cursor = db.conn.execute(
        f"""
        SELECT {", ".join(TODO_COLUMNS)}
        FROM todo_items
        WHERE id = {marker} AND user_id = {marker} AND deleted_at IS NULL
        LIMIT 1
        """,
        (todo_id, user_id),
    )
    row = cursor.fetchone()
    return _row_to_todo(row) if row is not None else None


def create_todo_item(
    db: DBClient,
    *,
    user_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    todo_id = str(uuid4())
    now = now_iso()
    metadata = _json_dumps(payload.get("metadata"))
    calendar_sync_status = "linked" if payload.get("calendar_event_id") else "none"

    values = {
        "id": todo_id,
        "user_id": user_id,
        "title": payload["title"],
        "description": payload.get("description"),
        "status": "open",
        "priority": payload.get("priority", 3),
        "due_at": payload.get("due_at"),
        "remind_at": payload.get("remind_at"),
        "timezone": payload.get("timezone"),
        "calendar_provider": payload.get("calendar_provider"),
        "calendar_id": payload.get("calendar_id"),
        "calendar_event_id": payload.get("calendar_event_id"),
        "calendar_sync_status": calendar_sync_status,
        "chat_id": payload.get("chat_id"),
        "source_message_id": payload.get("source_message_id"),
        "metadata": metadata,
        "created_at": now,
        "updated_at": now,
        "completed_at": None,
    }
    marker = _placeholder(db)
    metadata_expr = f"{marker}::jsonb" if db.backend == "postgres" else marker
    placeholders = [marker] * len(TODO_COLUMNS)
    placeholders[TODO_COLUMNS.index("metadata")] = metadata_expr

    try:
        db.conn.execute(
            f"""
            INSERT INTO todo_items ({", ".join(TODO_COLUMNS)})
            VALUES ({", ".join(placeholders)})
            """,
            tuple(values[column] for column in TODO_COLUMNS),
        )
        db.conn.commit()
    except Exception:
        db.conn.rollback()
        raise

    created = _select_by_id(db, user_id=user_id, todo_id=todo_id)
    if created is None:
        raise RuntimeError("created todo item could not be read")
    return created


def list_todo_items(
    db: DBClient,
    *,
    user_id: str,
    status: str | None = None,
    include_deleted: bool = False,
    limit: int = 50,
) -> list[dict[str, Any]]:
    marker = _placeholder(db)
    filters = [f"user_id = {marker}"]
    params: list[Any] = [user_id]
    if status:
        filters.append(f"status = {marker}")
        params.append(status)
    if not include_deleted:
        filters.append("deleted_at IS NULL")
    params.append(max(1, min(limit, 200)))
    cursor = db.conn.execute(
        f"""
        SELECT {", ".join(TODO_COLUMNS)}
        FROM todo_items
        WHERE {" AND ".join(filters)}
        ORDER BY
          CASE WHEN due_at IS NULL THEN 1 ELSE 0 END,
          due_at ASC,
          updated_at DESC
        LIMIT {marker}
        """,
        tuple(params),
    )
    return [_row_to_todo(row) for row in cursor.fetchall()]


def get_todo_item(
    db: DBClient,
    *,
    user_id: str,
    todo_id: str,
) -> dict[str, Any] | None:
    return _select_by_id(db, user_id=user_id, todo_id=todo_id)


def update_todo_item(
    db: DBClient,
    *,
    user_id: str,
    todo_id: str,
    updates: dict[str, Any],
) -> dict[str, Any] | None:
    filtered = {key: value for key, value in updates.items() if key in MUTABLE_COLUMNS}
    if not filtered:
        return _select_by_id(db, user_id=user_id, todo_id=todo_id)

    now = now_iso()
    if "metadata" in filtered:
        filtered["metadata"] = _json_dumps(filtered["metadata"])
    if filtered.get("status") == "completed":
        filtered["completed_at"] = now
    elif "status" in filtered and filtered["status"] != "completed":
        filtered["completed_at"] = None
    filtered["updated_at"] = now

    marker = _placeholder(db)
    assignments: list[str] = []
    params: list[Any] = []
    for column, value in filtered.items():
        if column == "metadata" and db.backend == "postgres":
            assignments.append(f"{column} = {marker}::jsonb")
        else:
            assignments.append(f"{column} = {marker}")
        params.append(value)
    params.extend([todo_id, user_id])

    try:
        db.conn.execute(
            f"""
            UPDATE todo_items
            SET {", ".join(assignments)}
            WHERE id = {marker} AND user_id = {marker} AND deleted_at IS NULL
            """,
            tuple(params),
        )
        db.conn.commit()
    except Exception:
        db.conn.rollback()
        raise

    return _select_by_id(db, user_id=user_id, todo_id=todo_id)


def delete_todo_item(
    db: DBClient,
    *,
    user_id: str,
    todo_id: str,
) -> bool:
    marker = _placeholder(db)
    now = now_iso()
    try:
        cursor = db.conn.execute(
            f"""
            UPDATE todo_items
            SET deleted_at = {marker}, updated_at = {marker}
            WHERE id = {marker} AND user_id = {marker} AND deleted_at IS NULL
            """,
            (now, now, todo_id, user_id),
        )
        db.conn.commit()
    except Exception:
        db.conn.rollback()
        raise
    return bool(getattr(cursor, "rowcount", 0))
