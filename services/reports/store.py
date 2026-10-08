"""analysis_reports 读写。独立事务，避免 HTTPException 回滚失败记录。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from services.persistence.database import last_insert_id

_SHANGHAI = ZoneInfo("Asia/Shanghai")
_UTC = timezone.utc


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _parse_json(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    text_value = str(value).strip()
    if not text_value:
        return default
    return json.loads(text_value)


def format_report_datetime(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text_value = str(value).strip().replace(" ", "T", 1)
        try:
            dt = datetime.fromisoformat(text_value)
        except ValueError:
            return str(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_UTC)
    return dt.astimezone(_SHANGHAI).isoformat()


def utc_now_naive() -> datetime:
    return datetime.now(_UTC).replace(tzinfo=None)


def insert_generating(
    engine: Engine,
    *,
    user_id: int,
    report_type: str,
    report_date: Optional[str],
    timezone_name: str,
    scope: dict[str, Any],
    report_data_json: str,
) -> int:
    now = utc_now_naive()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO analysis_reports (
                  created_by, report_type, report_date, timezone,
                  scope_json, report_data_json, status, created_at, updated_at
                ) VALUES (
                  :created_by, :report_type, :report_date, :timezone,
                  :scope_json, :report_data_json, 'generating', :created_at, :updated_at
                )
                """
            ),
            {
                "created_by": int(user_id),
                "report_type": report_type,
                "report_date": report_date,
                "timezone": timezone_name,
                "scope_json": _json_text(scope),
                "report_data_json": report_data_json,
                "created_at": now,
                "updated_at": now,
            },
        )
        return last_insert_id(conn)


def mark_completed(
    engine: Engine,
    report_id: int,
    *,
    title: str,
    report_markdown: str,
    warnings: list[str],
    workflow_run_id: Optional[str],
    dify_output: dict[str, Any],
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE analysis_reports
                SET title = :title,
                    report_markdown = :report_markdown,
                    warnings_json = :warnings_json,
                    dify_workflow_run_id = :run_id,
                    dify_output_json = :dify_output_json,
                    status = 'completed',
                    error_message = NULL,
                    updated_at = :updated_at
                WHERE id = :id
                """
            ),
            {
                "title": title,
                "report_markdown": report_markdown,
                "warnings_json": _json_text(warnings),
                "run_id": workflow_run_id,
                "dify_output_json": _json_text(dify_output),
                "updated_at": utc_now_naive(),
                "id": int(report_id),
            },
        )


def mark_failed(engine: Engine, report_id: int, error_message: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE analysis_reports
                SET status = 'failed',
                    error_message = :error_message,
                    updated_at = :updated_at
                WHERE id = :id
                """
            ),
            {
                "error_message": (error_message or "报告生成失败")[:1000],
                "updated_at": utc_now_naive(),
                "id": int(report_id),
            },
        )


def _is_deleted(row: dict[str, Any]) -> bool:
    return row.get("deleted_at") is not None


def get_report(
    conn: Connection,
    report_id: int,
    *,
    include_deleted: bool = False,
) -> Optional[dict[str, Any]]:
    row = conn.execute(
        text("SELECT * FROM analysis_reports WHERE id = :id"),
        {"id": int(report_id)},
    ).mappings().first()
    if row is None:
        return None
    payload = dict(row)
    if not include_deleted and _is_deleted(payload):
        return None
    return payload


def fetch_report(
    engine: Engine,
    report_id: int,
    *,
    include_deleted: bool = False,
) -> Optional[dict[str, Any]]:
    with engine.connect() as conn:
        return get_report(conn, report_id, include_deleted=include_deleted)


def soft_delete(engine: Engine, report_id: int, user_id: int) -> None:
    now = utc_now_naive()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE analysis_reports
                SET deleted_at = :deleted_at,
                    deleted_by = :deleted_by,
                    updated_at = :updated_at
                WHERE id = :id
                  AND deleted_at IS NULL
                """
            ),
            {
                "deleted_at": now,
                "deleted_by": int(user_id),
                "updated_at": now,
                "id": int(report_id),
            },
        )


def list_reports(
    engine: Engine,
    *,
    user_id: int,
    is_admin: bool,
    status: Optional[str] = None,
    report_date: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[dict[str, Any]], int]:
    page = max(1, int(page))
    page_size = min(100, max(1, int(page_size)))
    where = ["deleted_at IS NULL"]
    filters: dict[str, Any] = {}
    if not is_admin:
        where.append("created_by = :user_id")
        filters["user_id"] = int(user_id)
    if status:
        where.append("status = :status")
        filters["status"] = status
    if report_date:
        where.append("report_date = :report_date")
        filters["report_date"] = report_date
    clause = " AND ".join(where)
    with engine.connect() as conn:
        total = int(
            conn.execute(
                text(f"SELECT COUNT(*) FROM analysis_reports WHERE {clause}"),
                filters,
            ).scalar()
            or 0
        )
        rows = conn.execute(
            text(
                f"""
                SELECT id, created_by, report_type, report_date, timezone, title,
                       warnings_json, status, error_message, created_at, updated_at,
                       dify_workflow_run_id, scope_json
                FROM analysis_reports
                WHERE {clause}
                ORDER BY created_at DESC, id DESC
                LIMIT :limit OFFSET :offset
                """
            ),
            {**filters, "limit": page_size, "offset": (page - 1) * page_size},
        ).mappings().all()
    return [dict(row) for row in rows], total


def to_public_dict(row: dict[str, Any], *, detail: bool = False) -> dict[str, Any]:
    report_date = row.get("report_date")
    if hasattr(report_date, "isoformat"):
        report_date = report_date.isoformat()
    elif report_date is not None:
        report_date = str(report_date)[:10]
    payload = {
        "id": int(row["id"]),
        "status": row.get("status"),
        "report_type": row.get("report_type"),
        "report_date": report_date,
        "timezone": row.get("timezone"),
        "title": row.get("title"),
        "warnings": _parse_json(row.get("warnings_json"), []) or [],
        "scope": _parse_json(row.get("scope_json"), {}) or {},
        "created_at": format_report_datetime(row.get("created_at")),
        "updated_at": format_report_datetime(row.get("updated_at")),
        "error_message": row.get("error_message"),
        "dify_workflow_run_id": row.get("dify_workflow_run_id"),
    }
    if detail:
        payload["report_markdown"] = row.get("report_markdown")
        payload["report_data_json"] = _parse_json(row.get("report_data_json"), None)
    return payload
