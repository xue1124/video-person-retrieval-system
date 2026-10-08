"""medical_audit_v3 业务库连接与视频任务辅助。

运行时只使用 TRACKING_DB_URL，禁止再连 siglip_v2。
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url

from services.media.time_utils import db_datetime_to_utc_iso, parse_iso_captured_at
from services.config import PROJECT_ROOT

TARGET_DATABASE = "medical_audit_v3"
_ENGINE: Engine | None = None

_ENV_FILE = PROJECT_ROOT / "deploy" / "env" / "siglip.env"
_LOCAL_ENV_FILE = PROJECT_ROOT / "deploy" / "env" / "siglip.local.env"

ORIGIN_TO_ENUM = {
    "手动上传": "upload",
    "NVR流": "rtsp",
    "NVR_ISAPI": "isapi",
    "RTSP实时": "rtsp",
}

TASK_FIELD_MAP = {
    "raw_path": "source_path",
    "source_type": "origin_label",
}

MEDIA_LOOKUP_SQL = """
SELECT
  id AS video_id,
  file_name,
  status,
  failure_reason,
  origin_label AS source_type,
  task_id,
  source_path AS raw_path,
  fps,
  duration,
  progress,
  completed_at,
  target_count,
  created_at,
  updated_at,
  processing_started_at
FROM videos
WHERE file_name = :v AND is_media_source = 1
LIMIT 1
"""

MEDIA_LIST_SQL = """
SELECT
  v.id AS video_id,
  v.file_name,
  v.status,
  v.failure_reason,
  v.origin_label AS source_type,
  v.task_id,
  v.source_path AS raw_path,
  v.fps,
  v.duration,
  v.progress,
  v.completed_at,
  v.target_count,
  v.created_at,
  v.updated_at,
  v.processing_started_at,
  TIMESTAMPDIFF(
    SECOND,
    COALESCE(v.processing_started_at, v.created_at),
    COALESCE(v.completed_at, v.updated_at, v.created_at)
  ) AS elapsed_seconds
FROM videos v
WHERE v.is_media_source = 1
ORDER BY v.created_at DESC
"""


def _load_env_file(path: Path, *, override: bool) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if override or key not in os.environ:
            os.environ[key] = value


def load_siglip_env() -> None:
    """加载共享 env，再用本地 secrets 覆盖。本地文件不入库 Git。"""
    _load_env_file(_ENV_FILE, override=False)
    _load_env_file(_LOCAL_ENV_FILE, override=True)


def require_tracking_db_url() -> str:
    load_siglip_env()
    url = os.environ.get("TRACKING_DB_URL", "").strip()
    if not url:
        raise RuntimeError("未配置 TRACKING_DB_URL，业务库必须指向 medical_audit_v3")
    database = (make_url(url).database or "").lower()
    if database != TARGET_DATABASE:
        raise RuntimeError(
            f"TRACKING_DB_URL 必须指向 {TARGET_DATABASE}，当前是 {database or '(空)'}"
        )
    return url


def get_engine() -> Engine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine(
            require_tracking_db_url(),
            pool_recycle=3600,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
    return _ENGINE


def origin_to_enum(label: str) -> str:
    text_label = (label or "").strip()
    if text_label in ORIGIN_TO_ENUM:
        return ORIGIN_TO_ENUM[text_label]
    low = text_label.lower()
    if "isapi" in low:
        return "isapi"
    if "rtsp" in low:
        return "rtsp"
    if "nvr" in low:
        return "rtsp"
    if "upload" in low or "手动" in text_label:
        return "upload"
    return "other"


def stable_source_key(file_name: str, source_path: str = "") -> str:
    identity = f"media:{file_name}|{source_path or file_name}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def get_video_by_file_name(conn, file_name: str) -> Optional[dict[str, Any]]:
    row = conn.execute(
        text(
            """
            SELECT id, file_name, source_type, origin_label, source_path, task_id,
                   status, failure_reason, progress, duration, fps, target_count,
                   captured_at, completed_at, processing_started_at, is_media_source
            FROM videos
            WHERE file_name = :n
            ORDER BY is_media_source DESC, id DESC
            LIMIT 1
            """
        ),
        {"n": file_name},
    ).mappings().first()
    return dict(row) if row else None


def get_video_id(conn, file_name: str) -> Optional[int]:
    row = get_video_by_file_name(conn, file_name)
    return int(row["id"]) if row else None


def last_insert_id(conn) -> int:
    if conn.dialect.name == "sqlite":
        return int(conn.execute(text("SELECT last_insert_rowid()")).scalar())
    return int(conn.execute(text("SELECT LAST_INSERT_ID()")).scalar_one())


def get_captured_at(conn, file_name: str) -> Optional[Any]:
    row = conn.execute(
        text(
            """
            SELECT captured_at FROM videos
            WHERE file_name = :n
            ORDER BY is_media_source DESC, id DESC
            LIMIT 1
            """
        ),
        {"n": file_name},
    ).first()
    return row[0] if row else None


def captured_at_iso_for_import(conn, file_name: str) -> Optional[str]:
    return db_datetime_to_utc_iso(get_captured_at(conn, file_name))


def stamp_captured_at_if_missing(conn, file_name: str, captured_at: Any) -> None:
    """仅在 captured_at 为空时写入，供 RTSP 直播实际开始时打戳。"""
    captured_at = parse_iso_captured_at(captured_at)
    if captured_at is None:
        return
    conn.execute(
        text(
            """
            UPDATE videos
            SET captured_at = COALESCE(captured_at, :captured_at)
            WHERE file_name=:v AND is_media_source=1
            """
        ),
        {"v": file_name, "captured_at": captured_at},
    )


def upsert_media_video(
    conn,
    *,
    file_name: str,
    origin_label: str,
    task_id: str,
    source_path: str,
    status: str = "pending",
    captured_at: Any = None,
) -> int:
    """插入或更新媒体源视频行，返回 video_id。

    captured_at 必须由调用方显式传入（可为 None）。同名重传会写入新值，
    包括清空为 NULL；不要在这里用 COALESCE 保留旧拍摄时间。
    """
    existing = get_video_by_file_name(conn, file_name)
    enum_type = origin_to_enum(origin_label)
    captured_at = parse_iso_captured_at(captured_at)
    if existing:
        conn.execute(
            text(
                """
                UPDATE videos
                SET origin_label = :origin_label,
                    source_type = :source_type,
                    source_path = :source_path,
                    task_id = :task_id,
                    status = :status,
                    captured_at = :captured_at,
                    is_media_source = 1,
                    progress = 0,
                    target_count = 0,
                    failure_reason = NULL,
                    completed_at = NULL,
                    processing_started_at = NULL
                WHERE id = :id
                """
            ),
            {
                "origin_label": origin_label,
                "source_type": enum_type,
                "source_path": source_path,
                "task_id": task_id,
                "status": status,
                "captured_at": captured_at,
                "id": int(existing["id"]),
            },
        )
        return int(existing["id"])

    conn.execute(
        text(
            """
            INSERT INTO videos (
              source_key, file_name, source_type, origin_label, source_path,
              task_id, status, is_media_source, progress, target_count, captured_at
            ) VALUES (
              :source_key, :file_name, :source_type, :origin_label, :source_path,
              :task_id, :status, 1, 0, 0, :captured_at
            )
            """
        ),
        {
            "source_key": stable_source_key(file_name, source_path),
            "file_name": file_name,
            "source_type": enum_type,
            "origin_label": origin_label,
            "source_path": source_path,
            "task_id": task_id,
            "status": status,
            "captured_at": captured_at,
        },
    )
    return last_insert_id(conn)


def update_video_task(conn, file_name: str, **fields: Any) -> None:
    if not fields:
        return
    sets: list[str] = []
    params: dict[str, Any] = {"file_name": file_name}
    for key, value in fields.items():
        column = TASK_FIELD_MAP.get(key, key)
        sets.append(f"{column} = :{column}")
        params[column] = value
    sql = (
        "UPDATE videos SET "
        + ", ".join(sets)
        + " WHERE file_name = :file_name AND is_media_source = 1"
    )
    conn.execute(text(sql), params)


def media_source_exists(conn, file_name: str) -> bool:
    row = conn.execute(
        text(
            "SELECT 1 FROM videos WHERE file_name=:n AND is_media_source=1 LIMIT 1"
        ),
        {"n": file_name},
    ).first()
    return row is not None
