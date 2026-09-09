#!/usr/bin/env python3
"""把 siglip_v2 业务数据回填进 medical_audit_v3。不删除、不修改旧库数据。"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db import (  # noqa: E402
    load_siglip_env,
    origin_to_enum,
    require_tracking_db_url,
    stable_source_key,
)

V2_TABLES = ("users", "processed_files", "video_rooms", "search_logs", "gallery_meta")


def _v2_url() -> str:
    load_siglip_env()
    url = os.environ.get("DB_URL", "").strip()
    if not url:
        raise SystemExit("回填需要 DB_URL 指向 siglip_v2（只读旧数据，不会删除）")
    database = (make_url(url).database or "").lower()
    if database != "siglip_v2":
        raise SystemExit(f"DB_URL 必须指向 siglip_v2，当前是 {database or '(空)'}")
    return url


def _has_column(conn, table: str, column: str) -> bool:
    row = conn.execute(
        text(
            """
            SELECT 1 FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = :t AND COLUMN_NAME = :c
            """
        ),
        {"t": table, "c": column},
    ).first()
    return row is not None


def _has_index(conn, table: str, index: str) -> bool:
    row = conn.execute(
        text(
            """
            SELECT 1 FROM information_schema.STATISTICS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = :t AND INDEX_NAME = :i
            LIMIT 1
            """
        ),
        {"t": table, "i": index},
    ).first()
    return row is not None


def apply_schema(conn) -> None:
    alters = [
        ("origin_label", "ALTER TABLE videos ADD COLUMN origin_label VARCHAR(50) NOT NULL DEFAULT '手动上传' AFTER source_type"),
        ("task_id", "ALTER TABLE videos ADD COLUMN task_id VARCHAR(100) NULL AFTER origin_label"),
        ("status", "ALTER TABLE videos ADD COLUMN status VARCHAR(32) NOT NULL DEFAULT 'pending' AFTER task_id"),
        ("failure_reason", "ALTER TABLE videos ADD COLUMN failure_reason VARCHAR(512) NULL AFTER status"),
        ("progress", "ALTER TABLE videos ADD COLUMN progress INT NOT NULL DEFAULT 0 AFTER failure_reason"),
        ("duration", "ALTER TABLE videos ADD COLUMN duration VARCHAR(32) NULL AFTER duration_sec"),
        ("target_count", "ALTER TABLE videos ADD COLUMN target_count INT NOT NULL DEFAULT 0 AFTER duration"),
        ("completed_at", "ALTER TABLE videos ADD COLUMN completed_at DATETIME(3) NULL AFTER target_count"),
        ("processing_started_at", "ALTER TABLE videos ADD COLUMN processing_started_at DATETIME(3) NULL AFTER completed_at"),
        ("updated_at", "ALTER TABLE videos ADD COLUMN updated_at DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3) AFTER created_at"),
        ("is_media_source", "ALTER TABLE videos ADD COLUMN is_media_source TINYINT(1) NOT NULL DEFAULT 0 AFTER source_path"),
    ]
    for column, sql in alters:
        if not _has_column(conn, "videos", column):
            conn.execute(text(sql))
            print(f"  + videos.{column}")

    indexes = [
        ("videos", "uq_videos_file_name", "ALTER TABLE videos ADD UNIQUE KEY uq_videos_file_name (file_name)"),
        ("videos", "idx_videos_media", "ALTER TABLE videos ADD KEY idx_videos_media (is_media_source, created_at)"),
        ("videos", "idx_videos_task_id", "ALTER TABLE videos ADD KEY idx_videos_task_id (task_id)"),
        ("videos", "idx_videos_status", "ALTER TABLE videos ADD KEY idx_videos_status (status)"),
        ("rooms", "uq_rooms_video_name", "ALTER TABLE rooms ADD UNIQUE KEY uq_rooms_video_name (video_id, name)"),
    ]
    for table, name, sql in indexes:
        if not _has_index(conn, table, name):
            conn.execute(text(sql))
            print(f"  + {table}.{name}")

    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS users (
              id INT UNSIGNED NOT NULL AUTO_INCREMENT,
              username VARCHAR(64) NOT NULL,
              password_hash VARCHAR(255) NOT NULL,
              role VARCHAR(20) NOT NULL DEFAULT 'user',
              is_active TINYINT(1) NOT NULL DEFAULT 1,
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY (id),
              UNIQUE KEY uk_username (username)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
        )
    )
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS search_logs (
              id INT UNSIGNED NOT NULL AUTO_INCREMENT,
              user_id INT UNSIGNED NULL,
              query_image_paths TEXT,
              search_query VARCHAR(500) DEFAULT NULL,
              algorithm VARCHAR(50) NOT NULL,
              threshold FLOAT NOT NULL DEFAULT 0.82,
              results_json LONGTEXT,
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY (id),
              KEY idx_created_at (created_at),
              KEY idx_algorithm (algorithm),
              KEY idx_search_logs_user (user_id),
              CONSTRAINT fk_search_logs_user
                FOREIGN KEY (user_id) REFERENCES users(id)
                ON DELETE SET NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
        )
    )
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS gallery_meta (
              id INT UNSIGNED NOT NULL AUTO_INCREMENT,
              video_id BIGINT UNSIGNED NULL,
              video_name VARCHAR(255) NOT NULL,
              timestamp FLOAT NOT NULL,
              image_path VARCHAR(500) NOT NULL,
              bbox_x1 INT DEFAULT NULL,
              bbox_y1 INT DEFAULT NULL,
              bbox_x2 INT DEFAULT NULL,
              bbox_y2 INT DEFAULT NULL,
              feature_vector LONGBLOB,
              clip_feature LONGBLOB,
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY (id),
              KEY idx_video_name (video_name),
              KEY idx_timestamp (timestamp),
              KEY idx_gallery_video_id (video_id),
              CONSTRAINT fk_gallery_meta_video
                FOREIGN KEY (video_id) REFERENCES videos(id)
                ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
        )
    )


def count_v2(conn) -> dict[str, int]:
    out = {}
    for table in V2_TABLES:
        out[table] = int(conn.execute(text(f"SELECT COUNT(*) FROM `{table}`")).scalar() or 0)
    return out


def count_v3(conn) -> dict[str, int]:
    def n(sql: str) -> int:
        return int(conn.execute(text(sql)).scalar() or 0)

    return {
        "users": n("SELECT COUNT(*) FROM users"),
        "media_videos": n("SELECT COUNT(*) FROM videos WHERE is_media_source=1"),
        "videos_all": n("SELECT COUNT(*) FROM videos"),
        "rooms_media": n(
            """
            SELECT COUNT(*) FROM rooms r
            JOIN videos v ON v.id = r.video_id
            WHERE v.is_media_source = 1
            """
        ),
        "rooms_all": n("SELECT COUNT(*) FROM rooms"),
        "search_logs": n("SELECT COUNT(*) FROM search_logs"),
        "gallery_meta": n("SELECT COUNT(*) FROM gallery_meta"),
    }


def backfill_users(v2, v3) -> int:
    rows = v2.execute(text("SELECT id, username, password_hash, role, is_active, created_at FROM users")).mappings().all()
    for row in rows:
        v3.execute(
            text(
                """
                INSERT INTO users (id, username, password_hash, role, is_active, created_at)
                VALUES (:id, :username, :password_hash, :role, :is_active, :created_at)
                ON DUPLICATE KEY UPDATE
                  password_hash=VALUES(password_hash),
                  role=VALUES(role),
                  is_active=VALUES(is_active)
                """
            ),
            dict(row),
        )
    if rows:
        max_id = max(int(r["id"]) for r in rows)
        v3.execute(text(f"ALTER TABLE users AUTO_INCREMENT = {max_id + 1}"))
    return len(rows)


def backfill_search_logs(v2, v3) -> int:
    rows = v2.execute(
        text(
            """
            SELECT id, query_image_paths, search_query, algorithm, threshold, results_json, created_at
            FROM search_logs
            """
        )
    ).mappings().all()
    for row in rows:
        v3.execute(
            text(
                """
                INSERT INTO search_logs
                  (id, query_image_paths, search_query, algorithm, threshold, results_json, created_at)
                VALUES
                  (:id, :query_image_paths, :search_query, :algorithm, :threshold, :results_json, :created_at)
                ON DUPLICATE KEY UPDATE
                  query_image_paths=VALUES(query_image_paths),
                  search_query=VALUES(search_query),
                  algorithm=VALUES(algorithm),
                  threshold=VALUES(threshold),
                  results_json=VALUES(results_json)
                """
            ),
            dict(row),
        )
    if rows:
        max_id = max(int(r["id"]) for r in rows)
        v3.execute(text(f"ALTER TABLE search_logs AUTO_INCREMENT = {max_id + 1}"))
    return len(rows)


def backfill_videos(v2, v3) -> int:
    rows = v2.execute(text("SELECT * FROM processed_files")).mappings().all()
    for row in rows:
        file_name = str(row["file_name"])
        origin = str(row["source_type"] or "手动上传")
        existing = v3.execute(
            text("SELECT id FROM videos WHERE file_name=:n LIMIT 1"),
            {"n": file_name},
        ).scalar()
        payload = {
            "file_name": file_name,
            "origin_label": origin,
            "source_type": origin_to_enum(origin),
            "source_path": row.get("raw_path"),
            "task_id": row.get("task_id"),
            "status": row.get("status") or "pending",
            "failure_reason": row.get("failure_reason"),
            "progress": int(row.get("progress") or 0),
            "duration": row.get("duration"),
            "fps": row.get("fps"),
            "target_count": int(row.get("target_count") or 0),
            "completed_at": row.get("completed_at"),
            "processing_started_at": row.get("processing_started_at"),
            "created_at": row.get("created_at"),
        }
        if existing:
            v3.execute(
                text(
                    """
                    UPDATE videos SET
                      origin_label=:origin_label,
                      source_type=:source_type,
                      source_path=COALESCE(:source_path, source_path),
                      task_id=:task_id,
                      status=:status,
                      failure_reason=:failure_reason,
                      progress=:progress,
                      duration=:duration,
                      fps=COALESCE(:fps, fps),
                      target_count=:target_count,
                      completed_at=:completed_at,
                      processing_started_at=:processing_started_at,
                      is_media_source=1
                    WHERE id=:id
                    """
                ),
                {**payload, "id": int(existing)},
            )
        else:
            v3.execute(
                text(
                    """
                    INSERT INTO videos (
                      source_key, file_name, source_type, origin_label, source_path,
                      task_id, status, failure_reason, progress, duration, fps,
                      target_count, completed_at, processing_started_at, created_at,
                      is_media_source
                    ) VALUES (
                      :source_key, :file_name, :source_type, :origin_label, :source_path,
                      :task_id, :status, :failure_reason, :progress, :duration, :fps,
                      :target_count, :completed_at, :processing_started_at, :created_at,
                      1
                    )
                    """
                ),
                {**payload, "source_key": stable_source_key(file_name, str(payload["source_path"] or ""))},
            )
    return len(rows)


def backfill_rooms(v2, v3) -> int:
    rows = v2.execute(
        text("SELECT video_name, room_name, polygon_json, created_at, updated_at FROM video_rooms")
    ).mappings().all()
    written = 0
    for row in rows:
        video_id = v3.execute(
            text("SELECT id FROM videos WHERE file_name=:n LIMIT 1"),
            {"n": row["video_name"]},
        ).scalar()
        if not video_id:
            print(f"  ! 跳过房间 {row['video_name']}/{row['room_name']}：v3 无对应 videos 行")
            continue
        raw = row["polygon_json"]
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                raw = json.dumps(parsed, ensure_ascii=False)
            except json.JSONDecodeError:
                pass
        v3.execute(
            text(
                """
                INSERT INTO rooms (video_id, name, polygon_json, created_at, updated_at)
                VALUES (:video_id, :name, :polygon_json, :created_at, :updated_at)
                ON DUPLICATE KEY UPDATE
                  polygon_json=VALUES(polygon_json),
                  updated_at=VALUES(updated_at)
                """
            ),
            {
                "video_id": int(video_id),
                "name": row["room_name"],
                "polygon_json": raw,
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
            },
        )
        written += 1
    return written


def backfill_gallery(v2, v3) -> int:
    rows = v2.execute(text("SELECT * FROM gallery_meta")).mappings().all()
    for row in rows:
        video_id = v3.execute(
            text("SELECT id FROM videos WHERE file_name=:n LIMIT 1"),
            {"n": row["video_name"]},
        ).scalar()
        v3.execute(
            text(
                """
                INSERT INTO gallery_meta (
                  id, video_id, video_name, timestamp, image_path,
                  bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                  feature_vector, clip_feature, created_at
                ) VALUES (
                  :id, :video_id, :video_name, :timestamp, :image_path,
                  :bbox_x1, :bbox_y1, :bbox_x2, :bbox_y2,
                  :feature_vector, :clip_feature, :created_at
                )
                ON DUPLICATE KEY UPDATE
                  video_id=VALUES(video_id),
                  image_path=VALUES(image_path),
                  feature_vector=VALUES(feature_vector),
                  clip_feature=VALUES(clip_feature)
                """
            ),
            {
                "id": row["id"],
                "video_id": int(video_id) if video_id else None,
                "video_name": row["video_name"],
                "timestamp": row["timestamp"],
                "image_path": row["image_path"],
                "bbox_x1": row.get("bbox_x1"),
                "bbox_y1": row.get("bbox_y1"),
                "bbox_x2": row.get("bbox_x2"),
                "bbox_y2": row.get("bbox_y2"),
                "feature_vector": row.get("feature_vector"),
                "clip_feature": row.get("clip_feature"),
                "created_at": row.get("created_at"),
            },
        )
    if rows:
        max_id = max(int(r["id"]) for r in rows)
        v3.execute(text(f"ALTER TABLE gallery_meta AUTO_INCREMENT = {max_id + 1}"))
    return len(rows)


def main() -> int:
    v2_engine = create_engine(_v2_url(), pool_pre_ping=True)
    v3_engine = create_engine(require_tracking_db_url(), pool_pre_ping=True)
    with v2_engine.connect() as v2:
        before_v2 = count_v2(v2)
    print("=== 回填前 siglip_v2 ===")
    for k, v in before_v2.items():
        print(f"  {k}: {v}")

    with v3_engine.begin() as v3:
        print("=== 应用 v3 schema ===")
        apply_schema(v3)
        before_v3 = count_v3(v3)
        print("=== 回填前 medical_audit_v3 ===")
        for k, v in before_v3.items():
            print(f"  {k}: {v}")

    with v2_engine.connect() as v2, v3_engine.begin() as v3:
        print("=== 回填 ===")
        print(f"  users: {backfill_users(v2, v3)}")
        print(f"  processed_files -> videos: {backfill_videos(v2, v3)}")
        print(f"  video_rooms -> rooms: {backfill_rooms(v2, v3)}")
        print(f"  search_logs: {backfill_search_logs(v2, v3)}")
        print(f"  gallery_meta: {backfill_gallery(v2, v3)}")

    with v3_engine.connect() as v3:
        after_v3 = count_v3(v3)
    print("=== 回填后 medical_audit_v3 ===")
    for k, v in after_v3.items():
        print(f"  {k}: {v}")

    report = {
        "v2": before_v2,
        "v3_after": after_v3,
        "checks": {
            "users": after_v3["users"] == before_v2["users"],
            "tasks": after_v3["media_videos"] == before_v2["processed_files"],
            "rooms": after_v3["rooms_media"] >= before_v2["video_rooms"],
            "logs": after_v3["search_logs"] == before_v2["search_logs"],
            "gallery": after_v3["gallery_meta"] == before_v2["gallery_meta"],
        },
    }
    ok = all(report["checks"].values())
    print("=== 校验 ===")
    for k, v in report["checks"].items():
        print(f"  {k}: {'OK' if v else 'FAIL'}")
    print("旧库 siglip_v2 未改动。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
