#!/usr/bin/env python3
"""把 medical_audit_v3 业务表回写到 siglip_v2（不删 v3、不重启服务）。

默认 --dry-run。真正写入需显式 --execute。

这不是完整回滚：回写后仍需检出切库前代码并重启 API/Celery，旧进程才会再读 v2。
轨迹档案（global_people/tracks 等）只存在于 v3，不会写回 v2。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db import load_siglip_env, require_tracking_db_url  # noqa: E402


def _v2_url() -> str:
    load_siglip_env()
    url = os.environ.get("DB_URL", "").strip()
    database = (make_url(url).database or "").lower() if url else ""
    if database != "siglip_v2":
        raise SystemExit("回滚回写需要 DB_URL 指向 siglip_v2")
    return url


def _count(conn, sql: str) -> int:
    return int(conn.execute(text(sql)).scalar() or 0)


def rollback(execute: bool) -> int:
    v2_engine = create_engine(_v2_url(), pool_pre_ping=True)
    v3_engine = create_engine(require_tracking_db_url(), pool_pre_ping=True)

    with v3_engine.connect() as v3, v2_engine.connect() as v2:
        plan = {
            "users": _count(v3, "SELECT COUNT(*) FROM users"),
            "media_videos": _count(v3, "SELECT COUNT(*) FROM videos WHERE is_media_source=1"),
            "rooms": _count(
                v3,
                """
                SELECT COUNT(*) FROM rooms r
                JOIN videos v ON v.id = r.video_id
                WHERE v.is_media_source = 1
                """,
            ),
            "search_logs": _count(v3, "SELECT COUNT(*) FROM search_logs"),
            "gallery_meta": _count(v3, "SELECT COUNT(*) FROM gallery_meta"),
        }
        current_v2 = {
            "users": _count(v2, "SELECT COUNT(*) FROM users"),
            "processed_files": _count(v2, "SELECT COUNT(*) FROM processed_files"),
            "video_rooms": _count(v2, "SELECT COUNT(*) FROM video_rooms"),
            "search_logs": _count(v2, "SELECT COUNT(*) FROM search_logs"),
            "gallery_meta": _count(v2, "SELECT COUNT(*) FROM gallery_meta"),
        }

    print("将从 v3 回写到 v2（UPSERT，不删 v3）：")
    print(f"  v3 users={plan['users']} videos(media)={plan['media_videos']} rooms={plan['rooms']} logs={plan['search_logs']} gallery={plan['gallery_meta']}")
    print(f"  当前 v2 users={current_v2['users']} processed_files={current_v2['processed_files']} video_rooms={current_v2['video_rooms']} logs={current_v2['search_logs']} gallery={current_v2['gallery_meta']}")
    print("不会回写：cameras / processing_runs / tracks / global_people / stay_segments / person_snapshots")

    if not execute:
        print("dry-run：未写入。确认后执行：python migrations/rollback_v3_to_v2.py --execute")
        return 0

    with v3_engine.connect() as v3, v2_engine.begin() as v2:
        users = v3.execute(
            text("SELECT id, username, password_hash, role, is_active, created_at FROM users")
        ).mappings().all()
        for row in users:
            v2.execute(
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

        videos = v3.execute(
            text(
                """
                SELECT file_name, status, failure_reason, origin_label, task_id, source_path,
                       fps, duration, progress, completed_at, target_count, created_at,
                       updated_at, processing_started_at
                FROM videos WHERE is_media_source=1
                """
            )
        ).mappings().all()
        for row in videos:
            v2.execute(
                text(
                    """
                    INSERT INTO processed_files (
                      file_name, status, failure_reason, source_type, task_id, raw_path,
                      fps, duration, progress, completed_at, target_count, created_at,
                      updated_at, processing_started_at
                    ) VALUES (
                      :file_name, :status, :failure_reason, :origin_label, :task_id, :source_path,
                      :fps, :duration, :progress, :completed_at, :target_count, :created_at,
                      :updated_at, :processing_started_at
                    )
                    ON DUPLICATE KEY UPDATE
                      status=VALUES(status),
                      failure_reason=VALUES(failure_reason),
                      source_type=VALUES(source_type),
                      task_id=VALUES(task_id),
                      raw_path=VALUES(raw_path),
                      fps=VALUES(fps),
                      duration=VALUES(duration),
                      progress=VALUES(progress),
                      completed_at=VALUES(completed_at),
                      target_count=VALUES(target_count),
                      processing_started_at=VALUES(processing_started_at)
                    """
                ),
                dict(row),
            )

        rooms = v3.execute(
            text(
                """
                SELECT v.file_name AS video_name, r.name AS room_name, r.polygon_json,
                       r.created_at, r.updated_at
                FROM rooms r
                JOIN videos v ON v.id = r.video_id
                WHERE v.is_media_source = 1
                """
            )
        ).mappings().all()
        for row in rooms:
            raw = row["polygon_json"]
            if not isinstance(raw, str):
                raw = json.dumps(raw, ensure_ascii=False)
            v2.execute(
                text(
                    """
                    INSERT INTO video_rooms (video_name, room_name, polygon_json, created_at, updated_at)
                    VALUES (:video_name, :room_name, :polygon_json, :created_at, :updated_at)
                    ON DUPLICATE KEY UPDATE
                      polygon_json=VALUES(polygon_json),
                      updated_at=VALUES(updated_at)
                    """
                ),
                {
                    "video_name": row["video_name"],
                    "room_name": row["room_name"],
                    "polygon_json": raw,
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                },
            )

        logs = v3.execute(
            text(
                """
                SELECT id, query_image_paths, search_query, algorithm, threshold,
                       results_json, created_at
                FROM search_logs
                """
            )
        ).mappings().all()
        for row in logs:
            v2.execute(
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

        gallery = v3.execute(text("SELECT * FROM gallery_meta")).mappings().all()
        for row in gallery:
            v2.execute(
                text(
                    """
                    INSERT INTO gallery_meta (
                      id, video_name, timestamp, image_path,
                      bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                      feature_vector, clip_feature, created_at
                    ) VALUES (
                      :id, :video_name, :timestamp, :image_path,
                      :bbox_x1, :bbox_y1, :bbox_x2, :bbox_y2,
                      :feature_vector, :clip_feature, :created_at
                    )
                    ON DUPLICATE KEY UPDATE
                      video_name=VALUES(video_name),
                      image_path=VALUES(image_path),
                      feature_vector=VALUES(feature_vector),
                      clip_feature=VALUES(clip_feature)
                    """
                ),
                {
                    "id": row["id"],
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

    print("已 UPSERT 写入 siglip_v2。medical_audit_v3 未删除。")
    print("接下来仍需：检出切库前代码 → 重启 API/Celery。仅回写数据不会让旧进程改读 v2。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="v3 业务表回写 v2")
    parser.add_argument("--execute", action="store_true", help="真正写入 siglip_v2")
    args = parser.parse_args()
    return rollback(execute=bool(args.execute))


if __name__ == "__main__":
    raise SystemExit(main())
