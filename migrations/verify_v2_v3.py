#!/usr/bin/env python3
"""只读校验：用户、视频、任务状态、房间、日志、轨迹外键。不改库。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db import load_siglip_env, require_tracking_db_url  # noqa: E402

REQUIRED_VIDEO_COLS = {
    "origin_label",
    "task_id",
    "status",
    "failure_reason",
    "progress",
    "duration",
    "target_count",
    "completed_at",
    "processing_started_at",
    "updated_at",
    "is_media_source",
    "source_path",
    "file_name",
}
REQUIRED_INDEXES = {
    ("videos", "uq_videos_file_name"),
    ("videos", "idx_videos_media"),
    ("videos", "idx_videos_task_id"),
    ("videos", "idx_videos_status"),
    ("rooms", "uq_rooms_video_name"),
    ("rooms", "idx_rooms_video"),
    ("users", "uk_username"),
    ("search_logs", "idx_created_at"),
    ("gallery_meta", "idx_gallery_video_id"),
    ("gallery_meta", "idx_video_name"),
}
REQUIRED_FKS = {
    ("rooms", "fk_rooms_video"),
    ("gallery_meta", "fk_gallery_meta_video"),
    ("search_logs", "fk_search_logs_user"),
    ("processing_runs", "fk_processing_runs_video"),
    ("stay_segments", "fk_stay_room"),
    ("track_points", "fk_track_points_room"),
}


def _v2_url() -> str:
    load_siglip_env()
    url = os.environ.get("DB_URL", "").strip()
    database = (make_url(url).database or "").lower() if url else ""
    if database != "siglip_v2":
        raise SystemExit("校验需要 DB_URL 指向 siglip_v2")
    return url


def _fail(failed: list[str], msg: str) -> None:
    failed.append(msg)
    print(f"  FAIL  {msg}")


def _ok(msg: str) -> None:
    print(f"  OK    {msg}")


def _warn(msg: str) -> None:
    print(f"  WARN  {msg}")


def main() -> int:
    v2 = create_engine(_v2_url(), pool_pre_ping=True)
    v3 = create_engine(require_tracking_db_url(), pool_pre_ping=True)
    failed: list[str] = []

    def n(eng, sql: str, params: dict | None = None) -> int:
        with eng.connect() as conn:
            return int(conn.execute(text(sql), params or {}).scalar() or 0)

    print("=== 数量 ===")
    pairs = [
        ("用户", "SELECT COUNT(*) FROM users", "SELECT COUNT(*) FROM users", "eq"),
        (
            "任务/媒体源",
            "SELECT COUNT(*) FROM processed_files",
            "SELECT COUNT(*) FROM videos WHERE is_media_source=1",
            "eq",
        ),
        (
            "房间(业务视频)",
            "SELECT COUNT(*) FROM video_rooms",
            """
            SELECT COUNT(*) FROM rooms r
            JOIN videos v ON v.id = r.video_id
            WHERE v.is_media_source = 1
            """,
            "ge",
        ),
        ("检索日志", "SELECT COUNT(*) FROM search_logs", "SELECT COUNT(*) FROM search_logs", "eq"),
        ("gallery_meta", "SELECT COUNT(*) FROM gallery_meta", "SELECT COUNT(*) FROM gallery_meta", "eq"),
    ]
    print(f"{'项目':<16} {'v2':>8} {'v3':>8}")
    for label, sql_v2, sql_v3, mode in pairs:
        a, b = n(v2, sql_v2), n(v3, sql_v3)
        ok = (a == b) if mode == "eq" else (b >= a)
        print(f"{label:<16} {a:>8} {b:>8}  {'OK' if ok else 'FAIL'}")
        if not ok:
            failed.append(f"{label} 数量不一致 v2={a} v3={b}")

    videos_all = n(v3, "SELECT COUNT(*) FROM videos")
    media = n(v3, "SELECT COUNT(*) FROM videos WHERE is_media_source=1")
    print(f"{'videos 全部':<16} {'—':>8} {videos_all:>8}  (媒体源 {media})")

    print("\n=== 用户 ===")
    with v2.connect() as c2, v3.connect() as c3:
        v2_users = {
            r[0]: (r[1], int(r[2]))
            for r in c2.execute(text("SELECT username, role, is_active FROM users"))
        }
        v3_users = {
            r[0]: (r[1], int(r[2]))
            for r in c3.execute(text("SELECT username, role, is_active FROM users"))
        }
        if set(v2_users) != set(v3_users):
            _fail(failed, f"用户名不一致 missing={sorted(set(v2_users)-set(v3_users))} extra={sorted(set(v3_users)-set(v2_users))}")
        else:
            _ok(f"用户名对齐 {sorted(v2_users)}")
        for name, meta in v2_users.items():
            if v3_users.get(name) != meta:
                _fail(failed, f"用户 {name} role/is_active 不一致 v2={meta} v3={v3_users.get(name)}")

    print("\n=== 任务状态 ===")
    with v2.connect() as c2, v3.connect() as c3:
        v2_tasks = {
            r["file_name"]: dict(r)
            for r in c2.execute(
                text(
                    "SELECT file_name, status, source_type, task_id, progress, target_count FROM processed_files"
                )
            ).mappings()
        }
        v3_tasks = {
            r["file_name"]: dict(r)
            for r in c3.execute(
                text(
                    """
                    SELECT file_name, status, origin_label, task_id, progress, target_count
                    FROM videos WHERE is_media_source=1
                    """
                )
            ).mappings()
        }
        missing = sorted(set(v2_tasks) - set(v3_tasks))
        extra = sorted(set(v3_tasks) - set(v2_tasks))
        if missing or extra:
            _fail(failed, f"任务 file_name missing={missing or '无'} extra={extra or '无'}")
        else:
            _ok(f"任务 file_name 对齐 {len(v2_tasks)} 条")
        status_mismatch = []
        for name, row in v2_tasks.items():
            other = v3_tasks.get(name)
            if not other:
                continue
            if str(row["status"]) != str(other["status"]):
                status_mismatch.append(
                    f"{name}: v2={row['status']} v3={other['status']}"
                )
            if str(row["source_type"] or "") != str(other["origin_label"] or ""):
                status_mismatch.append(
                    f"{name} 来源: v2={row['source_type']!r} v3={other['origin_label']!r}"
                )
            if int(row["progress"] or 0) != int(other["progress"] or 0):
                _warn(f"{name} progress v2={row['progress']} v3={other['progress']}（切库前旧进程仍可能写 v2）")
        if status_mismatch:
            for item in status_mismatch:
                _fail(failed, f"任务状态 {item}")
        else:
            _ok("任务 status / origin_label 与 v2 一致")

    print("\n=== 房间 ===")
    with v2.connect() as c2, v3.connect() as c3:
        v2_rooms = {
            (r[0], r[1])
            for r in c2.execute(text("SELECT video_name, room_name FROM video_rooms"))
        }
        v3_rooms = {
            (r[0], r[1])
            for r in c3.execute(
                text(
                    """
                    SELECT v.file_name, r.name
                    FROM rooms r
                    JOIN videos v ON v.id = r.video_id
                    WHERE v.is_media_source = 1
                    """
                )
            )
        }
        room_missing = sorted(v2_rooms - v3_rooms)
        if room_missing:
            _fail(failed, f"房间缺失 {room_missing}")
        else:
            _ok(f"业务房间对齐 {len(v2_rooms)} 条")
        orphan_rooms = n(
            v3,
            "SELECT COUNT(*) FROM rooms WHERE video_id IS NOT NULL AND video_id NOT IN (SELECT id FROM videos)",
        )
        if orphan_rooms:
            _fail(failed, f"rooms.video_id 孤儿 {orphan_rooms}")
        else:
            _ok("rooms.video_id 均可关联 videos.id")

    print("\n=== 日志 ===")
    with v2.connect() as c2, v3.connect() as c3:
        v2_ids = {int(r[0]) for r in c2.execute(text("SELECT id FROM search_logs"))}
        v3_ids = {int(r[0]) for r in c3.execute(text("SELECT id FROM search_logs"))}
        if v2_ids != v3_ids:
            _fail(failed, f"search_logs id 不一致 missing={sorted(v2_ids-v3_ids)} extra={sorted(v3_ids-v2_ids)}")
        else:
            _ok(f"search_logs id 对齐 {sorted(v2_ids)}")

    print("\n=== 轨迹关联 ===")
    with v3.connect() as c3:
        bad_runs = n(
            v3,
            "SELECT COUNT(*) FROM processing_runs WHERE video_id NOT IN (SELECT id FROM videos)",
        )
        bad_stays = n(
            v3,
            """
            SELECT COUNT(*) FROM stay_segments ss
            WHERE ss.room_id IS NOT NULL
              AND ss.room_id NOT IN (SELECT id FROM rooms)
            """,
        )
        bad_points = n(
            v3,
            """
            SELECT COUNT(*) FROM track_points tp
            WHERE tp.room_id IS NOT NULL
              AND tp.room_id NOT IN (SELECT id FROM rooms)
            """,
        )
        media_with_runs = n(
            v3,
            """
            SELECT COUNT(DISTINCT v.id)
            FROM videos v
            JOIN processing_runs pr ON pr.video_id = v.id
            WHERE v.is_media_source = 1
            """,
        )
        media_completed = n(
            v3,
            "SELECT COUNT(*) FROM videos WHERE is_media_source=1 AND status='completed'",
        )
        if bad_runs:
            _fail(failed, f"processing_runs 孤儿 video_id {bad_runs}")
        else:
            _ok("processing_runs.video_id 均可关联 videos")
        if bad_stays:
            _fail(failed, f"stay_segments.room_id 孤儿 {bad_stays}")
        else:
            _ok("stay_segments.room_id 为空或指向有效 rooms")
        if bad_points:
            _fail(failed, f"track_points.room_id 孤儿 {bad_points}")
        else:
            _ok("track_points.room_id 为空或指向有效 rooms")
        print(f"  INFO  已完成媒体源 {media_completed}，其中有 processing_runs 的 {media_with_runs}")
        if media_completed and media_with_runs < media_completed:
            _warn("部分 completed 媒体源没有 processing_runs（可能尚未建模或导入失败）")

    print("\n=== schema / 索引 / 外键 ===")
    with v3.connect() as conn:
        cols = {
            str(r[0])
            for r in conn.execute(text("SHOW COLUMNS FROM videos"))
        }
        missing_cols = sorted(REQUIRED_VIDEO_COLS - cols)
        if missing_cols:
            _fail(failed, f"videos 缺列 {missing_cols}")
        else:
            _ok("videos 业务列齐全")

        indexes = {
            (str(r[0]), str(r[1]))
            for r in conn.execute(
                text(
                    """
                    SELECT TABLE_NAME, INDEX_NAME
                    FROM information_schema.STATISTICS
                    WHERE TABLE_SCHEMA = DATABASE()
                    """
                )
            )
        }
        missing_idx = sorted(REQUIRED_INDEXES - indexes)
        if missing_idx:
            _fail(failed, f"缺索引 {missing_idx}")
        else:
            _ok("常用唯一键/查询索引齐全")

        fks = {
            (str(r[0]), str(r[1]))
            for r in conn.execute(
                text(
                    """
                    SELECT TABLE_NAME, CONSTRAINT_NAME
                    FROM information_schema.TABLE_CONSTRAINTS
                    WHERE TABLE_SCHEMA = DATABASE()
                      AND CONSTRAINT_TYPE = 'FOREIGN KEY'
                    """
                )
            )
        }
        missing_fk = sorted(REQUIRED_FKS - fks)
        if missing_fk:
            _fail(failed, f"缺外键 {missing_fk}")
        else:
            _ok("房间/日志/底库/轨迹外键齐全")

    print("\n=== 结论 ===")
    if failed:
        print(f"失败 {len(failed)} 项。旧库未删除。")
        print("不要用「恢复旧代码」当作完整回滚，见 migrations/README.md 回滚策略。")
        return 1
    print("校验通过。旧库未删除。")
    print("若新代码尚未重启上线，v2 仍可能被旧进程写入；切库后回滚必须先做 v3→v2 回填。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
