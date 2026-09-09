"""房间标注已直接写 medical_audit_v3.rooms；这里按 video_id 重算停留与快照。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

TARGET_DATABASE = "medical_audit_v3"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tracking_url() -> str | None:
    url = os.environ.get("TRACKING_DB_URL", "").strip()
    if not url:
        return None
    database = (make_url(url).database or "").lower()
    if database != TARGET_DATABASE:
        return None
    return url


def _parse_polygon(raw: Any) -> list[list[float]]:
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, list) or len(raw) < 3:
        raise ValueError("polygon 无效")
    out: list[list[float]] = []
    for point in raw:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            raise ValueError("polygon 顶点无效")
        out.append([float(point[0]), float(point[1])])
    return out


def find_tracking_video_ids(conn: Any, video_name: str) -> list[int]:
    rows = conn.execute(
        text(
            """
            SELECT id
            FROM videos
            WHERE file_name = :name
            ORDER BY id ASC
            """
        ),
        {"name": video_name},
    ).scalars().all()
    return [int(v) for v in rows]


def replace_rooms_for_video(
    conn: Any,
    video_id: int,
    rooms: list[dict[str, Any]],
) -> int:
    # 先清停留（依赖 room_id），再清房间，避免外键挂起
    conn.execute(
        text(
            """
            DELETE ss FROM stay_segments ss
            JOIN video_people vp ON vp.id = ss.video_person_id
            JOIN processing_runs pr ON pr.id = vp.processing_run_id
            WHERE pr.video_id = :vid
            """
        ),
        {"vid": video_id},
    )
    conn.execute(
        text(
            """
            UPDATE track_points tp
            JOIN tracks t ON t.id = tp.track_id
            JOIN video_people vp ON vp.id = t.video_person_id
            JOIN processing_runs pr ON pr.id = vp.processing_run_id
            SET tp.room_id = NULL
            WHERE pr.video_id = :vid
            """
        ),
        {"vid": video_id},
    )
    conn.execute(
        text("DELETE FROM rooms WHERE video_id = :vid"),
        {"vid": video_id},
    )
    for room in rooms:
        conn.execute(
            text(
                """
                INSERT INTO rooms (camera_id, video_id, name, polygon_json)
                VALUES (NULL, :video_id, :name, :polygon_json)
                """
            ),
            {
                "video_id": video_id,
                "name": room["name"],
                "polygon_json": json.dumps(room["polygon"], ensure_ascii=False),
            },
        )
    return len(rooms)


def _stay_params() -> tuple[float, float]:
    time_gap = float(os.environ.get("TRACKING_V3_STAY_TIME_GAP", "2") or "2")
    nearest = float(os.environ.get("TRACKING_V3_NEAREST_MAX_PX", "0") or "0")
    return time_gap, nearest


def _import_stays():
    import sys

    experiments = PROJECT_ROOT / "tracking_experiments"
    if str(experiments) not in sys.path:
        sys.path.insert(0, str(experiments))
    import compute_stays as stays  # type: ignore

    return stays


def compute_stays_on_conn(conn: Any, video_id: int) -> dict[str, Any]:
    """用当前 rooms 给轨迹点赋 room_id 并生成 stay_segments。应与换房在同一事务里。"""
    stays = _import_stays()
    time_gap, nearest = _stay_params()
    rooms = stays.load_rooms(conn, video_id)
    if not rooms:
        return {"skipped": "no_rooms", "video_id": video_id}
    stays.delete_stays_for_video(conn, video_id)
    assign_stats = stays.assign_room_ids(
        conn,
        video_id,
        rooms,
        nearest_max_px=nearest,
    )
    stay_count = stays.build_stay_segments(
        conn,
        video_id,
        time_gap=time_gap,
    )
    return {
        **assign_stats,
        "stay_segments": stay_count,
        "video_id": video_id,
    }


def recompute_stays_for_video(video_id: int) -> dict[str, Any]:
    url = _tracking_url()
    if not url:
        return {"skipped": "no_tracking_db"}
    engine = create_engine(url, pool_pre_ping=True)
    try:
        with engine.begin() as conn:
            return compute_stays_on_conn(conn, video_id)
    finally:
        engine.dispose()


def _snapshots_enabled() -> bool:
    return os.environ.get("TRACKING_V3_GENERATE_SNAPSHOTS", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def recompute_snapshots_for_video(video_id: int) -> dict[str, Any]:
    url = _tracking_url()
    if not url:
        return {"skipped": "no_tracking_db"}
    if not _snapshots_enabled():
        return {"skipped": "disabled", "video_id": video_id}
    import sys

    experiments = PROJECT_ROOT / "tracking_experiments"
    if str(experiments) not in sys.path:
        sys.path.insert(0, str(experiments))
    import generate_snapshots as snapshots  # type: ignore

    return snapshots.generate_snapshots_for_video(url, video_id)


def recompute_stays_and_snapshots(video_id: int) -> dict[str, Any]:
    return {
        "stays": recompute_stays_for_video(video_id),
        "snapshots": recompute_snapshots_for_video(video_id),
    }


def generate_snapshots_for_video_name(video_name: str) -> dict[str, Any]:
    """只切图，不改房间、不清停留。供保存房间后的后台任务使用。"""
    url = _tracking_url()
    if not url:
        return {"skipped": "no_tracking_db", "video_name": video_name}
    engine = create_engine(url, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            video_ids = find_tracking_video_ids(conn, video_name)
    finally:
        engine.dispose()
    if not video_ids:
        return {
            "skipped": "medical_audit_v3 尚无该视频轨迹记录",
            "video_name": video_name,
        }
    return {
        "video_name": video_name,
        "snapshots": [recompute_snapshots_for_video(vid) for vid in video_ids],
    }


def recompute_stays_for_video_name(video_name: str) -> dict[str, Any]:
    """按当前 rooms 重算停留，不再从旧库拷贝多边形。"""
    url = _tracking_url()
    if not url:
        return {"skipped": "no_tracking_db", "video_name": video_name}
    engine = create_engine(url, pool_pre_ping=True)
    stay_reports: list[dict[str, Any]] = []
    video_ids: list[int] = []
    try:
        with engine.begin() as conn:
            video_ids = find_tracking_video_ids(conn, video_name)
            if not video_ids:
                return {
                    "skipped": "尚无该视频记录",
                    "video_name": video_name,
                }
            for vid in video_ids:
                stay_reports.append(compute_stays_on_conn(conn, vid))
    finally:
        engine.dispose()
    return {
        "video_name": video_name,
        "video_ids": video_ids,
        "stays": stay_reports,
        "recomputed": stay_reports,
    }


def sync_video_rooms_to_tracking(
    video_name: str,
    *,
    recompute: bool | None = None,
    recompute_stays: bool | None = None,
    recompute_snapshots: bool | None = None,
) -> dict[str, Any]:
    """兼容旧调用名：房间已在 v3，这里只重算停留。"""
    del recompute, recompute_stays, recompute_snapshots
    return recompute_stays_for_video_name(video_name)
