"""medical_audit_v3 人物轨迹档案 API。"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import text
from sqlalchemy.engine import Engine

from services.auth.dependencies import get_current_user
from services.persistence import database as app_db
from services.config import PROJECT_ROOT

SNAPSHOT_ROOT = Path(
    os.environ.get("TRACKING_SNAPSHOT_ROOT", str(PROJECT_ROOT / "tracking_snapshots"))
).expanduser().resolve()

router = APIRouter(prefix="/tracking", tags=["tracking-archive"])

_tracking_engine: Engine | None = None


def get_tracking_engine() -> Engine:
    global _tracking_engine
    if _tracking_engine is not None:
        return _tracking_engine
    _tracking_engine = app_db.get_engine()
    return _tracking_engine


def _fmt_dt(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="seconds")
    return str(value)


def _fmt_sec(value: Any) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 3)


def _fmt_score(value: Any) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 4)


def _fmt_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _hms(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"
    total = max(0, int(round(float(seconds))))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _snapshot_url(snapshot_id: Any) -> Optional[str]:
    if snapshot_id is None:
        return None
    return f"/tracking/snapshots/{int(snapshot_id)}"


def _serialize_snapshot(row: Any) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "url": _snapshot_url(row["id"]),
        "snapshot_type": row["snapshot_type"],
        "is_primary": bool(row["is_primary"]),
        "timestamp_sec": _fmt_sec(row["timestamp_sec"]),
        "timestamp_hms": _hms(_fmt_sec(row["timestamp_sec"])),
        "quality_score": _fmt_sec(row["quality_score"]),
        "video_person_id": int(row["video_person_id"]),
        "stay_segment_id": (
            int(row["stay_segment_id"]) if row["stay_segment_id"] is not None else None
        ),
        "video_id": int(row["video_id"]),
    }


@router.get("/status")
def tracking_status(user: dict = Depends(get_current_user)):
    url = os.environ.get("TRACKING_DB_URL", "").strip()
    enabled = bool(url)
    ready = False
    detail = "未配置 TRACKING_DB_URL"
    counts: dict[str, int] = {}
    if enabled:
        try:
            eng = get_tracking_engine()
            with eng.connect() as conn:
                counts = {
                    "videos": int(conn.execute(text("SELECT COUNT(*) FROM videos")).scalar_one()),
                    "global_people": int(
                        conn.execute(
                            text(
                                "SELECT COUNT(*) FROM global_people WHERE status <> 'rejected'"
                            )
                        ).scalar_one()
                    ),
                    "stay_segments": int(
                        conn.execute(text("SELECT COUNT(*) FROM stay_segments")).scalar_one()
                    ),
                }
                try:
                    counts["snapshots"] = int(
                        conn.execute(text("SELECT COUNT(*) FROM person_snapshots")).scalar_one()
                    )
                except Exception:
                    counts["snapshots"] = 0
            ready = True
            detail = "ok"
        except Exception as exc:
            detail = str(exc)
    return {
        "enabled": enabled,
        "ready": ready,
        "detail": detail,
        "counts": counts,
        "user": user.get("username"),
    }


@router.get("/snapshots/{snapshot_id}")
def serve_tracking_snapshot(snapshot_id: int, user: dict = Depends(get_current_user)):
    eng = get_tracking_engine()
    with eng.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT id, image_path
                FROM person_snapshots
                WHERE id = :id
                """
            ),
            {"id": snapshot_id},
        ).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="快照不存在")
    rel = str(row["image_path"] or "").lstrip("/").replace("\\", "/")
    if not rel or ".." in rel.split("/"):
        raise HTTPException(status_code=400, detail="非法快照路径")
    path = (SNAPSHOT_ROOT / rel).resolve()
    try:
        path.relative_to(SNAPSHOT_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="非法快照路径") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="快照文件不存在")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/videos")
def list_tracking_videos(user: dict = Depends(get_current_user)):
    eng = get_tracking_engine()
    with eng.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT
                  v.id,
                  v.file_name,
                  v.source_type,
                  v.captured_at,
                  v.fps,
                  v.duration_sec,
                  c.name AS camera_name,
                  c.channel_no,
                  (
                    SELECT COUNT(DISTINCT vp.global_person_id)
                    FROM processing_runs pr
                    JOIN video_people vp ON vp.processing_run_id = pr.id
                    WHERE pr.video_id = v.id
                      AND vp.global_person_id IS NOT NULL
                  ) AS person_count,
                  (
                    SELECT COUNT(*)
                    FROM processing_runs pr2
                    WHERE pr2.video_id = v.id
                  ) AS run_count
                FROM videos v
                LEFT JOIN cameras c ON c.id = v.camera_id
                ORDER BY v.id DESC
                """
            )
        ).mappings()
        items = []
        for row in rows:
            items.append(
                {
                    "id": int(row["id"]),
                    "file_name": row["file_name"],
                    "source_type": row["source_type"],
                    "captured_at": _fmt_dt(row["captured_at"]),
                    "fps": _fmt_sec(row["fps"]),
                    "duration_sec": _fmt_sec(row["duration_sec"]),
                    "camera_name": row["camera_name"],
                    "channel_no": row["channel_no"],
                    "person_count": int(row["person_count"] or 0),
                    "run_count": int(row["run_count"] or 0),
                }
            )
    return {"items": items}


@router.get("/rooms")
def list_tracking_rooms(
    video_id: Optional[int] = Query(None),
    user: dict = Depends(get_current_user),
):
    eng = get_tracking_engine()
    sql = """
        SELECT r.id, r.name, r.video_id, v.file_name
        FROM rooms r
        LEFT JOIN videos v ON v.id = r.video_id
    """
    params: dict[str, Any] = {}
    if video_id is not None:
        sql += " WHERE r.video_id = :video_id"
        params["video_id"] = video_id
    sql += " ORDER BY r.video_id, r.id"
    with eng.connect() as conn:
        rows = conn.execute(text(sql), params).mappings()
        items = [
            {
                "id": int(row["id"]),
                "name": row["name"],
                "video_id": int(row["video_id"]) if row["video_id"] is not None else None,
                "file_name": row["file_name"],
            }
            for row in rows
        ]
    return {"items": items}


@router.get("/people")
def list_tracking_people(
    video_id: Optional[int] = Query(None),
    room_id: Optional[int] = Query(None),
    user: dict = Depends(get_current_user),
):
    """人物下拉：必须先选视频，避免一次拉全库 G。"""
    if video_id is None:
        return {"items": []}

    eng = get_tracking_engine()
    params: dict[str, Any] = {"video_id": int(video_id)}
    room_join = ""
    room_where = ""
    if room_id is not None:
        room_join = """
            JOIN stay_segments ss_f ON ss_f.video_person_id = vp.id
        """
        room_where = " AND ss_f.room_id = :room_id"
        params["room_id"] = int(room_id)

    sql = f"""
        SELECT
          gp.id,
          gp.status,
          gp.sample_count,
          gp.first_seen_at,
          gp.last_seen_at,
          scoped.video_person_count,
          scoped.local_person_no
        FROM (
            SELECT
              vp.global_person_id,
              COUNT(DISTINCT vp.id) AS video_person_count,
              MIN(vp.local_person_no) AS local_person_no
            FROM video_people vp
            JOIN processing_runs pr ON pr.id = vp.processing_run_id
            {room_join}
            WHERE pr.video_id = :video_id
              AND vp.global_person_id IS NOT NULL
              {room_where}
            GROUP BY vp.global_person_id
        ) scoped
        JOIN global_people gp ON gp.id = scoped.global_person_id
        WHERE gp.status <> 'rejected'
        ORDER BY scoped.local_person_no, gp.id
    """
    with eng.connect() as conn:
        rows = conn.execute(text(sql), params).mappings()
        items = []
        for row in rows:
            local_label = row["local_person_no"]
            items.append(
                {
                    "id": int(row["id"]),
                    "label": f"G{row['id']}"
                    + (
                        f"（视频内P{local_label}）"
                        if local_label is not None
                        else ""
                    ),
                    "status": row["status"],
                    "sample_count": int(row["sample_count"] or 0),
                    "first_seen_at": _fmt_dt(row["first_seen_at"]),
                    "last_seen_at": _fmt_dt(row["last_seen_at"]),
                    "video_person_count": int(row["video_person_count"] or 0),
                    "video_count": 1,
                    "track_count": 0,
                    "stay_sec_total": 0.0,
                    "members": None,
                    "local_person_no": int(local_label) if local_label is not None else None,
                    "thumb_url": None,
                }
            )
    return {"items": items}


@router.get("/people/{global_person_id}")
def get_person_archive(
    global_person_id: int,
    video_id: Optional[int] = Query(None),
    room_id: Optional[int] = Query(None),
    include_points: bool = Query(False),
    points_limit: int = Query(300, ge=1, le=5000),
    user: dict = Depends(get_current_user),
):
    eng = get_tracking_engine()
    with eng.connect() as conn:
        person = conn.execute(
            text(
                """
                SELECT id, status, sample_count, first_seen_at, last_seen_at,
                       embedding_dim, created_at
                FROM global_people
                WHERE id = :id
                """
            ),
            {"id": global_person_id},
        ).mappings().first()
        if not person:
            raise HTTPException(status_code=404, detail=f"找不到人物 G{global_person_id}")

        vp_params: dict[str, Any] = {"gid": global_person_id}
        vp_filter = ""
        if video_id is not None:
            vp_filter = " AND pr.video_id = :video_id"
            vp_params["video_id"] = video_id

        video_people = conn.execute(
            text(
                f"""
                SELECT
                  vp.id,
                  vp.local_person_no,
                  vp.start_sec,
                  vp.end_sec,
                  vp.assignment_method,
                  vp.assignment_score,
                  pr.id AS processing_run_id,
                  v.id AS video_id,
                  v.file_name,
                  v.captured_at,
                  c.name AS camera_name
                FROM video_people vp
                JOIN processing_runs pr ON pr.id = vp.processing_run_id
                JOIN videos v ON v.id = pr.video_id
                LEFT JOIN cameras c ON c.id = v.camera_id
                WHERE vp.global_person_id = :gid
                {vp_filter}
                ORDER BY v.id, vp.local_person_no
                """
            ),
            vp_params,
        ).mappings().all()

        stay_params: dict[str, Any] = {"gid": global_person_id}
        stay_filters = ["vp.global_person_id = :gid"]
        if video_id is not None:
            stay_filters.append("pr.video_id = :video_id")
            stay_params["video_id"] = video_id
        if room_id is not None:
            stay_filters.append("ss.room_id = :room_id")
            stay_params["room_id"] = room_id
        stay_where = " AND ".join(stay_filters)

        stays = conn.execute(
            text(
                f"""
                SELECT
                  ss.id,
                  ss.start_sec,
                  ss.end_sec,
                  ss.duration_sec,
                  ss.entered_at,
                  ss.exited_at,
                  ss.point_count,
                  r.id AS room_id,
                  r.name AS room_name,
                  v.id AS video_id,
                  v.file_name,
                  vp.local_person_no,
                  COALESCE(
                    (
                      SELECT ps.id
                      FROM person_snapshots ps
                      WHERE ps.stay_segment_id = ss.id
                      ORDER BY ps.quality_score DESC, ps.id ASC
                      LIMIT 1
                    ),
                    (
                      SELECT ps.id
                      FROM person_snapshots ps
                      WHERE ps.video_person_id = vp.id
                        AND ps.snapshot_type = 'candidate'
                      ORDER BY ABS(ps.timestamp_sec - ss.start_sec) ASC,
                               ps.quality_score DESC, ps.id ASC
                      LIMIT 1
                    )
                  ) AS snapshot_id
                FROM stay_segments ss
                JOIN video_people vp ON vp.id = ss.video_person_id
                JOIN processing_runs pr ON pr.id = vp.processing_run_id
                JOIN videos v ON v.id = pr.video_id
                LEFT JOIN rooms r ON r.id = ss.room_id
                WHERE {stay_where}
                ORDER BY v.id, ss.start_sec, ss.id
                """
            ),
            stay_params,
        ).mappings().all()

        stay_items = []
        room_totals: dict[str, float] = {}
        for row in stays:
            duration = float(row["duration_sec"] or 0)
            room_name = row["room_name"] or "未标注区域"
            room_totals[room_name] = room_totals.get(room_name, 0.0) + duration
            stay_items.append(
                {
                    "id": int(row["id"]),
                    "video_id": int(row["video_id"]),
                    "file_name": row["file_name"],
                    "local_person_no": int(row["local_person_no"]),
                    "room_id": int(row["room_id"]) if row["room_id"] is not None else None,
                    "room_name": row["room_name"],
                    "start_sec": _fmt_sec(row["start_sec"]),
                    "end_sec": _fmt_sec(row["end_sec"]),
                    "duration_sec": _fmt_sec(row["duration_sec"]),
                    "start_hms": _hms(_fmt_sec(row["start_sec"])),
                    "end_hms": _hms(_fmt_sec(row["end_sec"])),
                    "duration_hms": _hms(_fmt_sec(row["duration_sec"])),
                    "entered_at": _fmt_dt(row["entered_at"]),
                    "exited_at": _fmt_dt(row["exited_at"]),
                    "point_count": int(row["point_count"] or 0),
                    "snapshot_url": _snapshot_url(row["snapshot_id"]),
                }
            )

        track_count = conn.execute(
            text(
                f"""
                SELECT COUNT(*)
                FROM tracks t
                JOIN video_people vp ON vp.id = t.video_person_id
                JOIN processing_runs pr ON pr.id = vp.processing_run_id
                WHERE vp.global_person_id = :gid
                {vp_filter}
                """
            ),
            vp_params,
        ).scalar_one()

        points: list[dict[str, Any]] = []
        if include_points:
            point_params = dict(vp_params)
            point_params["limit"] = points_limit
            if room_id is not None:
                point_params["room_id"] = room_id
            room_clause = " AND tp.room_id = :room_id" if room_id is not None else ""
            point_rows = conn.execute(
                text(
                    f"""
                    SELECT
                      tp.timestamp_sec,
                      tp.occurred_at,
                      tp.confidence,
                      tp.pre_merge_id,
                      tp.assign_score,
                      tp.merge_score,
                      tp.bbox_x1, tp.bbox_y1, tp.bbox_x2, tp.bbox_y2,
                      tp.foot_x, tp.foot_y,
                      r.name AS room_name,
                      t.local_track_no,
                      v.file_name,
                      vp.local_person_no
                    FROM track_points tp
                    JOIN tracks t ON t.id = tp.track_id
                    JOIN video_people vp ON vp.id = t.video_person_id
                    JOIN processing_runs pr ON pr.id = vp.processing_run_id
                    JOIN videos v ON v.id = pr.video_id
                    LEFT JOIN rooms r ON r.id = tp.room_id
                    WHERE vp.global_person_id = :gid
                    {vp_filter}
                    {room_clause}
                    ORDER BY v.id, tp.timestamp_sec, tp.id
                    LIMIT :limit
                    """
                ),
                point_params,
            ).mappings()
            for row in point_rows:
                points.append(
                    {
                        "file_name": row["file_name"],
                        "local_person_no": int(row["local_person_no"]),
                        "local_track_no": int(row["local_track_no"]),
                        "timestamp_sec": _fmt_sec(row["timestamp_sec"]),
                        "timestamp_hms": _hms(_fmt_sec(row["timestamp_sec"])),
                        "occurred_at": _fmt_dt(row["occurred_at"]),
                        "room_name": row["room_name"],
                        "confidence": _fmt_score(row["confidence"]),
                        "pre_merge_id": _fmt_int(row["pre_merge_id"]),
                        "assign_score": _fmt_score(row["assign_score"]),
                        "merge_score": _fmt_score(row["merge_score"]),
                        "bbox": [
                            _fmt_sec(row["bbox_x1"]),
                            _fmt_sec(row["bbox_y1"]),
                            _fmt_sec(row["bbox_x2"]),
                            _fmt_sec(row["bbox_y2"]),
                        ],
                        "foot": [_fmt_sec(row["foot_x"]), _fmt_sec(row["foot_y"])],
                    }
                )

        snap_params: dict[str, Any] = {"gid": global_person_id}
        snap_filter = ""
        if video_id is not None:
            snap_filter = " AND ps.video_id = :video_id"
            snap_params["video_id"] = video_id
        snapshot_rows = conn.execute(
            text(
                f"""
                SELECT
                  ps.id, ps.snapshot_type, ps.is_primary, ps.timestamp_sec,
                  ps.quality_score, ps.video_person_id, ps.stay_segment_id,
                  ps.video_id
                FROM person_snapshots ps
                WHERE ps.global_person_id = :gid
                  AND ps.snapshot_type = 'candidate'
                {snap_filter}
                ORDER BY ps.is_primary DESC, ps.quality_score DESC, ps.id ASC
                """
            ),
            snap_params,
        ).mappings().all()
        candidate_snapshots = [_serialize_snapshot(row) for row in snapshot_rows]
        primary_snapshot = next(
            (s for s in candidate_snapshots if s["is_primary"]),
            candidate_snapshots[0] if candidate_snapshots else None,
        )

        # 每个 video_person 取一张最佳候选图
        member_thumb: dict[int, str] = {}
        for snap in candidate_snapshots:
            vp_id = int(snap["video_person_id"])
            if vp_id not in member_thumb and snap["url"]:
                member_thumb[vp_id] = snap["url"]

    stay_total = round(sum(float(s["duration_sec"] or 0) for s in stay_items), 3)
    overview = {
        "id": int(person["id"]),
        "status": person["status"],
        "sample_count": int(person["sample_count"] or 0),
        "first_seen_at": _fmt_dt(person["first_seen_at"]),
        "last_seen_at": _fmt_dt(person["last_seen_at"]),
        "video_count": len({int(v["video_id"]) for v in video_people}),
        "video_person_count": len(video_people),
        "track_count": int(track_count or 0),
        "stay_segment_count": len(stay_items),
        "stay_sec_total": stay_total,
        "stay_hms_total": _hms(stay_total),
        "primary_snapshot": primary_snapshot,
        "candidate_snapshots": candidate_snapshots[:12],
        "rooms_visited": [
            {"name": name, "stay_sec": round(sec, 3), "stay_hms": _hms(sec)}
            for name, sec in sorted(room_totals.items(), key=lambda x: -x[1])
        ],
        "members": [
            {
                "video_person_id": int(v["id"]),
                "video_id": int(v["video_id"]),
                "file_name": v["file_name"],
                "local_person_no": int(v["local_person_no"]),
                "label": f"P{v['local_person_no']}",
                "start_sec": _fmt_sec(v["start_sec"]),
                "end_sec": _fmt_sec(v["end_sec"]),
                "camera_name": v["camera_name"],
                "assignment_method": v["assignment_method"],
                "assignment_score": _fmt_sec(v["assignment_score"]),
                "thumb_url": member_thumb.get(int(v["id"])),
            }
            for v in video_people
        ],
    }

    # 简易时间线：按视频分组的停留段
    timeline_by_video: dict[str, list[dict[str, Any]]] = {}
    for stay in stay_items:
        timeline_by_video.setdefault(stay["file_name"], []).append(stay)

    return {
        "overview": overview,
        "stays": stay_items,
        "timeline_by_video": [
            {"file_name": name, "segments": segs}
            for name, segs in timeline_by_video.items()
        ],
        "points": points,
        "points_truncated": include_points,
        "points_limit": points_limit if include_points else 0,
    }
