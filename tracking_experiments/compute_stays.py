#!/usr/bin/env python3
"""根据 rooms 多边形给 track_points 填 room_id，并生成 stay_segments。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from room_geometry import classify_point_to_room  # noqa: E402


TARGET_DATABASE = "medical_audit_v3"
SYSTEM_DATABASES = {"mysql", "information_schema", "performance_schema", "sys"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="轨迹点房间归属 + 停留片段生成")
    parser.add_argument("--db-url", default=os.getenv("TRACKING_DB_URL"))
    parser.add_argument(
        "--video-id",
        type=int,
        action="append",
        dest="video_ids",
        help="可重复传入；不传则处理所有 videos",
    )
    parser.add_argument(
        "--rooms-video-id",
        type=int,
        help="房间定义所在 video_id；默认与每个目标视频自身 rooms 一致",
    )
    parser.add_argument(
        "--time-gap",
        type=float,
        default=2.0,
        help="同房间相邻观测间隔不超过该秒数时合并为同一停留段",
    )
    parser.add_argument(
        "--nearest-max-px",
        type=float,
        default=0.0,
        help="允许最近房间匹配的最大像素距离；0 表示只认落在多边形内",
    )
    parser.add_argument(
        "--replace-stays",
        action="store_true",
        help="删除目标人物已有 stay_segments 后再生成",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_rooms(conn: Any, video_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        text(
            """
            SELECT id, name, polygon_json
            FROM rooms
            WHERE video_id = :video_id
            ORDER BY id
            """
        ),
        {"video_id": video_id},
    )
    rooms: list[dict[str, Any]] = []
    for row in rows:
        polygon = row.polygon_json
        if isinstance(polygon, (bytes, bytearray)):
            polygon = polygon.decode("utf-8")
        if isinstance(polygon, str):
            polygon = json.loads(polygon)
        rooms.append(
            {
                "id": int(row.id),
                "name": str(row.name),
                "polygon": polygon,
            }
        )
    return rooms


def assign_room_ids(
    conn: Any,
    video_id: int,
    rooms: list[dict[str, Any]],
    nearest_max_px: float,
) -> dict[str, int]:
    name_to_id = {room["name"]: room["id"] for room in rooms}
    points = conn.execute(
        text(
            """
            SELECT tp.id, tp.foot_x, tp.foot_y, tp.bbox_x1, tp.bbox_x2, tp.bbox_y2
            FROM track_points tp
            JOIN tracks t ON t.id = tp.track_id
            JOIN processing_runs pr ON pr.id = t.processing_run_id
            WHERE pr.video_id = :video_id
            """
        ),
        {"video_id": video_id},
    ).mappings().all()

    updates: list[dict[str, Any]] = []
    inside = nearest = none = 0
    for point in points:
        foot_x = point["foot_x"]
        foot_y = point["foot_y"]
        if foot_x is None or foot_y is None:
            foot_x = (float(point["bbox_x1"]) + float(point["bbox_x2"])) / 2.0
            foot_y = float(point["bbox_y2"])
        result = classify_point_to_room(float(foot_x), float(foot_y), rooms)
        room_id = None
        if result["method"] == "inside":
            room_id = name_to_id.get(result["name"])
            inside += 1
        elif (
            result["method"] == "nearest"
            and nearest_max_px > 0
            and float(result["dist_px"]) <= nearest_max_px
        ):
            room_id = name_to_id.get(result["name"])
            nearest += 1
        else:
            none += 1
        updates.append({"id": int(point["id"]), "room_id": room_id})

    if updates:
        conn.execute(
            text("UPDATE track_points SET room_id=:room_id WHERE id=:id"),
            updates,
        )
        conn.execute(
            text(
                """
                UPDATE observations o
                INNER JOIN track_points tp ON tp.id = o.track_point_id
                SET o.room_id = tp.room_id
                WHERE o.video_id = :video_id
                """
            ),
            {"video_id": int(video_id)},
        )
    return {"points": len(points), "inside": inside, "nearest": nearest, "none": none}


def build_stay_segments(
    conn: Any,
    video_id: int,
    time_gap: float,
) -> int:
    rows = conn.execute(
        text(
            """
            SELECT
              vp.id AS video_person_id,
              tp.room_id,
              tp.timestamp_sec,
              tp.occurred_at
            FROM track_points tp
            JOIN tracks t ON t.id = tp.track_id
            JOIN video_people vp ON vp.id = t.video_person_id
            JOIN processing_runs pr ON pr.id = t.processing_run_id
            WHERE pr.video_id = :video_id
              AND tp.room_id IS NOT NULL
            ORDER BY vp.id, tp.timestamp_sec, tp.id
            """
        ),
        {"video_id": video_id},
    ).mappings().all()

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["video_person_id"])].append(dict(row))

    inserts: list[dict[str, Any]] = []
    for video_person_id, points in grouped.items():
        if not points:
            continue
        current = {
            "video_person_id": video_person_id,
            "room_id": int(points[0]["room_id"]),
            "start_sec": float(points[0]["timestamp_sec"]),
            "end_sec": float(points[0]["timestamp_sec"]),
            "entered_at": points[0]["occurred_at"],
            "exited_at": points[0]["occurred_at"],
            "point_count": 1,
        }
        for point in points[1:]:
            room_id = int(point["room_id"])
            ts = float(point["timestamp_sec"])
            same_room = room_id == current["room_id"]
            close_enough = (ts - current["end_sec"]) <= time_gap
            if same_room and close_enough:
                current["end_sec"] = ts
                current["exited_at"] = point["occurred_at"]
                current["point_count"] += 1
            else:
                inserts.append(_finalize_segment(current))
                current = {
                    "video_person_id": video_person_id,
                    "room_id": room_id,
                    "start_sec": ts,
                    "end_sec": ts,
                    "entered_at": point["occurred_at"],
                    "exited_at": point["occurred_at"],
                    "point_count": 1,
                }
        inserts.append(_finalize_segment(current))

    if inserts:
        conn.execute(
            text(
                """
                INSERT INTO stay_segments
                  (video_person_id, room_id, start_sec, end_sec, duration_sec,
                   entered_at, exited_at, point_count, algorithm_version)
                VALUES
                  (:video_person_id, :room_id, :start_sec, :end_sec, :duration_sec,
                   :entered_at, :exited_at, :point_count, 'offline_v1')
                """
            ),
            inserts,
        )
    return len(inserts)


def _finalize_segment(segment: dict[str, Any]) -> dict[str, Any]:
    start = float(segment["start_sec"])
    end = float(segment["end_sec"])
    return {
        "video_person_id": segment["video_person_id"],
        "room_id": segment["room_id"],
        "start_sec": start,
        "end_sec": end,
        "duration_sec": round(end - start, 3),
        "entered_at": segment["entered_at"],
        "exited_at": segment["exited_at"],
        "point_count": int(segment["point_count"]),
    }


def delete_stays_for_video(conn: Any, video_id: int) -> None:
    conn.execute(
        text(
            """
            DELETE ss FROM stay_segments ss
            JOIN video_people vp ON vp.id = ss.video_person_id
            JOIN processing_runs pr ON pr.id = vp.processing_run_id
            WHERE pr.video_id = :video_id
            """
        ),
        {"video_id": video_id},
    )


def main() -> int:
    args = parse_args()
    if not args.db_url:
        raise ValueError("请提供 --db-url 或 TRACKING_DB_URL")
    database = (make_url(args.db_url).database or "").lower()
    if database in SYSTEM_DATABASES or database != TARGET_DATABASE:
        raise ValueError(f"只允许写入 {TARGET_DATABASE}")
    if args.time_gap < 0:
        raise ValueError("--time-gap 不能为负")
    if args.nearest_max_px < 0:
        raise ValueError("--nearest-max-px 不能为负")

    engine = create_engine(args.db_url, pool_pre_ping=True)
    with engine.begin() as conn:
        if args.video_ids:
            video_ids = list(args.video_ids)
        else:
            video_ids = [
                int(row[0])
                for row in conn.execute(text("SELECT id FROM videos ORDER BY id"))
            ]
        if not video_ids:
            raise ValueError("没有可处理的视频")

        report: list[dict[str, Any]] = []
        for video_id in video_ids:
            rooms_video_id = args.rooms_video_id or video_id
            rooms = load_rooms(conn, rooms_video_id)
            if not rooms:
                raise ValueError(f"video_id={rooms_video_id} 没有 rooms 数据")

            # 若目标视频没有自己的 rooms，但指定了模板 rooms_video_id，则复制一份绑定
            if rooms_video_id != video_id:
                existing = load_rooms(conn, video_id)
                if not existing:
                    for room in rooms:
                        conn.execute(
                            text(
                                """
                                INSERT INTO rooms (camera_id, video_id, name, polygon_json)
                                SELECT camera_id, :video_id, name, polygon_json
                                FROM rooms WHERE id=:room_id
                                """
                            ),
                            {"video_id": video_id, "room_id": room["id"]},
                        )
                    rooms = load_rooms(conn, video_id)

            if args.dry_run:
                report.append(
                    {
                        "video_id": video_id,
                        "rooms": [r["name"] for r in rooms],
                        "dry_run": True,
                    }
                )
                continue

            if args.replace_stays:
                delete_stays_for_video(conn, video_id)
            assign_stats = assign_room_ids(
                conn, video_id, rooms, args.nearest_max_px
            )
            stay_count = build_stay_segments(
                conn,
                video_id,
                time_gap=args.time_gap,
            )
            report.append(
                {
                    "video_id": video_id,
                    "rooms": [r["name"] for r in rooms],
                    **assign_stats,
                    "stay_segments": stay_count,
                }
            )

        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.dry_run:
            conn.rollback()
    engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
