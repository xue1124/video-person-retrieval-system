#!/usr/bin/env python3
"""离线房间标注：从 JSON 导入 medical_audit_v3.rooms，也可导出覆盖图。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url


TARGET_DATABASE = "medical_audit_v3"
SYSTEM_DATABASES = {"mysql", "information_schema", "performance_schema", "sys"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导入/预览离线房间标注")
    parser.add_argument("rooms_json", type=Path)
    parser.add_argument("--db-url", default=os.getenv("TRACKING_DB_URL"))
    parser.add_argument(
        "--video-id",
        type=int,
        help="绑定到 videos.id；不传则按 --file-name 查找",
    )
    parser.add_argument("--file-name", help="例如 test3_part_a.mp4")
    parser.add_argument(
        "--camera-id",
        type=int,
        help="可选：同时挂到摄像头，后续同摄像头视频可复用",
    )
    parser.add_argument(
        "--reference-image",
        type=Path,
        help="用于生成覆盖预览图",
    )
    parser.add_argument(
        "--preview-out",
        type=Path,
        default=Path("tracking_experiments/outputs/room_demo/rooms_overlay.jpg"),
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="先删除该 video_id 下旧房间再导入",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_rooms(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rooms = payload.get("rooms") or []
    cleaned: list[dict[str, Any]] = []
    for room in rooms:
        name = str(room.get("name") or "").strip()
        polygon = room.get("polygon") or []
        if not name or len(polygon) < 3:
            raise ValueError(f"房间无效: {room}")
        cleaned.append(
            {
                "name": name,
                "polygon": [[float(p[0]), float(p[1])] for p in polygon],
            }
        )
    return cleaned


def draw_preview(image_path: Path, rooms: list[dict[str, Any]], out_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取参考图: {image_path}")
    overlay = image.copy()
    colors = [
        (80, 180, 80),
        (80, 160, 220),
        (220, 140, 60),
        (180, 80, 200),
        (60, 200, 200),
    ]
    for idx, room in enumerate(rooms):
        color = colors[idx % len(colors)]
        pts = np.asarray(room["polygon"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(image, [pts], True, color, 2)
        cx = int(np.mean([p[0] for p in room["polygon"]]))
        cy = int(np.mean([p[1] for p in room["polygon"]]))
        cv2.putText(
            image,
            room["name"],
            (cx - 40, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    blended = cv2.addWeighted(overlay, 0.28, image, 0.72, 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), blended)


def resolve_video_id(conn: Any, args: argparse.Namespace) -> int:
    if args.video_id is not None:
        return int(args.video_id)
    if not args.file_name:
        raise ValueError("需要 --video-id 或 --file-name")
    row = conn.execute(
        text("SELECT id FROM videos WHERE file_name=:name ORDER BY id DESC LIMIT 1"),
        {"name": args.file_name},
    ).scalar_one_or_none()
    if row is None:
        raise ValueError(f"找不到视频: {args.file_name}")
    return int(row)


def main() -> int:
    args = parse_args()
    rooms = load_rooms(args.rooms_json.resolve())
    print(f"读取到 {len(rooms)} 个房间: {[r['name'] for r in rooms]}")

    if args.reference_image:
        draw_preview(args.reference_image.resolve(), rooms, args.preview_out.resolve())
        print(f"预览图: {args.preview_out.resolve()}")

    if args.dry_run:
        return 0
    if not args.db_url:
        raise ValueError("请提供 --db-url 或 TRACKING_DB_URL")
    database = (make_url(args.db_url).database or "").lower()
    if database in SYSTEM_DATABASES or database != TARGET_DATABASE:
        raise ValueError(f"只允许写入 {TARGET_DATABASE}")

    engine = create_engine(args.db_url, pool_pre_ping=True)
    with engine.begin() as conn:
        video_id = resolve_video_id(conn, args)
        if args.replace:
            conn.execute(
                text("DELETE FROM rooms WHERE video_id=:video_id"),
                {"video_id": video_id},
            )
        for room in rooms:
            conn.execute(
                text(
                    """
                    INSERT INTO rooms (camera_id, video_id, name, polygon_json)
                    VALUES (:camera_id, :video_id, :name, :polygon_json)
                    """
                ),
                {
                    "camera_id": args.camera_id,
                    "video_id": video_id,
                    "name": room["name"],
                    "polygon_json": json.dumps(room["polygon"], ensure_ascii=False),
                },
            )
        count = conn.execute(
            text("SELECT COUNT(*) FROM rooms WHERE video_id=:video_id"),
            {"video_id": video_id},
        ).scalar_one()
        print(f"已写入 video_id={video_id}，当前房间数={count}")
    engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
