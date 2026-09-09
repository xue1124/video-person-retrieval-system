#!/usr/bin/env python3
"""根据 medical_audit_v3 中的最终 G，重绘两段视频便于肉眼核对。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url


TARGET_DATABASE = "medical_audit_v3"


def color_for_id(person_id: int) -> tuple[int, int, int]:
    return (
        64 + (person_id * 47) % 192,
        64 + (person_id * 89) % 192,
        64 + (person_id * 137) % 192,
    )


def load_run_mapping(db_url: str, run_id: int) -> dict[int, int]:
    """local_person_no(P) -> global_person_id(G)"""
    engine = create_engine(db_url, pool_pre_ping=True)
    mapping: dict[int, int] = {}
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT local_person_no, global_person_id
                FROM video_people
                WHERE processing_run_id = :run_id
                  AND global_person_id IS NOT NULL
                """
            ),
            {"run_id": run_id},
        )
        for row in rows:
            mapping[int(row.local_person_no)] = int(row.global_person_id)
    engine.dispose()
    return mapping


def render_video(
    source_video: Path,
    tracks_json: Path,
    output_video: Path,
    person_to_global: dict[int, int],
) -> None:
    payload = json.loads(tracks_json.read_text(encoding="utf-8"))
    metadata = payload["metadata"]
    frame_step = int(metadata["frame_step"])
    effective_fps = float(metadata["effective_fps"])
    width, height = (int(v) for v in metadata["frame_size"])
    processed_frames = int(metadata["processed_frames"])

    points_by_frame: dict[int, list[tuple[list[float], int, int, float]]] = {}
    for track in payload["tracks"]:
        local_p = int(track["global_person_id"])
        local_t = int(track["local_track_id"])
        global_g = person_to_global.get(local_p)
        if global_g is None:
            continue
        for point in track["points"]:
            frame_index = int(point["frame_index"])
            points_by_frame.setdefault(frame_index, []).append(
                (
                    [float(v) for v in point["bbox"]],
                    global_g,
                    local_t,
                    float(point["confidence"]),
                )
            )

    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {source_video}")
    temporary = output_video.with_name(f"{output_video.stem}.tmp.mp4")
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        effective_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"无法创建输出: {temporary}")

    source_frame_index = -1
    written = 0
    try:
        while written < processed_frames:
            ok, frame = capture.read()
            if not ok:
                break
            source_frame_index += 1
            if source_frame_index % frame_step != 0:
                continue
            annotated = frame.copy()
            for bbox, global_g, local_t, conf in points_by_frame.get(
                source_frame_index, []
            ):
                x1, y1, x2, y2 = (int(round(v)) for v in bbox)
                color = color_for_id(global_g)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"G{global_g} / T{local_t}  {conf:.2f}"
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(20, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            writer.write(annotated)
            written += 1
    finally:
        capture.release()
        writer.release()

    if written != processed_frames:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"帧数不一致: 期望 {processed_frames}，实际 {written}")
    temporary.replace(output_video)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-url", default=os.getenv("TRACKING_DB_URL"))
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--tracks-json", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.db_url:
        raise ValueError("需要 TRACKING_DB_URL 或 --db-url")
    database = (make_url(args.db_url).database or "").lower()
    if database != TARGET_DATABASE:
        raise ValueError(f"只允许写入/读取 {TARGET_DATABASE}")

    mapping = load_run_mapping(args.db_url, args.run_id)
    print(f"run {args.run_id} 映射: {mapping}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render_video(
        args.source_video.resolve(),
        args.tracks_json.resolve(),
        args.output.resolve(),
        mapping,
    )
    print(f"已生成: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
