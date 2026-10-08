#!/usr/bin/env python3
"""从原视频按轨迹点生成人物代表 crop / 停留现场图。"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.engine import Engine, make_url

TARGET_DATABASE = "medical_audit_v3"
SYSTEM_DATABASES = {"mysql", "information_schema", "performance_schema", "sys"}
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOT_ROOT = PROJECT_ROOT / "tracking_snapshots"


def configured_snapshot_root() -> Path:
    """Return the shared snapshot root used by both writers and the API."""
    configured = os.environ.get("TRACKING_SNAPSHOT_ROOT", "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_SNAPSHOT_ROOT


@dataclass
class CropJob:
    frame_index: int
    bbox: tuple[float, float, float, float]
    quality_score: float
    timestamp_sec: float
    global_person_id: int | None
    video_person_id: int
    track_id: int | None
    track_point_id: int | None
    stay_segment_id: int | None
    video_id: int
    snapshot_type: str  # candidate | stay
    rel_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为 medical_audit_v3 生成人物快照 crop")
    parser.add_argument(
        "--db-url",
        default=os.getenv("TRACKING_DB_URL"),
        help="也可通过 TRACKING_DB_URL 提供",
    )
    parser.add_argument("--video-id", type=int, help="只处理指定视频；默认全部")
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=configured_snapshot_root(),
        help="快照根目录；默认读取 TRACKING_SNAPSHOT_ROOT",
    )
    parser.add_argument("--candidates-per-person", type=int, default=4)
    parser.add_argument("--min-gap-sec", type=float, default=2.0)
    parser.add_argument("--pad-ratio", type=float, default=0.08)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计将生成的数量，不写盘不入库",
    )
    return parser.parse_args()


def require_db_url(url: str | None) -> str:
    if not url:
        raise ValueError("请通过 --db-url 或 TRACKING_DB_URL 提供数据库地址")
    database = (make_url(url).database or "").lower()
    if database in SYSTEM_DATABASES or database != TARGET_DATABASE:
        raise ValueError(f"目标必须是 {TARGET_DATABASE}，当前是 {database or '(空)'}")
    return url


def quality_score(
    confidence: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_w: int,
    frame_h: int,
) -> float:
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    area = bw * bh
    frame_area = max(1, frame_w * frame_h)
    area_ratio = area / frame_area
    size_score = min(1.0, area_ratio / 0.035)
    margin_x = 0.03 * frame_w
    margin_y = 0.03 * frame_h
    edge = 0.65 if (
        x1 < margin_x
        or y1 < margin_y
        or x2 > frame_w - margin_x
        or y2 > frame_h - margin_y
    ) else 1.0
    height_bonus = 1.0 + min(bh, 280.0) / 280.0 * 0.25
    return float(confidence) * size_score * edge * height_bonus


def pick_diverse(
    scored: list[dict[str, Any]],
    limit: int,
    min_gap_sec: float,
) -> list[dict[str, Any]]:
    if limit <= 0 or not scored:
        return []
    ranked = sorted(scored, key=lambda row: (-row["quality"], row["timestamp_sec"]))
    chosen: list[dict[str, Any]] = []
    for row in ranked:
        if any(abs(row["timestamp_sec"] - c["timestamp_sec"]) < min_gap_sec for c in chosen):
            continue
        chosen.append(row)
        if len(chosen) >= limit:
            break
    chosen.sort(key=lambda row: row["timestamp_sec"])
    return chosen


def padded_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_w: int,
    frame_h: int,
    pad_ratio: float,
) -> tuple[int, int, int, int]:
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    px = bw * pad_ratio
    py = bh * pad_ratio
    left = int(max(0, np.floor(x1 - px)))
    top = int(max(0, np.floor(y1 - py)))
    right = int(min(frame_w, np.ceil(x2 + px)))
    bottom = int(min(frame_h, np.ceil(y2 + py)))
    if right <= left:
        right = min(frame_w, left + 1)
    if bottom <= top:
        bottom = min(frame_h, top + 1)
    return left, top, right, bottom


def list_video_snapshot_rows(conn: Any, video_id: int) -> list[dict[str, Any]]:
    return list(
        conn.execute(
            text("SELECT id, image_path FROM person_snapshots WHERE video_id=:vid"),
            {"vid": video_id},
        ).mappings()
    )


def delete_snapshot_rows(
    conn: Any,
    snapshot_ids: list[int],
    snapshot_root: Path,
    *,
    keep_paths: set[str],
) -> int:
    if not snapshot_ids:
        return 0
    stmt = text(
        "SELECT id, image_path FROM person_snapshots WHERE id IN :ids"
    ).bindparams(bindparam("ids", expanding=True))
    rows = conn.execute(stmt, {"ids": snapshot_ids}).mappings().all()
    conn.execute(
        text("DELETE FROM person_snapshots WHERE id IN :ids").bindparams(
            bindparam("ids", expanding=True)
        ),
        {"ids": snapshot_ids},
    )
    for row in rows:
        rel = str(row["image_path"] or "")
        if rel in keep_paths:
            continue
        path = snapshot_root / rel
        if path.is_file():
            try:
                path.unlink()
            except OSError:
                pass
    return len(rows)


def delete_video_snapshots(conn: Any, video_id: int, snapshot_root: Path) -> int:
    rows = list_video_snapshot_rows(conn, video_id)
    return delete_snapshot_rows(
        conn,
        [int(row["id"]) for row in rows],
        snapshot_root,
        keep_paths=set(),
    )


def live_stay_segment_ids(conn: Any, video_id: int) -> set[int]:
    rows = conn.execute(
        text(
            """
            SELECT ss.id
            FROM stay_segments ss
            JOIN video_people vp ON vp.id = ss.video_person_id
            JOIN processing_runs pr ON pr.id = vp.processing_run_id
            WHERE pr.video_id = :vid
            """
        ),
        {"vid": video_id},
    ).scalars()
    return {int(value) for value in rows}


def load_video_meta(conn: Any, video_id: int | None) -> list[dict[str, Any]]:
    sql = """
        SELECT id, file_name, source_path, width, height, fps
        FROM videos
    """
    params: dict[str, Any] = {}
    if video_id is not None:
        sql += " WHERE id = :video_id"
        params["video_id"] = video_id
    sql += " ORDER BY id"
    return list(conn.execute(text(sql), params).mappings())


def build_jobs_for_video(
    conn: Any,
    video: dict[str, Any],
    *,
    candidates_per_person: int,
    min_gap_sec: float,
) -> list[CropJob]:
    video_id = int(video["id"])
    frame_w = int(video["width"] or 0)
    frame_h = int(video["height"] or 0)
    if frame_w <= 0 or frame_h <= 0:
        # 从任意点推不了宽高时，用 bbox 最大值兜底
        dims = conn.execute(
            text(
                """
                SELECT MAX(bbox_x2) AS mw, MAX(bbox_y2) AS mh
                FROM track_points tp
                JOIN tracks t ON t.id = tp.track_id
                JOIN video_people vp ON vp.id = t.video_person_id
                JOIN processing_runs pr ON pr.id = vp.processing_run_id
                WHERE pr.video_id = :vid
                """
            ),
            {"vid": video_id},
        ).mappings().first()
        frame_w = int((dims["mw"] or 1280) + 1) if dims else 1280
        frame_h = int((dims["mh"] or 720) + 1) if dims else 720

    points = conn.execute(
        text(
            """
            SELECT
              tp.id AS track_point_id,
              tp.track_id,
              tp.frame_index,
              tp.timestamp_sec,
              tp.bbox_x1, tp.bbox_y1, tp.bbox_x2, tp.bbox_y2,
              tp.confidence,
              vp.id AS video_person_id,
              vp.global_person_id,
              vp.local_person_no
            FROM track_points tp
            JOIN tracks t ON t.id = tp.track_id
            JOIN video_people vp ON vp.id = t.video_person_id
            JOIN processing_runs pr ON pr.id = vp.processing_run_id
            WHERE pr.video_id = :vid
            ORDER BY vp.id, tp.timestamp_sec, tp.id
            """
        ),
        {"vid": video_id},
    ).mappings().all()

    by_person: dict[int, list[dict[str, Any]]] = {}
    for row in points:
        scored = quality_score(
            float(row["confidence"]),
            float(row["bbox_x1"]),
            float(row["bbox_y1"]),
            float(row["bbox_x2"]),
            float(row["bbox_y2"]),
            frame_w,
            frame_h,
        )
        by_person.setdefault(int(row["video_person_id"]), []).append(
            {
                **dict(row),
                "quality": scored,
            }
        )

    jobs: list[CropJob] = []
    for video_person_id, rows in by_person.items():
        selected = pick_diverse(rows, candidates_per_person, min_gap_sec)
        for idx, row in enumerate(selected):
            jobs.append(
                CropJob(
                    frame_index=int(row["frame_index"]),
                    bbox=(
                        float(row["bbox_x1"]),
                        float(row["bbox_y1"]),
                        float(row["bbox_x2"]),
                        float(row["bbox_y2"]),
                    ),
                    quality_score=float(row["quality"]),
                    timestamp_sec=float(row["timestamp_sec"]),
                    global_person_id=(
                        int(row["global_person_id"])
                        if row["global_person_id"] is not None
                        else None
                    ),
                    video_person_id=video_person_id,
                    track_id=int(row["track_id"]),
                    track_point_id=int(row["track_point_id"]),
                    stay_segment_id=None,
                    video_id=video_id,
                    snapshot_type="candidate",
                    rel_name=(
                        f"v{video_id}/vp{video_person_id}_cand{idx}_"
                        f"tp{int(row['track_point_id'])}.jpg"
                    ),
                )
            )

    stays = conn.execute(
        text(
            """
            SELECT
              ss.id AS stay_segment_id,
              ss.start_sec,
              ss.end_sec,
              vp.id AS video_person_id,
              vp.global_person_id
            FROM stay_segments ss
            JOIN video_people vp ON vp.id = ss.video_person_id
            JOIN processing_runs pr ON pr.id = vp.processing_run_id
            WHERE pr.video_id = :vid
            ORDER BY ss.id
            """
        ),
        {"vid": video_id},
    ).mappings().all()

    for stay in stays:
        vp_id = int(stay["video_person_id"])
        start_sec = float(stay["start_sec"])
        end_sec = float(stay["end_sec"])
        candidates = [
            row
            for row in by_person.get(vp_id, [])
            if start_sec - 0.05 <= float(row["timestamp_sec"]) <= end_sec + 0.05
        ]
        if not candidates:
            continue
        best = max(candidates, key=lambda row: row["quality"])
        jobs.append(
            CropJob(
                frame_index=int(best["frame_index"]),
                bbox=(
                    float(best["bbox_x1"]),
                    float(best["bbox_y1"]),
                    float(best["bbox_x2"]),
                    float(best["bbox_y2"]),
                ),
                quality_score=float(best["quality"]),
                timestamp_sec=float(best["timestamp_sec"]),
                global_person_id=(
                    int(stay["global_person_id"])
                    if stay["global_person_id"] is not None
                    else None
                ),
                video_person_id=vp_id,
                track_id=int(best["track_id"]),
                track_point_id=int(best["track_point_id"]),
                stay_segment_id=int(stay["stay_segment_id"]),
                video_id=video_id,
                snapshot_type="stay",
                rel_name=(
                    f"v{video_id}/stay{int(stay['stay_segment_id'])}_"
                    f"tp{int(best['track_point_id'])}.jpg"
                ),
            )
        )

    return jobs


def _reuse_existing_crop(job: CropJob, snapshot_root: Path) -> bool:
    dest = snapshot_root / job.rel_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return True
    if job.track_point_id is None:
        return False
    folder = snapshot_root / f"v{job.video_id}"
    if not folder.is_dir():
        return False
    needle = f"tp{int(job.track_point_id)}.jpg"
    for existing in folder.glob(f"*{needle}"):
        if not existing.is_file() or existing.stat().st_size <= 0:
            continue
        if existing.resolve() == dest.resolve():
            return True
        try:
            shutil.copy2(existing, dest)
        except OSError:
            continue
        return dest.is_file() and dest.stat().st_size > 0
    return False


def _split_reuse_jobs(
    jobs: list[CropJob],
    snapshot_root: Path,
) -> tuple[list[CropJob], list[CropJob]]:
    reused: list[CropJob] = []
    pending: list[CropJob] = []
    for job in jobs:
        if _reuse_existing_crop(job, snapshot_root):
            reused.append(job)
        else:
            pending.append(job)
    return reused, pending


def _crop_jobs_from_frame(
    frame: Any,
    jobs: list[CropJob],
    snapshot_root: Path,
    *,
    frame_w: int,
    frame_h: int,
    pad_ratio: float,
) -> list[CropJob]:
    saved: list[CropJob] = []
    h, w = frame.shape[:2]
    use_w = frame_w if frame_w > 0 else w
    use_h = frame_h if frame_h > 0 else h
    for job in jobs:
        x1, y1, x2, y2 = padded_bbox(*job.bbox, use_w, use_h, pad_ratio)
        x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
        x2, y2 = max(x1 + 1, min(x2, w)), max(y1 + 1, min(y2, h))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        out_path = snapshot_root / job.rel_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_path), crop):
            continue
        if not out_path.is_file() or out_path.stat().st_size <= 0:
            continue
        saved.append(job)
    return saved


def extract_pending_frames(
    source_path: Path,
    pending: list[CropJob],
    snapshot_root: Path,
    *,
    frame_w: int,
    frame_h: int,
    pad_ratio: float,
) -> list[CropJob]:
    """只解码需要的帧：大跨度用 seek，避免把整段长视频读一遍。"""
    if not pending:
        return []
    if not source_path.is_file():
        raise FileNotFoundError(f"找不到源视频: {source_path}")

    by_frame: dict[int, list[CropJob]] = {}
    for job in pending:
        by_frame.setdefault(job.frame_index, []).append(job)
    targets = sorted(by_frame)

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {source_path}")

    saved: list[CropJob] = []
    current = -1
    seek_gap = 8
    try:
        for idx, target in enumerate(targets, start=1):
            if target < current or target - current > seek_gap:
                capture.set(cv2.CAP_PROP_POS_FRAMES, float(max(target, 0)))
                current = target - 1
            frame = None
            while current < target:
                ok, frame = capture.read()
                if not ok:
                    frame = None
                    break
                current += 1
            if frame is None:
                continue
            saved.extend(
                _crop_jobs_from_frame(
                    frame,
                    by_frame[target],
                    snapshot_root,
                    frame_w=frame_w,
                    frame_h=frame_h,
                    pad_ratio=pad_ratio,
                )
            )
            if idx == 1 or idx == len(targets) or idx % 25 == 0:
                print(
                    f"[snapshots] 切图 {idx}/{len(targets)} frame={target}",
                    flush=True,
                )
    finally:
        capture.release()
    return saved


def extract_and_save(
    source_path: Path,
    jobs: list[CropJob],
    snapshot_root: Path,
    *,
    frame_w: int,
    frame_h: int,
    pad_ratio: float,
) -> list[CropJob]:
    if not jobs:
        return []
    reused, pending = _split_reuse_jobs(jobs, snapshot_root)
    extracted = extract_pending_frames(
        source_path,
        pending,
        snapshot_root,
        frame_w=frame_w,
        frame_h=frame_h,
        pad_ratio=pad_ratio,
    )
    return reused + extracted


def insert_snapshots(conn: Any, jobs: list[CropJob]) -> int:
    if not jobs:
        return 0
    rows = [
        {
            "global_person_id": job.global_person_id,
            "video_person_id": job.video_person_id,
            "track_id": job.track_id,
            "track_point_id": job.track_point_id,
            "stay_segment_id": job.stay_segment_id,
            "video_id": job.video_id,
            "image_path": job.rel_name,
            "timestamp_sec": job.timestamp_sec,
            "quality_score": job.quality_score,
            "snapshot_type": job.snapshot_type,
            "is_primary": False,
            "bbox_x1": job.bbox[0],
            "bbox_y1": job.bbox[1],
            "bbox_x2": job.bbox[2],
            "bbox_y2": job.bbox[3],
        }
        for job in jobs
    ]
    conn.execute(
        text(
            """
            INSERT INTO person_snapshots
              (global_person_id, video_person_id, track_id, track_point_id,
               stay_segment_id, video_id, image_path, timestamp_sec,
               quality_score, snapshot_type, is_primary,
               bbox_x1, bbox_y1, bbox_x2, bbox_y2)
            VALUES
              (:global_person_id, :video_person_id, :track_id, :track_point_id,
               :stay_segment_id, :video_id, :image_path, :timestamp_sec,
               :quality_score, :snapshot_type, :is_primary,
               :bbox_x1, :bbox_y1, :bbox_x2, :bbox_y2)
            """
        ),
        rows,
    )
    return len(rows)


def insert_snapshots_safe(conn: Any, jobs: list[CropJob], video_id: int) -> tuple[int, int]:
    """候选图和停留图分开插入；过期停留段只跳过现场图，不影响头像。"""
    candidates = [job for job in jobs if job.snapshot_type == "candidate"]
    stay_jobs = [job for job in jobs if job.snapshot_type == "stay"]
    inserted_candidates = insert_snapshots(conn, candidates)
    live_stays = live_stay_segment_ids(conn, video_id)
    valid_stays = [
        job
        for job in stay_jobs
        if job.stay_segment_id is not None and int(job.stay_segment_id) in live_stays
    ]
    skipped = len(stay_jobs) - len(valid_stays)
    if skipped:
        print(
            f"[snapshots] 跳过 {skipped} 张过期停留图，保留 {len(valid_stays)} 张",
            flush=True,
        )
    inserted_stays = insert_snapshots(conn, valid_stays)
    return inserted_candidates, inserted_stays


def refresh_primary_flags(conn: Any, global_person_ids: set[int]) -> None:
    if not global_person_ids:
        return
    for gid in sorted(global_person_ids):
        conn.execute(
            text(
                """
                UPDATE person_snapshots
                SET is_primary = FALSE
                WHERE global_person_id = :gid
                """
            ),
            {"gid": gid},
        )
        best_id = conn.execute(
            text(
                """
                SELECT id
                FROM person_snapshots
                WHERE global_person_id = :gid
                  AND snapshot_type = 'candidate'
                ORDER BY quality_score DESC, id ASC
                LIMIT 1
                """
            ),
            {"gid": gid},
        ).scalar_one_or_none()
        if best_id is not None:
            conn.execute(
                text(
                    """
                    UPDATE person_snapshots
                    SET is_primary = TRUE
                    WHERE id = :id
                    """
                ),
                {"id": int(best_id)},
            )


def process_video(
    engine: Engine,
    video: dict[str, Any],
    *,
    snapshot_root: Path,
    candidates_per_person: int,
    min_gap_sec: float,
    pad_ratio: float,
    dry_run: bool,
) -> dict[str, Any]:
    video_id = int(video["id"])
    source_path = Path(str(video["source_path"] or ""))
    with engine.connect() as conn:
        old_rows = list_video_snapshot_rows(conn, video_id)
        jobs = build_jobs_for_video(
            conn,
            video,
            candidates_per_person=candidates_per_person,
            min_gap_sec=min_gap_sec,
        )
    old_ids = [int(row["id"]) for row in old_rows]
    if dry_run:
        return {
            "video_id": video_id,
            "file_name": video["file_name"],
            "planned": len(jobs),
            "candidates": sum(1 for j in jobs if j.snapshot_type == "candidate"),
            "stays": sum(1 for j in jobs if j.snapshot_type == "stay"),
            "dry_run": True,
        }

    frame_w = int(video["width"] or 0)
    frame_h = int(video["height"] or 0)
    reused, pending = _split_reuse_jobs(jobs, snapshot_root)
    inserted_candidates = 0
    inserted_stays = 0
    keep_paths = {job.rel_name for job in reused}

    if reused:
        with engine.begin() as conn:
            inserted_candidates, inserted_stays = insert_snapshots_safe(
                conn, reused, video_id
            )
            print(
                f"[snapshots] 先入库可复用切图 candidates={inserted_candidates} stays={inserted_stays}",
                flush=True,
            )

    extracted = extract_pending_frames(
        source_path,
        pending,
        snapshot_root,
        frame_w=frame_w,
        frame_h=frame_h,
        pad_ratio=pad_ratio,
    )
    keep_paths.update(job.rel_name for job in extracted)
    saved = reused + extracted

    with engine.begin() as conn:
        if extracted:
            extra_c, extra_s = insert_snapshots_safe(conn, extracted, video_id)
            inserted_candidates += extra_c
            inserted_stays += extra_s
        inserted = inserted_candidates + inserted_stays
        if old_ids and inserted:
            delete_snapshot_rows(
                conn,
                old_ids,
                snapshot_root,
                keep_paths=keep_paths,
            )
        gids = {
            int(row["global_person_id"])
            for row in conn.execute(
                text(
                    """
                    SELECT DISTINCT global_person_id
                    FROM video_people vp
                    JOIN processing_runs pr ON pr.id = vp.processing_run_id
                    WHERE pr.video_id = :vid
                      AND vp.global_person_id IS NOT NULL
                    """
                ),
                {"vid": video_id},
            ).mappings()
            if row["global_person_id"] is not None
        }
        refresh_primary_flags(conn, gids)

    return {
        "video_id": video_id,
        "file_name": video["file_name"],
        "deleted": len(old_ids) if inserted else 0,
        "planned": len(jobs),
        "saved": len(saved),
        "inserted": inserted,
        "candidates": inserted_candidates,
        "stays": inserted_stays,
    }


def generate_snapshots_for_video(
    db_url: str,
    video_id: int,
    *,
    snapshot_root: Path | None = None,
    candidates_per_person: int = 4,
    min_gap_sec: float = 2.0,
    pad_ratio: float = 0.08,
) -> dict[str, Any]:
    """供正式分析工作流或其他模块调用。"""
    url = require_db_url(db_url)
    root = (snapshot_root or configured_snapshot_root()).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            videos = load_video_meta(conn, video_id)
        if not videos:
            return {"video_id": video_id, "skipped": "video_not_found"}
        return process_video(
            engine,
            dict(videos[0]),
            snapshot_root=root,
            candidates_per_person=candidates_per_person,
            min_gap_sec=min_gap_sec,
            pad_ratio=pad_ratio,
            dry_run=False,
        )
    finally:
        engine.dispose()


def main() -> int:
    args = parse_args()
    url = require_db_url(args.db_url)
    root = args.snapshot_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            videos = load_video_meta(conn, args.video_id)
        if not videos:
            print("没有可处理的视频")
            return 0
        for video in videos:
            report = process_video(
                engine,
                dict(video),
                snapshot_root=root,
                candidates_per_person=args.candidates_per_person,
                min_gap_sec=args.min_gap_sec,
                pad_ratio=args.pad_ratio,
                dry_run=args.dry_run,
            )
            print(report, flush=True)
    finally:
        engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
