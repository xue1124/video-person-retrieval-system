#!/usr/bin/env python3
"""将视频分析的结构化结果安全写入正式业务数据库。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.media.time_utils import parse_iso_captured_at  # noqa: E402

from services.analysis.pipeline.reid_match import pack_embedding_matrix


TARGET_DATABASE = "medical_audit_v3"
SYSTEM_DATABASES = {"mysql", "information_schema", "performance_schema", "sys"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="导入轨迹 JSON；只允许写入 medical_audit_v3。"
    )
    parser.add_argument("tracks_json", type=Path)
    parser.add_argument(
        "--embeddings",
        type=Path,
        help="默认使用 tracks.json 同目录下的 track_embeddings.npz",
    )
    parser.add_argument(
        "--db-url",
        default=os.getenv("TRACKING_DB_URL"),
        help="也可通过 TRACKING_DB_URL 提供",
    )
    parser.add_argument("--camera-code")
    parser.add_argument("--camera-name")
    parser.add_argument("--channel-no")
    parser.add_argument("--location")
    parser.add_argument(
        "--camera-source-type",
        choices=("isapi", "rtsp", "other"),
        default="other",
    )
    parser.add_argument(
        "--video-source-type",
        choices=("upload", "isapi", "rtsp", "other"),
        default="upload",
    )
    parser.add_argument(
        "--captured-at",
        help="视频实际开始时间，例如 2026-08-06T10:00:00+08:00",
    )
    parser.add_argument(
        "--video-id",
        type=int,
        default=None,
        help="复用已有 videos.id，不新建视频行（建模任务用）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只校验文件和统计数据，不连接数据库",
    )
    return parser.parse_args()


def parse_datetime(value: str | None) -> datetime | None:
    """内部 ISO / 库值：有时区则转到 UTC，无时区视为已是 UTC。"""
    return parse_iso_captured_at(value)


def source_key(source_video: Path) -> str:
    resolved = source_video.expanduser().resolve()
    if resolved.is_file():
        stat = resolved.stat()
        identity = f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}"
    else:
        identity = str(resolved)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def load_and_validate(
    json_path: Path,
    embeddings_path: Path,
) -> tuple[dict[str, Any], np.lib.npyio.NpzFile]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) < 3:
        raise ValueError("仅支持 schema_version >= 3 的 tracks.json")
    required = {"source_video", "metadata", "persons", "tracks"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"tracks.json 缺少字段: {sorted(missing)}")

    embeddings = np.load(embeddings_path, allow_pickle=False)
    person_ids = {int(person["global_person_id"]) for person in payload["persons"]}
    if len(person_ids) != len(payload["persons"]):
        raise ValueError("persons 中存在重复的 global_person_id")

    track_ids: set[int] = set()
    point_count = 0
    for track in payload["tracks"]:
        track_id = int(track["local_track_id"])
        if track_id in track_ids:
            raise ValueError(f"tracks 中存在重复 local_track_id: {track_id}")
        track_ids.add(track_id)
        if int(track["global_person_id"]) not in person_ids:
            raise ValueError(f"轨迹 T{track_id} 没有对应的视频内人物 P")
        point_count += len(track["points"])
        embedding_key = track.get("embedding_key")
        if embedding_key and embedding_key not in embeddings:
            raise ValueError(f"缺少轨迹特征: {embedding_key}")

    for person in payload["persons"]:
        embedding_key = person.get("embedding_key")
        if embedding_key and embedding_key not in embeddings:
            raise ValueError(f"缺少人物特征: {embedding_key}")

    if "observation_osnet" in embeddings.files:
        n_obs = int(embeddings["observation_osnet"].shape[0])
        for key in ("observation_local_track_id", "observation_frame_index"):
            if key not in embeddings.files:
                raise ValueError(f"缺少观测索引: {key}")
            if int(embeddings[key].shape[0]) != n_obs:
                raise ValueError(f"{key} 行数与 observation_osnet 不一致")
        if "observation_siglip" in embeddings.files:
            if int(embeddings["observation_siglip"].shape[0]) != n_obs:
                raise ValueError("observation_siglip 行数与 observation_osnet 不一致")

    payload["_statistics"] = {
        "person_count": len(person_ids),
        "track_count": len(track_ids),
        "point_count": point_count,
    }
    return payload, embeddings


def embedding_bytes(
    embeddings: np.lib.npyio.NpzFile,
    key: str | None,
) -> tuple[bytes | None, int | None, int]:
    """返回 (blob, dim, sample_count)。人物特征可为 (N, D) 单图矩阵。"""
    if not key:
        return None, None, 0
    matrix = np.asarray(embeddings[key], dtype=np.float32)
    return pack_embedding_matrix(matrix)


def last_insert_id(conn: Any, sql: str, params: dict[str, Any]) -> int:
    conn.execute(text(sql), params)
    return int(conn.execute(text("SELECT LAST_INSERT_ID()")).scalar_one())


def snapshot_root() -> Path:
    raw = os.environ.get("TRACKING_SNAPSHOT_ROOT", "").strip()
    root = Path(raw) if raw else PROJECT_ROOT / "tracking_snapshots"
    root.mkdir(parents=True, exist_ok=True)
    return root.expanduser().resolve()


def observation_feature_map(embeddings: np.lib.npyio.NpzFile) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    if "observation_osnet" not in embeddings.files:
        return {}
    osnet = np.asarray(embeddings["observation_osnet"], dtype=np.float32)
    siglip = (
        np.asarray(embeddings["observation_siglip"], dtype=np.float32)
        if "observation_siglip" in embeddings.files
        else None
    )
    tids = np.asarray(embeddings["observation_local_track_id"])
    frames = np.asarray(embeddings["observation_frame_index"])
    out: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for i in range(len(tids)):
        item: dict[str, np.ndarray] = {"osnet": osnet[i]}
        if siglip is not None:
            item["siglip"] = siglip[i]
        out[(int(tids[i]), int(frames[i]))] = item
    return out


def _copy_crop(src: Path, dest: Path) -> bool:
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return dest.is_file() and dest.stat().st_size > 0
    except OSError:
        return False


def insert_observations_for_track(
    conn: Any,
    *,
    video_id: int,
    processing_run_id: int,
    video_person_id: int,
    global_person_id: int,
    track_id: int,
    local_track_no: int,
    points: list[dict[str, Any]],
    crop_root: Path,
    feat_map: dict[tuple[int, int], dict[str, np.ndarray]],
    osnet_model: str,
    osnet_version: str,
    siglip_model: str,
    siglip_version: str,
) -> int:
    if not points:
        return 0
    point_ids = {
        int(r["frame_index"]): int(r["id"])
        for r in conn.execute(
            text(
                "SELECT id, frame_index FROM track_points WHERE track_id=:tid"
            ),
            {"tid": track_id},
        ).mappings()
    }
    snap = snapshot_root()
    obs_rows: list[dict[str, Any]] = []
    for point in points:
        frame_index = int(point["frame_index"])
        crop_file = point.get("crop_file")
        feats = feat_map.get((local_track_no, frame_index))
        if not crop_file or feats is None or "osnet" not in feats:
            continue
        src = crop_root / str(crop_file)
        if not src.is_file():
            continue
        rel = f"v{video_id}/obs/t{track_id}_f{frame_index}.jpg"
        if not _copy_crop(src, snap / rel):
            continue
        x1, y1, x2, y2 = (float(v) for v in point["bbox"])
        obs_rows.append(
            {
                "video_id": video_id,
                "processing_run_id": processing_run_id,
                "video_person_id": video_person_id,
                "global_person_id": global_person_id,
                "track_id": track_id,
                "track_point_id": point_ids.get(frame_index),
                "timestamp_sec": float(point["timestamp_sec"]),
                "frame_index": frame_index,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "crop_path": rel,
                "osnet": np.asarray(feats["osnet"], dtype=np.float32).reshape(-1),
                "siglip": (
                    np.asarray(feats["siglip"], dtype=np.float32).reshape(-1)
                    if "siglip" in feats
                    else None
                ),
            }
        )
    if not obs_rows:
        return 0
    conn.execute(
        text(
            """
            INSERT INTO observations
              (video_id, processing_run_id, video_person_id, global_person_id,
               track_id, track_point_id, timestamp_sec, frame_index,
               bbox_x1, bbox_y1, bbox_x2, bbox_y2, crop_path)
            VALUES
              (:video_id, :processing_run_id, :video_person_id, :global_person_id,
               :track_id, :track_point_id, :timestamp_sec, :frame_index,
               :x1, :y1, :x2, :y2, :crop_path)
            """
        ),
        [{k: r[k] for k in (
            "video_id", "processing_run_id", "video_person_id", "global_person_id",
            "track_id", "track_point_id", "timestamp_sec", "frame_index",
            "x1", "y1", "x2", "y2", "crop_path",
        )} for r in obs_rows],
    )
    stored = {
        int(r["frame_index"]): int(r["id"])
        for r in conn.execute(
            text(
                """
                SELECT id, frame_index FROM observations
                WHERE processing_run_id=:run AND track_id=:tid
                """
            ),
            {"run": processing_run_id, "tid": track_id},
        ).mappings()
    }
    emb_rows: list[dict[str, Any]] = []
    for r in obs_rows:
        oid = stored.get(int(r["frame_index"]))
        if oid is None:
            continue
        osnet = r["osnet"]
        emb_rows.append(
            {
                "observation_id": oid,
                "modality": "osnet",
                "embedding": osnet.tobytes(),
                "model_name": osnet_model,
                "model_version": osnet_version or None,
                "embedding_dim": int(osnet.shape[0]),
            }
        )
        siglip = r["siglip"]
        if siglip is not None:
            emb_rows.append(
                {
                    "observation_id": oid,
                    "modality": "siglip_image",
                    "embedding": siglip.tobytes(),
                    "model_name": siglip_model or "siglip_vision",
                    "model_version": siglip_version or None,
                    "embedding_dim": int(siglip.shape[0]),
                }
            )
    if emb_rows:
        conn.execute(
            text(
                """
                INSERT INTO observation_embeddings
                  (observation_id, modality, embedding, model_name, model_version, embedding_dim)
                VALUES
                  (:observation_id, :modality, :embedding, :model_name, :model_version, :embedding_dim)
                """
            ),
            emb_rows,
        )
    return len(obs_rows)


def optional_score(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def import_result(
    args: argparse.Namespace,
    payload: dict[str, Any],
    embeddings: np.lib.npyio.NpzFile,
) -> dict[str, int]:
    if not args.db_url:
        raise ValueError("请通过 --db-url 或 TRACKING_DB_URL 提供新数据库地址")
    url = make_url(args.db_url)
    database = (url.database or "").lower()
    if database in SYSTEM_DATABASES or database != TARGET_DATABASE:
        raise ValueError(
            f"安全拒绝：目标必须是 {TARGET_DATABASE}，当前是 {database or '(空)'}"
        )

    captured_at = parse_datetime(args.captured_at)
    json_path = args.tracks_json.resolve()
    metadata = payload["metadata"]
    source_video = Path(payload["source_video"])
    run_key = hashlib.sha256(json_path.read_bytes()).hexdigest()
    osnet_model = str(metadata["osnet_model"])
    source_fps = float(metadata["source_fps"])
    duration_sec = (
        float(metadata["source_total_frames"]) / source_fps if source_fps > 0 else None
    )
    width, height = (int(value) for value in metadata["frame_size"])

    engine = create_engine(args.db_url, pool_pre_ping=True)
    with engine.begin() as conn:
        existing_run = conn.execute(
            text("SELECT id FROM processing_runs WHERE run_key=:run_key"),
            {"run_key": run_key},
        ).scalar_one_or_none()
        if existing_run is not None:
            raise ValueError(f"该结果已经导入，processing_run_id={existing_run}")

        camera_id: int | None = None
        if args.camera_code:
            camera_id = last_insert_id(
                conn,
                """
                INSERT INTO cameras
                  (camera_code, name, channel_no, location, source_type)
                VALUES
                  (:code, :name, :channel_no, :location, :source_type)
                ON DUPLICATE KEY UPDATE
                  id=LAST_INSERT_ID(id),
                  name=VALUES(name),
                  channel_no=VALUES(channel_no),
                  location=VALUES(location),
                  source_type=VALUES(source_type)
                """,
                {
                    "code": args.camera_code,
                    "name": args.camera_name or args.camera_code,
                    "channel_no": args.channel_no,
                    "location": args.location,
                    "source_type": args.camera_source_type,
                },
            )

        reuse_video_id = getattr(args, "video_id", None)
        if reuse_video_id:
            video_id = int(reuse_video_id)
            exists = conn.execute(
                text("SELECT id FROM videos WHERE id=:id"),
                {"id": video_id},
            ).scalar()
            if exists is None:
                raise ValueError(f"videos.id={video_id} 不存在，无法复用")
            conn.execute(
                text(
                    """
                    UPDATE videos
                    SET camera_id = COALESCE(:camera_id, camera_id),
                        source_path = COALESCE(:source_path, source_path),
                        captured_at = COALESCE(:captured_at, captured_at),
                        fps = :fps,
                        duration_sec = :duration_sec,
                        width = :width,
                        height = :height
                    WHERE id = :id
                    """
                ),
                {
                    "camera_id": camera_id,
                    "source_path": str(source_video),
                    "captured_at": captured_at,
                    "fps": source_fps,
                    "duration_sec": duration_sec,
                    "width": width,
                    "height": height,
                    "id": video_id,
                },
            )
        else:
            video_id = last_insert_id(
                conn,
                """
                INSERT INTO videos
                  (source_key, camera_id, file_name, source_type, source_path,
                   captured_at, fps, duration_sec, width, height)
                VALUES
                  (:source_key, :camera_id, :file_name, :source_type, :source_path,
                   :captured_at, :fps, :duration_sec, :width, :height)
                ON DUPLICATE KEY UPDATE
                  id=LAST_INSERT_ID(id),
                  camera_id=COALESCE(VALUES(camera_id), camera_id),
                  captured_at=COALESCE(VALUES(captured_at), captured_at),
                  fps=VALUES(fps),
                  duration_sec=VALUES(duration_sec),
                  width=VALUES(width),
                  height=VALUES(height)
                """,
                {
                    "source_key": source_key(source_video),
                    "camera_id": camera_id,
                    "file_name": source_video.name,
                    "source_type": args.video_source_type,
                    "source_path": str(source_video),
                    "captured_at": captured_at,
                    "fps": source_fps,
                    "duration_sec": duration_sec,
                    "width": width,
                    "height": height,
                },
            )

        processing_run_id = last_insert_id(
            conn,
            """
            INSERT INTO processing_runs
              (run_key, video_id, detector_model, tracker_name, reid_model,
               process_fps, parameters_json, status, completed_at)
            VALUES
              (:run_key, :video_id, :detector_model, :tracker_name, :reid_model,
               :process_fps, :parameters_json, 'completed', NOW(3))
            """,
            {
                "run_key": run_key,
                "video_id": video_id,
                "detector_model": str(metadata["detector_model"]),
                "tracker_name": str(metadata.get("tracker_name") or "BoT-SORT"),
                "reid_model": osnet_model,
                "process_fps": float(metadata["effective_fps"]),
                "parameters_json": json.dumps(metadata, ensure_ascii=False),
            },
        )

        video_person_ids: dict[int, int] = {}
        global_person_ids: dict[int, int] = {}
        for person in payload["persons"]:
            local_person_no = int(person["global_person_id"])
            feature, feature_dim, sample_count = embedding_bytes(
                embeddings, person.get("embedding_key")
            )
            first_seen = (
                captured_at + timedelta(seconds=float(person["start_sec"]))
                if captured_at
                else None
            )
            last_seen = (
                captured_at + timedelta(seconds=float(person["end_sec"]))
                if captured_at
                else None
            )
            stored_samples = int(
                person.get("sample_count")
                or sample_count
                or person["profile_updates"]
            )
            global_person_id = last_insert_id(
                conn,
                """
                INSERT INTO global_people
                  (representative_embedding, embedding_model, embedding_dim,
                   sample_count, first_seen_at, last_seen_at)
                VALUES
                  (:embedding, :model, :dim, :samples, :first_seen, :last_seen)
                """,
                {
                    "embedding": feature,
                    "model": osnet_model,
                    "dim": feature_dim,
                    "samples": stored_samples,
                    "first_seen": first_seen,
                    "last_seen": last_seen,
                },
            )
            video_person_id = last_insert_id(
                conn,
                """
                INSERT INTO video_people
                  (processing_run_id, global_person_id, local_person_no,
                   start_sec, end_sec, representative_embedding,
                   embedding_model, embedding_dim, profile_updates,
                   assignment_method)
                VALUES
                  (:run_id, :global_id, :local_no, :start_sec, :end_sec,
                   :embedding, :model, :dim, :updates, 'new')
                """,
                {
                    "run_id": processing_run_id,
                    "global_id": global_person_id,
                    "local_no": local_person_no,
                    "start_sec": float(person["start_sec"]),
                    "end_sec": float(person["end_sec"]),
                    "embedding": feature,
                    "model": osnet_model,
                    "dim": feature_dim,
                    "updates": int(person["profile_updates"]),
                },
            )
            conn.execute(
                text(
                    """
                    INSERT INTO person_identity_assignments
                      (video_person_id, global_person_id, method, note)
                    VALUES
                      (:video_person_id, :global_person_id, 'new',
                       '首次导入，尚未执行跨视频关联')
                    """
                ),
                {
                    "video_person_id": video_person_id,
                    "global_person_id": global_person_id,
                },
            )
            video_person_ids[local_person_no] = video_person_id
            global_person_ids[local_person_no] = global_person_id

        feat_map = observation_feature_map(embeddings)
        crop_root = json_path.parent
        osnet_version = str(metadata.get("osnet_provider") or "")
        siglip_model = str(metadata.get("siglip_model") or "siglip_vision")
        siglip_version = str(metadata.get("siglip_model_version") or "")
        inserted_observations = 0

        inserted_points = 0
        for track in payload["tracks"]:
            local_track_no = int(track["local_track_id"])
            local_person_no = int(track["global_person_id"])
            feature, feature_dim, _sample_count = embedding_bytes(
                embeddings, track.get("embedding_key")
            )
            track_id = last_insert_id(
                conn,
                """
                INSERT INTO tracks
                  (processing_run_id, video_person_id, local_track_no,
                   start_sec, end_sec, observation_count,
                   representative_embedding, embedding_model, embedding_dim,
                   online_match_score)
                VALUES
                  (:run_id, :video_person_id, :local_no, :start_sec, :end_sec,
                   :observations, :embedding, :model, :dim, :match_score)
                """,
                {
                    "run_id": processing_run_id,
                    "video_person_id": video_person_ids[local_person_no],
                    "local_no": local_track_no,
                    "start_sec": float(track["start_sec"]),
                    "end_sec": float(track["end_sec"]),
                    "observations": int(track["observation_count"]),
                    "embedding": feature,
                    "model": osnet_model,
                    "dim": feature_dim,
                    "match_score": track.get("global_match_similarity"),
                },
            )

            point_rows: list[dict[str, Any]] = []
            for point in track["points"]:
                x1, y1, x2, y2 = (float(value) for value in point["bbox"])
                timestamp_sec = float(point["timestamp_sec"])
                point_rows.append(
                    {
                        "track_id": track_id,
                        "frame_index": int(point["frame_index"]),
                        "timestamp_sec": timestamp_sec,
                        "occurred_at": (
                            captured_at + timedelta(seconds=timestamp_sec)
                            if captured_at
                            else None
                        ),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "foot_x": (x1 + x2) / 2.0,
                        "foot_y": y2,
                        "confidence": float(point["confidence"]),
                        "pre_merge_id": optional_int(point.get("pre_merge_id")),
                        "assign_score": optional_score(point.get("assign_score")),
                        "merge_score": optional_score(point.get("merge_score")),
                    }
                )
            if point_rows:
                conn.execute(
                    text(
                        """
                        INSERT INTO track_points
                          (track_id, frame_index, timestamp_sec, occurred_at,
                           bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                           foot_x, foot_y, confidence,
                           pre_merge_id, assign_score, merge_score)
                        VALUES
                          (:track_id, :frame_index, :timestamp_sec, :occurred_at,
                           :x1, :y1, :x2, :y2, :foot_x, :foot_y, :confidence,
                           :pre_merge_id, :assign_score, :merge_score)
                        """
                    ),
                    point_rows,
                )
                inserted_points += len(point_rows)
            inserted_observations += insert_observations_for_track(
                conn,
                video_id=video_id,
                processing_run_id=processing_run_id,
                video_person_id=video_person_ids[local_person_no],
                global_person_id=global_person_ids[local_person_no],
                track_id=track_id,
                local_track_no=local_track_no,
                points=track["points"],
                crop_root=crop_root,
                feat_map=feat_map,
                osnet_model=osnet_model,
                osnet_version=osnet_version,
                siglip_model=siglip_model,
                siglip_version=siglip_version,
            )

    engine.dispose()
    return {
        "video_id": video_id,
        "processing_run_id": processing_run_id,
        "person_count": len(video_person_ids),
        "track_count": len(payload["tracks"]),
        "point_count": inserted_points,
        "observation_count": inserted_observations,
    }


def main() -> int:
    args = parse_args()
    json_path = args.tracks_json.expanduser().resolve()
    embeddings_path = (
        args.embeddings.expanduser().resolve()
        if args.embeddings
        else json_path.with_name("track_embeddings.npz")
    )
    if not json_path.is_file():
        raise FileNotFoundError(f"轨迹文件不存在: {json_path}")
    if not embeddings_path.is_file():
        raise FileNotFoundError(f"特征文件不存在: {embeddings_path}")

    args.tracks_json = json_path
    payload, embeddings = load_and_validate(json_path, embeddings_path)
    stats = payload["_statistics"]
    print(
        f"校验通过：{stats['person_count']} 个人物，"
        f"{stats['track_count']} 条轨迹，{stats['point_count']} 个轨迹点"
    )
    if args.dry_run:
        return 0

    result = import_result(args, payload, embeddings)
    print(
        f"导入完成：video_id={result['video_id']}，"
        f"processing_run_id={result['processing_run_id']}，"
        f"{result['person_count']} 个人物，{result['track_count']} 条轨迹，"
        f"{result['point_count']} 个轨迹点"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
