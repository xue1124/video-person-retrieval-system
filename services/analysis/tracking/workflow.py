#!/usr/bin/env python3
"""编排视频检测、人物聚类、入库、停留计算和检索索引更新。

上传任务跑 YOLO + OSNet 聚类（身份合并只用 OSNet），同时提取 SigLIP 图像向量写入 observations。
不再写旧 gallery_meta。
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from services.config import PROJECT_ROOT, TENSORRT_MODEL_DIR

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "run" / "analysis"


def _uuid_suffix() -> str:
    return uuid.uuid4().hex[:8]


def analysis_enabled() -> bool:
    return os.environ.get("TRACKING_V3_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def tracking_db_url() -> str:
    url = os.environ.get("TRACKING_DB_URL", "").strip()
    if not url:
        raise ValueError("已启用 TRACKING_V3_ENABLED，但未设置 TRACKING_DB_URL")
    database = (make_url(url).database or "").lower()
    if database != "medical_audit_v3":
        raise ValueError(
            f"TRACKING_DB_URL 必须指向 medical_audit_v3，当前是 {database or '(空)'}"
        )
    return url


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _safe_stem(video_name: str) -> str:
    stem = Path(str(video_name)).stem
    cleaned = re.sub(r"[^\w\-.]+", "_", stem, flags=re.UNICODE).strip("._")
    return cleaned or "video"


def _infer_source_type(raw_path: str | None) -> str:
    text_path = str(raw_path or "")
    if text_path.startswith("isapi://"):
        return "isapi"
    if text_path.startswith("rtsp://"):
        return "rtsp"
    return "upload"


def _infer_camera(video_name: str, source_type: str) -> dict[str, Any]:
    """尽量从文件名提取通道号；提取不到则不写 cameras。"""
    match = re.search(r"通道\s*0*(\d+)", str(video_name))
    if not match:
        return {}
    channel = match.group(1)
    return {
        "camera_code": f"channel-{channel}",
        "camera_name": f"通道{channel}",
        "channel_no": channel,
        "camera_source_type": source_type if source_type in {"isapi", "rtsp"} else "other",
    }


def _captured_at_from_video_row(
    db_url: str, video_name: str, video_id: int | None
) -> str | None:
    from services.media.time_utils import db_datetime_to_utc_iso

    engine = create_engine(db_url, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            if video_id is not None:
                row = conn.execute(
                    text("SELECT captured_at FROM videos WHERE id=:id"),
                    {"id": video_id},
                ).first()
            else:
                row = conn.execute(
                    text(
                        """
                        SELECT captured_at FROM videos
                        WHERE file_name=:n
                        ORDER BY is_media_source DESC, id DESC
                        LIMIT 1
                        """
                    ),
                    {"n": video_name},
                ).first()
        return db_datetime_to_utc_iso(row[0] if row else None)
    finally:
        engine.dispose()


def _run_tracking(
    video_path: Path,
    output_dir: Path,
    progress_callback: Any | None = None,
    cancel_check: Any | None = None,
    *,
    skip_frames: int | None = None,
    conf_val: float | None = None,
) -> Path:
    """进程内常驻 YOLO(TRT) + OSNet(TRT) 无跟踪聚类。"""
    from services.analysis.tracking.engines import get_tracking_engines, reset_encoder_stats
    from services.analysis.tracking.siglip_vision import get_siglip_encoder
    from services.analysis.pipeline import detect_reid_video as reid_mod

    engines = get_tracking_engines()
    reset_encoder_stats()
    siglip_encoder = get_siglip_encoder()

    model = os.environ.get(
        "TRACKING_V3_YOLO_MODEL",
        str(TENSORRT_MODEL_DIR / "yolo11m.engine"),
    )
    process_fps = float(os.environ.get("TRACKING_V3_PROCESS_FPS", "12") or 12)
    imgsz = int(os.environ.get("TRACKING_V3_IMGSZ", "640") or 640)
    conf = float(conf_val) if conf_val is not None else float(
        os.environ.get("TRACKING_V3_CONF", "0.60") or 0.60
    )
    crop_batch_size = int(
        os.environ.get("TRACKING_V3_CROP_BATCH_SIZE", "128") or 128
    )
    neighbor_k = int(os.environ.get("TRACKING_V3_REID_NEIGHBOR_K", "256") or 256)
    max_frames_raw = os.environ.get("TRACKING_V3_MAX_FRAMES", "").strip()
    max_frames = int(max_frames_raw) if max_frames_raw and int(max_frames_raw) > 0 else 0
    frame_step = max(1, int(skip_frames)) if skip_frames is not None else None

    print(
        f"[tracking_v3] 无跟踪人物聚类: video={video_path.name} | "
        f"yolo={Path(model).name} | imgsz={imgsz} | "
        f"skip_frames={frame_step or 'env_process_fps'} | "
        f"process_fps_fallback={process_fps} | conf={conf} | "
        f"crop_batch={crop_batch_size}",
        flush=True,
    )

    def _progress(ratio: float, person_count: int | None = None) -> None:
        if cancel_check is not None and not cancel_check():
            raise reid_mod.TrackingCancelled("cancelled during progress callback")
        if progress_callback is None:
            return
        try:
            progress_callback(ratio, person_count)
        except TypeError:
            progress_callback(ratio)

    tracks_json, _ = reid_mod.run_detection_reid(
        video_path,
        output_dir=output_dir,
        detector=engines["detector"],
        encoder=engines["encoder"],
        infer_lock=engines["lock"],
        detector_model=model,
        osnet_model=os.environ.get(
            "TRACKING_V3_OSNET_ENGINE",
            os.environ.get("TRACKING_V3_OSNET_MODEL", "OSNet"),
        ),
        process_fps=process_fps,
        frame_step=frame_step,
        imgsz=imgsz,
        conf=conf,
        max_frames=max_frames,
        crop_batch_size=crop_batch_size,
        similarity_threshold=_env_float("TRACKING_V3_GLOBAL_MATCH_THRESH", 0.70),
        max_samples=int(_env_float("TRACKING_V3_REID_MAX_SAMPLES", 15) or 15),
        top_k=int(_env_float("TRACKING_V3_REID_TOP_K", 3) or 3),
        min_pairs=int(_env_float("TRACKING_V3_REID_MIN_PAIRS", 2) or 2),
        neighbor_k=neighbor_k,
        progress_callback=_progress,
        cancel_check=cancel_check,
        siglip_encoder=siglip_encoder,
    )
    if not tracks_json.is_file():
        raise FileNotFoundError(f"跟踪未产出 tracks.json: {tracks_json}")
    return tracks_json


def _import_tracks(
    tracks_json: Path,
    *,
    db_url: str,
    video_source_type: str,
    camera_meta: dict[str, Any],
    captured_at: str | None,
    video_id: int | None = None,
) -> dict[str, int]:
    from services.analysis.pipeline import persist_results as importer

    embeddings = tracks_json.with_name("track_embeddings.npz")
    payload, emb = importer.load_and_validate(tracks_json, embeddings)
    args = SimpleNamespace(
        db_url=db_url,
        tracks_json=tracks_json,
        captured_at=captured_at,
        video_source_type=video_source_type,
        camera_code=camera_meta.get("camera_code"),
        camera_name=camera_meta.get("camera_name"),
        channel_no=camera_meta.get("channel_no"),
        location=camera_meta.get("location"),
        camera_source_type=camera_meta.get("camera_source_type", "other"),
        video_id=video_id,
    )
    return importer.import_result(args, payload, emb)


def _match_cross_video(db_url: str, target_run_id: int) -> dict[str, Any]:
    from services.analysis.pipeline import match_cross_video as matcher

    threshold = _env_float("TRACKING_V3_CROSS_MATCH_THRESH", 0.70)
    max_samples = int(_env_float("TRACKING_V3_REID_MAX_SAMPLES", 15) or 15)
    min_pairs = int(_env_float("TRACKING_V3_REID_MIN_PAIRS", 2) or 2)
    engine = create_engine(db_url, pool_pre_ping=True)
    try:
        with engine.begin() as conn:
            queries = matcher.load_people(conn, run_id=target_run_id)
            gallery = matcher.load_people(conn, exclude_run_id=target_run_id)
            if not queries:
                return {"match_count": 0, "unmatched_count": 0, "skipped": "no_queries"}
            if not gallery:
                return {
                    "match_count": 0,
                    "unmatched_count": len(queries),
                    "skipped": "empty_gallery",
                }
            matches = matcher.greedy_match(
                queries,
                gallery,
                threshold,
                max_samples=max_samples,
                min_pairs=min_pairs,
            )
            matcher.apply_matches(
                conn,
                matches,
                queries,
                max_samples=max_samples,
            )
            return {
                "match_count": len(matches),
                "unmatched_count": len(queries) - len(matches),
                "threshold": threshold,
                "reid_max_samples": max_samples,
                "reid_min_pairs": min_pairs,
            }
    finally:
        engine.dispose()


def _compute_stays_if_rooms(db_url: str, video_id: int) -> dict[str, Any]:
    from services.analysis.pipeline import compute_stays as stays

    engine = create_engine(db_url, pool_pre_ping=True)
    try:
        with engine.begin() as conn:
            rooms = stays.load_rooms(conn, video_id)
            if not rooms:
                return {"skipped": "no_rooms"}
            stays.delete_stays_for_video(conn, video_id)
            assign_stats = stays.assign_room_ids(
                conn,
                video_id,
                rooms,
                nearest_max_px=_env_float("TRACKING_V3_NEAREST_MAX_PX", 0.0),
            )
            stay_count = stays.build_stay_segments(
                conn,
                video_id,
                time_gap=_env_float("TRACKING_V3_STAY_TIME_GAP", 2.0),
            )
            return {**assign_stats, "stay_segments": stay_count}
    finally:
        engine.dispose()


def _generate_snapshots(db_url: str, video_id: int) -> dict[str, Any]:
    from services.analysis.pipeline import generate_snapshots as snapshots

    return snapshots.generate_snapshots_for_video(
        db_url,
        video_id,
        candidates_per_person=int(
            os.environ.get("TRACKING_V3_SNAPSHOT_CANDIDATES", "4") or "4"
        ),
        min_gap_sec=_env_float("TRACKING_V3_SNAPSHOT_MIN_GAP", 2.0),
    )


def run_video_analysis(
    *,
    video_path: str | Path,
    video_name: str,
    raw_path: str | None = None,
    captured_at: str | None = None,
    commit_check: Any | None = None,
    progress_callback: Any | None = None,
    skip_frames: int | None = None,
    conf_val: float | None = None,
) -> dict[str, Any] | None:
    """跟踪并入库。commit_check 返回 False 时跳过入库（用于同名重传导致的过期任务）。"""
    if not analysis_enabled():
        return None

    db_url = tracking_db_url()
    source = Path(video_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"tracking_v3 找不到视频文件: {source}")

    output_dir = (
        DEFAULT_OUTPUT_ROOT / f"{_safe_stem(video_name)}_{os.getpid()}_{_uuid_suffix()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    def _emit(ratio: float, person_count: int | None = None) -> None:
        if progress_callback is None:
            return
        if commit_check is not None and not commit_check():
            return
        try:
            progress_callback(ratio, person_count)
        except TypeError:
            progress_callback(ratio)

    tracks_json = _run_tracking(
        source,
        output_dir,
        progress_callback=_emit,
        cancel_check=commit_check,
        skip_frames=skip_frames,
        conf_val=conf_val,
    )

    if commit_check is not None and not commit_check():
        print(
            f"[tracking_v3] 任务已过期，跳过入库: video_name={video_name}",
            flush=True,
        )
        return {
            "tracks_json": str(tracks_json),
            "skipped": "superseded",
            "import": {"person_count": 0},
        }

    # 复用上传时创建的 videos 行，只清旧轨迹，保留房间标注
    from services.analysis.tracking.purge import purge_processing_keep_video, purge_tracking_by_file_name

    reuse_video_id: int | None = None
    lookup_engine = create_engine(db_url, pool_pre_ping=True)
    try:
        with lookup_engine.connect() as conn:
            reuse_video_id = conn.execute(
                text("SELECT id FROM videos WHERE file_name=:n LIMIT 1"),
                {"n": str(video_name)},
            ).scalar()
            if reuse_video_id is not None:
                reuse_video_id = int(reuse_video_id)
    finally:
        lookup_engine.dispose()

    if reuse_video_id is not None:
        purge_report = purge_processing_keep_video(reuse_video_id)
        print(
            f"[tracking_v3] 复用 videos.id={reuse_video_id}，已清旧轨迹: {purge_report}",
            flush=True,
        )
    else:
        purge_report = purge_tracking_by_file_name(str(video_name))
        print(f"[tracking_v3] 导入前清理同名视频: {purge_report}", flush=True)

    if not captured_at:
        captured_at = _captured_at_from_video_row(db_url, video_name, reuse_video_id)

    if commit_check is not None and not commit_check():
        print(
            f"[tracking_v3] 清理后检测任务已过期，中止导入: video_name={video_name}",
            flush=True,
        )
        return {
            "tracks_json": str(tracks_json),
            "skipped": "superseded_after_purge",
            "import": {"person_count": 0},
            "purge": purge_report,
        }

    source_type = _infer_source_type(raw_path)
    camera_meta = _infer_camera(video_name, source_type)
    imported = _import_tracks(
        tracks_json,
        db_url=db_url,
        video_source_type=source_type,
        camera_meta=camera_meta,
        captured_at=captured_at,
        video_id=reuse_video_id,
    )
    person_count = int(imported.get("person_count") or 0)
    _emit(0.93, person_count)

    match_report: dict[str, Any] = {"skipped": "disabled"}
    if _env_bool("TRACKING_V3_CROSS_MATCH", True):
        match_report = _match_cross_video(db_url, int(imported["processing_run_id"]))
    _emit(0.96, person_count)

    stay_report: dict[str, Any] = {"skipped": "disabled"}
    if _env_bool("TRACKING_V3_COMPUTE_STAYS", True):
        stay_report = _compute_stays_if_rooms(db_url, int(imported["video_id"]))
    _emit(0.97, person_count)

    snapshot_report: dict[str, Any] = {"skipped": "disabled"}
    if _env_bool("TRACKING_V3_GENERATE_SNAPSHOTS", True):
        snapshot_report = _generate_snapshots(db_url, int(imported["video_id"]))
    _emit(0.99, person_count)

    index_report: dict[str, Any] = {"skipped": "no_observations"}
    try:
        from services.analysis.tracking.search_index import add_video_observations

        idx_engine = create_engine(db_url, pool_pre_ping=True)
        try:
            index_report = add_video_observations(idx_engine, int(imported["video_id"]))
        finally:
            idx_engine.dispose()
    except Exception as exc:
        from services.analysis.tracking.search_index import mark_dirty

        mark_dirty()
        index_report = {"error": str(exc), "dirty": True}
        print(f"[tracking_v3] FAISS 增量失败，已标记 dirty: {exc}", flush=True)

    result = {
        "tracks_json": str(tracks_json),
        "purge": purge_report,
        "import": imported,
        "cross_match": match_report,
        "stays": stay_report,
        "snapshots": snapshot_report,
        "search_index": index_report,
    }
    print(f"[tracking_v3] 完成: {result}", flush=True)
    return result
