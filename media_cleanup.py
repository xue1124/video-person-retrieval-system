"""同一视频重新建模 / 删除媒体源时的本地文件与数据库清理。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from video_playback import cleanup_finalize_sidecars

BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "video_archives"
GALLERY_DIR = BASE_DIR / "video_crops"


def normalize_video_name(video_name: str) -> str:
    """与 api_server._norm_video_name_key / 上传 file_name 一致：仅允许 basename。"""
    s = (video_name or "").strip()
    base = os.path.basename(s.replace("\\", "/"))
    if not base or base != s:
        raise ValueError(f"非法 video_name: {video_name!r}")
    return base


def _remove_file(path: Optional[str]) -> None:
    if not path:
        return
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def _remove_gallery_crop_files(video_name: str) -> None:
    try:
        for p in GALLERY_DIR.glob(f"{video_name}_*.jpg"):
            try:
                p.unlink()
            except OSError:
                pass
        # 房间标注参考帧（由归档视频抽帧生成）
        for p in GALLERY_DIR.glob(f"{video_name}__room_ref.jpg"):
            try:
                p.unlink()
            except OSError:
                pass
    except OSError:
        pass


def _remove_legacy_web_sidecars(stem: str, parent: Path) -> None:
    for suffix in ("_web.mp4", "_web.mp4.lock", "_web.mp4.part.mp4"):
        p = parent / f"{stem}{suffix}"
        if p.is_file():
            try:
                p.unlink()
            except OSError:
                pass


def _remove_archived_videos(video_name: str, raw_path: Optional[str] = None) -> None:
    removed_paths: list[str] = []
    for rel in (VIDEO_DIR / video_name, VIDEO_DIR / f"{video_name}.mp4"):
        if rel.is_file():
            removed_paths.append(str(rel))
            _remove_legacy_web_sidecars(rel.stem, rel.parent)
            try:
                rel.unlink()
            except OSError:
                pass
    if raw_path:
        rp = str(raw_path).strip()
        if rp and not rp.startswith(("rtsp://", "isapi://")):
            rp_norm = os.path.normpath(rp)
            _remove_legacy_web_sidecars(Path(rp_norm).stem, Path(rp_norm).parent)
            if os.path.isfile(rp_norm):
                removed_paths.append(rp_norm)
            _remove_file(rp_norm)
    for p in removed_paths:
        cleanup_finalize_sidecars(p)


def purge_video_modeling_artifacts(
    db_engine: Engine,
    video_name: str,
    *,
    raw_path: Optional[str] = None,
    remove_archived_video: bool = True,
    reset_target_count: bool = True,
    remove_rooms: bool = False,
) -> None:
    """
    删除某视频关联的建模产物：gallery_meta、裁切图；可选删除归档视频。
    默认保留 rooms（房间标注挂在 videos.id 上）。
    不删除 videos 行（由调用方 DELETE 或 UPSERT）。
    """
    vn = normalize_video_name(video_name)

    with db_engine.begin() as conn:
        video_id = conn.execute(
            text("SELECT id FROM videos WHERE file_name=:v LIMIT 1"),
            {"v": vn},
        ).scalar()
        imgs = conn.execute(
            text(
                """
                SELECT image_path FROM gallery_meta
                WHERE video_name=:v OR video_id=:vid
                """
            ),
            {"v": vn, "vid": int(video_id) if video_id else 0},
        ).fetchall()
        for row in imgs:
            ip = row[0] if row else None
            if not ip:
                continue
            ip_str = str(ip)
            _remove_file(ip_str)
            base = os.path.basename(ip_str.replace("\\", "/"))
            disk = GALLERY_DIR / base
            if disk.is_file():
                try:
                    disk.unlink()
                except OSError:
                    pass
        conn.execute(
            text("DELETE FROM gallery_meta WHERE video_name=:v OR video_id=:vid"),
            {"v": vn, "vid": int(video_id) if video_id else 0},
        )
        if remove_rooms and video_id:
            conn.execute(text("DELETE FROM rooms WHERE video_id=:vid"), {"vid": int(video_id)})
        if reset_target_count:
            conn.execute(
                text(
                    "UPDATE videos SET target_count=0, progress=0 "
                    "WHERE file_name=:v AND is_media_source=1"
                ),
                {"v": vn},
            )

    _remove_gallery_crop_files(vn)
    if remove_archived_video:
        _remove_archived_videos(vn, raw_path)
