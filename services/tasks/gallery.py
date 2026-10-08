"""RTSP实时流的截图落盘与 gallery_meta 批量写入。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sqlalchemy import text

from services.inference.runtime import (
    extract_osnet_feat_batch,
    extract_siglip_feat_img_batch,
)
from services.persistence import database as app_db

_write_pool = ThreadPoolExecutor(max_workers=4)


def _write_gallery_frame(path: str, image: np.ndarray) -> bool:
    """图片写入成功后才允许创建数据库记录。"""
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(target), image)
        return bool(ok) and target.is_file() and target.stat().st_size > 0
    except (OSError, cv2.error):
        return False


def flush_gallery_buffer(
    crop_buffer: list[np.ndarray],
    full_frame_buffer: list[np.ndarray],
    meta_buffer: list[dict[str, Any]],
    db_engine: Any,
) -> int:
    """提取两类向量、保存截图并批量写入实时流兼容表。"""
    if not crop_buffer:
        return 0

    osnet_features = extract_osnet_feat_batch(crop_buffer)
    siglip_features = extract_siglip_feat_img_batch(crop_buffer)
    write_ok = [False] * len(crop_buffer)
    futures = {
        _write_pool.submit(
            _write_gallery_frame, meta_buffer[index]["p"], full_frame_buffer[index]
        ): index
        for index in range(len(crop_buffer))
    }
    for future in as_completed(futures):
        index = futures[future]
        try:
            write_ok[index] = bool(future.result())
        except Exception:
            write_ok[index] = False

    rows: list[dict[str, Any]] = []
    for index, ok in enumerate(write_ok):
        if not ok:
            continue
        meta = meta_buffer[index]
        rows.append(
            {
                "v": meta["v"],
                "t": meta["t"],
                "p": meta["p"],
                "x1": int(meta["x1"]),
                "y1": int(meta["y1"]),
                "x2": int(meta["x2"]),
                "y2": int(meta["y2"]),
                "feat": osnet_features[index].tobytes(),
                "cfeat": siglip_features[index].tobytes(),
            }
        )

    if rows:
        with db_engine.begin() as conn:
            video_ids: dict[str, int | None] = {}
            for row in rows:
                name = str(row["v"])
                if name not in video_ids:
                    video_ids[name] = app_db.get_video_id(conn, name)
                row["video_id"] = video_ids[name]
            conn.execute(
                text(
                    "INSERT INTO gallery_meta (video_id, video_name, timestamp, image_path, "
                    "bbox_x1, bbox_y1, bbox_x2, bbox_y2, feature_vector, clip_feature) "
                    "VALUES (:video_id, :v, :t, :p, :x1, :y1, :x2, :y2, :feat, :cfeat)"
                ),
                rows,
            )
    return len(rows)
