"""双人房间共现：患者轨迹与单人检索相同，按停留房间筛选展示。"""
from __future__ import annotations

import io
import os
import uuid
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from search_service import (
    QUERY_STORAGE,
    _load_rooms_for_videos,
    attach_image_urls_to_stay_segments,
    build_stay_segments_by_video,
    run_multi_target_search,
)
from tasks import extract_osnet_feat, extract_siglip_feat_img


def _ensure_query_storage():
    if not os.path.exists(QUERY_STORAGE):
        os.makedirs(QUERY_STORAGE, exist_ok=True)


def _extract_feat_from_bytes(raw: bytes, is_osnet: bool) -> np.ndarray:
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    if is_osnet:
        f = extract_osnet_feat(bgr)
    else:
        f = extract_siglip_feat_img(bgr)
    if f is None:
        raise ValueError("无法从查询图提取特征，请换一张更清晰的人物图")
    arr = np.asarray(f, dtype=np.float32).reshape(-1)
    arr /= np.linalg.norm(arr) + 1e-8
    return arr


def _save_query_image(raw: bytes, name: str) -> str:
    _ensure_query_storage()
    ext = os.path.splitext(name)[1] or ".jpg"
    unique_name = f"query_{uuid.uuid4().hex}{ext}"
    path = os.path.join(QUERY_STORAGE, unique_name)
    with open(path, "wb") as wf:
        wf.write(raw)
    return unique_name


def _build_feats_and_saved(
    images_bytes: List[bytes],
    image_names: List[str],
    is_osnet: bool,
) -> Tuple[List[np.ndarray], List[str]]:
    """多张查询图：各提特征，检索时由 run_multi_target_search 按 meta_id 取 max(sim) 并集。"""
    feats: List[np.ndarray] = []
    saved: List[str] = []
    for i, raw in enumerate(images_bytes):
        if not raw:
            continue
        name = image_names[i] if i < len(image_names) else f"upload_{i}.jpg"
        feats.append(_extract_feat_from_bytes(raw, is_osnet))
        saved.append(_save_query_image(raw, name))
    return feats, saved


def _stay_room_names_from_segments(segments: List[dict]) -> List[str]:
    names: set[str] = set()
    for s in segments:
        for key in ("exit_room", "entry_room"):
            v = (str(s.get(key) or "")).strip()
            if v:
                names.add(v)
        primary = (str(s.get("room_name") or "")).strip()
        if primary and "→" not in primary:
            names.add(primary)
        elif primary and "→" in primary:
            names.add(primary.split("→")[-1].strip())
    return sorted(names)


def _stay_room_names_for_video(
    segments: List[dict], annotated_rooms: List[dict]
) -> List[str]:
    """下拉选项 = 媒体源已标注房间 ∪ 各段判定出的房间名。"""
    names = {str(r.get("name") or "").strip() for r in annotated_rooms}
    names.discard("")
    names.update(_stay_room_names_from_segments(segments))
    return sorted(names)


def perform_room_copresence(
    *,
    engine,
    algorithm: str,
    threshold: float,
    time_gap: float,
    images_a_bytes: List[bytes],
    images_a_names: Optional[List[str]] = None,
    images_b_bytes: Optional[List[bytes]] = None,
    images_b_names: Optional[List[str]] = None,
    role_a_label: str = "患者",
    role_b_label: str = "医生",
) -> dict:
    """
    返回患者/医生 stay_segments。每角色可多张查询图，检索逻辑与单人轨迹检索一致。
    """
    if not images_a_bytes:
        raise ValueError("请至少上传一张患者查询图")

    algo_upper = (algorithm or "OSNet").upper()
    is_osnet = "OSNET" in algo_upper or algorithm == "OSNet"
    db_col = "feature_vector" if is_osnet else "clip_feature"
    algo_label = "OSNet" if is_osnet else "SIGLIP"

    names_a = images_a_names or [f"role_a_{i}.jpg" for i in range(len(images_a_bytes))]
    feats_a, saved_a = _build_feats_and_saved(images_a_bytes, names_a, is_osnet)
    if not feats_a:
        raise ValueError("患者查询图均无法提取特征，请更换图片")

    feats_b: List[np.ndarray] = []
    saved_b: List[str] = []
    if images_b_bytes:
        names_b = images_b_names or [f"role_b_{i}.jpg" for i in range(len(images_b_bytes))]
        feats_b, saved_b = _build_feats_and_saved(images_b_bytes, names_b, is_osnet)

    _, raw_a = run_multi_target_search(
        feats_a,
        db_col,
        threshold,
        time_gap,
        False,
        0.0,
        engine,
        candidate_k=500,
        top_k_per_query=None,
    )

    if not raw_a:
        raise ValueError("底库为空或未找到满足阈值的患者命中，请先完成视频建模并调整阈值")

    raw_b: Dict[str, List[dict]] = {}
    if feats_b:
        _, raw_b = run_multi_target_search(
            feats_b,
            db_col,
            threshold,
            time_gap,
            False,
            0.0,
            engine,
            candidate_k=500,
            top_k_per_query=None,
        )

    stay_by_video = attach_image_urls_to_stay_segments(
        build_stay_segments_by_video(raw_a, engine, time_gap)
    )
    stay_doctor_by_video = (
        attach_image_urls_to_stay_segments(
            build_stay_segments_by_video(raw_b, engine, time_gap)
        )
        if raw_b
        else {}
    )
    vnames = sorted(set(stay_by_video.keys()) | set(stay_doctor_by_video.keys()))
    rooms_map = _load_rooms_for_videos(engine, vnames)

    videos_out: List[dict] = []
    for vn in vnames:
        segs = stay_by_video.get(vn, [])
        segs_dr = stay_doctor_by_video.get(vn, [])
        annotated = rooms_map.get(vn, [])
        videos_out.append(
            {
                "video_name": vn,
                "room_annotated": len(annotated) > 0,
                "stay_segments": segs,
                "stay_segments_doctor": segs_dr,
                "stay_room_names": _stay_room_names_for_video(segs, annotated),
            }
        )

    return {
        "algorithm": algo_label,
        "role_a_label": role_a_label.strip() or "患者",
        "role_b_label": role_b_label.strip() or "医生",
        "query_image_a": ",".join(saved_a),
        "query_image_b": ",".join(saved_b),
        "threshold": float(threshold),
        "time_gap": float(time_gap),
        "videos": videos_out,
    }
