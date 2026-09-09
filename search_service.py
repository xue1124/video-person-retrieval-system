"""服务端轨迹检索：复用 tasks 中的特征提取，底库检索逻辑与 app_ui 一致。"""
from __future__ import annotations

import io
import json
import math
import os
import re
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from sqlalchemy import bindparam, text

from room_geometry import bbox_foot_xy, classify_point_to_room
from tasks import extract_osnet_feat, extract_siglip_feat_img, extract_siglip_feat_text

QUERY_STORAGE = "query_storage"
TEXT_RECALL_CANDIDATE_K = 500
TEXT_MIXED_TEXT_THRESHOLD = 80.0
# 监控行人 crop 的 raw cosine 通常仅 0.01~0.13
TEXT_SCORE_TEMP = 0.05
# 对比 rerank：margin = min(正词) - max(负词)，再用 sigmoid 映射展示分
TEXT_MARGIN_TEMP = 0.02
TEXT_NEGATIVE_KEYWORDS = (
    "狗",
    "香蕉",
    "汽车",
    "树",
    "天空",
    "路人",
    "桌子",
    "椅子",
)


def _cosine_to_display_score(raw_score: float) -> float:
    """
    将原始余弦相似度[-1,1]映射到更直观的[0,1]区间用于展示与阈值筛选。
    """
    s = float(np.clip(raw_score, -1.0, 1.0))
    return (s + 1.0) * 0.5


def _display_to_cosine_threshold(display_threshold: float) -> float:
    """
    将外部传入阈值(0~1)反向映射回余弦空间，保证筛选与展示口径一致。
    """
    t = float(np.clip(display_threshold, 0.0, 1.0))
    return t * 2.0 - 1.0


def _text_display_score(raw_score: float) -> float:
    """
    文搜图单词展示分：sigmoid(raw / T) × 100（用于 keyword_scores 展示）。
    """
    s = float(np.clip(raw_score, -1.0, 1.0))
    return round(100.0 / (1.0 + math.exp(-s / TEXT_SCORE_TEMP)), 2)


def _text_margin_display_score(pos_raw: float, neg_max: float) -> Tuple[float, float]:
    """
    对比 rerank 展示分：margin = pos_raw - neg_max，再 sigmoid(margin / Tm) × 100。
    返回 (margin, display_score)。
    """
    margin = float(pos_raw) - float(neg_max)
    m = float(np.clip(margin, -1.0, 1.0))
    display = round(100.0 / (1.0 + math.exp(-m / TEXT_MARGIN_TEMP)), 2)
    return margin, display


def _extract_text_feats(words: List[str]) -> Tuple[List[str], List[np.ndarray]]:
    valid: List[str] = []
    feats: List[np.ndarray] = []
    for w in words:
        f = extract_siglip_feat_text(w)
        if f is None:
            continue
        valid.append(w)
        feats.append(_norm_feat(f))
    return valid, feats


def _split_query_keywords(query: str) -> List[str]:
    """将用户输入拆成短词；单句无分隔符时整句作为一个关键词。"""
    q = (query or "").strip()
    if not q:
        return []
    parts = re.split(r"[\s,，、;；/|]+", q)
    kws = [p.strip() for p in parts if p.strip()]
    return kws if kws else [q]


def _ensure_query_storage():
    if not os.path.exists(QUERY_STORAGE):
        os.makedirs(QUERY_STORAGE, exist_ok=True)


def _fetch_gallery_rows(conn, db_column: str):
    """已废弃：检索改为 observations + FAISS。保留以免外部误调用。"""
    raise RuntimeError("v3 检索不再读取 gallery_meta，请使用 observation 索引")


def _hit_from_observation(
    row: dict,
    dist: float,
    *,
    score_percent: bool = False,
    keyword_scores: Optional[Dict[str, float]] = None,
) -> dict:
    raw = float(dist)
    display_score = _text_display_score(raw) if score_percent else _cosine_to_display_score(raw)
    oid = int(row["id"])
    hit = {
        "meta_id": oid,
        "observation_id": oid,
        "global_person_id": row.get("global_person_id"),
        "video_person_id": row.get("video_person_id"),
        "track_id": row.get("track_id"),
        "video_id": row.get("video_id"),
        "vid": row["video_name"],
        "time": float(row["timestamp_sec"]),
        "path": row.get("crop_path") or "",
        "score": display_score,
        "raw_score": raw,
        "bbox": row.get("bbox"),
        "room_id": row.get("room_id"),
        "room_name": row.get("room_name") or "",
    }
    if keyword_scores:
        hit["keyword_scores"] = {
            k: _text_display_score(float(v)) for k, v in keyword_scores.items()
        }
    return hit


def _hit_from_observation_text(
    row: dict,
    *,
    pos_raw: float,
    neg_max: float,
    margin: float,
    display_score: float,
    keyword_scores: Optional[Dict[str, float]] = None,
    negative_scores: Optional[Dict[str, float]] = None,
) -> dict:
    hit = _hit_from_observation(row, margin, score_percent=True, keyword_scores=keyword_scores)
    hit["score"] = display_score
    hit["raw_score"] = margin
    hit["pos_raw"] = float(pos_raw)
    hit["neg_max"] = float(neg_max)
    if negative_scores:
        hit["negative_scores"] = {
            k: _text_display_score(float(v)) for k, v in negative_scores.items()
        }
    return hit


def _hit_dedup_key(hit: dict):
    """同一底库帧：优先 meta_id，否则用 video+时间+路径。"""
    mid = hit.get("meta_id")
    if mid is not None:
        return ("meta_id", int(mid))
    return ("frame", hit["vid"], float(hit["time"]), hit.get("path") or "")


def _union_hits_max_score(hit_lists: List[List[dict]]) -> List[dict]:
    """
    多查询各自检索后的命中并集；同一 meta_id（或同一帧）保留展示相似度最高的那条。
    """
    best: Dict[Any, dict] = {}
    for hits in hit_lists:
        for hit in hits:
            k = _hit_dedup_key(hit)
            prev = best.get(k)
            if prev is None:
                best[k] = hit
                continue
            if float(hit["score"]) > float(prev["score"]):
                best[k] = hit
            elif float(hit["score"]) == float(prev["score"]) and float(
                hit.get("raw_score", 0.0)
            ) > float(prev.get("raw_score", 0.0)):
                best[k] = hit
    return list(best.values())


def _gallery_bbox_from_row(row: dict) -> Optional[List[int]]:
    if "bbox" in row and row.get("bbox"):
        return row["bbox"]
    if not all(k in row for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")):
        return None
    b = [row.get("bbox_x1"), row.get("bbox_y1"), row.get("bbox_x2"), row.get("bbox_y2")]
    if b[0] is None or b[1] is None or b[2] is None or b[3] is None:
        return None
    return [int(float(b[0])), int(float(b[1])), int(float(b[2])), int(float(b[3]))]


def _hit_from_gallery_row_text(
    row: dict,
    *,
    pos_raw: float,
    neg_max: float,
    margin: float,
    display_score: float,
    keyword_scores: Optional[Dict[str, float]] = None,
    negative_scores: Optional[Dict[str, float]] = None,
) -> dict:
    if "video_name" in row and "timestamp_sec" in row:
        return _hit_from_observation_text(
            row,
            pos_raw=pos_raw,
            neg_max=neg_max,
            margin=margin,
            display_score=display_score,
            keyword_scores=keyword_scores,
            negative_scores=negative_scores,
        )
    hit = {
        "meta_id": int(row["id"]),
        "vid": row["video_name"],
        "time": float(row["timestamp"]),
        "path": row["image_path"],
        "score": display_score,
        "raw_score": margin,
        "pos_raw": float(pos_raw),
        "neg_max": float(neg_max),
        "bbox": _gallery_bbox_from_row(row),
    }
    if keyword_scores:
        hit["keyword_scores"] = {
            k: _text_display_score(float(v)) for k, v in keyword_scores.items()
        }
    if negative_scores:
        hit["negative_scores"] = {
            k: _text_display_score(float(v)) for k, v in negative_scores.items()
        }
    return hit


def _hit_from_gallery_row(
    row: dict,
    dist: float,
    *,
    score_percent: bool = False,
    keyword_scores: Optional[Dict[str, float]] = None,
) -> dict:
    if "video_name" in row and "timestamp_sec" in row:
        return _hit_from_observation(
            row, dist, score_percent=score_percent, keyword_scores=keyword_scores
        )
    raw = float(dist)
    display_score = _text_display_score(raw) if score_percent else _cosine_to_display_score(raw)
    hit = {
        "meta_id": int(row["id"]),
        "vid": row["video_name"],
        "time": float(row["timestamp"]),
        "path": row["image_path"],
        "score": display_score,
        "raw_score": raw,
        "bbox": _gallery_bbox_from_row(row),
    }
    if keyword_scores:
        hit["keyword_scores"] = {
            k: _text_display_score(float(v)) for k, v in keyword_scores.items()
        }
    return hit


def _intersect_siglip_image_text_hits(
    img_hits: List[dict], text_hits: List[dict]
) -> List[dict]:
    """图文混搜：同一 meta_id 必须同时过图像与文本检索，综合分取两者归一化后的 min。"""
    text_by_id: Dict[int, dict] = {}
    for hit in text_hits:
        mid = hit.get("meta_id")
        if mid is None:
            continue
        prev = text_by_id.get(int(mid))
        if prev is None or float(hit.get("score", 0.0)) > float(prev.get("score", 0.0)):
            text_by_id[int(mid)] = hit

    merged: List[dict] = []
    for ih in img_hits:
        mid = ih.get("meta_id")
        if mid is None:
            continue
        th = text_by_id.get(int(mid))
        if th is None:
            continue
        img_pct = round(float(ih.get("score", 0.0)) * 100.0, 2)
        text_pct = float(th.get("score", 0.0))
        combined = round(min(img_pct, text_pct), 2)
        hit = dict(th)
        hit["score"] = combined
        hit["image_score"] = img_pct
        hit["text_score"] = text_pct
        hit["search_mode"] = "mixed_intersection"
        merged.append(hit)

    merged.sort(
        key=lambda x: (
            float(x.get("score", 0.0)),
            float(x.get("text_score", 0.0)),
            float(x.get("image_score", 0.0)),
        ),
        reverse=True,
    )
    return merged


def _hits_to_raw_by_video(hits: List[dict]) -> Dict[str, List[dict]]:
    raw: Dict[str, List[dict]] = {}
    for hit in hits:
        raw.setdefault(hit["vid"], []).append(hit)
    for v in raw:
        raw[v].sort(key=lambda x: float(x["time"]))
    return raw


def _nms_hits_by_video(hits: List[dict], time_gap: float) -> Dict[str, List[dict]]:
    fg: Dict[str, List[dict]] = {}
    for hit in hits:
        v = hit["vid"]
        if v not in fg:
            fg[v] = []
        if not any(abs(it["time"] - hit["time"]) < time_gap for it in fg[v]):
            fg[v].append(hit)
    return fg


def run_siglip_mixed_search(
    *,
    img_feats: List[np.ndarray],
    keywords: List[str],
    image_threshold: float,
    text_threshold_percent: float,
    time_gap: float,
    engine,
) -> Tuple[Optional[Dict[str, List[dict]]], Dict[str, List[dict]]]:
    """
    SIGLIP 图文混搜：图、文分别检索后按 meta_id 取交集（AND），综合分取两者较低分。
    """
    if not img_feats:
        raise ValueError("图文混搜需要至少一张查询图")
    kws = [k.strip() for k in keywords if k and str(k).strip()]
    if not kws:
        raise ValueError("图文混搜需要文字描述")

    _, raw_img = run_multi_target_search(
        img_feats,
        "clip_feature",
        float(image_threshold),
        time_gap,
        False,
        0.0,
        engine,
        candidate_k=500,
        top_k_per_query=None,
    )
    img_hits = [h for hits in (raw_img or {}).values() for h in hits]

    _, raw_text = run_text_siglip_search(
        kws,
        float(text_threshold_percent),
        time_gap,
        engine,
    )
    text_hits = [h for hits in (raw_text or {}).values() for h in hits]

    merged = _intersect_siglip_image_text_hits(img_hits, text_hits)
    if not merged:
        return {}, {}
    raw_by_video = _hits_to_raw_by_video(merged)
    return _nms_hits_by_video(merged, time_gap), raw_by_video


def run_text_siglip_search(
    keywords: List[str],
    threshold_percent: float,
    time_gap: float,
    engine,
    candidate_k: int = TEXT_RECALL_CANDIDATE_K,
) -> Tuple[Optional[Dict[str, List[dict]]], Dict[str, List[dict]]]:
    """
    文搜图：FAISS 宽召回 → 正词 min + 负词 max 对比 rerank → sigmoid(margin) 展示分。
    展示分 = sigmoid((min(正词cos) - max(负词cos)) / TEXT_MARGIN_TEMP) × 100。
    """
    kws = [k.strip() for k in keywords if k and str(k).strip()]
    if not kws:
        raise ValueError("请提供有效的文字关键词")

    valid_kws, kw_feats = _extract_text_feats(kws)
    if not kw_feats:
        raise ValueError("无法提取文本特征，请检查 SigLIP 模型")

    pos_set = set(valid_kws)
    neg_words = [n for n in TEXT_NEGATIVE_KEYWORDS if n not in pos_set]
    neg_labels, neg_feats = _extract_text_feats(neg_words)
    neg_matrix = np.vstack(neg_feats) if neg_feats else None

    from tracking_v3.search_index import (
        MODALITY_SIGLIP,
        db_embedding_count,
        hydrate_observations,
        load_embeddings_by_ids,
        search as faiss_search,
    )

    if db_embedding_count(engine, MODALITY_SIGLIP) == 0:
        return None, {}

    recall_ids: set[int] = set()
    per_kw_k = max(100, candidate_k // max(len(kw_feats), 1))
    for feat in kw_feats:
        _, idxs = faiss_search(engine, MODALITY_SIGLIP, feat, candidate_k=per_kw_k)
        for idx in idxs:
            if int(idx) >= 0:
                recall_ids.add(int(idx))

    ids_list = list(recall_ids)
    vectors_by_id = load_embeddings_by_ids(engine, ids_list, MODALITY_SIGLIP)
    rows_by_id = hydrate_observations(engine, ids_list)

    kw_matrix = np.vstack(kw_feats)
    score_threshold = float(np.clip(threshold_percent, 0.0, 100.0))
    hits: List[dict] = []
    for oid in ids_list:
        img_vec = vectors_by_id.get(oid)
        row = rows_by_id.get(oid)
        if img_vec is None or row is None:
            continue
        pos_scores = kw_matrix @ img_vec
        pos_raw = float(np.min(pos_scores))
        neg_max = 0.0
        neg_score_map: Dict[str, float] = {}
        if neg_matrix is not None and neg_matrix.size > 0:
            neg_scores = neg_matrix @ img_vec
            neg_max = float(np.max(neg_scores))
            neg_score_map = {neg_labels[i]: float(neg_scores[i]) for i in range(len(neg_labels))}
        margin, display = _text_margin_display_score(pos_raw, neg_max)
        if display < score_threshold:
            continue
        kw_score_map = {valid_kws[i]: float(pos_scores[i]) for i in range(len(valid_kws))}
        hits.append(
            _hit_from_observation_text(
                row,
                pos_raw=pos_raw,
                neg_max=neg_max,
                margin=margin,
                display_score=display,
                keyword_scores=kw_score_map,
                negative_scores=neg_score_map if neg_score_map else None,
            )
        )

    hits.sort(
        key=lambda h: (float(h["score"]), float(h.get("pos_raw", 0.0)), float(h.get("raw_score", 0.0))),
        reverse=True,
    )

    raw_by_video = _hits_to_raw_by_video(hits)

    return _nms_hits_by_video(hits, time_gap), raw_by_video


def run_multi_target_search(
    feats_list: List[np.ndarray],
    db_column: str,
    threshold: float,
    time_gap: float,
    group_mode: bool,
    co_time_threshold: float,
    engine,
    candidate_k: int = 500,
    top_k_per_query: Optional[int] = None,
) -> Tuple[Optional[Dict[str, List[dict]]], Dict[str, List[dict]]]:
    """
    返回 (NMS 后的结果用于卡片展示, 按视频的「阈值内全部命中」用于停留段合并)。
    后者与 tests/yolo_osnet_query_search_streamlit 中按 gap 合并段一致。

    非 group_mode 且多条查询特征时：每条特征单独检索，按 meta_id 取 max(sim) 并集后再 NMS / 合并段。
    group_mode 仍为多人共现（各查询分别检索后做时空交集）。
    """
    from tracking_v3.search_index import (
        MODALITY_OSNET,
        MODALITY_SIGLIP,
        db_embedding_count,
        hydrate_observations,
        search as faiss_search,
    )

    modality = MODALITY_OSNET if db_column == "feature_vector" else MODALITY_SIGLIP
    if db_embedding_count(engine, modality) == 0:
        return None, {}

    target_results: List[List[dict]] = []
    raw_threshold = _display_to_cosine_threshold(threshold)
    per_query_pairs: List[List[Tuple[int, float]]] = []
    all_ids: List[int] = []
    for feat in feats_list:
        feat = feat.flatten().astype("float32")
        scores, ids = faiss_search(engine, modality, feat, candidate_k=candidate_k)
        pairs: List[Tuple[int, float]] = []
        for dist, oid in zip(scores, ids):
            if int(oid) < 0:
                continue
            if float(dist) < raw_threshold:
                continue
            pairs.append((int(oid), float(dist)))
        if top_k_per_query is not None and len(pairs) > top_k_per_query:
            pairs = pairs[:top_k_per_query]
        per_query_pairs.append(pairs)
        all_ids.extend(p[0] for p in pairs)

    meta = hydrate_observations(engine, all_ids)
    for pairs in per_query_pairs:
        hits: List[dict] = []
        for oid, dist in pairs:
            row = meta.get(oid)
            if not row:
                continue
            hits.append(_hit_from_observation(row, dist))
        target_results.append(hits)

    raw_by_video: Dict[str, List[dict]] = {}
    final_groups: Dict[str, List[dict]] = {}

    if group_mode and len(target_results) > 1:
        base_person = target_results[0]
        others = target_results[1:]
        co_hits: List[dict] = []
        for hit in base_person:
            is_co = True
            for other_person in others:
                if not any(
                    it["vid"] == hit["vid"]
                    and abs(it["time"] - hit["time"]) <= co_time_threshold
                    for it in other_person
                ):
                    is_co = False
                    break
            if is_co:
                co_hits.append(hit)
        for h in co_hits:
            raw_by_video.setdefault(h["vid"], []).append(h)
        for v in raw_by_video:
            raw_by_video[v].sort(key=lambda x: float(x["time"]))
        final_groups = _nms_hits_by_video(co_hits, time_gap)
    else:
        if len(target_results) > 1:
            merged_hits = _union_hits_max_score(target_results)
        elif target_results:
            merged_hits = target_results[0]
        else:
            merged_hits = []
        for hit in merged_hits:
            raw_by_video.setdefault(hit["vid"], []).append(hit)
        for v in raw_by_video:
            raw_by_video[v].sort(key=lambda x: float(x["time"]))
        final_groups = _nms_hits_by_video(merged_hits, time_gap)

    return final_groups, raw_by_video


def _path_to_url(path: str) -> str:
    """将磁盘路径转为 axios baseURL=/api 下的相对路径（如 /files/crop/xxx）。"""
    norm = os.path.normpath(path).replace("\\", "/")
    if "/obs/" in norm or norm.startswith("v") and "/obs/" in norm:
        return path
    if "video_crops" in norm:
        base = os.path.basename(norm)
        return f"/files/crop/{base}"
    if "query_storage" in norm:
        base = os.path.basename(norm)
        return f"/files/query/{base}"
    base = os.path.basename(norm)
    return f"/files/crop/{base}"


def _hit_image_url(hit: dict) -> str:
    oid = hit.get("observation_id") if hit.get("observation_id") is not None else hit.get("meta_id")
    if oid is not None:
        return f"/files/obs/{int(oid)}"
    return _path_to_url(hit.get("path") or "")


def attach_image_urls(results: Optional[Dict[str, List[dict]]]) -> Dict[str, Any]:
    if not results:
        return {}
    out = {}
    for vid, items in results.items():
        out[vid] = []
        for it in items:
            row = dict(it)
            row["imageUrl"] = _hit_image_url(it)
            out[vid].append(row)
    return out


def _fmt_hms(sec: float) -> str:
    total = int(round(max(0.0, float(sec))))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _polygon_from_db_cell(val: Any) -> List[List[float]]:
    if val is None:
        return []
    if isinstance(val, list):
        raw = val
    elif isinstance(val, str):
        try:
            raw = json.loads(val)
        except json.JSONDecodeError:
            return []
    else:
        return []
    if not isinstance(raw, list):
        return []
    clean: List[List[float]] = []
    for p in raw:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            clean.append([float(p[0]), float(p[1])])
    return clean


def _load_rooms_for_videos(engine, video_names: List[str]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    names = sorted({n for n in video_names if n})
    if not names:
        return out
    stmt = text(
        """
        SELECT v.file_name AS video_name, r.name AS room_name, r.polygon_json
        FROM rooms r
        JOIN videos v ON v.id = r.video_id
        WHERE v.file_name IN :vnames
        """
    ).bindparams(bindparam("vnames", expanding=True))
    with engine.connect() as conn:
        rows = conn.execute(stmt, {"vnames": names}).mappings().all()
    for r in rows:
        vn = str(r["video_name"])
        poly = _polygon_from_db_cell(r.get("polygon_json"))
        if len(poly) < 3:
            continue
        rn = (str(r.get("room_name") or "")).strip() or "未命名"
        out.setdefault(vn, []).append({"name": rn, "polygon": poly})
    return out


def _segment_primary_room(
    crop_names: List[str], entry_room: str, exit_room: str
) -> str:
    trimmed = [x.strip() for x in crop_names if x and str(x).strip()]
    if trimmed:
        top, _ = Counter(trimmed).most_common(1)[0]
        return top
    if entry_room and exit_room and entry_room != exit_room:
        return f"{entry_room}→{exit_room}"
    return entry_room or exit_room or ""


def merge_hits_to_stay_segments(
    hits: List[dict],
    gap_sec: float,
    rooms: List[dict],
) -> List[dict]:
    """按时间排序后，相邻命中间隔不超过 gap_sec 的合并为一段（与 Streamlit _build_segments 一致）。"""
    if not hits:
        return []
    ordered = sorted(hits, key=lambda r: float(r["time"]))

    def _finalize(rows: List[dict], seg_idx: int) -> dict:
        t0 = float(rows[0]["time"])
        t1 = float(rows[-1]["time"])
        duration = max(0.0, t1 - t0)
        best_sim = max(float(r["score"]) for r in rows)

        first_hit, last_hit = rows[0], rows[-1]
        fe = fv = le = lv = 0.0
        if first_hit.get("bbox") and len(first_hit["bbox"]) >= 4:
            fe, fv = bbox_foot_xy(first_hit["bbox"])
        if last_hit.get("bbox") and len(last_hit["bbox"]) >= 4:
            le, lv = bbox_foot_xy(last_hit["bbox"])

        stored_entry = str(first_hit.get("room_name") or "").strip()
        stored_exit = str(last_hit.get("room_name") or "").strip()
        ce = (
            {"name": stored_entry, "method": "stored", "dist_px": 0.0}
            if stored_entry
            else (
                classify_point_to_room(fe, fv, rooms)
                if (first_hit.get("bbox") and len(first_hit["bbox"]) >= 4)
                else {"name": "", "method": "none", "dist_px": 0.0}
            )
        )
        cx = (
            {"name": stored_exit, "method": "stored", "dist_px": 0.0}
            if stored_exit
            else (
                classify_point_to_room(le, lv, rooms)
                if (last_hit.get("bbox") and len(last_hit["bbox"]) >= 4)
                else {"name": "", "method": "none", "dist_px": 0.0}
            )
        )

        crop_names: List[str] = []
        crops: List[dict] = []
        for r in rows:
            rn = str(r.get("room_name") or "").strip()
            rm = "stored" if rn else "none"
            rd = 0.0
            if not rn and r.get("bbox") and len(r["bbox"]) >= 4:
                fx, fy = bbox_foot_xy(r["bbox"])
                c = classify_point_to_room(fx, fy, rooms)
                rn, rm, rd = str(c["name"]), str(c["method"]), float(c["dist_px"])
            crop_names.append(rn)
            crops.append(
                {
                    "meta_id": r.get("meta_id"),
                    "observation_id": r.get("observation_id") or r.get("meta_id"),
                    "global_person_id": r.get("global_person_id"),
                    "video_person_id": r.get("video_person_id"),
                    "track_id": r.get("track_id"),
                    "vid": r.get("vid"),
                    "time": float(r["time"]),
                    "score": float(r["score"]),
                    "raw_score": float(r.get("raw_score", 0.0)),
                    "path": r["path"],
                    "bbox": r.get("bbox"),
                    "room_id": r.get("room_id"),
                    "room_name": rn or "",
                    "room_method": rm,
                    "room_dist_px": rd,
                }
            )

        primary = _segment_primary_room(crop_names, ce["name"], cx["name"])
        g_ids = [int(c["global_person_id"]) for c in crops if c.get("global_person_id") is not None]
        primary_g = Counter(g_ids).most_common(1)[0][0] if g_ids else None

        return {
            "segment_idx": seg_idx,
            "global_person_id": primary_g,
            "room_name": primary,
            "entry_room": ce["name"] or "",
            "exit_room": cx["name"] or "",
            "entry_room_method": ce["method"],
            "exit_room_method": cx["method"],
            "entry_room_dist_px": ce["dist_px"],
            "exit_room_dist_px": cx["dist_px"],
            "start_sec": round(t0, 3),
            "end_sec": round(t1, 3),
            "duration_sec": round(duration, 3),
            "start_time": _fmt_hms(t0),
            "end_time": _fmt_hms(t1),
            "duration_time": _fmt_hms(duration),
            "hit_count": len(rows),
            "best_similarity": round(best_sim, 4),
            "entry_foot_x": round(fe, 1),
            "entry_foot_y": round(fv, 1),
            "exit_foot_x": round(le, 1),
            "exit_foot_y": round(lv, 1),
            "crops": crops,
        }

    segments: List[dict] = []
    seg_i = 1
    cur_rows: List[dict] = [ordered[0]]
    cur_end = float(ordered[0]["time"])

    for row in ordered[1:]:
        ts = float(row["time"])
        if ts <= cur_end + gap_sec:
            cur_end = max(cur_end, ts)
            cur_rows.append(row)
            continue
        segments.append(_finalize(cur_rows, seg_i))
        seg_i += 1
        cur_rows = [row]
        cur_start = ts
        cur_end = ts

    segments.append(_finalize(cur_rows, seg_i))
    return segments


def build_stay_segments_by_video(
    raw_by_video: Dict[str, List[dict]],
    engine,
    gap_sec: float,
) -> Dict[str, List[dict]]:
    vnames = list(raw_by_video.keys())
    rooms_map = _load_rooms_for_videos(engine, vnames)
    out: Dict[str, List[dict]] = {}
    for vn, hits in raw_by_video.items():
        if not hits:
            continue
        rooms = rooms_map.get(vn, [])
        out[vn] = merge_hits_to_stay_segments(hits, gap_sec, rooms)
    return out


def attach_image_urls_to_stay_segments(stay: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    if not stay:
        return {}
    out: Dict[str, List[dict]] = {}
    for vn, segs in stay.items():
        out[vn] = []
        for seg in segs:
            s = dict(seg)
            cs = []
            for c in seg.get("crops") or []:
                cc = dict(c)
                cc["imageUrl"] = _hit_image_url(c)
                cs.append(cc)
            s["crops"] = cs
            out[vn].append(s)
    return out


def _norm_feat(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    arr /= np.linalg.norm(arr) + 1e-8
    return arr


def perform_search(
    *,
    engine,
    algorithm: str,
    threshold: float,
    time_gap: float,
    group_mode: bool,
    co_time_threshold: float,
    q_text: Optional[str],
    image_bytes_list: List[bytes],
    image_names: List[str],
    include_stay_segments: bool = True,
) -> dict:
    """
    algorithm: 'OSNet' | 'SIGLIP'
    返回 dict: results（NMS 卡片）、stay_segments（阈值内全量命中按 gap 合并的段及段内 crops）、等。

    多张查询图：每条单独检索后并集（max）。
    SIGLIP 图文混搜（有图有文）：分别检索后按 meta_id 交集（见 run_siglip_mixed_search）。
    """
    _ensure_query_storage()
    saved_names: List[str] = []
    img_feats: List[np.ndarray] = []

    algo_upper = algorithm.upper()
    is_osnet = "OSNET" in algo_upper or algorithm == "OSNet"

    text_only_siglip = bool(q_text and (not image_bytes_list) and (not is_osnet))
    siglip_mixed = bool(q_text and image_bytes_list and (not is_osnet))

    for i, raw in enumerate(image_bytes_list):
        name = image_names[i] if i < len(image_names) else f"upload_{i}.jpg"
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        if is_osnet:
            f = extract_osnet_feat(bgr)
        else:
            f = extract_siglip_feat_img(bgr)
        if f is not None:
            img_feats.append(f)
        ext = os.path.splitext(name)[1] or ".jpg"
        unique_name = f"query_{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(QUERY_STORAGE, unique_name)
        with open(save_path, "wb") as wf:
            wf.write(raw)
        saved_names.append(unique_name)

    if text_only_siglip:
        keywords = _split_query_keywords(q_text or "")
        final, raw_by_video = run_text_siglip_search(
            keywords,
            float(threshold),
            time_gap,
            engine,
        )
        if final is None:
            raise ValueError("底库为空，请先完成视频建模")
        stay_segments: Dict[str, List[dict]] = {}
        if include_stay_segments and raw_by_video:
            stay_segments = attach_image_urls_to_stay_segments(
                build_stay_segments_by_video(raw_by_video, engine, time_gap)
            )
        return {
            "results": attach_image_urls(final),
            "stay_segments": stay_segments,
            "results_json": json.dumps(final) if final else "{}",
            "query_image_names": "",
            "algorithm": "SIGLIP",
            "score_scale": "contrastive_percent",
            "keywords": keywords,
            "rerank": "contrastive_min_pos",
        }

    if siglip_mixed and not group_mode:
        keywords = _split_query_keywords(q_text or "")
        final, raw_by_video = run_siglip_mixed_search(
            img_feats=img_feats,
            keywords=keywords,
            image_threshold=float(threshold),
            text_threshold_percent=TEXT_MIXED_TEXT_THRESHOLD,
            time_gap=time_gap,
            engine=engine,
        )
        if final is None:
            raise ValueError("底库为空，请先完成视频建模")
        stay_segments: Dict[str, List[dict]] = {}
        if include_stay_segments and raw_by_video:
            stay_segments = attach_image_urls_to_stay_segments(
                build_stay_segments_by_video(raw_by_video, engine, time_gap)
            )
        return {
            "results": attach_image_urls(final),
            "stay_segments": stay_segments,
            "results_json": json.dumps(final) if final else "{}",
            "query_image_names": ",".join(saved_names) if saved_names else "",
            "algorithm": "SIGLIP",
            "score_scale": "mixed_min_percent",
            "keywords": keywords,
            "rerank": "image_text_intersection",
            "text_threshold": TEXT_MIXED_TEXT_THRESHOLD,
        }

    feats: List[np.ndarray] = list(img_feats)
    if q_text and not is_osnet and not siglip_mixed:
        f = extract_siglip_feat_text(q_text.strip())
        if f is not None:
            feats.append(_norm_feat(f))

    if not feats:
        raise ValueError("请提供文字描述（SIGLIP）或至少一张有效查询图片")

    db_col = "feature_vector" if is_osnet else "clip_feature"
    algo_label = "OSNet" if is_osnet else "SIGLIP"

    if group_mode:
        final, raw_by_video = run_multi_target_search(
            feats,
            db_col,
            threshold,
            time_gap,
            True,
            co_time_threshold,
            engine,
            candidate_k=500,
            top_k_per_query=None,
        )
    else:
        final, raw_by_video = run_multi_target_search(
            feats,
            db_col,
            float(threshold),
            time_gap,
            False,
            0.0,
            engine,
            candidate_k=500,
            top_k_per_query=None,
        )

    if final is None:
        raise ValueError("底库为空，请先完成视频建模")

    stay_segments: Dict[str, List[dict]] = {}
    if include_stay_segments and raw_by_video:
        stay_segments = attach_image_urls_to_stay_segments(
            build_stay_segments_by_video(raw_by_video, engine, time_gap)
        )

    return {
        "results": attach_image_urls(final),
        "stay_segments": stay_segments,
        "results_json": json.dumps(final) if final else "{}",
        "query_image_names": ",".join(saved_names) if saved_names else "",
        "algorithm": algo_label,
    }
