"""无跟踪人物建模：YOLO 检测、OSNet 跨帧批处理、单图特征聚类。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import faiss
import numpy as np

from services.analysis.pipeline.reid_match import (
    DEFAULT_MAX_SAMPLES,
    DEFAULT_MIN_MATCH_PAIRS,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    identity_match,
    normalize_rows,
    select_sample_indices,
)
from services.analysis.osnet_encoder import TrackingCancelled


def _sample_members(
    members: list[int],
    observations: list[dict[str, Any]],
    max_samples: int,
) -> list[int]:
    if len(members) <= max_samples:
        return list(members)
    areas = np.asarray([observations[index]["area"] for index in members], dtype=np.float32)
    chosen = select_sample_indices(len(members), max_samples=max_samples, areas=areas)
    return [members[int(index)] for index in chosen]


def _cluster_observations(
    observations: list[dict[str, Any]],
    *,
    threshold: float,
    max_samples: int,
    top_k: int,
    neighbor_k: int,
    min_pairs: int = DEFAULT_MIN_MATCH_PAIRS,
    progress_callback: Any | None = None,
) -> tuple[dict[int, list[int]], list[int]]:
    """按时间分配 ID，再用一对一两两匹配做受同帧约束的后合并。

    只用 OSNet `row["feature"]`。`siglip_feature` 即使存在也不得参与本函数。
    """
    if not observations:
        return {}, []

    features = normalize_rows(np.stack([row["feature"] for row in observations]))
    # HNSW 只找近邻候选，避免对所有 crops 构造 N² 相似度矩阵。
    index = faiss.IndexHNSWFlat(
        int(features.shape[1]), 32, faiss.METRIC_INNER_PRODUCT
    )
    index.hnsw.efConstruction = 80
    index.hnsw.efSearch = max(64, min(int(neighbor_k), 256))
    index.add(np.ascontiguousarray(features))
    k = min(max(1, neighbor_k), len(observations))

    assignments = [-1] * len(observations)
    clusters: dict[int, list[int]] = {}
    frame_occupancy: dict[int, set[int]] = {}
    next_person_id = 1
    search_block = 4096
    n_obs = len(observations)
    emit_every = max(500, n_obs // 20)

    for block_start in range(0, len(observations), search_block):
        block_end = min(len(observations), block_start + search_block)
        _scores, block_neighbors = index.search(
            np.ascontiguousarray(features[block_start:block_end]), k
        )
        for offset, neighbors in enumerate(block_neighbors):
            obs_index = block_start + offset
            observation = observations[obs_index]
            frame_index = int(observation["frame_index"])
            occupied = frame_occupancy.setdefault(frame_index, set())
            candidate_ids: list[int] = []
            seen: set[int] = set()
            for neighbor_index in neighbors:
                candidate_index = int(neighbor_index)
                if candidate_index >= obs_index:
                    continue
                person_id = assignments[candidate_index]
                if (
                    person_id > 0
                    and person_id not in occupied
                    and person_id not in seen
                ):
                    seen.add(person_id)
                    candidate_ids.append(person_id)

            best_person_id = -1
            best_score = -1.0
            query = features[obs_index : obs_index + 1]
            for person_id in candidate_ids:
                gallery_indices = _sample_members(
                    clusters[person_id], observations, max_samples
                )
                matched, score = identity_match(
                    query,
                    features[gallery_indices],
                    threshold=threshold,
                    min_pairs=min_pairs,
                )
                if matched and score > best_score:
                    best_score = score
                    best_person_id = person_id

            assign_score = None
            if best_person_id < 0:
                best_person_id = next_person_id
                next_person_id += 1
                clusters[best_person_id] = []
            else:
                assign_score = float(best_score)

            assignments[obs_index] = best_person_id
            clusters[best_person_id].append(obs_index)
            occupied.add(best_person_id)
            observations[obs_index]["assign_score"] = assign_score
            if progress_callback is not None and (
                (obs_index + 1) % emit_every == 0 or obs_index + 1 == n_obs
            ):
                progress_callback(0.75 + 0.08 * ((obs_index + 1) / n_obs))

    for obs_index, person_id in enumerate(assignments):
        observations[obs_index]["pre_merge_id"] = int(person_id)
        observations[obs_index]["merge_score"] = None

    # 后合并：只考察近邻命中过的候选组；一对一独立匹配对。
    candidate_pairs: set[tuple[int, int]] = set()
    for block_start in range(0, len(observations), search_block):
        block_end = min(len(observations), block_start + search_block)
        _scores, block_neighbors = index.search(
            np.ascontiguousarray(features[block_start:block_end]), k
        )
        for offset, neighbors in enumerate(block_neighbors):
            obs_index = block_start + offset
            first_id = assignments[obs_index]
            for neighbor_index in neighbors:
                second_index = int(neighbor_index)
                if second_index == obs_index:
                    continue
                second_id = assignments[second_index]
                if first_id != second_id:
                    candidate_pairs.add(tuple(sorted((first_id, second_id))))

    parent = {person_id: person_id for person_id in clusters}

    def root(person_id: int) -> int:
        while parent[person_id] != person_id:
            parent[person_id] = parent[parent[person_id]]
            person_id = parent[person_id]
        return person_id

    scored_pairs: list[tuple[float, int, int]] = []
    for first_id, second_id in candidate_pairs:
        first_samples = _sample_members(clusters[first_id], observations, max_samples)
        second_samples = _sample_members(clusters[second_id], observations, max_samples)
        matched, score = identity_match(
            features[first_samples],
            features[second_samples],
            threshold=threshold,
            min_pairs=min_pairs,
        )
        if matched:
            scored_pairs.append((score, first_id, second_id))
    scored_pairs.sort(reverse=True)

    for _score, first_id, second_id in scored_pairs:
        first_root, second_root = root(first_id), root(second_id)
        if first_root == second_root:
            continue
        first_samples = _sample_members(
            clusters[first_root], observations, max_samples
        )
        second_samples = _sample_members(
            clusters[second_root], observations, max_samples
        )
        matched, score = identity_match(
            features[first_samples],
            features[second_samples],
            threshold=threshold,
            min_pairs=min_pairs,
        )
        if not matched:
            continue
        keep, remove = sorted((first_root, second_root))
        parent[remove] = keep
        merge_score = float(score)
        for member_index in clusters[remove]:
            if observations[member_index].get("merge_score") is None:
                observations[member_index]["merge_score"] = merge_score
        clusters[keep].extend(clusters[remove])
        del clusters[remove]

    if progress_callback is not None:
        progress_callback(0.86)

    # 最终 ID 连续编号，便于前端展示。
    ordered_roots = sorted(
        clusters,
        key=lambda person_id: min(clusters[person_id]),
    )
    compact = {old_id: new_id for new_id, old_id in enumerate(ordered_roots, 1)}
    final_clusters = {
        compact[old_id]: sorted(members)
        for old_id, members in clusters.items()
    }
    final_assignments = [0] * len(observations)
    for person_id, members in final_clusters.items():
        for index_value in members:
            final_assignments[index_value] = person_id
    return final_clusters, final_assignments


def run_detection_reid(
    input_path: Path,
    *,
    output_dir: Path,
    detector: Any,
    encoder: Any,
    infer_lock: Any,
    detector_model: str,
    osnet_model: str,
    process_fps: float = 12.0,
    frame_step: int | None = None,
    imgsz: int = 640,
    conf: float = 0.60,
    max_frames: int = 0,
    crop_batch_size: int = 128,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    top_k: int = DEFAULT_TOP_K,
    min_pairs: int = DEFAULT_MIN_MATCH_PAIRS,
    neighbor_k: int = 256,
    progress_callback: Any | None = None,
    cancel_check: Any | None = None,
    siglip_encoder: Any | None = None,
) -> tuple[Path, Path]:
    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.is_file():
        raise FileNotFoundError(f"输入视频不存在: {input_path}")
    if crop_batch_size <= 0 or max_samples <= 0 or min_pairs <= 0:
        raise ValueError("crop_batch_size、max_samples、min_pairs 必须大于 0")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {input_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    if source_fps <= 0:
        source_fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # 优先使用前端 skip_frames（每 N 帧取 1 帧）；否则按 process_fps 换算。
    if frame_step is not None and int(frame_step) > 0:
        frame_step = max(1, int(frame_step))
    else:
        if process_fps <= 0:
            raise ValueError("process_fps 必须大于 0")
        frame_step = max(1, int(round(source_fps / process_fps)))
    effective_fps = source_fps / frame_step
    planned_frames = max(
        1,
        (
            min(total_frames, max_frames * frame_step)
            if max_frames
            else total_frames
        )
        // frame_step,
    )

    observations: list[dict[str, Any]] = []
    pending_crops: list[np.ndarray] = []
    pending_rows: list[dict[str, Any]] = []
    crop_dir = output_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop_seq = 0
    siglip_model_name = ""
    siglip_model_version = ""
    siglip_dim = 0
    if siglip_encoder is not None:
        siglip_model_name = str(getattr(siglip_encoder, "model_name", "") or "")
        siglip_model_version = str(getattr(siglip_encoder, "model_version", "") or "")

    def flush_crops() -> None:
        nonlocal crop_seq, siglip_dim
        if not pending_crops:
            return
        inner = getattr(encoder, "_inner", encoder)
        with infer_lock:
            batch_features = inner.encode_crops(pending_crops)
            batch_siglip = None
            if siglip_encoder is not None:
                batch_siglip = siglip_encoder.encode_crops(pending_crops)
        if len(batch_features) != len(pending_rows):
            raise RuntimeError("OSNet 输出数量与 crop 数量不一致")
        if batch_siglip is not None and len(batch_siglip) != len(pending_rows):
            raise RuntimeError("SigLIP 输出数量与 crop 数量不一致")
        for i, (row, feature) in enumerate(zip(pending_rows, batch_features)):
            crop_seq += 1
            rel = f"crops/{crop_seq:08d}.jpg"
            dest = output_dir / rel
            wrote = bool(
                cv2.imwrite(str(dest), pending_crops[i], [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            )
            row["feature"] = np.asarray(feature, dtype=np.float32)
            if batch_siglip is not None:
                row["siglip_feature"] = np.asarray(batch_siglip[i], dtype=np.float32)
                if siglip_dim == 0:
                    siglip_dim = int(row["siglip_feature"].reshape(-1).shape[0])
            row["crop_file"] = rel if wrote and dest.is_file() and dest.stat().st_size > 0 else None
            observations.append(row)
        pending_crops.clear()
        pending_rows.clear()

    def _emit_progress(ratio: float, person_count: int | None = None) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(float(ratio), person_count)
        except TypeError:
            progress_callback(float(ratio))

    source_frame_index = -1
    processed_frames = 0
    started_at = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            source_frame_index += 1
            if source_frame_index % frame_step != 0:
                continue
            if max_frames and processed_frames >= max_frames:
                break

            with infer_lock:
                prediction = detector.predict(
                    frame,
                    classes=[0],
                    conf=conf,
                    imgsz=imgsz,
                    verbose=False,
                )[0]
            boxes = prediction.boxes
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.detach().cpu().numpy()
                confidences = boxes.conf.detach().cpu().numpy()
                for bbox, confidence in zip(xyxy, confidences):
                    x1, y1, x2, y2 = (float(value) for value in bbox)
                    left = max(0, min(width - 1, int(np.floor(x1))))
                    top = max(0, min(height - 1, int(np.floor(y1))))
                    right = max(left + 1, min(width, int(np.ceil(x2))))
                    bottom = max(top + 1, min(height, int(np.ceil(y2))))
                    crop = frame[top:bottom, left:right]
                    if crop.size == 0:
                        continue
                    pending_crops.append(crop.copy())
                    pending_rows.append(
                        {
                            "frame_index": source_frame_index,
                            "timestamp_sec": source_frame_index / source_fps,
                            "bbox": [x1, y1, x2, y2],
                            "confidence": float(confidence),
                            "area": max(1.0, (x2 - x1) * (y2 - y1)),
                        }
                    )
                    if len(pending_crops) >= crop_batch_size:
                        flush_crops()

            processed_frames += 1
            if processed_frames % 50 == 0 or processed_frames == planned_frames:
                ratio = min(1.0, processed_frames / planned_frames)
                print(
                    f"PROGRESS {processed_frames}/{planned_frames} ratio={ratio:.4f} "
                    f"crops={len(observations) + len(pending_rows)}",
                    flush=True,
                )
                if cancel_check is not None and not bool(cancel_check()):
                    raise TrackingCancelled(
                        f"cancelled after {processed_frames} frames"
                    )
                if progress_callback is not None:
                    # 抽帧检测+提特征：0 → 75%
                    _emit_progress(0.75 * ratio)
    finally:
        capture.release()
    flush_crops()

    print(f"[detection_reid] 开始聚类 {len(observations)} 个 crops", flush=True)
    _emit_progress(0.75)
    clusters, assignments = _cluster_observations(
        observations,
        threshold=similarity_threshold,
        max_samples=max_samples,
        top_k=top_k,
        neighbor_k=neighbor_k,
        min_pairs=min_pairs,
        progress_callback=_emit_progress,
    )
    kept = {
        person_id: members
        for person_id, members in clusters.items()
        if len(members) >= 2
    }
    dropped = len(clusters) - len(kept)
    if dropped:
        print(
            f"[detection_reid] 丢弃仅 1 帧人物 {dropped} 个，剩余 {len(kept)}",
            flush=True,
        )
    clusters = {
        new_id: sorted(members)
        for new_id, (_old_id, members) in enumerate(
            sorted(kept.items(), key=lambda item: min(item[1])),
            1,
        )
    }
    _emit_progress(0.88, len(clusters))
    features = (
        normalize_rows(np.stack([row["feature"] for row in observations]))
        if observations
        else np.empty((0, 512), dtype=np.float32)
    )

    embeddings: dict[str, np.ndarray] = {}
    persons_json: list[dict[str, Any]] = []
    tracks_json: list[dict[str, Any]] = []
    for person_id, members in sorted(clusters.items()):
        sample_members = _sample_members(members, observations, max_samples)
        gallery = features[sample_members]
        person_key = f"person_{person_id:06d}"
        track_key = f"track_{person_id:06d}"
        embeddings[person_key] = gallery
        embeddings[track_key] = gallery
        points_by_frame: dict[int, dict[str, Any]] = {}
        merge_scores: list[float] = []
        for index in members:
            observation = observations[index]
            assign_score = observation.get("assign_score")
            merge_score = observation.get("merge_score")
            if merge_score is not None:
                merge_scores.append(float(merge_score))
            frame_index = int(observation["frame_index"])
            confidence = round(float(observation["confidence"]), 4)
            existing = points_by_frame.get(frame_index)
            if existing is not None and float(existing["confidence"]) >= confidence:
                continue
            points_by_frame[frame_index] = {
                "frame_index": frame_index,
                "timestamp_sec": round(float(observation["timestamp_sec"]), 3),
                "bbox": [
                    round(float(value), 2)
                    for value in observation["bbox"]
                ],
                "confidence": confidence,
                "crop_file": observation.get("crop_file"),
                "pre_merge_id": observation.get("pre_merge_id"),
                "assign_score": (
                    round(float(assign_score), 4)
                    if assign_score is not None
                    else None
                ),
                "merge_score": (
                    round(float(merge_score), 4)
                    if merge_score is not None
                    else None
                ),
                "_osnet": observation["feature"],
                "_siglip": observation.get("siglip_feature"),
            }
        points = [points_by_frame[key] for key in sorted(points_by_frame)]
        start_sec = min(point["timestamp_sec"] for point in points)
        end_sec = max(point["timestamp_sec"] for point in points)
        persons_json.append(
            {
                "global_person_id": person_id,
                "local_track_ids": [person_id],
                "start_sec": start_sec,
                "end_sec": end_sec,
                "profile_updates": len(members),
                "sample_count": len(sample_members),
                "embedding_key": person_key,
            }
        )
        tracks_json.append(
            {
                "local_track_id": person_id,
                "global_person_id": person_id,
                "preconsolidation_global_person_id": person_id,
                "global_match_similarity": (
                    round(max(merge_scores), 4) if merge_scores else None
                ),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": round(end_sec - start_sec, 3),
                "observation_count": len(points),
                "embedding_key": track_key,
                "points": points,
            }
        )

    obs_osnet: list[np.ndarray] = []
    obs_siglip: list[np.ndarray] = []
    obs_track_ids: list[int] = []
    obs_frames: list[int] = []
    for track in tracks_json:
        local_tid = int(track["local_track_id"])
        cleaned_points: list[dict[str, Any]] = []
        for point in track["points"]:
            osnet_vec = point.pop("_osnet", None)
            siglip_vec = point.pop("_siglip", None)
            cleaned_points.append(point)
            if osnet_vec is None or not point.get("crop_file"):
                continue
            obs_osnet.append(np.asarray(osnet_vec, dtype=np.float32).reshape(-1))
            obs_track_ids.append(local_tid)
            obs_frames.append(int(point["frame_index"]))
            if siglip_vec is not None:
                obs_siglip.append(np.asarray(siglip_vec, dtype=np.float32).reshape(-1))
        track["points"] = cleaned_points

    if siglip_encoder is not None and obs_osnet and len(obs_siglip) != len(obs_osnet):
        raise RuntimeError(
            f"SigLIP 向量数量 {len(obs_siglip)} 与 OSNet 观测 {len(obs_osnet)} 不一致，"
            "拒绝入库以免检索底库残缺"
        )
    if obs_osnet:
        embeddings["observation_osnet"] = np.stack(obs_osnet).astype(np.float32)
        embeddings["observation_local_track_id"] = np.asarray(obs_track_ids, dtype=np.int32)
        embeddings["observation_frame_index"] = np.asarray(obs_frames, dtype=np.int32)
    if obs_siglip:
        embeddings["observation_siglip"] = np.stack(obs_siglip).astype(np.float32)
        if siglip_dim == 0:
            siglip_dim = int(embeddings["observation_siglip"].shape[1])

    elapsed = time.perf_counter() - started_at
    osnet_dim = int(features.shape[1]) if features.size else 0
    metadata = {
        "detector_model": detector_model,
        "osnet_model": osnet_model,
        "osnet_dim": osnet_dim,
        "siglip_model": siglip_model_name,
        "siglip_model_version": siglip_model_version,
        "siglip_dim": siglip_dim,
        "dual_embeddings": bool(obs_siglip),
        "identity_merge_backend": "osnet",
        "tracker_name": "none",
        "pipeline": "yolo_osnet_clustering",
        "source_fps": round(source_fps, 4),
        "effective_fps": round(effective_fps, 4),
        "process_fps_request": process_fps,
        "frame_step": frame_step,
        "skip_frames": frame_step,
        "frame_size": [width, height],
        "source_total_frames": total_frames,
        "processed_frames": processed_frames,
        "crop_count": len(observations),
        "observation_count": len(obs_osnet),
        "elapsed_sec": round(elapsed, 3),
        "processing_fps": round(processed_frames / max(elapsed, 1e-6), 3),
        "similarity_threshold": similarity_threshold,
        "reid_max_samples": max_samples,
        "reid_min_pairs": min_pairs,
        "reid_similarity_mode": "pairwise_independent_pairs",
        "same_frame_merge_forbidden": False,
        "osnet_provider": getattr(getattr(encoder, "_inner", encoder), "provider", None),
    }
    embeddings_path = output_dir / "track_embeddings.npz"
    np.savez_compressed(embeddings_path, **embeddings)
    json_path = output_dir / "tracks.json"
    json_path.write_text(
        json.dumps(
            {
                "schema_version": 4 if obs_siglip else 3,
                "source_video": str(input_path),
                "metadata": metadata,
                "global_person_count": len(persons_json),
                "posthoc_merges": [],
                "persons": persons_json,
                "tracks": tracks_json,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[detection_reid] 完成: {processed_frames} 帧, "
        f"{len(observations)} crops, {len(persons_json)} 人物, {elapsed:.1f}s",
        flush=True,
    )
    return json_path, embeddings_path
