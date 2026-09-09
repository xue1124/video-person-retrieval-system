#!/usr/bin/env python3
"""离线验证 YOLO + BoT-SORT + OSNet，并输出可审查视频与结构化轨迹。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import torch
import yaml
from ultralytics import YOLO
from ultralytics.trackers.bot_sort import BOTSORT

from reid_match import (
    DEFAULT_MAX_SAMPLES,
    DEFAULT_MIN_MATCH_PAIRS,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    identity_match,
    normalize_rows,
    select_feature_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OSNET = PROJECT_ROOT / "osnet_ain_msmt17_dynamic.onnx"
DEFAULT_TRACKER = Path(__file__).with_name("botsort_osnet.yaml")


class TrackingCancelled(Exception):
    """用户删除任务或任务被取代时中止跟踪。"""


class OSNetEncoder:
    """将 BoT-SORT 的检测框批量转换为 L2 归一化 OSNet 特征。"""

    def __init__(
        self,
        model_path: Path,
        *,
        prefer_trt: bool = False,
        session: ort.InferenceSession | None = None,
        trt_cache_dir: Path | None = None,
    ) -> None:
        if session is not None:
            self.session = session
        else:
            available = set(ort.get_available_providers())
            providers: list[Any] = []
            if prefer_trt and "TensorrtExecutionProvider" in available:
                cache = trt_cache_dir or (PROJECT_ROOT / "models" / "cache" / "tracking_v3")
                cache.mkdir(parents=True, exist_ok=True)
                providers.append(
                    (
                        "TensorrtExecutionProvider",
                        {
                            "device_id": 0,
                            "trt_fp16_enable": True,
                            "trt_engine_cache_enable": True,
                            "trt_engine_cache_path": str(cache),
                        },
                    )
                )
            if "CUDAExecutionProvider" in available:
                providers.append(("CUDAExecutionProvider", {"device_id": 0}))
            providers.append("CPUExecutionProvider")
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=providers,
            )
        self.input_name = self.session.get_inputs()[0].name
        self.provider = self.session.get_providers()[0]
        self.total_calls = 0
        self.total_crops = 0
        self.total_sec = 0.0
        self.preprocess_sec = 0.0
        self.infer_sec = 0.0

    def reset_stats(self) -> None:
        self.total_calls = 0
        self.total_crops = 0
        self.total_sec = 0.0
        self.preprocess_sec = 0.0
        self.infer_sec = 0.0

    def _preprocess_crops(self, image: np.ndarray, detections: np.ndarray) -> np.ndarray:
        crops: list[np.ndarray] = []
        height, width = image.shape[:2]

        for cx, cy, box_w, box_h, *_ in detections:
            x1 = max(0, min(width - 1, int(round(cx - box_w / 2))))
            y1 = max(0, min(height - 1, int(round(cy - box_h / 2))))
            x2 = max(x1 + 1, min(width, int(round(cx + box_w / 2))))
            y2 = max(y1 + 1, min(height, int(round(cy + box_h / 2))))
            crop = image[y1:y2, x1:x2]
            crop = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (128, 256))
            crops.append(crop)

        if not crops:
            return np.empty((0, 3, 256, 128), dtype=np.float32)

        batch = np.stack(crops).astype(np.float32) / 255.0
        batch -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
        batch /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return batch.transpose(0, 3, 1, 2)

    @staticmethod
    def preprocess_crop_images(crops: list[np.ndarray]) -> np.ndarray:
        """预处理已经裁好的 BGR 人像，供跨帧大 batch 推理。"""
        if not crops:
            return np.empty((0, 3, 256, 128), dtype=np.float32)
        resized = [
            cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (128, 256))
            for crop in crops
        ]
        batch = np.stack(resized).astype(np.float32) / 255.0
        batch -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
        batch /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return batch.transpose(0, 3, 1, 2)

    def encode_crops(self, crops: list[np.ndarray]) -> np.ndarray:
        """跨帧批量提取单图特征。"""
        t0 = time.perf_counter()
        batch = self.preprocess_crop_images(crops)
        t1 = time.perf_counter()
        if batch.shape[0] == 0:
            return np.empty((0, 512), dtype=np.float32)
        features = self.session.run(None, {self.input_name: batch})[0].astype(np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        t2 = time.perf_counter()
        self.total_calls += 1
        self.total_crops += int(batch.shape[0])
        self.preprocess_sec += t1 - t0
        self.infer_sec += t2 - t1
        self.total_sec += t2 - t0
        return features

    def __call__(self, image: np.ndarray, detections: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        batch = self._preprocess_crops(image, detections)
        t1 = time.perf_counter()
        if batch.shape[0] == 0:
            self.total_calls += 1
            self.total_sec += t1 - t0
            self.preprocess_sec += t1 - t0
            return np.empty((0, 512), dtype=np.float32)

        t2 = time.perf_counter()
        features = self.session.run(None, {self.input_name: batch})[0].astype(np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        t3 = time.perf_counter()

        self.total_calls += 1
        self.total_crops += int(batch.shape[0])
        self.preprocess_sec += t2 - t0
        self.infer_sec += t3 - t2
        self.total_sec += t3 - t0
        return features


class TensorRTOSNetEncoder(OSNetEncoder):
    """原生 TensorRT OSNet，支持动态 batch（与 Ultralytics YOLO.engine 同进程共存）。"""

    def __init__(self, engine_path: Path, *, max_batch: int | None = None) -> None:
        import tensorrt as trt

        self.engine_path = Path(engine_path)
        if not self.engine_path.is_file():
            raise FileNotFoundError(f"OSNet TensorRT engine 不存在: {self.engine_path}")
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        self._engine = self._runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self._engine is None:
            raise RuntimeError(f"无法反序列化 OSNet engine: {self.engine_path}")
        self._context = self._engine.create_execution_context()
        self._input_name = None
        self._output_name = None
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            mode = self._engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self._input_name = name
            else:
                self._output_name = name
        if not self._input_name or not self._output_name:
            raise RuntimeError("OSNet engine 缺少 input/output tensor")

        in_shape = tuple(self._engine.get_tensor_shape(self._input_name))
        out_shape = tuple(self._engine.get_tensor_shape(self._output_name))
        self._feat_dim = int(out_shape[-1]) if out_shape else 512
        # 动态 batch：shape[0] 为 -1；静态则为固定值
        engine_max = 1
        if in_shape and int(in_shape[0]) > 0:
            engine_max = int(in_shape[0])
            self._dynamic = False
        else:
            self._dynamic = True
            # 从 optimization profile 读取 max
            try:
                vmin, vopt, vmax = self._engine.get_tensor_profile_shape(self._input_name, 0)
                engine_max = int(vmax[0])
            except Exception:
                engine_max = 16
        if max_batch is not None and max_batch > 0:
            engine_max = min(engine_max, int(max_batch))
        elif self._dynamic:
            # 未指定时默认封顶 4（与 YOLO.engine 同卡更稳）
            engine_max = min(engine_max, 4)
        self._max_batch = max(1, engine_max)

        self._d_input = torch.empty(
            (self._max_batch, 3, 256, 128), device="cuda", dtype=torch.float32
        )
        self._d_output = torch.empty(
            (self._max_batch, self._feat_dim), device="cuda", dtype=torch.float32
        )
        self._context.set_tensor_address(self._input_name, int(self._d_input.data_ptr()))
        self._context.set_tensor_address(self._output_name, int(self._d_output.data_ptr()))
        self.provider = f"TensorRT(batch<={self._max_batch})"
        self.session = None  # type: ignore[assignment]
        self.input_name = self._input_name
        self.reset_stats()

    def _infer_batch_once(self, batch: np.ndarray) -> np.ndarray:
        n = int(batch.shape[0])
        if self._dynamic:
            ok_shape = self._context.set_input_shape(self._input_name, (n, 3, 256, 128))
            if ok_shape is False:
                raise RuntimeError(f"OSNet set_input_shape 失败: batch={n}")

        stream = torch.cuda.current_stream()
        self._d_input[:n].copy_(torch.from_numpy(batch), non_blocking=True)
        ok = self._context.execute_async_v3(stream.cuda_stream)
        if not ok:
            # 清掉可能残留的 CUDA error，便于上层拆半重试
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            raise RuntimeError("OSNet TensorRT execute_async_v3 失败")
        stream.synchronize()
        return self._d_output[:n].detach().float().cpu().numpy()

    def _infer_batch(self, batch: np.ndarray) -> np.ndarray:
        n = int(batch.shape[0])
        if n > self._max_batch:
            parts = [
                self._infer_batch(batch[i : i + self._max_batch])
                for i in range(0, n, self._max_batch)
            ]
            return np.concatenate(parts, axis=0)

        try:
            return self._infer_batch_once(batch)
        except Exception:
            if n <= 1:
                raise
            mid = max(1, n // 2)
            left = self._infer_batch(batch[:mid])
            right = self._infer_batch(batch[mid:])
            return np.concatenate([left, right], axis=0)

    def __call__(self, image: np.ndarray, detections: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        batch = self._preprocess_crops(image, detections)
        t1 = time.perf_counter()
        n = int(batch.shape[0])
        if n == 0:
            self.total_calls += 1
            self.total_sec += t1 - t0
            self.preprocess_sec += t1 - t0
            return np.empty((0, self._feat_dim), dtype=np.float32)

        t2 = time.perf_counter()
        features = self._infer_batch(batch).astype(np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        t3 = time.perf_counter()

        self.total_calls += 1
        self.total_crops += n
        self.preprocess_sec += t2 - t0
        self.infer_sec += t3 - t2
        self.total_sec += t3 - t0
        return features

    def encode_crops(self, crops: list[np.ndarray]) -> np.ndarray:
        """跨帧批量提取单图特征，按 TensorRT profile 自动分块。"""
        t0 = time.perf_counter()
        batch = self.preprocess_crop_images(crops)
        t1 = time.perf_counter()
        n = int(batch.shape[0])
        if n == 0:
            return np.empty((0, self._feat_dim), dtype=np.float32)
        features = self._infer_batch(batch).astype(np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        t2 = time.perf_counter()
        self.total_calls += 1
        self.total_crops += n
        self.preprocess_sec += t1 - t0
        self.infer_sec += t2 - t1
        self.total_sec += t2 - t0
        return features


class GlobalIdentityResolver:
    """在 BoT-SORT 本地轨迹断开后，用单图 OSNet 特征恢复视频内人物 ID。"""

    def __init__(
        self,
        similarity_threshold: float,
        max_gap_sec: float,
        min_observations: int,
        max_observations: int,
        *,
        max_samples: int = DEFAULT_MAX_SAMPLES,
        top_k: int = DEFAULT_TOP_K,
        min_pairs: int = DEFAULT_MIN_MATCH_PAIRS,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_gap_sec = max_gap_sec
        self.min_observations = min_observations
        self.max_observations = max_observations
        self.max_samples = max_samples
        self.top_k = top_k
        self.min_pairs = min_pairs
        self.local_to_global: dict[int, int] = {}
        self.local_match_similarity: dict[int, float | None] = {}
        # 每人保留多帧单图特征；匹配时再均匀抽 max_samples 张。
        self.gallery_features: dict[int, list[np.ndarray]] = {}
        self.gallery_areas: dict[int, list[float]] = {}
        self.last_seen_sec: dict[int, float] = {}
        self.profile_updates: dict[int, int] = {}
        self.pending_features: dict[int, list[np.ndarray]] = {}
        self.pending_areas: dict[int, list[float]] = {}
        self.identity_owner: dict[int, int] = {}
        self.local_first_seen_sec: dict[int, float] = {}
        self.local_last_seen_sec: dict[int, float] = {}
        self.preconsolidation_global_ids: dict[int, int] = {}
        self.posthoc_merges: list[dict[str, int | float]] = []
        self.next_global_id = 1

    @property
    def profiles(self) -> dict[int, np.ndarray]:
        """导出用：每人抽后的单图特征矩阵 (N, D)。"""
        return {
            global_id: self._gallery_matrix(global_id)
            for global_id in self.gallery_features
        }

    @staticmethod
    def _normalize(feature: np.ndarray) -> np.ndarray:
        return feature / (np.linalg.norm(feature) + 1e-8)

    def _gallery_matrix(self, global_id: int) -> np.ndarray:
        return select_feature_matrix(
            self.gallery_features[global_id],
            areas=self.gallery_areas.get(global_id),
            max_samples=self.max_samples,
        )

    def _pending_matrix(self, local_id: int) -> np.ndarray:
        return select_feature_matrix(
            self.pending_features[local_id],
            areas=self.pending_areas.get(local_id),
            max_samples=self.max_samples,
        )

    def _append_pending(self, local_id: int, feature: np.ndarray, area: float) -> None:
        self.pending_features.setdefault(local_id, []).append(self._normalize(feature.copy()))
        self.pending_areas.setdefault(local_id, []).append(float(area))

    def _append_gallery(self, global_id: int, feature: np.ndarray, area: float) -> None:
        self.gallery_features.setdefault(global_id, []).append(self._normalize(feature.copy()))
        self.gallery_areas.setdefault(global_id, []).append(float(area))
        self.profile_updates[global_id] = self.profile_updates.get(global_id, 0) + 1

    def _create_identity(
        self,
        local_id: int,
        features: list[np.ndarray] | np.ndarray,
        areas: list[float] | None = None,
    ) -> int:
        global_id = self.next_global_id
        self.next_global_id += 1
        if isinstance(features, np.ndarray):
            matrix = normalize_rows(features)
            feature_list = [matrix[index] for index in range(matrix.shape[0])]
        else:
            feature_list = [self._normalize(feature) for feature in features]
        if not feature_list:
            raise ValueError(f"无法为空轨迹 T{local_id} 创建全局人物")
        area_list = (
            [float(value) for value in areas]
            if areas is not None and len(areas) == len(feature_list)
            else [1.0] * len(feature_list)
        )
        self.gallery_features[global_id] = feature_list
        self.gallery_areas[global_id] = area_list
        self.profile_updates[global_id] = len(feature_list)
        self.identity_owner[global_id] = local_id
        self.local_to_global[local_id] = global_id
        self.local_match_similarity[local_id] = None
        self.last_seen_sec[global_id] = self.local_last_seen_sec.get(
            local_id, self.last_seen_sec.get(global_id, 0.0)
        )
        self.pending_features.pop(local_id, None)
        self.pending_areas.pop(local_id, None)
        return global_id

    def _create_identity_from_pending(self, local_id: int) -> int:
        return self._create_identity(
            local_id,
            self.pending_features[local_id],
            self.pending_areas.get(local_id),
        )

    def _overlaps_global(self, local_id: int, global_id: int) -> bool:
        new_start = self.local_first_seen_sec[local_id]
        new_end = self.local_last_seen_sec[local_id]
        return any(
            assigned_global_id == global_id
            and max(new_start, self.local_first_seen_sec[assigned_local_id])
            <= min(new_end, self.local_last_seen_sec[assigned_local_id])
            for assigned_local_id, assigned_global_id in self.local_to_global.items()
            if assigned_local_id != local_id
        )

    def _try_historical_match(
        self,
        local_id: int,
        timestamp_sec: float,
        occupied_global_ids: set[int],
    ) -> bool:
        query = self._pending_matrix(local_id)
        best_global_id: int | None = None
        best_similarity = -1.0
        for global_id in self.gallery_features:
            if global_id in occupied_global_ids:
                continue
            if self._overlaps_global(local_id, global_id):
                continue
            gap_sec = timestamp_sec - self.last_seen_sec.get(global_id, timestamp_sec)
            # max_gap_sec <= 0 表示不限制离开时长。
            if gap_sec < 0 or (
                self.max_gap_sec > 0 and gap_sec > self.max_gap_sec
            ):
                continue
            matched, similarity = identity_match(
                query,
                self._gallery_matrix(global_id),
                threshold=self.similarity_threshold,
                min_pairs=self.min_pairs,
            )
            if not matched:
                continue
            if similarity > best_similarity:
                best_similarity = similarity
                best_global_id = global_id

        if best_global_id is None:
            return False

        pending = self.pending_features.get(local_id, [])
        pending_areas = self.pending_areas.get(local_id, [])
        for index, feature in enumerate(pending):
            area = pending_areas[index] if index < len(pending_areas) else 1.0
            self._append_gallery(best_global_id, feature, area)
        self.local_to_global[local_id] = best_global_id
        self.local_match_similarity[local_id] = best_similarity
        self.pending_features.pop(local_id, None)
        self.pending_areas.pop(local_id, None)
        self.last_seen_sec[best_global_id] = timestamp_sec
        occupied_global_ids.add(best_global_id)
        return True

    def _resolve_active_conflicts(
        self,
        current_local_ids: set[int],
        active_embeddings: dict[int, np.ndarray],
        active_areas: dict[int, float],
    ) -> None:
        grouped: dict[int, list[int]] = {}
        for local_id in current_local_ids:
            global_id = self.local_to_global.get(local_id)
            if global_id is not None:
                grouped.setdefault(global_id, []).append(local_id)

        for global_id, local_ids in grouped.items():
            if len(local_ids) < 2:
                continue
            owner = self.identity_owner.get(global_id)
            keeper = owner if owner in local_ids else min(local_ids)
            for local_id in local_ids:
                if local_id == keeper:
                    continue
                feature = active_embeddings.get(local_id)
                if feature is not None:
                    self._create_identity(
                        local_id,
                        [self._normalize(feature)],
                        [active_areas.get(local_id, 1.0)],
                    )

    def finalize_pending(self) -> None:
        """处理视频结尾仍不足以完成判断的短轨迹。"""
        for local_id in list(self.pending_features):
            self._create_identity_from_pending(local_id)

    def consolidate_identities(
        self,
        summaries: dict[int, dict[str, Any]],
        similarity_threshold: float,
    ) -> None:
        """用单图一对一两两匹配二次合并时间不重叠的全局人物。"""
        self.preconsolidation_global_ids = dict(self.local_to_global)
        members: dict[int, list[int]] = {
            global_id: [
                local_id
                for local_id, assigned_id in self.local_to_global.items()
                if assigned_id == global_id
            ]
            for global_id in self.gallery_features
        }

        def start_sec(global_id: int) -> float:
            return min(float(summaries[local_id]["start_sec"]) for local_id in members[global_id])

        def groups_are_separate(first_id: int, second_id: int) -> bool:
            for first_local_id in members[first_id]:
                first = summaries[first_local_id]
                for second_local_id in members[second_id]:
                    second = summaries[second_local_id]
                    overlap = max(
                        float(first["start_sec"]), float(second["start_sec"])
                    ) <= min(float(first["end_sec"]), float(second["end_sec"]))
                    if overlap:
                        return False
                    gap = min(
                        abs(float(first["start_sec"]) - float(second["end_sec"])),
                        abs(float(second["start_sec"]) - float(first["end_sec"])),
                    )
                    # max_gap_sec <= 0 表示不限制离开时长。
                    if self.max_gap_sec > 0 and gap > self.max_gap_sec:
                        return False
            return True

        canonical_ids: list[int] = []
        for global_id in sorted(list(self.gallery_features), key=start_sec):
            best_target: int | None = None
            best_similarity = -1.0
            query = self._gallery_matrix(global_id)
            for candidate_id in canonical_ids:
                if not groups_are_separate(candidate_id, global_id):
                    continue
                matched, similarity = identity_match(
                    query,
                    self._gallery_matrix(candidate_id),
                    threshold=similarity_threshold,
                    min_pairs=self.min_pairs,
                )
                if not matched:
                    continue
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_target = candidate_id

            if best_target is None:
                canonical_ids.append(global_id)
                continue

            self.gallery_features[best_target].extend(self.gallery_features[global_id])
            self.gallery_areas[best_target].extend(self.gallery_areas[global_id])
            self.profile_updates[best_target] += self.profile_updates[global_id]
            self.last_seen_sec[best_target] = max(
                self.last_seen_sec.get(best_target, 0.0),
                self.last_seen_sec.get(global_id, 0.0),
            )
            for local_id in members[global_id]:
                self.local_to_global[local_id] = best_target
            members[best_target].extend(members[global_id])
            self.posthoc_merges.append(
                {
                    "from_global_person_id": global_id,
                    "to_global_person_id": best_target,
                    "similarity": round(best_similarity, 6),
                }
            )
            del members[global_id]
            del self.gallery_features[global_id]
            del self.gallery_areas[global_id]
            del self.profile_updates[global_id]
            self.last_seen_sec.pop(global_id, None)
            self.identity_owner.pop(global_id, None)

    def update(
        self,
        tracks: np.ndarray,
        tracker: BOTSORT,
        timestamp_sec: float,
    ) -> dict[int, int]:
        active_embeddings = {
            int(track.track_id): np.asarray(track.smooth_feat, dtype=np.float32)
            for track in tracker.tracked_stracks
            if track.is_activated and track.smooth_feat is not None
        }
        active_areas = {
            int(row[4]): max(1.0, float(row[2] - row[0]) * float(row[3] - row[1]))
            for row in tracks
        }
        current_local_ids = {int(row[4]) for row in tracks}
        for local_id in current_local_ids:
            self.local_first_seen_sec.setdefault(local_id, timestamp_sec)
            self.local_last_seen_sec[local_id] = timestamp_sec
        self._resolve_active_conflicts(
            current_local_ids, active_embeddings, active_areas
        )
        occupied_global_ids = {
            self.local_to_global[local_id]
            for local_id in current_local_ids
            if local_id in self.local_to_global
        }
        newly_matched: set[int] = set()

        for local_id in sorted(current_local_ids):
            if local_id in self.local_to_global:
                continue
            feature = active_embeddings.get(local_id)
            if feature is None:
                continue
            self._append_pending(local_id, feature, active_areas.get(local_id, 1.0))
            pending = self.pending_features[local_id]
            if len(pending) < self.min_observations:
                continue

            matched = self._try_historical_match(
                local_id,
                timestamp_sec,
                occupied_global_ids,
            )
            if matched:
                newly_matched.add(local_id)
            elif len(pending) >= self.max_observations:
                global_id = self._create_identity_from_pending(local_id)
                occupied_global_ids.add(global_id)
                newly_matched.add(local_id)

        for local_id in current_local_ids:
            if local_id in newly_matched:
                # 刚建档/合并时 pending 已写入 gallery，避免当前帧重复计入。
                continue
            global_id = self.local_to_global.get(local_id)
            feature = active_embeddings.get(local_id)
            if global_id is None or feature is None:
                continue
            self._append_gallery(global_id, feature, active_areas.get(local_id, 1.0))
            self.last_seen_sec[global_id] = timestamp_sec

        known_tracker_ids = {
            int(track.track_id)
            for track in [*tracker.tracked_stracks, *tracker.lost_stracks]
        }
        for local_id in list(self.pending_features):
            if local_id not in known_tracker_ids:
                self._create_identity_from_pending(local_id)

        return {
            local_id: self.local_to_global[local_id]
            for local_id in current_local_ids
            if local_id in self.local_to_global
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="独立验证 YOLO + BoT-SORT + OSNet，不写入业务数据库。"
    )
    parser.add_argument("input", type=Path, help="输入视频路径")
    parser.add_argument(
        "--model",
        default=str(PROJECT_ROOT / "yolo11m.pt"),
        help="YOLO 模型路径或名称，默认使用项目根目录的 yolo11m.pt",
    )
    parser.add_argument("--osnet-model", type=Path, default=DEFAULT_OSNET)
    parser.add_argument("--tracker", type=Path, default=DEFAULT_TRACKER)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--process-fps", type=float, default=12.0)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--half", action="store_true", help="CUDA 上使用 FP16 检测")
    parser.add_argument(
        "--global-match-thresh",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="断轨后合并为同一人物：独立匹配对阈值（单图余弦）",
    )
    parser.add_argument(
        "--global-max-gap",
        type=float,
        default=0.0,
        help="允许恢复同一人物 ID 的最大离开时长（秒）；0 表示不限制",
    )
    parser.add_argument(
        "--global-min-observations",
        type=int,
        default=5,
        help="开始尝试全局人物匹配前至少积累的轨迹帧数",
    )
    parser.add_argument(
        "--global-max-observations",
        type=int,
        default=15,
        help="持续尝试匹配的最大轨迹帧数，之后确认为新人物",
    )
    parser.add_argument(
        "--post-merge-thresh",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="分析结束后二次合并：独立匹配对阈值（单图余弦）",
    )
    parser.add_argument(
        "--reid-max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="每人/每轨迹参与匹配时最多保留的单图特征数",
    )
    parser.add_argument(
        "--reid-min-pairs",
        type=int,
        default=DEFAULT_MIN_MATCH_PAIRS,
        help="二次合并至少需要多少组独立匹配对；短轨迹则要求全部样本达标",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="最多分析多少个采样帧，0 表示处理完整视频",
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="跳过最终标注视频渲染，仅输出 JSON 与特征（正式入库推荐）",
    )
    parser.add_argument(
        "--prefer-trt",
        action="store_true",
        help="OSNet 优先使用 ONNX Runtime TensorRT EP（首次会编译 engine cache）",
    )
    return parser.parse_args()


def load_tracker(config_path: Path, encoder: OSNetEncoder, frame_rate: float) -> BOTSORT:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("tracker_type") != "botsort":
        raise ValueError("当前实验工具仅支持 tracker_type: botsort")

    # BOTSORT 原生 ReID 加载器仅接受 YOLO 分类模型；先跳过它，再注入 OSNet。
    config["with_reid"] = False
    tracker_args = SimpleNamespace(**config)
    tracker = BOTSORT(args=tracker_args, frame_rate=max(1, int(round(frame_rate))))
    tracker.args.with_reid = True
    tracker.encoder = encoder
    return tracker


def resolve_model(model: str) -> str:
    candidate = Path(model)
    if candidate.is_absolute() or candidate.exists():
        return str(candidate)
    project_candidate = PROJECT_ROOT / candidate
    return str(project_candidate) if project_candidate.exists() else model


def color_for_track(track_id: int) -> tuple[int, int, int]:
    return (
        64 + (track_id * 47) % 192,
        64 + (track_id * 89) % 192,
        64 + (track_id * 137) % 192,
    )


def draw_tracks(
    frame: np.ndarray,
    tracks: np.ndarray,
    global_ids: dict[int, int],
) -> np.ndarray:
    annotated = frame.copy()
    for row in tracks:
        x1, y1, x2, y2 = (int(round(value)) for value in row[:4])
        track_id = int(row[4])
        global_id = global_ids.get(track_id)
        confidence = float(row[5])
        color = color_for_track(global_id if global_id is not None else track_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = (
            f"P{global_id} / T{track_id}  {confidence:.2f}"
            if global_id is not None
            else f"T{track_id}  {confidence:.2f}"
        )
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
    return annotated


def update_summaries(
    summaries: dict[int, dict[str, Any]],
    tracks: np.ndarray,
    tracker: BOTSORT,
    identity_resolver: GlobalIdentityResolver,
    global_ids: dict[int, int],
    frame_index: int,
    timestamp_sec: float,
) -> None:
    active_embeddings = {
        int(track.track_id): np.asarray(track.smooth_feat, dtype=np.float32)
        for track in tracker.tracked_stracks
        if track.is_activated and track.smooth_feat is not None
    }

    for row in tracks:
        x1, y1, x2, y2 = (float(value) for value in row[:4])
        track_id = int(row[4])
        global_id = global_ids.get(track_id)
        confidence = float(row[5])
        item = summaries.setdefault(
            track_id,
            {
                "local_track_id": track_id,
                "global_person_id": global_id,
                "global_match_similarity": identity_resolver.local_match_similarity.get(
                    track_id
                ),
                "start_sec": timestamp_sec,
                "end_sec": timestamp_sec,
                "observation_count": 0,
                "points": [],
                "_embedding": None,
            },
        )
        if global_id is not None:
            item["global_person_id"] = global_id
            item["global_match_similarity"] = (
                identity_resolver.local_match_similarity.get(track_id)
            )
        item["end_sec"] = timestamp_sec
        item["observation_count"] += 1
        item["points"].append(
            {
                "frame_index": frame_index,
                "timestamp_sec": round(timestamp_sec, 3),
                "bbox": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2),
                ],
                "confidence": round(confidence, 4),
            }
        )
        if track_id in active_embeddings:
            item["_embedding"] = active_embeddings[track_id]


def render_final_video(
    input_path: Path,
    output_path: Path,
    summaries: dict[int, dict[str, Any]],
    identity_resolver: GlobalIdentityResolver,
    effective_fps: float,
    frame_step: int,
    frame_size: tuple[int, int],
    processed_frames: int,
) -> None:
    """根据最终全局人物映射重新绘制视频，确保二次合并反映到画面。"""
    points_by_frame: dict[int, list[list[float]]] = {}
    for local_id, item in summaries.items():
        for point in item["points"]:
            points_by_frame.setdefault(int(point["frame_index"]), []).append(
                [
                    *[float(value) for value in point["bbox"]],
                    float(local_id),
                    float(point["confidence"]),
                    0.0,
                    0.0,
                ]
            )

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法重新打开视频生成最终标注: {input_path}")
    temporary_path = output_path.with_name(f"{output_path.stem}.rendering.mp4")
    writer = cv2.VideoWriter(
        str(temporary_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        effective_fps,
        frame_size,
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"无法创建最终标注视频: {temporary_path}")

    source_frame_index = -1
    written_frames = 0
    final_global_ids = dict(identity_resolver.local_to_global)
    try:
        while written_frames < processed_frames:
            ok, frame = capture.read()
            if not ok:
                break
            source_frame_index += 1
            if source_frame_index % frame_step != 0:
                continue
            rows = points_by_frame.get(source_frame_index, [])
            tracks = (
                np.asarray(rows, dtype=np.float32)
                if rows
                else np.empty((0, 8), dtype=np.float32)
            )
            writer.write(draw_tracks(frame, tracks, final_global_ids))
            written_frames += 1
    finally:
        capture.release()
        writer.release()

    if written_frames != processed_frames:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"最终视频帧数不一致: 期望 {processed_frames}，实际 {written_frames}"
        )
    temporary_path.replace(output_path)


def write_outputs(
    output_dir: Path,
    input_path: Path,
    metadata: dict[str, Any],
    summaries: dict[int, dict[str, Any]],
    identity_resolver: GlobalIdentityResolver,
) -> tuple[Path, Path]:
    embeddings: dict[str, np.ndarray] = {}
    tracks_json: list[dict[str, Any]] = []

    for track_id in sorted(summaries):
        item = summaries[track_id]
        embedding = item.get("_embedding")
        embedding_key = f"track_{track_id:06d}"
        if embedding is not None:
            embeddings[embedding_key] = embedding
        public_item = {key: value for key, value in item.items() if not key.startswith("_")}
        public_item["global_person_id"] = identity_resolver.local_to_global.get(
            track_id, public_item.get("global_person_id")
        )
        public_item["preconsolidation_global_person_id"] = (
            identity_resolver.preconsolidation_global_ids.get(track_id)
        )
        public_item["global_match_similarity"] = (
            identity_resolver.local_match_similarity.get(track_id)
        )
        public_item["duration_sec"] = round(item["end_sec"] - item["start_sec"], 3)
        public_item["embedding_key"] = embedding_key if embedding is not None else None
        tracks_json.append(public_item)

    persons_json: list[dict[str, Any]] = []
    for global_id in sorted(identity_resolver.gallery_features):
        embedding_key = f"person_{global_id:06d}"
        sample_matrix = identity_resolver.profiles[global_id]
        embeddings[embedding_key] = sample_matrix
        person_tracks = [
            item
            for local_id, item in summaries.items()
            if identity_resolver.local_to_global.get(local_id) == global_id
        ]
        persons_json.append(
            {
                "global_person_id": global_id,
                "local_track_ids": sorted(
                    int(item["local_track_id"]) for item in person_tracks
                ),
                "start_sec": min(
                    (float(item["start_sec"]) for item in person_tracks),
                    default=None,
                ),
                "end_sec": max(
                    (float(item["end_sec"]) for item in person_tracks),
                    default=None,
                ),
                "profile_updates": identity_resolver.profile_updates[global_id],
                "sample_count": int(sample_matrix.shape[0]),
                "embedding_key": embedding_key,
            }
        )

    embeddings_path = output_dir / "track_embeddings.npz"
    np.savez_compressed(embeddings_path, **embeddings)

    json_path = output_dir / "tracks.json"
    payload = {
        "schema_version": 3,
        "source_video": str(input_path.resolve()),
        "metadata": metadata,
        "global_person_count": len(persons_json),
        "posthoc_merges": identity_resolver.posthoc_merges,
        "persons": persons_json,
        "tracks": tracks_json,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return json_path, embeddings_path


def run_tracking(
    input_path: Path,
    *,
    output_dir: Path | None = None,
    model: str | None = None,
    osnet_model: Path | None = None,
    tracker_config: Path | None = None,
    detector: Any | None = None,
    encoder: Any | None = None,
    process_fps: float = 12.0,
    conf: float = 0.20,
    iou: float = 0.70,
    imgsz: int = 640,
    device: str | None = None,
    half: bool = True,
    global_match_thresh: float = DEFAULT_SIMILARITY_THRESHOLD,
    global_max_gap: float = 0.0,
    global_min_observations: int = 5,
    global_max_observations: int = 15,
    post_merge_thresh: float = DEFAULT_SIMILARITY_THRESHOLD,
    reid_max_samples: int = DEFAULT_MAX_SAMPLES,
    reid_top_k: int = DEFAULT_TOP_K,
    reid_min_pairs: int = DEFAULT_MIN_MATCH_PAIRS,
    max_frames: int = 0,
    skip_render: bool = False,
    prefer_trt: bool = False,
    progress_callback: Any | None = None,
    cancel_check: Any | None = None,
    infer_lock: Any | None = None,
) -> tuple[Path, Path]:
    """进程内跟踪主入口；可注入常驻 detector/encoder。

    cancel_check: 返回 False 时立即中止（前端删除任务后 DB 行消失）。
    """
    input_path = Path(input_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"输入视频不存在: {input_path}")

    osnet_path = Path(osnet_model or DEFAULT_OSNET).expanduser().resolve()
    tracker_path = Path(tracker_config or DEFAULT_TRACKER).expanduser().resolve()
    if encoder is None and not osnet_path.is_file():
        raise FileNotFoundError(f"OSNet 模型不存在: {osnet_path}")
    if not tracker_path.is_file():
        raise FileNotFoundError(f"跟踪配置不存在: {tracker_path}")
    if process_fps <= 0:
        raise ValueError("process_fps 必须大于 0")
    if not 0.0 <= global_match_thresh <= 1.0:
        raise ValueError("global_match_thresh 必须在 0 到 1 之间")
    if global_max_gap < 0:
        raise ValueError("global_max_gap 不能为负数；0 表示不限制")
    if global_min_observations <= 0:
        raise ValueError("global_min_observations 必须大于 0")
    if global_max_observations < global_min_observations:
        raise ValueError("global_max_observations 不能小于 global_min_observations")
    if not 0.0 <= post_merge_thresh <= 1.0:
        raise ValueError("post_merge_thresh 必须在 0 到 1 之间")
    if reid_max_samples <= 0:
        raise ValueError("reid_max_samples 必须大于 0")
    if reid_min_pairs <= 0:
        raise ValueError("reid_min_pairs 必须大于 0")

    out_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else Path(__file__).with_name("outputs") / input_path.stem
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {input_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, int(round(source_fps / process_fps)))
    effective_fps = source_fps / frame_step
    device = device or ("0" if torch.cuda.is_available() else "cpu")
    model_name = model or str(PROJECT_ROOT / "yolo11m.pt")

    output_video = out_dir / "tracked.mp4"

    if encoder is None:
        encoder = OSNetEncoder(osnet_path, prefer_trt=prefer_trt)
    elif hasattr(encoder, "reset_stats"):
        encoder.reset_stats()
    tracker = load_tracker(tracker_path, encoder, effective_fps)
    identity_resolver = GlobalIdentityResolver(
        similarity_threshold=global_match_thresh,
        max_gap_sec=global_max_gap,
        min_observations=global_min_observations,
        max_observations=global_max_observations,
        max_samples=reid_max_samples,
        top_k=reid_top_k,
        min_pairs=reid_min_pairs,
    )
    if detector is None:
        detector = YOLO(resolve_model(model_name))

    print(
        f"输入={input_path.name} | 原始FPS={source_fps:.2f} | "
        f"分析FPS={effective_fps:.2f} | 计划帧数≈"
        f"{(min(total_frames, max_frames * frame_step) if max_frames else total_frames) // frame_step} | "
        f"OSNet={getattr(encoder, 'provider', '?')}",
        flush=True,
    )

    summaries: dict[int, dict[str, Any]] = {}
    source_frame_index = -1
    processed_frames = 0
    started_at = time.perf_counter()
    detect_sec = 0.0
    track_sec = 0.0
    other_sec = 0.0
    planned_frames = max(
        1,
        (
            min(total_frames, max_frames * frame_step)
            if max_frames
            else total_frames
        )
        // frame_step,
    )

    use_half = bool(half and device != "cpu" and not str(model_name).endswith(".engine"))

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

            t_detect0 = time.perf_counter()
            predict_kwargs = dict(
                classes=[0],
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                half=use_half,
                verbose=False,
            )
            if infer_lock is not None:
                with infer_lock:
                    prediction = detector.predict(frame, **predict_kwargs)[0]
            else:
                prediction = detector.predict(frame, **predict_kwargs)[0]
            detections = prediction.boxes.cpu().numpy()
            t_detect1 = time.perf_counter()
            detect_sec += t_detect1 - t_detect0

            # tracker.update 内部会调用 OSNet（帧内检测框批处理）
            tracks = tracker.update(detections, frame)
            t_track1 = time.perf_counter()
            track_sec += t_track1 - t_detect1

            timestamp_sec = source_frame_index / source_fps
            global_ids = identity_resolver.update(
                tracks,
                tracker,
                timestamp_sec,
            )

            update_summaries(
                summaries,
                tracks,
                tracker,
                identity_resolver,
                global_ids,
                source_frame_index,
                timestamp_sec,
            )
            other_sec += time.perf_counter() - t_track1
            processed_frames += 1

            # 线程池下 revoke 杀不掉任务；轮询 cancel_check（删除媒体源后返回 False）
            if cancel_check is not None and (
                processed_frames == 1 or processed_frames % 25 == 0
            ):
                try:
                    still_ok = bool(cancel_check())
                except Exception as exc:
                    print(f"cancel_check 失败，视为继续: {exc}", flush=True)
                    still_ok = True
                if not still_ok:
                    print(
                        f"跟踪已取消（用户删除或任务过期），已处理 {processed_frames} 帧",
                        flush=True,
                    )
                    raise TrackingCancelled(
                        f"cancelled after {processed_frames} frames"
                    )

            if processed_frames % 50 == 0 or processed_frames == planned_frames:
                elapsed = time.perf_counter() - started_at
                ratio = min(1.0, processed_frames / planned_frames)
                print(
                    f"PROGRESS {processed_frames}/{planned_frames} ratio={ratio:.4f}",
                    flush=True,
                )
                print(
                    f"已分析 {processed_frames}/{planned_frames} 帧，速度 "
                    f"{processed_frames / max(elapsed, 1e-6):.2f} FPS，"
                    f"本地轨迹 {len(summaries)}，"
                    f"全局人物 {len(identity_resolver.profiles)}，"
                    f"待判断 {len(identity_resolver.pending_features)}",
                    flush=True,
                )
                if progress_callback is not None:
                    try:
                        progress_callback(ratio)
                    except TrackingCancelled:
                        raise
                    except Exception as exc:
                        # 允许上层用异常取消
                        if exc.__class__.__name__ in {
                            "MediaSourceGone",
                            "TrackingCancelled",
                        }:
                            raise
                        print(f"progress_callback 失败: {exc}", flush=True)
    finally:
        capture.release()

    identity_resolver.finalize_pending()
    identity_resolver.consolidate_identities(
        summaries,
        similarity_threshold=post_merge_thresh,
    )
    elapsed = time.perf_counter() - started_at
    detect_pct = 100.0 * detect_sec / max(elapsed, 1e-6)
    track_pct = 100.0 * track_sec / max(elapsed, 1e-6)
    encoder_total = float(getattr(encoder, "total_sec", 0.0) or 0.0)
    encoder_infer = float(getattr(encoder, "infer_sec", 0.0) or 0.0)
    encoder_prep = float(getattr(encoder, "preprocess_sec", 0.0) or 0.0)
    encoder_crops = int(getattr(encoder, "total_crops", 0) or 0)
    osnet_pct = 100.0 * encoder_total / max(elapsed, 1e-6)
    print(
        f"耗时拆分：总 {elapsed:.1f}s | "
        f"YOLO检测 {detect_sec:.1f}s ({detect_pct:.0f}%) | "
        f"跟踪+ReID {track_sec:.1f}s ({track_pct:.0f}%) | "
        f"其中OSNet {encoder_total:.1f}s "
        f"(推理 {encoder_infer:.1f}s / 裁切预处理 {encoder_prep:.1f}s, "
        f"{encoder_crops} crops, {osnet_pct:.0f}%) | "
        f"其余 {other_sec:.1f}s",
        flush=True,
    )
    if skip_render:
        print("已跳过最终视频渲染。", flush=True)
    else:
        print(
            f"二次合并 {len(identity_resolver.posthoc_merges)} 组身份，正在生成最终视频...",
            flush=True,
        )
        render_final_video(
            input_path=input_path,
            output_path=output_video,
            summaries=summaries,
            identity_resolver=identity_resolver,
            effective_fps=effective_fps,
            frame_step=frame_step,
            frame_size=(width, height),
            processed_frames=processed_frames,
        )
    metadata = {
        "detector_model": model_name,
        "osnet_model": str(osnet_path),
        "tracker_config": str(tracker_path),
        "source_fps": round(source_fps, 4),
        "effective_fps": round(effective_fps, 4),
        "frame_step": frame_step,
        "frame_size": [width, height],
        "source_total_frames": total_frames,
        "processed_frames": processed_frames,
        "elapsed_sec": round(elapsed, 3),
        "processing_fps": round(processed_frames / max(elapsed, 1e-6), 3),
        "global_match_threshold": global_match_thresh,
        "global_max_gap_sec": global_max_gap,
        "global_min_observations": global_min_observations,
        "global_max_observations": global_max_observations,
        "post_merge_threshold": post_merge_thresh,
        "reid_max_samples": reid_max_samples,
        "reid_min_pairs": reid_min_pairs,
        "reid_similarity_mode": "pairwise_independent_pairs",
        "posthoc_merge_count": len(identity_resolver.posthoc_merges),
        "osnet_provider": getattr(encoder, "provider", None),
    }
    json_path, embeddings_path = write_outputs(
        out_dir,
        input_path,
        metadata,
        summaries,
        identity_resolver,
    )
    print(
        f"完成：{len(summaries)} 条本地轨迹，"
        f"{len(identity_resolver.profiles)} 个全局人物",
        flush=True,
    )
    if not skip_render:
        print(f"标注视频：{output_video}", flush=True)
    print(f"轨迹数据：{json_path}", flush=True)
    print(f"轨迹特征：{embeddings_path}", flush=True)
    return json_path, embeddings_path


def main() -> int:
    args = parse_args()
    prefer_trt = bool(getattr(args, "prefer_trt", False))
    run_tracking(
        args.input,
        output_dir=args.output_dir,
        model=args.model,
        osnet_model=args.osnet_model,
        tracker_config=args.tracker,
        process_fps=args.process_fps,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        half=args.half,
        global_match_thresh=args.global_match_thresh,
        global_max_gap=args.global_max_gap,
        global_min_observations=args.global_min_observations,
        global_max_observations=args.global_max_observations,
        post_merge_thresh=args.post_merge_thresh,
        reid_max_samples=args.reid_max_samples,
        reid_min_pairs=args.reid_min_pairs,
        max_frames=args.max_frames,
        skip_render=args.skip_render,
        prefer_trt=prefer_trt,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中止。", file=sys.stderr)
        raise SystemExit(130)
