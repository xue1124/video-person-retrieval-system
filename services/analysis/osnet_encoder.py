"""OSNet crop feature encoders shared by the formal analysis pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import torch

from services.config import PROJECT_ROOT


class TrackingCancelled(Exception):
    """Raised when a deleted or superseded analysis task should stop."""


class OSNetEncoder:
    """Convert person crops to L2-normalized OSNet feature vectors."""

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
        self.reset_stats()

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
        """Preprocess already-cropped BGR person images for batch inference."""
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
        """Extract features from a cross-frame batch of person crops."""
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
        features = self.session.run(None, {self.input_name: batch})[0].astype(np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        t2 = time.perf_counter()
        self.total_calls += 1
        self.total_crops += int(batch.shape[0])
        self.preprocess_sec += t1 - t0
        self.infer_sec += t2 - t1
        self.total_sec += t2 - t0
        return features


class TensorRTOSNetEncoder(OSNetEncoder):
    """Native TensorRT OSNet encoder with dynamic-batch support."""

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
        for index in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(index)
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
        engine_max = 1
        if in_shape and int(in_shape[0]) > 0:
            engine_max = int(in_shape[0])
            self._dynamic = False
        else:
            self._dynamic = True
            try:
                _, _, maximum = self._engine.get_tensor_profile_shape(self._input_name, 0)
                engine_max = int(maximum[0])
            except Exception:
                engine_max = 16
        if max_batch is not None and max_batch > 0:
            engine_max = min(engine_max, int(max_batch))
        elif self._dynamic:
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
        count = int(batch.shape[0])
        if self._dynamic:
            ok_shape = self._context.set_input_shape(
                self._input_name, (count, 3, 256, 128)
            )
            if ok_shape is False:
                raise RuntimeError(f"OSNet set_input_shape 失败: batch={count}")
        stream = torch.cuda.current_stream()
        self._d_input[:count].copy_(torch.from_numpy(batch), non_blocking=True)
        ok = self._context.execute_async_v3(stream.cuda_stream)
        if not ok:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            raise RuntimeError("OSNet TensorRT execute_async_v3 失败")
        stream.synchronize()
        return self._d_output[:count].detach().float().cpu().numpy()

    def _infer_batch(self, batch: np.ndarray) -> np.ndarray:
        count = int(batch.shape[0])
        if count > self._max_batch:
            parts = [
                self._infer_batch(batch[index : index + self._max_batch])
                for index in range(0, count, self._max_batch)
            ]
            return np.concatenate(parts, axis=0)
        try:
            return self._infer_batch_once(batch)
        except Exception:
            if count <= 1:
                raise
            midpoint = max(1, count // 2)
            return np.concatenate(
                [self._infer_batch(batch[:midpoint]), self._infer_batch(batch[midpoint:])],
                axis=0,
            )

    def __call__(self, image: np.ndarray, detections: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        batch = self._preprocess_crops(image, detections)
        t1 = time.perf_counter()
        count = int(batch.shape[0])
        if count == 0:
            self.total_calls += 1
            self.total_sec += t1 - t0
            self.preprocess_sec += t1 - t0
            return np.empty((0, self._feat_dim), dtype=np.float32)
        features = self._infer_batch(batch).astype(np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        t2 = time.perf_counter()
        self.total_calls += 1
        self.total_crops += count
        self.preprocess_sec += t1 - t0
        self.infer_sec += t2 - t1
        self.total_sec += t2 - t0
        return features

    def encode_crops(self, crops: list[np.ndarray]) -> np.ndarray:
        """Extract features in chunks that fit the TensorRT profile."""
        t0 = time.perf_counter()
        batch = self.preprocess_crop_images(crops)
        t1 = time.perf_counter()
        count = int(batch.shape[0])
        if count == 0:
            return np.empty((0, self._feat_dim), dtype=np.float32)
        features = self._infer_batch(batch).astype(np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        t2 = time.perf_counter()
        self.total_calls += 1
        self.total_crops += count
        self.preprocess_sec += t1 - t0
        self.infer_sec += t2 - t1
        self.total_sec += t2 - t0
        return features
