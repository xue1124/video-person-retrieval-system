"""建模用 SigLIP 视觉编码器。只提图像向量，禁止参与身份聚类/合并。"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from services.config import ONNX_MODEL_DIR, PROJECT_ROOT

_lock = threading.RLock()
_session: ort.InferenceSession | None = None
_model_name = ""
_model_version = ""
_output_name = "1726"


def _vision_onnx_path() -> Path:
    raw = os.environ.get("SIGLIP_VISION_ONNX", str(ONNX_MODEL_DIR / "siglip_vision.onnx")).strip()
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _model_version_for(path: Path) -> str:
    if not path.is_file():
        return ""
    stat = path.stat()
    return f"{stat.st_size:x}-{stat.st_mtime_ns:x}"[:32]


def _providers() -> list:
    available = set(ort.get_available_providers())
    providers: list = []
    # 与 YOLO TensorRT 同进程时不用 ORT-TRT，避免编译冲突
    if "CUDAExecutionProvider" in available:
        providers.append(("CUDAExecutionProvider", {"device_id": 0}))
    providers.append("CPUExecutionProvider")
    return providers


def get_siglip_vision_session() -> tuple[ort.InferenceSession, str, str]:
    global _session, _model_name, _model_version, _output_name
    with _lock:
        if _session is not None:
            return _session, _model_name, _model_version
        path = _vision_onnx_path()
        if not path.is_file():
            raise FileNotFoundError(f"找不到 SigLIP 视觉模型: {path}")
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        sess = ort.InferenceSession(
            str(path), sess_options=opts, providers=_providers()
        )
        outputs = [o.name for o in sess.get_outputs()]
        if "1726" in outputs:
            _output_name = "1726"
        else:
            _output_name = outputs[0]
        _session = sess
        _model_name = path.name
        _model_version = _model_version_for(path)
        print(
            f"[tracking_v3.siglip] vision ready | model={_model_name} | "
            f"ver={_model_version} | active={sess.get_providers()[0]} | "
            f"out={_output_name}",
            flush=True,
        )
        return _session, _model_name, _model_version


def preprocess_bgr_crops(crops: list[np.ndarray]) -> np.ndarray:
    """与统一推理服务保持一致：RGB / 256 / [-1,1]。"""
    resized = [
        cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (256, 256)) for img in crops
    ]
    batch = np.stack(resized, axis=0).astype(np.float32)
    batch /= 255.0
    batch = (batch - 0.5) / 0.5
    return np.transpose(batch, (0, 3, 1, 2))


def encode_siglip_crops(crops: list[np.ndarray]) -> np.ndarray:
    """返回 (N, D) float32 L2 归一化向量。不用于身份合并。"""
    if not crops:
        return np.zeros((0, 0), dtype=np.float32)
    sess, _, _ = get_siglip_vision_session()
    batch = preprocess_bgr_crops(crops)
    inp = sess.get_inputs()[0].name
    feats = sess.run([_output_name], {inp: batch})[0]
    feats = np.asarray(feats, dtype=np.float32)
    if feats.ndim == 1:
        feats = feats.reshape(1, -1)
    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
    return feats / norms


class SiglipVisionEncoder:
    """与 OSNet encoder 相同的 encode_crops 接口；输出不得送入聚类。"""

    def encode_crops(self, crops: list[np.ndarray]) -> np.ndarray:
        return encode_siglip_crops(crops)

    @property
    def model_name(self) -> str:
        _, name, _ = get_siglip_vision_session()
        return name

    @property
    def model_version(self) -> str:
        _, _, ver = get_siglip_vision_session()
        return ver


def get_siglip_encoder() -> SiglipVisionEncoder:
    get_siglip_vision_session()
    return SiglipVisionEncoder()


def siglip_model_info() -> dict[str, Any]:
    _, name, ver = get_siglip_vision_session()
    return {"model_name": name, "model_version": ver}
