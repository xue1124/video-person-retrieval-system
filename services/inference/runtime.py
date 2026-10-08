"""OSNet、SigLIP 与实时 YOLO 的进程内模型运行时。

本模块只负责模型的延迟加载和推理，不负责 HTTP、Celery 或数据库操作。
API 查询、Celery 视频任务和 RTSP 线程复用这里的同一套预处理逻辑，避免
业务服务反向依赖根目录 ``tasks.py``。
"""

from __future__ import annotations

import os
import threading
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoTokenizer
from ultralytics import YOLO

from services.config import ONNX_MODEL_DIR, PYTORCH_MODEL_DIR, TOKENIZER_DIR, model_path

_lock = threading.RLock()
_live_predict_lock = threading.Lock()
_reid_session: ort.InferenceSession | None = None
_siglip_vision_session: ort.InferenceSession | None = None
_siglip_text_session: ort.InferenceSession | None = None
_siglip_tokenizer: Any | None = None
_live_detector: Any | None = None


def _log(message: str) -> None:
    print(f"[INFERENCE][pid={os.getpid()}] {message}", flush=True)


def _providers(*, allow_tensorrt: bool = True) -> list[Any]:
    """仅选择当前 ONNX Runtime 实际提供的后端，最后始终保留 CPU。"""
    available = set(ort.get_available_providers())
    providers: list[Any] = []
    if allow_tensorrt and "TensorrtExecutionProvider" in available:
        providers.append(
            (
                "TensorrtExecutionProvider",
                {
                    "device_id": 0,
                    "trt_fp16_enable": False,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": "./models/cache",
                },
            )
        )
    if "CUDAExecutionProvider" in available:
        providers.append(("CUDAExecutionProvider", {"device_id": 0}))
    providers.append("CPUExecutionProvider")
    return providers


def _session(path: str, *, allow_tensorrt: bool = True) -> ort.InferenceSession:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"模型文件不存在: {path}")
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(
        path,
        sess_options=options,
        providers=_providers(allow_tensorrt=allow_tensorrt),
    )


def _get_osnet_session() -> ort.InferenceSession:
    global _reid_session
    with _lock:
        if _reid_session is None:
            path = str(
                model_path(
                    "OSNET_ONNX", ONNX_MODEL_DIR / "osnet_ain_msmt17_dynamic.onnx"
                )
            )
            _reid_session = _session(path)
            _log(f"OSNet ready | provider={_reid_session.get_providers()[0]}")
    return _reid_session


def _tokenizer_path() -> str:
    return str(model_path("SIGLIP_TOKENIZER_PATH", TOKENIZER_DIR))


def _load_local_tokenizer(path: str) -> Any:
    last_error: Exception | None = None
    for use_fast in (True, False):
        try:
            return AutoTokenizer.from_pretrained(
                path,
                local_files_only=True,
                trust_remote_code=True,
                use_fast=use_fast,
            )
        except Exception as exc:  # 两种 tokenizer 实现均失败后再统一报错
            last_error = exc
    raise RuntimeError(f"无法加载本地 SigLIP tokenizer: {path}") from last_error


def _get_siglip_vision_session() -> ort.InferenceSession:
    global _siglip_vision_session
    with _lock:
        if _siglip_vision_session is None:
            path = str(
                model_path(
                    "SIGLIP_VISION_ONNX", ONNX_MODEL_DIR / "siglip_vision.onnx"
                )
            )
            _siglip_vision_session = _session(path)
            _log(
                "SigLIP vision ready | "
                f"provider={_siglip_vision_session.get_providers()[0]}"
            )
    return _siglip_vision_session


def _get_siglip_text_resources() -> tuple[ort.InferenceSession, Any]:
    global _siglip_text_session, _siglip_tokenizer
    with _lock:
        if _siglip_text_session is None:
            path = str(
                model_path("SIGLIP_TEXT_ONNX", ONNX_MODEL_DIR / "siglip_text.onnx")
            )
            # 文本编码输入较小，不启用 TensorRT 编译缓存。
            _siglip_text_session = _session(path, allow_tensorrt=False)
            _log(
                "SigLIP text ready | "
                f"provider={_siglip_text_session.get_providers()[0]}"
            )
        if _siglip_tokenizer is None:
            _siglip_tokenizer = _load_local_tokenizer(_tokenizer_path())
    return _siglip_text_session, _siglip_tokenizer


def _session_info(session: ort.InferenceSession | None) -> dict[str, Any]:
    if session is None:
        return {"loaded": False}
    providers = session.get_providers()
    return {
        "loaded": True,
        "active_provider": providers[0] if providers else "unknown",
        "providers": providers,
        "inputs": [item.name for item in session.get_inputs()],
        "outputs": [item.name for item in session.get_outputs()],
    }


def get_runtime_backend_info() -> dict[str, Any]:
    """返回当前进程的加载状态；调用本函数不会主动加载模型。"""
    return {
        "pid": os.getpid(),
        "live_yolo_loaded": _live_detector is not None,
        "osnet": _session_info(_reid_session),
        "siglip_vision": _session_info(_siglip_vision_session),
        "siglip_text": _session_info(_siglip_text_session),
        "tokenizer_loaded": _siglip_tokenizer is not None,
        "tokenizer_path": _tokenizer_path(),
        "ort_available_providers": ort.get_available_providers(),
    }


def _output_name(session: ort.InferenceSession, preferred: str) -> str:
    names = [item.name for item in session.get_outputs()]
    return preferred if preferred in names else names[0]


def extract_siglip_feat_img_batch(images: list[np.ndarray]) -> np.ndarray:
    """批量提取并 L2 归一化 SigLIP 图像向量。"""
    if not images:
        return np.empty((0, 0), dtype=np.float32)
    session = _get_siglip_vision_session()
    resized = [
        cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (256, 256))
        for image in images
    ]
    batch = np.stack(resized).astype(np.float32) / 255.0
    batch = ((batch - 0.5) / 0.5).transpose(0, 3, 1, 2)
    features = session.run(
        [_output_name(session, "1726")],
        {session.get_inputs()[0].name: batch},
    )[0].astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    return features


def extract_siglip_feat_img(image: np.ndarray) -> np.ndarray | None:
    if image is None or image.size == 0:
        return None
    features = extract_siglip_feat_img_batch([image])
    return features[0] if len(features) else None


def _siglip_text_feed(session: ort.InferenceSession, tokenizer: Any, text: str) -> dict[str, np.ndarray]:
    tokens = tokenizer(
        [text],
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="np",
    )
    input_ids = tokens["input_ids"].astype(np.int64)
    attention = (
        tokens["attention_mask"].astype(np.int64)
        if "attention_mask" in tokens
        else None
    )
    feed: dict[str, np.ndarray] = {}
    for item in session.get_inputs():
        name = item.name.lower()
        if attention is not None and "attention" in name:
            feed[item.name] = attention
        elif "token_type" in name:
            feed[item.name] = np.zeros_like(input_ids, dtype=np.int64)
        else:
            feed[item.name] = input_ids
    return feed


def extract_siglip_feat_text(text: str) -> np.ndarray | None:
    """提取与 SigLIP 图像向量处于同一空间的文本向量。"""
    if not (text or "").strip():
        return None
    session, tokenizer = _get_siglip_text_resources()
    try:
        features = session.run(
            [_output_name(session, "text_embeds")],
            _siglip_text_feed(session, tokenizer, text),
        )[0].astype(np.float32)
        features = features.reshape(features.shape[0], -1)
        features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        return features[0]
    except Exception as exc:
        _log(f"SigLIP text inference failed: {exc}")
        return None


def extract_osnet_feat_batch(images: list[np.ndarray]) -> np.ndarray:
    """批量提取并 L2 归一化 OSNet 人物外观向量。"""
    if not images:
        return np.empty((0, 0), dtype=np.float32)
    session = _get_osnet_session()
    resized = [
        cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (128, 256))
        for image in images
    ]
    batch = np.stack(resized).astype(np.float32) / 255.0
    batch -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
    batch /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
    batch = batch.transpose(0, 3, 1, 2)
    features = session.run(None, {session.get_inputs()[0].name: batch})[0].astype(
        np.float32
    )
    features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    return features


def extract_osnet_feat(image: np.ndarray) -> np.ndarray | None:
    if image is None or image.size == 0:
        return None
    try:
        features = extract_osnet_feat_batch([image])
        return features[0] if len(features) else None
    except Exception as exc:
        _log(f"OSNet inference failed: {exc}")
        return None


def _get_live_detector() -> Any:
    global _live_detector
    with _lock:
        if _live_detector is None:
            live_model_path = str(
                model_path("LIVE_YOLO_MODEL", PYTORCH_MODEL_DIR / "yolov10n.pt")
            )
            if not os.path.isfile(live_model_path):
                raise FileNotFoundError(f"RTSP实时检测模型不存在: {live_model_path}")
            device: Any = 0 if torch.cuda.is_available() else "cpu"
            _live_detector = YOLO(live_model_path)
            if device != "cpu":
                _live_detector.to(device)
                _live_detector.model.float()
            _log(f"RTSP live YOLO ready | device={device}")
    return _live_detector


def live_yolo_predict(frame: np.ndarray, confidence: float) -> Any:
    """串行访问实时 YOLO，避免多个 RTSP 线程同时操作同一模型。"""
    detector = _get_live_detector()
    device: Any = 0 if torch.cuda.is_available() else "cpu"
    with _live_predict_lock:
        return detector.predict(
            frame,
            classes=0,
            conf=float(confidence),
            verbose=False,
            device=device,
            half=False,
        )


def warm_live_detector() -> None:
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    live_yolo_predict(dummy, 0.5)
    _log("RTSP live YOLO warmup complete")
