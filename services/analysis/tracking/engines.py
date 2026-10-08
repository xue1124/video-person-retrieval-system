"""常驻 YOLO(TensorRT) + OSNet(ORT TensorRT) 引擎。

Celery 使用 threads 池时，TRT/ORT 会话非线程安全，推理统一走同一把锁。
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from services.config import ONNX_MODEL_DIR, PROJECT_ROOT, PYTORCH_MODEL_DIR, TENSORRT_MODEL_DIR

_lock = threading.RLock()
_detector: Any = None
_encoder: Any = None
_loaded = False
_load_error: str | None = None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _default_yolo_path() -> Path:
    engine = TENSORRT_MODEL_DIR / "yolo11m.engine"
    try:
        import torch

        cuda_ready = torch.cuda.is_available()
    except Exception:
        cuda_ready = False
    if engine.is_file() and cuda_ready:
        return engine
    return PYTORCH_MODEL_DIR / "yolo11m.pt"


def _default_osnet_engine_path() -> Path:
    for name in (
        "osnet_ain_msmt17_dyn_b1-8_fp16.engine",
        "osnet_ain_msmt17_dyn_b1-16_fp16.engine",
        "osnet_ain_msmt17_static_1x3x256x128_fp16.engine",
    ):
        path = TENSORRT_MODEL_DIR / name
        if path.is_file():
            return path
    return TENSORRT_MODEL_DIR / "osnet_ain_msmt17_static_1x3x256x128_fp16.engine"


def _default_osnet_path() -> Path:
    return ONNX_MODEL_DIR / "osnet_ain_msmt17_dynamic.onnx"


def _trt_cache_dir() -> Path:
    raw = os.environ.get("TRACKING_V3_TRT_CACHE", "").strip()
    path = Path(raw) if raw else PROJECT_ROOT / "models" / "cache" / "tracking_v3"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _osnet_providers(*, prefer_trt: bool) -> list:
    available = set(ort.get_available_providers())
    providers: list = []
    if prefer_trt and "TensorrtExecutionProvider" in available:
        providers.append(
            (
                "TensorrtExecutionProvider",
                {
                    "device_id": 0,
                    "trt_fp16_enable": _env_bool("TRACKING_V3_OSNET_TRT_FP16", True),
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(_trt_cache_dir()),
                },
            )
        )
    if "CUDAExecutionProvider" in available:
        providers.append(("CUDAExecutionProvider", {"device_id": 0}))
    providers.append("CPUExecutionProvider")
    return providers


class LockedEncoder:
    """为共享 OSNet 编码器加锁，避免并发推理互相干扰。"""

    def __init__(self, inner: Any, lock: threading.RLock) -> None:
        self._inner = inner
        self._lock = lock

    def __call__(self, image: np.ndarray, detections: np.ndarray) -> np.ndarray:
        with self._lock:
            return self._inner(image, detections)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _build_encoder(model_path: Path, *, prefer_trt: bool, use_native_trt: bool) -> Any:
    # 延迟导入，避免 API 进程无谓拉起重依赖
    from services.analysis.osnet_encoder import OSNetEncoder, TensorRTOSNetEncoder

    if use_native_trt:
        max_batch_raw = os.environ.get("TRACKING_V3_OSNET_MAX_BATCH", "").strip()
        max_batch = int(max_batch_raw) if max_batch_raw else None
        return TensorRTOSNetEncoder(model_path, max_batch=max_batch)
    return OSNetEncoder(model_path, prefer_trt=prefer_trt)


def _build_detector(model_path: Path) -> Any:
    from ultralytics import YOLO

    path = str(model_path)
    if path.endswith(".engine"):
        return YOLO(path, task="detect")
    return YOLO(path)


def get_tracking_engines(*, force_reload: bool = False) -> dict[str, Any]:
    """返回常驻 detector / encoder（encoder 已加锁包装）。"""
    global _detector, _encoder, _loaded, _load_error

    with _lock:
        if _loaded and not force_reload:
            if _load_error:
                raise RuntimeError(_load_error)
            return {
                "detector": _detector,
                "encoder": _encoder,
                "lock": _lock,
            }

        try:
            yolo_path = Path(
                os.environ.get("TRACKING_V3_YOLO_MODEL", str(_default_yolo_path()))
            ).expanduser()
            if not yolo_path.is_file():
                yolo_path = _default_yolo_path()

            yolo_is_engine = yolo_path.suffix.lower() == ".engine"
            osnet_engine = Path(
                os.environ.get(
                    "TRACKING_V3_OSNET_ENGINE",
                    str(_default_osnet_engine_path()),
                )
            ).expanduser()
            # 与 YOLO.engine 同进程时，优先原生 TensorRT OSNet（避开 ORT-TRT 冲突）
            use_native_trt = bool(
                _env_bool("TRACKING_V3_OSNET_NATIVE_TRT", True)
                and osnet_engine.is_file()
                and (yolo_is_engine or _env_bool("TRACKING_V3_OSNET_NATIVE_TRT_FORCE", False))
            )

            if use_native_trt:
                osnet_path = osnet_engine
                prefer_trt = False
            else:
                osnet_path = Path(
                    os.environ.get(
                        "TRACKING_V3_OSNET_MODEL",
                        str(_default_osnet_path()),
                    )
                ).expanduser()
                if not osnet_path.is_file():
                    osnet_path = _default_osnet_path()
                # Ultralytics .engine 与 ORT-TRT 冲突：无原生 engine 时退回 CUDA EP
                want_osnet_trt = _env_bool("TRACKING_V3_OSNET_TRT", True)
                force_osnet_trt = _env_bool("TRACKING_V3_OSNET_TRT_FORCE", False)
                prefer_trt = want_osnet_trt and (force_osnet_trt or not yolo_is_engine)
                if yolo_is_engine and want_osnet_trt and not force_osnet_trt:
                    print(
                        "[tracking_v3.engines] 未找到原生 OSNet.engine，"
                        "YOLO TensorRT 下 OSNet 使用 CUDA EP",
                        flush=True,
                    )

            print(
                f"[tracking_v3.engines] 加载 YOLO={yolo_path.name} | "
                f"OSNet={osnet_path.name} | native_trt={use_native_trt} | "
                f"ort_trt={prefer_trt}",
                flush=True,
            )
            detector = _build_detector(yolo_path)
            inner = _build_encoder(
                osnet_path,
                prefer_trt=prefer_trt,
                use_native_trt=use_native_trt,
            )
            encoder = LockedEncoder(inner, _lock)
            _detector = detector
            _encoder = encoder
            _loaded = True
            _load_error = None
            print(
                f"[tracking_v3.engines] 就绪 | YOLO={yolo_path.name} | "
                f"OSNet provider={getattr(inner, 'provider', '?')}",
                flush=True,
            )
        except Exception as exc:
            _load_error = f"tracking engines 加载失败: {exc}"
            _loaded = True
            raise RuntimeError(_load_error) from exc

        return {
            "detector": _detector,
            "encoder": _encoder,
            "lock": _lock,
        }


def warm_tracking_engines() -> dict[str, Any]:
    """预热 YOLO + OSNet，触发 TRT engine cache 构建。"""
    engines = get_tracking_engines()
    detector = engines["detector"]
    encoder = engines["encoder"]
    lock = engines["lock"]
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    # 预热 batch=1 与 batch=8，覆盖动态 profile
    dets1 = np.array([[320.0, 320.0, 64.0, 128.0, 0.99]], dtype=np.float32)
    dets8 = np.array(
        [[80.0 + i * 60, 320.0, 48.0, 96.0, 0.9] for i in range(8)],
        dtype=np.float32,
    )
    imgsz = int(os.environ.get("TRACKING_V3_IMGSZ", "640") or 640)
    print("[tracking_v3.engines] 预热中…", flush=True)
    with lock:
        detector.predict(
            dummy,
            classes=[0],
            conf=float(os.environ.get("TRACKING_V3_CONF", "0.60") or 0.60),
            imgsz=imgsz,
            verbose=False,
        )
        encoder(dummy, dets1)
        encoder(dummy, dets8)
    info = {
        "yolo": os.environ.get("TRACKING_V3_YOLO_MODEL", str(_default_yolo_path())),
        "osnet_provider": getattr(getattr(encoder, "_inner", encoder), "provider", "?"),
    }
    print(f"[tracking_v3.engines] 预热完成 | {info}", flush=True)
    return info


def reset_encoder_stats() -> None:
    engines = get_tracking_engines()
    inner = getattr(engines["encoder"], "_inner", engines["encoder"])
    for attr in ("total_calls", "total_crops"):
        if hasattr(inner, attr):
            setattr(inner, attr, 0)
    for attr in ("total_sec", "preprocess_sec", "infer_sec"):
        if hasattr(inner, attr):
            setattr(inner, attr, 0.0)
