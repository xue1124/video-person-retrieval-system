"""项目统一路径。

业务代码从这里取得运行目录和模型目录，避免在各文件中重复写死路径。
"""
from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = Path(os.environ.get("MODEL_ROOT", PROJECT_ROOT / "models")).expanduser().resolve()
PYTORCH_MODEL_DIR = MODELS_ROOT / "pytorch"
ONNX_MODEL_DIR = MODELS_ROOT / "onnx"
TENSORRT_MODEL_DIR = MODELS_ROOT / "tensorrt"
RKNN_MODEL_DIR = MODELS_ROOT / "rknn"
TOKENIZER_DIR = MODELS_ROOT / "tokenizer" / "siglip_v1"

# 运行数据目录统一在这里定义。API、Celery Worker 与 RTSP 线程必须使用同一组路径，
# 否则会出现“数据库已有记录，但页面找不到图片或视频”的问题。
VIDEO_DIR = PROJECT_ROOT / "video_archives"
GALLERY_DIR = PROJECT_ROOT / "video_crops"
QUERY_STORAGE_DIR = PROJECT_ROOT / "query_storage"
TEMP_DIR = PROJECT_ROOT / "temp_clips"
SNAPSHOT_DIR = Path(
    os.environ.get("TRACKING_SNAPSHOT_ROOT", PROJECT_ROOT / "tracking_snapshots")
).expanduser().resolve()


def ensure_runtime_directories() -> None:
    """创建本地运行目录；目录内容均由 Git 忽略。"""
    for path in (VIDEO_DIR, GALLERY_DIR, QUERY_STORAGE_DIR, TEMP_DIR, SNAPSHOT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def model_path(env_name: str, default: Path) -> Path:
    """环境变量优先，否则返回项目约定的模型位置。"""
    raw = os.environ.get(env_name, "").strip()
    path = Path(raw).expanduser() if raw else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()
