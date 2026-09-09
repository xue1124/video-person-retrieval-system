import cv2, os, time, shutil, torch, subprocess, uuid, re, sys, threading
from typing import Callable, Optional
import numpy as np
import onnxruntime as ort
from celery import Celery
from ultralytics import YOLO
from sqlalchemy import text
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from decord import VideoReader, cpu
import requests
from requests import exceptions as requests_exc
import xml.etree.ElementTree as ET
from requests.auth import HTTPDigestAuth
from urllib.parse import urlparse, parse_qs, unquote

from media_cleanup import purge_video_modeling_artifacts
import db as app_db
from video_time import db_datetime_to_utc_iso, parse_user_captured_at

# 引入 transformers 用于处理 SigLIP 的文本分词
from transformers import AutoTokenizer
from celery.signals import task_failure, task_postrun, task_prerun, worker_ready

# API 与 Worker 必须用同一 broker/backend；可用环境变量覆盖
app_db.load_siglip_env()
_CELERY_BROKER = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
_CELERY_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", _CELERY_BROKER)

app = Celery("reid_tasks", broker=_CELERY_BROKER, backend=_CELERY_BACKEND)

BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "video_archives"
GALLERY_DIR = BASE_DIR / "video_crops"
TEMP_DIR = BASE_DIR / "temp_clips"
VIDEO_DIR.mkdir(exist_ok=True)
GALLERY_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

detector = None
live_detector = None
_live_detector_lock = threading.Lock()
_live_predict_lock = threading.Lock()
reid_sess = None
siglip_vis_sess = None
siglip_txt_sess = None
siglip_tokenizer = None
engine = None
_resource_summary_printed = False

_write_pool = ThreadPoolExecutor(max_workers=4)


@worker_ready.connect
def _on_celery_worker_ready(sender=None, **kwargs):
    _log_runtime(
        f"Celery worker 就绪 | broker={app.conf.broker_url} | "
        "Windows 下 --pool=solo 为严格单任务串行；若常排队可改用 "
        "--pool=threads --concurrency=2（注意 GPU 显存与设备稳定性）"
    )
    # 预热 tracking_v3 常驻 YOLO(TRT)+OSNet(TRT)，避免首个上传任务卡在编译/加载
    if os.environ.get("TRACKING_V3_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        try:
            from tracking_v3.engines import warm_tracking_engines

            info = warm_tracking_engines()
            _log_runtime(f"✅ tracking_v3 engines warmed | {info}")
        except Exception as exc:
            _log_runtime(f"⚠️ tracking_v3 engines 预热失败: {exc}")


@task_prerun.connect
def _on_celery_task_prerun(task_id=None, task=None, args=None, **_kw):
    line = f"▶ TASK_START | id={task_id} | {getattr(task, 'name', '')}"
    if args:
        try:
            vname = args[1] if len(args) > 1 else "?"
            line += f" | video_name={vname!r}"
        except Exception:
            pass
    print(line, flush=True)


@task_postrun.connect
def _on_celery_task_postrun(task_id=None, task=None, state=None, **kwargs):
    print(
        f"■ TASK_END | id={task_id} | {getattr(task, 'name', '')} | state={state}",
        flush=True,
    )


@task_failure.connect
def _on_celery_task_failure(task_id=None, exception=None, **kwargs):
    print(f"✖ TASK_FAIL | id={task_id} | {exception!r}", flush=True)


def _runtime_role() -> str:
    argv = " ".join(sys.argv).lower()
    if "celery" in argv:
        return "WORKER"
    if "uvicorn" in argv or "api_server.py" in argv:
        return "API"
    return "PROC"


def _log_runtime(msg: str) -> None:
    print(f"[{_runtime_role()}][pid={os.getpid()}] {msg}")


def _redact_source_url(value: str) -> str:
    """隐藏任务源地址中的密码，避免凭据进入 worker 日志。"""
    redacted = re.sub(r"([?&]password=)[^&]*", r"\1***", str(value), flags=re.I)
    return re.sub(
        r"(rtsp://[^:/@\s]+:)[^@/\s]+@",
        r"\1***@",
        redacted,
        flags=re.I,
    )


def _session_info(sess: Optional[ort.InferenceSession]) -> dict:
    if sess is None:
        return {"loaded": False}
    providers = sess.get_providers()
    return {
        "loaded": True,
        "active_provider": providers[0] if providers else "unknown",
        "providers": providers,
        "inputs": [i.name for i in sess.get_inputs()],
        "outputs": [o.name for o in sess.get_outputs()],
    }


def get_runtime_backend_info() -> dict:
    return {
        "role": _runtime_role(),
        "pid": os.getpid(),
        "yolo_loaded": detector is not None,
        "osnet": _session_info(reid_sess),
        "siglip_vision": _session_info(siglip_vis_sess),
        "siglip_text": _session_info(siglip_txt_sess),
        "tokenizer_loaded": siglip_tokenizer is not None,
        "tokenizer_path": os.environ.get("SIGLIP_TOKENIZER_PATH", "(default)"),
        "siglip_vision_onnx": os.environ.get("SIGLIP_VISION_ONNX", "siglip_vision.onnx"),
        "siglip_text_onnx": os.environ.get("SIGLIP_TEXT_ONNX", "siglip_text.onnx"),
        "ort_available_providers": ort.get_available_providers(),
        "trt_cache_path": "./models/cache",
    }

def _ensure_sqlalchemy_engine():
    """仅创建 DB 引擎（不加载 YOLO/ONNX），供任务开头尽快更新 videos 任务状态。"""
    global engine
    if engine is None:
        engine = app_db.get_engine()
    return engine


def _isapi_time_span_hint(cfg: dict) -> str:
    return f"{cfg.get('start_time', '')} ～ {cfg.get('end_time', '')}"


def _mark_task_completed(db_engine, video_name: str, count: int) -> None:
    with db_engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE videos SET status='completed', progress=100, "
                "completed_at=NOW(), target_count=:c WHERE file_name=:v AND is_media_source=1"
            ),
            {"c": count, "v": video_name},
        )


def _on_finalize_done(video_name: str, archive_path: str, success: bool, err: Optional[str]) -> None:
    """后台转码结束后将 transcoding → completed（转码失败时建模结果仍保留）。"""
    try:
        eng = _ensure_sqlalchemy_engine()
        with eng.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM videos WHERE file_name=:v AND is_media_source=1 LIMIT 1"),
                {"v": video_name},
            ).first()
        if not exists:
            return
        if success:
            with eng.begin() as conn:
                conn.execute(
                    text(
                        "UPDATE videos SET status='completed', progress=100, "
                        "completed_at=NOW() WHERE file_name=:v AND is_media_source=1"
                    ),
                    {"v": video_name},
                )
            return
        msg = f"建模完成；归档可播化失败: {(err or '未知错误')}"[:500]
        try:
            with eng.begin() as conn:
                conn.execute(
                    text(
                        "UPDATE videos SET status='completed', progress=100, "
                        "completed_at=NOW(), failure_reason=:m WHERE file_name=:v AND is_media_source=1"
                    ),
                    {"m": msg, "v": video_name},
                )
        except Exception:
            with eng.begin() as conn:
                conn.execute(
                    text(
                        "UPDATE videos SET status='completed', progress=100, "
                        "completed_at=NOW() WHERE file_name=:v AND is_media_source=1"
                    ),
                    {"v": video_name},
                )
        print(f"⚠ 归档可播化失败: {archive_path} | {err}", flush=True)
    except Exception as ex:
        print(f"[PLAYBACK] 更新完成状态失败: {video_name} | {ex}", flush=True)


def _schedule_transcoding_finalize(
    db_engine,
    video_name: str,
    count: int,
    archive_path: str,
) -> None:
    """建模结束后标为转码中，后台 ffmpeg 完成后再标 completed。"""
    with db_engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE videos SET status='transcoding', progress=99, "
                "target_count=:c WHERE file_name=:v AND is_media_source=1"
            ),
            {"c": count, "v": video_name},
        )
    from video_playback import schedule_finalize_archive

    schedule_finalize_archive(
        archive_path,
        on_done=lambda ok, err: _on_finalize_done(video_name, archive_path, ok, err),
    )


def _playback_finalize_enabled() -> bool:
    """归档可播化（faststart/转码）。默认关：设 VIDEO_PLAYBACK_FINALIZE=1 可恢复。"""
    raw = os.environ.get("VIDEO_PLAYBACK_FINALIZE", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _set_task_failed(db_engine, video_name: str, reason: str) -> None:
    """写入失败状态；若库表尚无 failure_reason 列则仅更新 status。"""
    msg = (reason or "")[:500]
    try:
        with db_engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE videos SET status='failed', failure_reason=:m "
                    "WHERE file_name=:v AND is_media_source=1"
                ),
                {"m": msg, "v": video_name},
            )
    except Exception:
        with db_engine.begin() as conn:
            conn.execute(
                text("UPDATE videos SET status='failed' WHERE file_name=:v AND is_media_source=1"),
                {"v": video_name},
            )


def get_resources():
    global detector, reid_sess, siglip_vis_sess, siglip_txt_sess, siglip_tokenizer, engine, _resource_summary_printed

    _ensure_sqlalchemy_engine()
        
    # 1. YOLO 模型加载
    if detector is None:
        yolo_engine = "yolov10n.engine"
        yolo_pt = "yolov10n.pt"
        if os.path.exists(yolo_engine):
            detector = YOLO(yolo_engine, task="detect")
            _log_runtime("✅ YOLO(TensorRT Engine) loaded")
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            detector = YOLO(yolo_pt)
            if device == "cuda":
                detector.to(device)
                # 强行统一为标准精度，避免 fuse 过程报错
                detector.model.float() 
            _log_runtime(f"✅ YOLO(.pt) loaded on {device}")

    # 2. 公共 ORT (SigLIP/OSNet) 调度策略
    providers = [
        ('TensorrtExecutionProvider', {
            'device_id': 0,
            'trt_fp16_enable': False,  
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': './models/cache'
        }),
        ('CUDAExecutionProvider', {'device_id': 0}),
        'CPUExecutionProvider'
    ]
    sess_options = ort.SessionOptions()
    # 抑制 ORT warning 级别刷屏（如大量 unused initializer），保留 error/fatal
    sess_options.log_severity_level = 3

    # 3. OSNet 模型加载
    if reid_sess is None:
        onnx_model_path = "osnet_ain_msmt17_dynamic.onnx"
        try:
            reid_sess = ort.InferenceSession(
                onnx_model_path,
                sess_options=sess_options,
                providers=providers
            )
            _log_runtime(
                f"✅ OSNet ready | active={reid_sess.get_providers()[0]} | providers={reid_sess.get_providers()}"
            )
        except Exception as e:
            reid_sess = ort.InferenceSession(
                onnx_model_path,
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )
            _log_runtime(f"⚠️ OSNet fallback to CPU due to: {e}")

    # 4. SigLIP 模型加载 (Vision & Text)
    if siglip_vis_sess is None:
        vis_path = os.environ.get("SIGLIP_VISION_ONNX", "siglip_vision.onnx")
        txt_path = os.environ.get("SIGLIP_TEXT_ONNX", "siglip_text.onnx")
        os.makedirs("./models/cache", exist_ok=True)
        
        try:
            _log_runtime(f"⏳ SigLIP Vision initializing from: {vis_path}")
            siglip_vis_sess = ort.InferenceSession(
                vis_path,
                sess_options=sess_options,
                providers=providers
            )
            _log_runtime(
                "✅ SigLIP Vision ready | "
                f"active={siglip_vis_sess.get_providers()[0]} | "
                f"providers={siglip_vis_sess.get_providers()} | "
                f"outputs={[o.name for o in siglip_vis_sess.get_outputs()]}"
            )
            
            _log_runtime(f"⏳ SigLIP Text initializing from: {txt_path}")
            # Text 用纯 CUDA，避免 TRT 编译开销
            txt_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            siglip_txt_sess = ort.InferenceSession(
                txt_path,
                sess_options=sess_options,
                providers=txt_providers
            )
            _log_runtime(
                "✅ SigLIP Text ready | "
                f"active={siglip_txt_sess.get_providers()[0]} | "
                f"providers={siglip_txt_sess.get_providers()} | "
                f"outputs={[o.name for o in siglip_txt_sess.get_outputs()]}"
            )
            
            tok_path = _default_siglip_tokenizer_path()
            siglip_tokenizer = _load_siglip_tokenizer(tok_path)
            _log_runtime(f"✅ SigLIP tokenizer ready: {tok_path}")
        except Exception as e:
            _log_runtime(f"⚠️ SigLIP load error: {e} | trying fallback...")
            siglip_vis_sess = ort.InferenceSession(
                vis_path,
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )
            # Text 兜底也用 CUDA，不用 TRT
            siglip_txt_sess = ort.InferenceSession(
                txt_path,
                sess_options=sess_options,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            try:
                tok_path = _default_siglip_tokenizer_path()
                siglip_tokenizer = _load_siglip_tokenizer(tok_path)
            except Exception:
                tok_path = os.environ.get("SIGLIP_TOKENIZER_PATH", "google/siglip-base-patch16-224")
                siglip_tokenizer = AutoTokenizer.from_pretrained(tok_path)
            _log_runtime(
                "✅ SigLIP fallback ready | "
                f"vision_active={siglip_vis_sess.get_providers()[0]} | "
                f"text_active={siglip_txt_sess.get_providers()[0]} | "
                f"tokenizer={tok_path}"
            )

    if not _resource_summary_printed:
        info = get_runtime_backend_info()
        _log_runtime(
            "🧭 Runtime summary | "
            f"vision_active={info['siglip_vision'].get('active_provider', 'n/a')} | "
            f"text_active={info['siglip_text'].get('active_provider', 'n/a')} | "
            f"osnet_active={info['osnet'].get('active_provider', 'n/a')}"
        )
        _resource_summary_printed = True
                
    return detector, reid_sess, engine, siglip_vis_sess, siglip_txt_sess, siglip_tokenizer


def _yolo_infer_device():
    return 0 if torch.cuda.is_available() else "cpu"


def get_live_detector():
    """
    RTSP 实时专用 YOLO（.pt）。
    与 API/Celery 共用的 TensorRT .engine 分离，避免子线程二次加载 TRT 导致进程崩溃。
    """
    global live_detector
    with _live_detector_lock:
        if live_detector is None:
            yolo_pt = "yolov10n.pt"
            if not os.path.isfile(yolo_pt):
                raise FileNotFoundError(f"RTSP 实时需要 {yolo_pt}，请放在项目根目录")
            device = _yolo_infer_device()
            live_detector = YOLO(yolo_pt)
            if device != "cpu":
                live_detector.to(device)
                live_detector.model.float()
            _log_runtime(f"✅ RTSP live YOLO(.pt) loaded on {device}")
    return live_detector


def live_yolo_predict(frame, conf_val: float):
    """线程安全；供 RTSP 拉流线程调用。"""
    det = get_live_detector()
    with _live_predict_lock:
        return det.predict(
            frame,
            classes=0,
            conf=float(conf_val),
            verbose=False,
            device=_yolo_infer_device(),
            half=False,
        )


def warm_live_detector() -> None:
    """主线程预热 RTSP 用 YOLO，避免首次推理在子线程初始化。"""
    import numpy as np

    det = get_live_detector()
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    with _live_predict_lock:
        det.predict(
            dummy,
            classes=0,
            conf=0.5,
            verbose=False,
            device=_yolo_infer_device(),
            half=False,
        )
    _log_runtime("✅ RTSP live YOLO warmup done")


def _default_siglip_tokenizer_path() -> str:
    p = os.environ.get("SIGLIP_TOKENIZER_PATH")
    if p and os.path.isdir(p) and os.path.isfile(os.path.join(p, "tokenizer_config.json")):
        return p
    demo = os.environ.get("SIGLIP_DEMO_ROOT", str(BASE_DIR))
    for rel in ("siglip_v1", os.path.join("models", "siglip_v1")):
        cand = os.path.join(demo, rel)
        if os.path.isfile(os.path.join(cand, "tokenizer_config.json")):
            return cand
    return os.path.join(demo, "siglip_v1")


def _load_siglip_tokenizer(path: str):
    """本地优先；fast 失败时用 slow，避免缺少 sentencepiece 时无法实例化。"""
    last_err = None
    for use_fast in (True, False):
        try:
            return AutoTokenizer.from_pretrained(
                path,
                local_files_only=True,
                trust_remote_code=True,
                use_fast=use_fast,
            )
        except Exception as e:
            last_err = e
    raise last_err


# ================= SigLIP 特征提取函数 =================
def extract_siglip_feat_img(cv2_img):
    """单图视觉特征提取"""
    _, _, _, vis_sess, _, _ = get_resources()
    if cv2_img is None: return None
    img = cv2.resize(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), (256, 256))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0) 
    
    feat = vis_sess.run(['1726'], {vis_sess.get_inputs()[0].name: img})[0]
    feat = feat.astype(np.float32)
        
    norm = np.linalg.norm(feat, axis=1, keepdims=True)
    feat = feat / (norm + 1e-8)
    return feat.flatten() 

def _siglip_text_onnx_feed(txt_sess, tokenizer, text_str):
    """按输入名绑定，避免导出时 attention_mask 排在 input_ids 之前导致喂错张量。"""
    tok = tokenizer(
        [text_str],
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="np",
    )
    input_ids = tok["input_ids"].astype(np.int64)
    att = tok["attention_mask"].astype(np.int64) if "attention_mask" in tok else None
    feed = {}
    for inp in txt_sess.get_inputs():
        n = inp.name.lower()
        if att is not None and "attention" in n:
            feed[inp.name] = att
        elif ("input" in n and "id" in n) or n in ("input_ids", "token_ids"):
            feed[inp.name] = input_ids
        elif "token_type" in n:
            feed[inp.name] = np.zeros_like(input_ids, dtype=np.int64)
        else:
            feed[inp.name] = input_ids
    return feed


def extract_siglip_feat_text(text_str):
    """文本特征提取：与底库 SigLIP 图像向量对齐（同维度、L2 归一化）。"""
    _, _, _, _, txt_sess, tokenizer = get_resources()
    try:
        feed = _siglip_text_onnx_feed(txt_sess, tokenizer, text_str)
        feat = txt_sess.run(['text_embeds'], feed)[0]
        feat = feat.astype(np.float32)

        if feat.ndim == 1:
            feat = feat / (np.linalg.norm(feat) + 1e-8)
            return feat.astype(np.float32)
        feat /= np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8
        return feat.flatten()
    except Exception as e:
        print(f"SigLIP Text Error: {e}")
        return None

def extract_siglip_feat_img_batch(cv2_imgs):
    """批量视觉特征提取"""
    _, _, _, vis_sess, _, _ = get_resources()
    if not cv2_imgs: return []
    
    resized_imgs = [cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (256, 256)) for img in cv2_imgs]
    batch_input = np.stack(resized_imgs, axis=0).astype(np.float32)
    batch_input /= 255.0
    batch_input = (batch_input - 0.5) / 0.5
    batch_input = batch_input.transpose(0, 3, 1, 2)
    
    feats = vis_sess.run(['1726'], {vis_sess.get_inputs()[0].name: batch_input})[0]
    feats = feats.astype(np.float32)
    
    norm = np.linalg.norm(feats, axis=1, keepdims=True)
    feats = feats / (norm + 1e-8)
    return feats

# ================= OSNet 特征提取函数 =================
def extract_osnet_feat_batch(cv2_imgs):
    _, sess, _, _, _, _ = get_resources()
    if not cv2_imgs: return []
    resized_imgs = [cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (128, 256)) for img in cv2_imgs]
    batch_input = np.stack(resized_imgs, axis=0).astype(np.float32)
    batch_input /= 255.0
    batch_input -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
    batch_input /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
    batch_input = batch_input.transpose(0, 3, 1, 2)
    feats = sess.run(None, {sess.get_inputs()[0].name: batch_input})[0]
    feats = feats.astype(np.float32)
    feats /= (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    return feats

def extract_osnet_feat(cv2_img):
    _, sess, _, _, _, _ = get_resources()
    if cv2_img is None or cv2_img.size == 0: return None
    try:
        img = cv2.resize(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), (128, 256))
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)
        feat = sess.run(None, {sess.get_inputs()[0].name: img})[0]
        feat = feat.astype(np.float32)
        feat /= (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
        return feat.flatten()
    except: return None

# ================= 辅助入库函数 =================
def _write_gallery_frame(path: str, image: np.ndarray) -> bool:
    """同步写 crop 到磁盘；成功才允许写入 gallery_meta。"""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(p), image)
        return bool(ok) and p.is_file() and p.stat().st_size > 0
    except Exception:
        return False


def _flush_buffer(crop_buffer, full_frame_buffer, meta_buffer, db_engine):
    """特征提取 → 落盘成功 → 再写库，保证 gallery_meta 与 video_crops 一致。"""
    if not crop_buffer:
        return 0

    osnet_feats = extract_osnet_feat_batch(crop_buffer)
    siglip_feats = extract_siglip_feat_img_batch(crop_buffer)

    write_ok = [False] * len(crop_buffer)
    futures = {
        _write_pool.submit(
            _write_gallery_frame, meta_buffer[i]["p"], full_frame_buffer[i]
        ): i
        for i in range(len(crop_buffer))
    }
    for fut in as_completed(futures):
        idx = futures[fut]
        try:
            write_ok[idx] = bool(fut.result())
        except Exception:
            write_ok[idx] = False

    db_insert_data = []
    for idx, ok in enumerate(write_ok):
        if not ok:
            continue
        m = meta_buffer[idx]
        db_insert_data.append(
            {
                "v": m["v"],
                "t": m["t"],
                "p": m["p"],
                "x1": int(m["x1"]),
                "y1": int(m["y1"]),
                "x2": int(m["x2"]),
                "y2": int(m["y2"]),
                "feat": osnet_feats[idx].tobytes(),
                "cfeat": siglip_feats[idx].tobytes(),
            }
        )

    if db_insert_data:
        with db_engine.begin() as conn:
            video_ids = {}
            for item in db_insert_data:
                name = item["v"]
                if name not in video_ids:
                    video_ids[name] = app_db.get_video_id(conn, name)
                item["video_id"] = video_ids[name]
            conn.execute(
                text(
                    "INSERT INTO gallery_meta (video_id, video_name, timestamp, image_path, "
                    "bbox_x1, bbox_y1, bbox_x2, bbox_y2, feature_vector, clip_feature) "
                    "VALUES (:video_id, :v, :t, :p, :x1, :y1, :x2, :y2, :feat, :cfeat)"
                ),
                db_insert_data,
            )
    return len(db_insert_data)

# ================= 任务主入口 =================
FFMPEG_PATH = os.environ.get("FFMPEG_PATH") or shutil.which("ffmpeg") or "/usr/bin/ffmpeg"

def _extract_playback_uris(xml_text):
    uris = []
    try:
        root = ET.fromstring(xml_text)
        for elem in root.iter():
            if elem.tag.endswith("playbackURI") and elem.text:
                u = elem.text.strip()
                if u:
                    uris.append(u)
    except Exception:
        pass
    if not uris:
        uris.extend([m.strip() for m in re.findall(r"<playbackURI>(.*?)</playbackURI>", xml_text, re.S)])
    # 去重并保持原顺序
    return list(dict.fromkeys(u for u in uris if u))


def _merge_isapi_segments(segment_paths, output_file):
    if not segment_paths:
        raise RuntimeError("ISAPI 未下载到任何片段")

    if len(segment_paths) == 1:
        cmd = [
            FFMPEG_PATH,
            "-y",
            "-i",
            segment_paths[0],
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-an",
            "-movflags",
            "+faststart",
            output_file,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=7200)
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode("utf-8", errors="ignore")
            raise RuntimeError(f"ISAPI 分段合并失败: {err}") from e
        if (not os.path.exists(output_file)) or os.path.getsize(output_file) == 0:
            raise RuntimeError("ISAPI 分段合并后输出为空")
        return

    list_file = f"{output_file}.concat.txt"
    try:
        with open(list_file, "w", encoding="utf-8") as f:
            for p in segment_paths:
                esc = p.replace("\\", "/").replace("'", "'\\''")
                f.write(f"file '{esc}'\n")
        cmd = [
            FFMPEG_PATH,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_file,
            # 海康回放常见 pcm_alaw 音轨，mp4 容器 copy 会失败；建模只需要视频帧，直接丢弃音频更稳。
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-an",
            "-movflags",
            "+faststart",
            output_file,
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=7200)
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ISAPI 分段合并失败: {err}") from e
    finally:
        if os.path.exists(list_file):
            try:
                os.remove(list_file)
            except OSError:
                pass

    if (not os.path.exists(output_file)) or os.path.getsize(output_file) == 0:
        raise RuntimeError("ISAPI 分段合并后输出为空")

def _parse_isapi_source(raw_path):
    """
    支持格式:
    isapi://host?user=admin&password=xxx&channel=1&start=2025-12-12T07:00:00&end=2025-12-12T08:00:00
    """
    if not raw_path.startswith("isapi://"):
        return None
    parsed = urlparse(raw_path)
    qs = parse_qs(parsed.query)
    host = parsed.netloc
    user = unquote(qs.get("user", [""])[0])
    password = unquote(qs.get("password", [""])[0])
    channel = int(qs.get("channel", ["1"])[0])
    start_time = qs.get("start", [""])[0]
    end_time = qs.get("end", [""])[0]
    if not (host and user and password and start_time and end_time):
        raise ValueError("ISAPI 参数不完整，需包含 host/user/password/start/end")
    return {
        "host": host,
        "user": user,
        "password": password,
        "channel": channel,
        "start_time": start_time,
        "end_time": end_time,
    }

class MediaSourceGone(Exception):
    """用户已从媒体源列表删除该任务，worker 应停止并做本地清理。"""


def _media_source_row_exists(engine, video_name: str) -> bool:
    """若 videos 中已无该媒体源任务，视为用户已删除（用于终止运行中的 worker）。"""
    try:
        with engine.connect() as conn:
            r = conn.execute(
                text("SELECT 1 FROM videos WHERE file_name=:v AND is_media_source=1 LIMIT 1"),
                {"v": video_name},
            ).fetchone()
        return r is not None
    except Exception:
        return True


def _task_still_current(engine, video_name: str, task_id: str | None) -> bool:
    """同名重传后旧 Celery 线程杀不掉时，用 task_id 判断是否已过期。"""
    if not task_id:
        return _media_source_row_exists(engine, video_name)
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT task_id FROM videos WHERE file_name=:v AND is_media_source=1 LIMIT 1"
                ),
                {"v": video_name},
            ).mappings().first()
        if not row:
            return False
        return str(row.get("task_id") or "") == str(task_id)
    except Exception:
        return True


def _cleanup_partial_task_files(video_name: str, temp_download_file: Optional[str]) -> None:
    if temp_download_file and os.path.exists(temp_download_file):
        try:
            os.remove(temp_download_file)
        except OSError:
            pass
    if engine is not None:
        try:
            purge_video_modeling_artifacts(
                engine, video_name, remove_archived_video=True
            )
        except Exception:
            pass


def _run_rtsp_ffmpeg_with_cancel(
    cmd: list, db_engine, video_name: str, timeout_sec: int = 3600
) -> None:
    """RTSP 下载过程中轮询 DB，用户删除媒体源则结束 ffmpeg。"""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    t0 = time.time()
    while True:
        if not _media_source_row_exists(db_engine, video_name):
            proc.kill()
            try:
                proc.wait(timeout=15)
            except Exception:
                pass
            raise MediaSourceGone()
        rc = proc.poll()
        if rc is not None:
            if rc != 0:
                raise RuntimeError(f"ffmpeg 退出码 {rc}")
            return
        if time.time() - t0 > timeout_sec:
            proc.kill()
            raise RuntimeError("RTSP 下载超时")
        time.sleep(0.5)


def _isapi_request_timeouts():
    """
    ISAPI 流式下载：单值 timeout 会限制「两次读到数据」的最大间隔。
    NVR 忙或片段大时易触发 ReadTimeout；用环境变量放宽（秒）。
    """
    connect = int(os.environ.get("ISAPI_CONNECT_TIMEOUT", "30"))
    search_read = int(os.environ.get("ISAPI_SEARCH_READ_TIMEOUT", "60"))
    download_read = int(os.environ.get("ISAPI_DOWNLOAD_READ_TIMEOUT", "3600"))
    return connect, search_read, download_read


def _download_video_via_isapi(
    raw_path,
    output_file,
    progress_callback=None,
    cancel_check: Optional[Callable[[], bool]] = None,
):
    cfg = _parse_isapi_source(raw_path)
    if cfg is None:
        raise ValueError("无效 ISAPI 源")

    connect_t, search_read_t, download_read_t = _isapi_request_timeouts()
    search_timeout = (connect_t, search_read_t)
    download_timeout = (connect_t, download_read_t)

    search_id = "{" + str(uuid.uuid4()).upper() + "}"
    search_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<CMSearchDescription version="1.0" xmlns="http://www.isapi.org/ver20/XMLSchema">
    <searchID>{search_id}</searchID>
    <trackIDList>
        <trackID>{cfg["channel"]}01</trackID>
    </trackIDList>
    <timeSpanList>
        <timeSpan>
            <startTime>{cfg["start_time"]}</startTime>
            <endTime>{cfg["end_time"]}</endTime>
        </timeSpan>
    </timeSpanList>
    <contentTypeList>
        <contentType>video</contentType>
    </contentTypeList>
    <maxResults>200</maxResults>
</CMSearchDescription>"""

    from box_tunnel import isapi_proxy_headers, isapi_proxy_url

    search_url = isapi_proxy_url("/ISAPI/ContentMgmt/search")
    proxy_headers = {
        "Content-Type": "application/xml",
        **isapi_proxy_headers(cfg["host"]),
    }
    search_resp = requests.post(
        search_url,
        auth=HTTPDigestAuth(cfg["user"], cfg["password"]),
        data=search_xml,
        headers=proxy_headers,
        timeout=search_timeout,
    )
    search_resp.raise_for_status()
    playback_uris = _extract_playback_uris(search_resp.text)
    if not playback_uris:
        span = _isapi_time_span_hint(cfg)
        raise RuntimeError(
            f"所选日期/时间段在设备上未返回可下载录像（{span}）。"
            "请确认该日是否有录像、通道号与起止时间是否正确。"
        )
    print(f"📦 ISAPI 命中片段数: {len(playback_uris)}")

    download_url = isapi_proxy_url("/ISAPI/ContentMgmt/download")
    max_attempts = int(os.environ.get("ISAPI_DOWNLOAD_RETRIES", "3"))
    seg_dir = os.path.join(
        os.path.dirname(output_file) or ".",
        f"isapi_parts_{uuid.uuid4().hex[:8]}",
    )
    os.makedirs(seg_dir, exist_ok=True)
    seg_paths = []
    total_units = max(1, len(playback_uris) * 1000)
    try:
        for idx, playback_uri in enumerate(playback_uris, start=1):
            if cancel_check and cancel_check():
                raise MediaSourceGone()
            download_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<downloadRequest version="1.0" xmlns="http://www.isapi.org/ver20/XMLSchema">
    <playbackURI>{playback_uri.replace("&", "&amp;")}</playbackURI>
</downloadRequest>"""
            seg_file = os.path.join(seg_dir, f"seg_{idx:04d}.mp4")
            for attempt in range(1, max_attempts + 1):
                if attempt > 1 and os.path.exists(seg_file):
                    try:
                        os.remove(seg_file)
                    except OSError:
                        pass
                try:
                    with requests.get(
                        download_url,
                        auth=HTTPDigestAuth(cfg["user"], cfg["password"]),
                        data=download_xml,
                        headers={
                            "Content-Type": "application/xml",
                            **isapi_proxy_headers(cfg["host"]),
                        },
                        stream=True,
                        timeout=download_timeout,
                    ) as resp:
                        resp.raise_for_status()
                        seg_total = int(resp.headers.get("Content-Length", 0) or 0)
                        seg_downloaded = 0
                        with open(seg_file, "wb") as f:
                            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                                if not chunk:
                                    continue
                                f.write(chunk)
                                seg_downloaded += len(chunk)
                                if progress_callback and seg_total > 0:
                                    base_units = (idx - 1) * 1000
                                    cur_units = base_units + int(min(1000, (seg_downloaded * 1000) / seg_total))
                                    progress_callback(cur_units, total_units)
                    if progress_callback:
                        progress_callback(idx * 1000, total_units)
                    break
                except (requests_exc.ReadTimeout, requests_exc.ConnectionError) as e:
                    if attempt >= max_attempts:
                        raise RuntimeError(
                            f"ISAPI 分段下载失败（片段 {idx}/{len(playback_uris)}，已重试 {max_attempts} 次）: {e}. "
                            f"可加大读间隔秒数: set ISAPI_DOWNLOAD_READ_TIMEOUT=7200 "
                            f"(当前 connect={connect_t}s read={download_read_t}s)"
                        ) from e
                    wait_s = min(30, 2 ** (attempt - 1))
                    print(f"⚠️ ISAPI 片段 {idx}/{len(playback_uris)} 第 {attempt} 次失败，{wait_s}s 后重试: {e}")
                    time.sleep(wait_s)
                except requests_exc.HTTPError as e:
                    raise RuntimeError(
                        f"ISAPI 下载 HTTP 错误（片段 {idx}/{len(playback_uris)}）: "
                        f"{e.response.status_code if e.response else ''} {e}"
                    ) from e
            seg_paths.append(seg_file)

        if os.path.exists(output_file):
            try:
                os.remove(output_file)
            except OSError:
                pass
        _merge_isapi_segments(seg_paths, output_file)
    finally:
        try:
            shutil.rmtree(seg_dir, ignore_errors=True)
        except Exception:
            pass

    if (not os.path.exists(output_file)) or os.path.getsize(output_file) == 0:
        span = _isapi_time_span_hint(cfg)
        raise RuntimeError(
            f"所选时间段录像下载为空（{span}），可能该时段无录像或设备未存盘。"
        )


def _captured_at_iso_for_tracking(engine, video_name: str, raw_path: str | None) -> str | None:
    """Celery 入库时沿用 API 已写入的 captured_at；ISAPI 可从回放开始时间补齐。"""
    try:
        with engine.connect() as conn:
            iso = app_db.captured_at_iso_for_import(conn, video_name)
            if iso:
                return iso
    except Exception as exc:
        print(f"[WORKER] 读取 captured_at 失败: {exc}", flush=True)
    if raw_path and str(raw_path).startswith("isapi://"):
        try:
            cfg = _parse_isapi_source(raw_path)
            captured = parse_user_captured_at(cfg.get("start_time"))
            return db_datetime_to_utc_iso(captured)
        except Exception:
            return None
    return None


@app.task(bind=True, name='tasks.process_video_task')
def process_video_task(self, raw_path, video_name, skip_frames, conf_val, duration=3600):
    print(
        f"[WORKER] process_video_task 已出队执行 | task_id={self.request.id} | video_name={video_name!r}",
        flush=True,
    )
    # 尽快把 DB 标为 processing，避免长时间卡在模型加载/ISAPI 下载时前端一直显示「等待」
    try:
        eng = _ensure_sqlalchemy_engine()
        with eng.begin() as conn:
            try:
                conn.execute(
                    text(
                        "UPDATE videos SET status='processing', "
                        "progress=IFNULL(progress, 0), "
                        "processing_started_at=COALESCE(processing_started_at, NOW()) "
                        "WHERE file_name=:v AND is_media_source=1"
                    ),
                    {"v": video_name},
                )
            except Exception as inner:
                if "unknown column" in str(inner).lower():
                    conn.execute(
                        text(
                            "UPDATE videos SET status='processing', "
                            "progress=IFNULL(progress, 0) WHERE file_name=:v AND is_media_source=1"
                        ),
                        {"v": video_name},
                    )
                else:
                    raise
    except Exception as ex:
        print(f"[WORKER] 早期 UPDATE videos 失败（可忽略）: {ex}", flush=True)

    # 1. 资源获取
    db_engine = _ensure_sqlalchemy_engine()

    if not _media_source_row_exists(db_engine, video_name):
        print(
            f"[WORKER] 媒体源「{video_name}」在库中已不存在，跳过（可能已被用户删除）",
            flush=True,
        )
        return {"status": "CANCELLED", "reason": "deleted"}

    # 支持两类网络源：历史 RTSP 与 ISAPI 回放；均先下载为本地文件再处理
    temp_download_file = None
    is_isapi_source = raw_path.startswith("isapi://")
    download_weight = 35 if is_isapi_source else 0
    if raw_path.startswith("rtsp://") and "starttime=" in raw_path:
        try:
            from box_tunnel import resolve_rtsp_playback_url

            raw_path = resolve_rtsp_playback_url(raw_path)
        except Exception as e:
            print(f"❌ RTSP 隧道中继失败: {e}")
            _set_task_failed(db_engine, video_name, str(e))
            return {"status": "FAILED", "error": str(e)}
        print(f"📹 检测到历史 RTSP，先下载到本地: {_redact_source_url(raw_path)}")
        temp_download_file = str(TEMP_DIR / f"temp_rtsp_{uuid.uuid4().hex[:8]}.mp4")
        
        download_cmd = [
            FFMPEG_PATH, "-y",
            "-rtsp_transport", "tcp",
            "-i", raw_path,
            "-t", str(duration),
            "-c:v", "copy",
            "-an",
            temp_download_file
        ]
        
        try:
            safe_download_cmd = [
                _redact_source_url(part) if isinstance(part, str) else part
                for part in download_cmd
            ]
            print(f"⏳ 执行下载: {' '.join(safe_download_cmd)}")
            _run_rtsp_ffmpeg_with_cancel(download_cmd, db_engine, video_name)
            print(f"✅ RTSP 下载完成: {temp_download_file}")
            raw_path = temp_download_file
        except MediaSourceGone:
            print(f"[WORKER] RTSP 下载已取消（媒体源删除）: {video_name}", flush=True)
            _cleanup_partial_task_files(video_name, temp_download_file)
            return {"status": "CANCELLED", "reason": "deleted"}
        except Exception as e:
            print(f"❌ RTSP 下载失败: {e}")
            _set_task_failed(db_engine, video_name, str(e))
            return {"status": "FAILED", "error": str(e)}
    elif raw_path.startswith("isapi://"):
        print("📹 检测到 ISAPI 历史回放，开始下载本地文件")
        temp_download_file = str(TEMP_DIR / f"temp_isapi_{uuid.uuid4().hex[:8]}.mp4")
        try:
            last_download_progress = {"p": -1}
            def _on_isapi_download_progress(downloaded, total_size):
                if not _media_source_row_exists(db_engine, video_name):
                    raise MediaSourceGone()
                if total_size > 0:
                    ratio = max(0.0, min(1.0, downloaded / total_size))
                    p = int(ratio * download_weight)
                else:
                    p = min(download_weight - 1, last_download_progress["p"] + 1)
                if p <= last_download_progress["p"]:
                    return
                last_download_progress["p"] = p
                if self.request.id:
                    self.update_state(state='PROGRESS', meta={'current': p})
                with db_engine.begin() as conn:
                    conn.execute(
                        text("UPDATE videos SET progress=:p WHERE file_name=:v AND is_media_source=1"),
                        {"p": p, "v": video_name}
                    )

            _download_video_via_isapi(
                raw_path,
                temp_download_file,
                progress_callback=_on_isapi_download_progress,
                cancel_check=lambda: not _media_source_row_exists(db_engine, video_name),
            )
            VIDEO_DIR.mkdir(exist_ok=True)
            vn_base = os.path.basename(str(video_name).replace("\\", "/"))
            persist_file = str(
                VIDEO_DIR / (vn_base if vn_base.lower().endswith(".mp4") else f"{vn_base}.mp4")
            )
            if os.path.exists(persist_file):
                os.remove(persist_file)
            shutil.move(temp_download_file, persist_file)
            print(f"✅ ISAPI 下载完成并落盘: {persist_file}")
            raw_path = persist_file
            temp_download_file = None
            with db_engine.begin() as conn:
                conn.execute(
                    text("UPDATE videos SET source_path=:rp, progress=:p WHERE file_name=:v AND is_media_source=1"),
                    {"rp": persist_file, "p": download_weight, "v": video_name}
                )
        except MediaSourceGone:
            print(f"[WORKER] ISAPI 下载已取消（媒体源删除）: {video_name}", flush=True)
            _cleanup_partial_task_files(video_name, temp_download_file)
            return {"status": "CANCELLED", "reason": "deleted"}
        except Exception as e:
            print(f"❌ ISAPI 下载失败: {e}")
            err = str(e)
            cfg = _parse_isapi_source(raw_path)
            span = _isapi_time_span_hint(cfg) if cfg else ""
            if span and span not in err:
                reason = f"{err}（查询时间段：{span}）"
            else:
                reason = err
            _set_task_failed(db_engine, video_name, reason)
            return {"status": "FAILED", "error": err}
    
    # 本地视频路径校验
    if not raw_path.startswith(("rtsp://", "isapi://")):
        if not os.path.isabs(raw_path):
            raw_path = str(BASE_DIR / raw_path)
        if not os.path.exists(raw_path):
            reason = f"视频文件不存在: {raw_path}（worker cwd={os.getcwd()}）"
            print(f"❌ {reason}")
            _set_task_failed(db_engine, video_name, reason)
            return {"status": "FAILED", "error": reason}
        if os.path.getsize(raw_path) == 0:
            reason = f"视频文件为空: {raw_path}"
            print(f"❌ {reason}")
            _set_task_failed(db_engine, video_name, reason)
            return {"status": "FAILED", "error": reason}

    # 读取基本元信息（不再跑旧 gallery / SigLIP 抽帧建模）
    capture = cv2.VideoCapture(raw_path)
    if not capture.isOpened():
        reason = f"视频读取失败: {raw_path}"
        print(f"❌ {reason}")
        _set_task_failed(db_engine, video_name, reason)
        return {"status": "FAILED", "error": reason}
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    if fps <= 0:
        fps = 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    if total > 0:
        total_seconds = int(total / fps)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        duration_str = "未知"

    # 清理旧 gallery 产物；房间标注保留，便于标注后建模仍能同步进 medical_audit_v3
    purge_video_modeling_artifacts(
        db_engine, video_name, remove_archived_video=False
    )
    with db_engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE videos SET status='processing', fps=:f, duration=:d, "
                "target_count=0, progress=:p WHERE file_name=:v AND is_media_source=1"
            ),
            {
                "f": round(fps, 1),
                "d": duration_str,
                "p": download_weight,
                "v": video_name,
            },
        )

    from tracking_v3 import run_tracking_v3_sidecar, tracking_v3_enabled

    if not tracking_v3_enabled():
        reason = "已改为仅 OSNet 跟踪入库，请设置 TRACKING_V3_ENABLED=1"
        print(f"❌ {reason}", flush=True)
        _set_task_failed(db_engine, video_name, reason)
        return {"status": "FAILED", "error": reason}

    person_count = 0

    def _set_progress(p: int, count: int | None = None) -> None:
        nonlocal person_count
        prog = max(0, min(99, int(p)))
        if count is not None:
            person_count = int(count)
        if self.request.id:
            self.update_state(state="PROGRESS", meta={"current": prog})
        with db_engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE videos SET progress=:p, target_count=:c WHERE file_name=:v AND is_media_source=1"
                ),
                {"p": prog, "c": person_count, "v": video_name},
            )

    archive_path = raw_path
    if archive_path and not str(archive_path).startswith(("rtsp://", "isapi://")):
        if not os.path.isabs(archive_path):
            archive_path = str(BASE_DIR / archive_path)

    if not (archive_path and os.path.isfile(archive_path) and os.path.getsize(archive_path) > 0):
        reason = f"本地视频不可用，无法执行 OSNet 跟踪: {archive_path}"
        print(f"❌ {reason}", flush=True)
        _set_task_failed(db_engine, video_name, reason)
        return {"status": "FAILED", "error": reason}

    try:
        if not _task_still_current(db_engine, video_name, self.request.id):
            print(
                f"[WORKER] 任务已过期（同名重传或已删除），跳过跟踪: {video_name} "
                f"task_id={self.request.id}",
                flush=True,
            )
            return {"status": "CANCELLED", "reason": "superseded"}
        skip_n = max(1, int(skip_frames or 1))
        print(
            f"[WORKER] 仅跑 tracking_v3（YOLO+OSNet 聚类）: {archive_path} | "
            f"skip_frames={skip_n} | conf={conf_val}",
            flush=True,
        )

        def _on_track_progress(ratio: float, count: int | None = None) -> None:
            if not _task_still_current(db_engine, video_name, self.request.id):
                raise MediaSourceGone()
            # 0-75 抽帧，75-88 聚类，88-93 入库，93-96 跨视频，96-99 停留/切图
            span = max(1, 99 - download_weight)
            p = download_weight + int(max(0.0, min(1.0, float(ratio))) * span)
            _set_progress(p, count)

        try:
            v3_result = run_tracking_v3_sidecar(
                video_path=archive_path,
                video_name=video_name,
                raw_path=raw_path,
                captured_at=_captured_at_iso_for_tracking(
                    db_engine, video_name, raw_path
                ),
                commit_check=lambda: _task_still_current(
                    db_engine, video_name, self.request.id
                ),
                progress_callback=_on_track_progress,
                skip_frames=skip_n,
                conf_val=float(conf_val),
            )
        except MediaSourceGone:
            print(
                f"[WORKER] 跟踪中检测到任务已删除/过期，停止: {video_name}",
                flush=True,
            )
            return {"status": "CANCELLED", "reason": "deleted_during_track"}
        except Exception as track_exc:
            # track_video.TrackingCancelled
            if track_exc.__class__.__name__ == "TrackingCancelled":
                print(
                    f"[WORKER] 跟踪已取消: {video_name} | {track_exc}",
                    flush=True,
                )
                return {"status": "CANCELLED", "reason": "tracking_cancelled"}
            raise
        if isinstance(v3_result, dict) and v3_result.get("skipped"):
            return {"status": "CANCELLED", "reason": str(v3_result.get("skipped"))}
        if not _task_still_current(db_engine, video_name, self.request.id):
            print(
                f"[WORKER] 跟踪入库后任务已过期，不写回人数/完成态: {video_name}",
                flush=True,
            )
            return {"status": "CANCELLED", "reason": "superseded_after_track"}
        if isinstance(v3_result, dict):
            person_count = int(
                (v3_result.get("import") or {}).get("person_count") or 0
            )
        # 立即写回人数，避免前端长时间看到 0
        with db_engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE videos SET target_count=:c, progress=99 "
                    "WHERE file_name=:v AND is_media_source=1"
                ),
                {"c": person_count, "v": video_name},
            )
        _set_progress(99, person_count)
    except Exception as v3_err:
        if not _task_still_current(db_engine, video_name, self.request.id):
            return {"status": "CANCELLED", "reason": "superseded"}
        reason = f"OSNet 跟踪入库失败: {v3_err}"
        print(f"❌ {reason}", flush=True)
        _set_task_failed(db_engine, video_name, reason)
        return {"status": "FAILED", "error": reason}

    if not _task_still_current(db_engine, video_name, self.request.id):
        return {"status": "CANCELLED", "reason": "superseded"}

    # 有本地归档则可选后台转码（VIDEO_PLAYBACK_FINALIZE=1）；默认跳过，直接 completed
    needs_finalize = os.path.isfile(archive_path) and os.path.getsize(archive_path) > 0
    if needs_finalize and _playback_finalize_enabled():
        _schedule_transcoding_finalize(db_engine, video_name, person_count, archive_path)
    else:
        if needs_finalize and not _playback_finalize_enabled():
            print(
                f"[PLAYBACK] 已跳过归档可播化（VIDEO_PLAYBACK_FINALIZE=0）: {archive_path}",
                flush=True,
            )
        _mark_task_completed(db_engine, video_name, person_count)

    if temp_download_file and os.path.exists(temp_download_file):
        try:
            os.remove(temp_download_file)
            print(f"🗑️ 清理临时文件: {temp_download_file}")
        except Exception:
            pass

    return {"status": "SUCCESS", "count": person_count, "mode": "tracking_v3_only"}
