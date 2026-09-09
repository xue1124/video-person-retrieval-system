import json
import os
import re
import shutil
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Dict, List, Optional


def _bootstrap_siglip_env() -> None:
    """
    在 import tasks / search_service 之前执行。
    若未设置环境变量且本机存在 SigLip_demo 默认布局，则自动填充（可被已有环境变量覆盖）。
    """
    demo = os.environ.get("SIGLIP_DEMO_ROOT", str(Path(__file__).resolve().parent))
    tok = os.path.join(demo, "siglip_v1")
    if os.path.isfile(os.path.join(tok, "tokenizer_config.json")):
        os.environ.setdefault("SIGLIP_TOKENIZER_PATH", tok)
    for path, key in (
        (os.path.join(demo, "siglip_text.onnx"), "SIGLIP_TEXT_ONNX"),
        (os.path.join(demo, "siglip_vision.onnx"), "SIGLIP_VISION_ONNX"),
    ):
        if os.path.isfile(path):
            os.environ.setdefault(key, path)


_bootstrap_siglip_env()

import threading

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests
from sqlalchemy import text
from sqlalchemy.engine import Engine
import xml.etree.ElementTree as ET

from auth_deps import (
    create_access_token,
    fetch_user_by_username,
    get_current_user,
    get_current_user_media,
    require_admin,
    verify_password,
)
import db as app_db
from media_cleanup import purge_video_modeling_artifacts
from room_copresence import perform_room_copresence
from search_service import perform_search
from video_time import parse_user_captured_at
from tasks import (
    process_video_task,
    app as celery_app,
    get_runtime_backend_info,
    get_resources,
    warm_live_detector,
)
from requests.auth import HTTPDigestAuth
from urllib.parse import quote_plus

import cv2
import numpy as np

from video_playback import (
    is_browser_friendly_mp4 as _is_browser_friendly_mp4,
    is_finalize_in_progress,
    playback_revision,
)
from room_geometry import polygons_from_freedraw_rgba, stroke_polygon_to_natural
from rtsp_live import (
    SOURCE_TYPE_RTSP_LIVE,
    STATUS_STREAMING,
    STATUS_STOPPED,
    active_live_thread_count,
    is_live_thread_running,
    live_archive_path,
    request_stop,
    request_stop_and_wait,
    shutdown_all,
    start_live_stream,
)

engine: Engine = app_db.get_engine()

FFMPEG_PATH = (
    os.environ.get("FFMPEG_PATH") or shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
)
BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = str(BASE_DIR / "video_archives")
CLIP_DIR = str(BASE_DIR / "temp_clips")
GALLERY_DIR = str(BASE_DIR / "video_crops")
QUERY_STORAGE = str(BASE_DIR / "query_storage")
SNAPSHOT_ROOT = Path(
    os.environ.get("TRACKING_SNAPSHOT_ROOT", str(BASE_DIR / "tracking_snapshots"))
).expanduser().resolve()

for d in [VIDEO_DIR, CLIP_DIR, GALLERY_DIR, QUERY_STORAGE]:
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

if not os.path.exists(FFMPEG_PATH):
    raise RuntimeError(
        f"❌ FFmpeg 未找到: {FFMPEG_PATH}（将 ffmpeg 加入 PATH 或设置环境变量 FFMPEG_PATH）"
    )


def _safe_upload_filename(filename: Optional[str]) -> str:
    name = os.path.basename((filename or "").strip())
    return name or f"upload_{uuid.uuid4().hex[:8]}.mp4"


def _allocate_upload_target(filename: Optional[str]) -> tuple[str, str]:
    """
    本地上传始终使用客户端原始文件名（仅 basename 清洗）。
    同名再次上传：覆盖 video_archives 下同名文件；videos 由 UPSERT 更新；
    调用方应在写入前 revoke 旧任务并 purge 旧底库/crops/房间标注/归档视频。
    """
    safe_name = _safe_upload_filename(filename)
    target_path = os.path.join(VIDEO_DIR, safe_name)
    return safe_name, target_path


def _resolve_ffprobe_for_boot_log() -> str:
    env = os.environ.get("FFPROBE_PATH")
    if env and os.path.isfile(env):
        return env
    if FFMPEG_PATH:
        sibling = os.path.join(
            os.path.dirname(FFMPEG_PATH),
            "ffprobe.exe" if os.name == "nt" else "ffprobe",
        )
        if os.path.isfile(sibling):
            return sibling
    return shutil.which("ffprobe") or "(not found)"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = engine
    print(
        "🧩 SigLIP 环境: "
        f"SIGLIP_TOKENIZER_PATH={os.environ.get('SIGLIP_TOKENIZER_PATH', '(未设置)')} | "
        f"SIGLIP_TEXT_ONNX={os.environ.get('SIGLIP_TEXT_ONNX', 'siglip_text.onnx')} | "
        f"SIGLIP_VISION_ONNX={os.environ.get('SIGLIP_VISION_ONNX', 'siglip_vision.onnx')}"
    )
    print(
        "🛠 媒体工具解析: "
        f"FFMPEG_PATH={FFMPEG_PATH} | "
        f"FFPROBE_PATH={_resolve_ffprobe_for_boot_log()} | "
        f"which(ffmpeg)={shutil.which('ffmpeg') or '(not found)'} | "
        f"which(ffprobe)={shutil.which('ffprobe') or '(not found)'}"
    )
    yield
    shutdown_all()


app = FastAPI(title="SigLIP ISAPI API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from tracking_api import router as tracking_router  # noqa: E402
from reports.api import router as reports_router  # noqa: E402

app.include_router(tracking_router)
app.include_router(reports_router)


def redact_raw_path(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return raw
    redacted = re.sub(r"([?&]password=)[^&]*", r"\1***", raw, flags=re.I)
    # RTSP 常见格式为 rtsp://user:password@host/path，返回前隐藏密码。
    return re.sub(r"(rtsp://[^:/@\s]+:)[^@/\s]+@", r"\1***@", redacted, flags=re.I)


def _upsert_processed_pending_row(
    conn,
    *,
    file_name: str,
    source_type: str,
    task_id: str,
    raw_path: str,
    captured_at=None,
) -> None:
    app_db.upsert_media_video(
        conn,
        file_name=file_name,
        origin_label=source_type,
        task_id=task_id,
        source_path=raw_path,
        status="pending",
        captured_at=captured_at,
    )


def _upsert_processed_pending_upload(
    conn,
    *,
    file_name: str,
    task_id: str,
    raw_path: str,
    captured_at=None,
) -> None:
    app_db.upsert_media_video(
        conn,
        file_name=file_name,
        origin_label="手动上传",
        task_id=task_id,
        source_path=raw_path,
        status="pending",
        captured_at=captured_at,
    )


def _revoke_celery_task(task_id: str) -> None:
    try:
        celery_app.control.revoke(task_id, terminate=True, signal="SIGTERM")
    except Exception:
        pass


def safe_file_token(name: str) -> str:
    base = os.path.basename(name.replace("\\", "/"))
    if not base or ".." in name or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="非法文件名")
    return base


def _normalize_isapi_channel_id(raw_id: str) -> Optional[int]:
    s = (raw_id or "").strip()
    if not s:
        return None
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    val = int(m.group(1))
    if val >= 100 and val % 100 in (1, 2):
        return val // 100
    return val


def _extract_isapi_channels(xml_text: str) -> List[dict]:
    items: List[dict] = []
    try:
        root = ET.fromstring(xml_text)
        for elem in root.iter():
            tag = elem.tag.lower()
            if not (tag.endswith("streamingchannel") or tag.endswith("inputproxychannel")):
                continue
            children = list(elem)
            raw_id = ""
            name = ""
            for ch in children:
                ctag = ch.tag.lower()
                txt = (ch.text or "").strip()
                if not txt:
                    continue
                if ctag.endswith("id"):
                    raw_id = txt
                elif ctag.endswith("name"):
                    name = txt
            ch_id = _normalize_isapi_channel_id(raw_id)
            if ch_id is None:
                continue
            items.append(
                {
                    "id": ch_id,
                    "name": name or f"通道{ch_id}",
                    "raw_id": raw_id or str(ch_id),
                }
            )
    except Exception:
        pass
    uniq = {}
    for it in items:
        uniq[it["id"]] = it
    return [uniq[k] for k in sorted(uniq)]


def _discover_isapi_channels(host: str, username: str, password: str) -> List[dict]:
    from box_tunnel import isapi_proxy_headers, isapi_proxy_url, normalize_nvr_host

    auth = HTTPDigestAuth(username, password)
    target = normalize_nvr_host(host)
    headers = isapi_proxy_headers(target)
    endpoints = [
        isapi_proxy_url("/ISAPI/Streaming/channels"),
        isapi_proxy_url("/ISAPI/ContentMgmt/InputProxy/channels"),
    ]
    last_err = None
    for url in endpoints:
        try:
            resp = requests.get(url, auth=auth, headers=headers, timeout=(10, 30))
            resp.raise_for_status()
            channels = _extract_isapi_channels(resp.text)
            if channels:
                return channels
        except Exception as e:
            last_err = e
    raise HTTPException(
        status_code=502,
        detail=f"无法识别设备通道，请确认 IP/账号密码正确且设备支持 ISAPI。{last_err or ''}",
    )


# MySQL naive datetime 在容器里多为 UTC；序列化为东八区 ISO，避免完成时间/耗时偏差 8 小时。
_UTC = ZoneInfo("UTC")
_SHANGHAI = ZoneInfo("Asia/Shanghai")


def serialize_datetime(dt) -> Optional[str]:
    if dt is None:
        return None
    if hasattr(dt, "isoformat"):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_UTC)
        return dt.astimezone(_SHANGHAI).isoformat()
    return str(dt)


def _format_task_time_label(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts)
        # 文件名中不能包含冒号，用下划线和横杠替代时间部分
        return dt.strftime("%Y-%m-%d_%H-%M")
    except Exception:
        return ts.replace("T", "_").replace(":", "-")[:16]


def _auto_isapi_task_name(channel: int, start: str, end: str) -> str:
    return f"海康回放_通道{channel:02d}_{_format_task_time_label(start)}_{_format_task_time_label(end)}"


# ---------- 健康检查（无需登录）----------
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/runtime/backends")
def runtime_backends(user: dict = Depends(get_current_user)):
    """
    返回当前进程中的模型加载状态与实际 Provider（TRT/CUDA/CPU）。
    注意：API 与 Celery Worker 为不同进程，应分别查看各自日志。
    """
    return get_runtime_backend_info()


@app.post("/isapi/channels")
def detect_isapi_channels(
    host: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    user: dict = Depends(get_current_user),
):
    if not host or not username or not str(password).strip():
        raise HTTPException(status_code=400, detail="请先填写设备 IP、用户名和密码")
    return {"channels": _discover_isapi_channels(host, username, password)}


@app.get("/celery/inspect")
def celery_inspect(user: dict = Depends(get_current_user)):
    """
    查看 Celery worker 是否在消费任务、当前在执行什么。
    Redis 里通常看不到「任务列表」：消息进默认队列后很快被 worker 取走。
    """
    broker = str(celery_app.conf.broker_url or "")
    insp = celery_app.control.inspect(timeout=3.0)
    if not insp:
        return {
            "ok": False,
            "broker": broker,
            "hint": "没有 worker 在超时内响应：请启动 celery worker，并确保与 API 使用同一 CELERY_BROKER_URL。",
        }
    out = {
        "ok": True,
        "broker": broker,
        "active": insp.active() or {},
        "reserved": insp.reserved() or {},
        "scheduled": insp.scheduled() or {},
        "stats": insp.stats() or {},
    }
    try:
        import redis as redis_lib

        r = redis_lib.Redis.from_url(
            os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
        )
        out["redis_llen_celery_queue"] = int(r.llen("celery"))
    except Exception as e:
        out["redis_llen_celery_queue"] = None
        out["redis_queue_note"] = str(e)
    return out


# ---------- 认证 ----------
class LoginBody(BaseModel):
    username: str
    password: str


@app.post("/auth/login")
def login(body: LoginBody, request: Request):
    eng = request.app.state.engine
    row = fetch_user_by_username(eng, body.username)
    if not row or not row["is_active"]:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    if not verify_password(body.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_access_token(row["username"], row["role"])
    return {
        "access_token": token,
        "token_type": "bearer",
        "username": row["username"],
        "role": row["role"],
    }


@app.get("/auth/me")
def me(user: dict = Depends(get_current_user)):
    return {"username": user["username"], "role": user["role"]}


# ---------- 静态资源（需登录）----------
@app.get("/files/crop/{filename}")
def serve_crop(filename: str, user: dict = Depends(get_current_user)):
    fn = safe_file_token(filename)
    path = os.path.join(GALLERY_DIR, fn)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(path)


@app.get("/files/query/{filename}")
def serve_query(filename: str, user: dict = Depends(get_current_user)):
    fn = safe_file_token(filename)
    path = os.path.join(QUERY_STORAGE, fn)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(path)


@app.get("/files/obs/{observation_id}")
def serve_observation_crop(observation_id: int, user: dict = Depends(get_current_user)):
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT id, crop_path FROM observations WHERE id=:id"),
            {"id": int(observation_id)},
        ).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="观测图片不存在")
    rel = str(row["crop_path"] or "").lstrip("/").replace("\\", "/")
    if not rel or ".." in rel.split("/"):
        raise HTTPException(status_code=400, detail="非法观测图片路径")
    path = (SNAPSHOT_ROOT / rel).resolve()
    try:
        path.relative_to(SNAPSHOT_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="非法观测图片路径") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="观测图片文件不存在")
    return FileResponse(path, media_type="image/jpeg")


def _norm_video_file_key(name: str) -> str:
    s = (name or "").strip()
    if not s or s != os.path.basename(s.replace("\\", "/")):
        raise HTTPException(status_code=400, detail="非法 video_name")
    return s


def _ffprobe_bin() -> str:
    env = os.environ.get("FFPROBE_PATH")
    if env and os.path.isfile(env):
        return env
    if FFMPEG_PATH:
        sibling = os.path.join(os.path.dirname(FFMPEG_PATH), "ffprobe.exe" if os.name == "nt" else "ffprobe")
        if os.path.isfile(sibling):
            return sibling
    found = shutil.which("ffprobe")
    return found or "ffprobe"


def _ffprobe_video_codec(path: str) -> Optional[str]:
    try:
        out = subprocess.run(
            [
                _ffprobe_bin(),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=nw=1:nk=1",
                path,
            ],
            capture_output=True,
            timeout=30,
            check=False,
        )
        if out.returncode != 0:
            return None
        name = (out.stdout or b"").decode("utf-8", errors="ignore").strip().lower()
        return name or None
    except Exception:
        return None


def _is_live_recording_playable(
    source_path: str,
    status: Optional[str],
    source_type: Optional[str],
) -> bool:
    """实时任务边录边播：fMP4 文件在增长中也可尝试播放。"""
    if source_type != SOURCE_TYPE_RTSP_LIVE:
        return False
    if status not in (STATUS_STREAMING, STATUS_STOPPED, "completed"):
        return False
    try:
        return os.path.isfile(source_path) and os.path.getsize(source_path) > 32768
    except OSError:
        return False


def _resolve_archived_video_path(video_name: str, raw_path: Optional[str] = None) -> str:
    """解析可播放的完整视频路径：优先 DB raw_path（本地文件），再兜底 video_archives。"""
    vn = _norm_video_file_key(video_name)
    candidates: List[str] = []
    rp = (raw_path or "").strip()
    if rp and not rp.startswith(("rtsp://", "isapi://")):
        p = rp if os.path.isabs(rp) else os.path.join(str(BASE_DIR), rp)
        candidates.append(os.path.normpath(p))

    candidates.append(os.path.join(VIDEO_DIR, vn))
    stem, ext = os.path.splitext(vn)
    if not ext:
        candidates.append(os.path.join(VIDEO_DIR, f"{vn}.mp4"))
    elif ext.lower() == ".mp4":
        candidates.append(os.path.join(VIDEO_DIR, f"{stem}.mp4"))

    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            return p

    raise HTTPException(
        status_code=404,
        detail="未找到可播放的完整视频（请确认本地上传或 ISAPI 已落盘到 video_archives）",
    )


@app.get("/files/video/check")
def check_full_video(
    video_name: str = Query(...),
    user: dict = Depends(get_current_user),
):
    """播放前检查归档视频是否存在（不触发转码）。"""
    vn = _norm_video_file_key(video_name)
    raw_path: Optional[str] = None
    status: Optional[str] = None
    source_type: Optional[str] = None
    with engine.connect() as conn:
        row = conn.execute(text(app_db.MEDIA_LOOKUP_SQL), {"v": vn}).mappings().first()
    if row:
        raw_path = row.get("raw_path")
        status = str(row.get("status") or "")
        source_type = str(row.get("source_type") or "")
    try:
        source_path = _resolve_archived_video_path(vn, str(raw_path) if raw_path else None)
    except HTTPException:
        if status == STATUS_STREAMING:
            raise HTTPException(
                status_code=404,
                detail="录像文件尚未就绪，请稍等片刻后再播放",
            )
        raise
    live_ok = _is_live_recording_playable(source_path, status, source_type)
    is_live = live_ok and status == STATUS_STREAMING
    optimizing = is_finalize_in_progress(source_path)
    browser_friendly = _is_browser_friendly_mp4(source_path) or is_live
    return {
        "ok": True,
        "video_name": vn,
        "source_file": os.path.basename(source_path),
        "browser_friendly": browser_friendly,
        "needs_transcode": False,
        "is_live_recording": is_live,
        "needs_faststart": optimizing,
        "playback_revision": playback_revision(source_path) if is_live else None,
    }


@app.get("/files/video")
def serve_full_video(
    video_name: str = Query(..., description="与 videos.file_name 一致"),
    user: dict = Depends(get_current_user_media),
):
    vn = _norm_video_file_key(video_name)
    raw_path: Optional[str] = None
    status: Optional[str] = None
    source_type: Optional[str] = None
    with engine.connect() as conn:
        row = conn.execute(text(app_db.MEDIA_LOOKUP_SQL), {"v": vn}).mappings().first()
    if row:
        raw_path = row.get("raw_path")
        status = str(row.get("status") or "")
        source_type = str(row.get("source_type") or "")
    source_path = _resolve_archived_video_path(vn, str(raw_path) if raw_path else None)
    if is_finalize_in_progress(source_path):
        raise HTTPException(status_code=503, detail="视频正在优化为可播格式，请稍后重试")
    playback_path = source_path
    headers: Dict[str, str] = {}
    if status == STATUS_STREAMING:
        headers["Cache-Control"] = "no-store"
    return FileResponse(
        playback_path,
        media_type="video/mp4",
        filename=os.path.basename(playback_path),
        headers=headers,
    )


# ---------- 媒体源 ----------
@app.get("/media/sources")
def list_media_sources(user: dict = Depends(get_current_user)):
    with engine.connect() as conn:
        rows = conn.execute(text(app_db.MEDIA_LIST_SQL)).mappings().all()
        count_rows = conn.execute(
            text(
                """
                SELECT v.file_name, COUNT(r.id) AS cnt
                FROM rooms r
                JOIN videos v ON v.id = r.video_id
                WHERE v.is_media_source = 1
                GROUP BY v.file_name
                """
            )
        ).mappings().all()
    room_counts: Dict[str, int] = {
        str(cr["file_name"]): int(cr["cnt"]) for cr in count_rows
    }

    out = []
    for r in rows:
        d = dict(r)
        d["raw_path"] = redact_raw_path(d.get("raw_path"))
        for k in ("completed_at", "created_at", "updated_at", "processing_started_at"):
            if d.get(k) is not None:
                d[k] = serialize_datetime(d[k])
        d["room_count"] = room_counts.get(str(d.get("file_name") or ""), 0)
        out.append(d)
    return out


def _delete_media_source_internal(
    *,
    file_name: str,
    error_if_missing: bool = True,
) -> bool:
    """与「视频源列表-删除」一致的后端清理逻辑。返回是否找到并清理了记录。"""
    if not file_name or file_name != os.path.basename(file_name):
        raise HTTPException(status_code=400, detail="非法 file_name")
    eng = engine
    with eng.connect() as conn:
        row = conn.execute(text(app_db.MEDIA_LOOKUP_SQL), {"v": file_name}).mappings().first()
    if not row:
        if error_if_missing:
            raise HTTPException(status_code=404, detail="记录不存在")
        return False

    raw_path_db = row.get("raw_path")
    st = str(row.get("status") or "")
    src = str(row.get("source_type") or "")
    if st in (STATUS_STREAMING, "pending") or src == SOURCE_TYPE_RTSP_LIVE:
        request_stop_and_wait(file_name, timeout=10.0)
    elif st == "transcoding":
        try:
            from video_playback import cleanup_finalize_sidecars

            rp = str(raw_path_db or "").strip()
            if rp and not rp.startswith(("rtsp://", "isapi://")):
                p = rp if os.path.isabs(rp) else os.path.join(str(BASE_DIR), rp)
                cleanup_finalize_sidecars(p)
        except Exception:
            pass
    elif row["task_id"] and st in ("pending", "processing", "transcoding"):
        try:
            # Windows/线程池下 terminate 往往杀不掉线程，worker 内会轮询 DB 自行退出
            celery_app.control.revoke(str(row["task_id"]), terminate=True, signal="SIGTERM")
            print(
                f"[media] 已 revoke Celery 任务 {row['task_id']}（线程池需靠轮询退出）",
                flush=True,
            )
        except Exception as exc:
            print(f"[media] revoke 失败: {exc}", flush=True)

    purge_video_modeling_artifacts(
        eng,
        file_name,
        raw_path=str(raw_path_db) if raw_path_db else None,
        remove_archived_video=True,
        remove_rooms=False,
    )
    try:
        from tracking_v3.purge import purge_tracking_by_file_name

        v3_purge = purge_tracking_by_file_name(file_name)
        print(f"[media] 已清理 medical_audit_v3: {v3_purge}", flush=True)
    except Exception as exc:
        print(f"[media] 清理 medical_audit_v3 失败: {exc}", flush=True)
        with eng.begin() as conn:
            conn.execute(
                text("DELETE FROM videos WHERE file_name=:v AND is_media_source=1"),
                {"v": file_name},
            )
    return True


@app.delete("/media/sources")
def delete_media_source(
    file_name: str = Query(..., description="视频任务名 file_name"),
    user: dict = Depends(require_admin),
):
    _delete_media_source_internal(file_name=file_name, error_if_missing=True)
    return {"ok": True}


@app.post("/stream/live/start")
async def start_rtsp_live_stream(
    rtsp_url: str = Form(...),
    file_name: str = Form(...),
    skip_frames: int = Form(...),
    conf_val: float = Form(...),
    user: dict = Depends(get_current_user),
):
    """启动 RTSP 实时建模；与离线任务不同，长驻线程边读边入库。"""
    name = safe_file_token(file_name.strip())
    url = (rtsp_url or "").strip()
    if not url.lower().startswith("rtsp://"):
        raise HTTPException(status_code=400, detail="RTSP 地址必须以 rtsp:// 开头")
    try:
        from box_tunnel import resolve_rtsp_playback_url

        url = resolve_rtsp_playback_url(url)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"RTSP 隧道中继失败: {e}") from e
    if skip_frames < 1:
        raise HTTPException(status_code=400, detail="抽帧步长至少为 1")

    if is_live_thread_running(name):
        raise HTTPException(status_code=409, detail="该任务已在直播中")

    try:
        _delete_media_source_internal(file_name=name, error_if_missing=False)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清理旧任务失败: {e}")

    task_id = uuid.uuid4().hex
    archive = str(live_archive_path(name))
    try:
        with engine.begin() as conn:
            _upsert_processed_pending_row(
                conn,
                file_name=name,
                source_type=SOURCE_TYPE_RTSP_LIVE,
                task_id=task_id,
                raw_path=archive,
                captured_at=None,
            )
            conn.execute(
                text(
                    "UPDATE videos SET status='pending', progress=0, target_count=0, "
                    "failure_reason=NULL, duration='排队中…', fps=NULL, completed_at=NULL, "
                    "processing_started_at=NOW(3) "
                    "WHERE file_name=:v AND is_media_source=1"
                ),
                {"v": name},
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"媒体源记录写入失败: {e}")

    try:
        get_resources()
        warm_live_detector()
        start_live_stream(
            engine=engine,
            file_name=name,
            rtsp_url=url,
            skip_frames=skip_frames,
            conf_val=conf_val,
            task_id=task_id,
        )
    except Exception as e:
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM videos WHERE file_name=:v AND is_media_source=1"),
                {"v": name},
            )
        raise HTTPException(status_code=500, detail=str(e))

    return {"task_id": task_id, "file_name": name, "status": STATUS_STREAMING}


@app.post("/stream/live/stop")
async def stop_rtsp_live_stream(
    file_name: str = Form(...),
    user: dict = Depends(get_current_user),
):
    """终止实时流：停止后续处理，保留已入库的检索数据。"""
    name = safe_file_token(file_name.strip())
    with engine.connect() as conn:
        row = conn.execute(text(app_db.MEDIA_LOOKUP_SQL), {"v": name}).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="记录不存在")
    if str(row.get("source_type") or "") != SOURCE_TYPE_RTSP_LIVE:
        raise HTTPException(status_code=400, detail="仅 RTSP 实时任务可终止")
    st = str(row.get("status") or "")
    if st == STATUS_STOPPED:
        return {"ok": True, "file_name": name, "status": STATUS_STOPPED}
    if st not in (STATUS_STREAMING, "pending"):
        raise HTTPException(status_code=400, detail=f"当前状态不可终止: {st}")

    request_stop(name)
    request_stop_and_wait(name, timeout=15.0)
    if active_live_thread_count() == 0:
        try:
            from box_tunnel import stop_rtsp_relay

            stop_rtsp_relay()
        except Exception:
            pass
    return {"ok": True, "file_name": name, "status": STATUS_STOPPED}


# ---------- 房间标注 ----------
_room_recompute_guard = threading.Lock()
_room_recompute_locks: dict[str, threading.Lock] = {}
_room_recompute_gen: dict[str, int] = {}
_room_recompute_done: dict[str, int] = {}


def _room_video_lock(video_name: str) -> threading.Lock:
    with _room_recompute_guard:
        lock = _room_recompute_locks.get(video_name)
        if lock is None:
            lock = threading.Lock()
            _room_recompute_locks[video_name] = lock
        return lock


def _bump_room_recompute(video_name: str) -> None:
    with _room_recompute_guard:
        _room_recompute_gen[video_name] = _room_recompute_gen.get(video_name, 0) + 1


def _sync_rooms_fast(video_name: str) -> dict:
    """用当前 rooms 当场重算停留；切图放到后台。"""
    from tracking_v3.sync_rooms import recompute_stays_for_video_name

    return recompute_stays_for_video_name(video_name)


def _recompute_rooms_background(video_name: str) -> None:
    """停留已写入后，后台只重切快照，不再清 stay_segments。"""
    lock = _room_video_lock(video_name)
    with lock:
        while True:
            with _room_recompute_guard:
                start_gen = _room_recompute_gen.get(video_name, 0)
                if _room_recompute_done.get(video_name, -1) == start_gen:
                    return
            try:
                from tracking_v3.sync_rooms import generate_snapshots_for_video_name

                report = generate_snapshots_for_video_name(video_name)
                print(f"[rooms] 后台快照重切完成: {report}", flush=True)
                with _room_recompute_guard:
                    _room_recompute_done[video_name] = start_gen
            except Exception as exc:
                print(f"[rooms] 后台快照重切失败: {exc}", flush=True)
            with _room_recompute_guard:
                if _room_recompute_gen.get(video_name, 0) == start_gen:
                    return
            print(
                f"[rooms] 房间又有变更，按最新停留再切一遍快照: {video_name}",
                flush=True,
            )


def _norm_video_name_key(name: str) -> str:
    s = (name or "").strip()
    if not s or s != os.path.basename(s.replace("\\", "/")):
        raise HTTPException(status_code=400, detail="非法 video_name")
    return s


def _polygon_from_db_cell(val: Any) -> List[List[float]]:
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            out = json.loads(val)
            return out if isinstance(out, list) else []
        except json.JSONDecodeError:
            return []
    return []


class RoomUpsertBody(BaseModel):
    video_name: str
    room_name: str
    polygon: List[List[float]]


@app.get("/rooms")
def list_rooms(
    video_name: str = Query(..., description="与 videos.file_name 一致"),
    user: dict = Depends(get_current_user),
):
    vn = _norm_video_name_key(video_name)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT r.id, v.file_name AS video_name, r.name AS room_name,
                       r.polygon_json, r.created_at, r.updated_at, r.video_id
                FROM rooms r
                JOIN videos v ON v.id = r.video_id
                WHERE v.file_name = :v
                ORDER BY r.id ASC
                """
            ),
            {"v": vn},
        ).mappings().all()
    out = []
    for r in rows:
        d = dict(r)
        d["polygon"] = _polygon_from_db_cell(d.pop("polygon_json", None))
        for k in ("created_at", "updated_at"):
            if d.get(k) is not None:
                d[k] = serialize_datetime(d[k])
        out.append(d)
    return {"video_name": vn, "rooms": out}


@app.get("/rooms/reference")
def room_reference_image(
    video_name: str = Query(..., description="用于画布底图：优先归档视频抽帧，其次旧 gallery crop"),
    user: dict = Depends(get_current_user),
):
    vn = _norm_video_name_key(video_name)

    # 1) 从归档视频抽一帧（新 OSNet-only 建模不再写 gallery_meta）
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT source_path AS raw_path FROM videos WHERE file_name=:v LIMIT 1"),
            {"v": vn},
        ).mappings().first()
    candidates: list[str] = []
    if row and row.get("raw_path"):
        rp = str(row["raw_path"]).strip()
        if rp and not rp.startswith(("rtsp://", "isapi://")):
            candidates.append(rp if os.path.isabs(rp) else os.path.join(str(BASE_DIR), rp))
    candidates.extend(
        [
            os.path.join(VIDEO_DIR, vn),
            os.path.join(VIDEO_DIR, f"{vn}.mp4" if not vn.lower().endswith(".mp4") else vn),
        ]
    )
    ref_name = f"{vn}__room_ref.jpg"
    ref_disk = os.path.join(GALLERY_DIR, ref_name)
    for video_path in candidates:
        if not video_path or not os.path.isfile(video_path) or os.path.getsize(video_path) <= 0:
            continue
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue
        ok, frame = cap.read()
        if not ok:
            # 尝试中间帧
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
                ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            continue
        os.makedirs(GALLERY_DIR, exist_ok=True)
        if cv2.imwrite(ref_disk, frame) and os.path.isfile(ref_disk):
            return {
                "video_name": vn,
                "image_path": ref_disk,
                "filename": ref_name,
                "image_url": f"/files/crop/{ref_name}",
                "source": "video_frame",
            }

    # 2) 兼容旧建模：仍有 gallery crop 时沿用
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT image_path FROM gallery_meta WHERE video_name=:v "
                "ORDER BY timestamp ASC, id ASC LIMIT 64"
            ),
            {"v": vn},
        ).mappings().all()
    for row in rows:
        ip = str(row.get("image_path") or "")
        if not ip:
            continue
        base = os.path.basename(ip.replace("\\", "/"))
        disk = os.path.join(GALLERY_DIR, base)
        if os.path.isfile(disk):
            return {
                "video_name": vn,
                "image_path": ip,
                "filename": base,
                "image_url": f"/files/crop/{base}",
                "source": "gallery_meta",
            }
    raise HTTPException(
        status_code=404,
        detail="找不到可用于房间标注的参考图（请确认视频已上传归档）",
    )


@app.post("/rooms/extract_freedraw")
async def rooms_extract_freedraw(
    file: UploadFile = File(...),
    natural_w: int = Form(...),
    natural_h: int = Form(...),
    canvas_w: int = Form(...),
    canvas_h: int = Form(...),
    stroke_r: int = Form(255),
    stroke_g: int = Form(0),
    stroke_b: int = Form(128),
    color_tol: float = Form(60.0),
    user: dict = Depends(get_current_user),
):
    """
    接收仅含描边的 PNG（RGBA，与画布同尺寸），用与 Streamlit demo 相同的颜色+轮廓算法提取多边形，
    并换算到原图 natural_w x natural_h 像素坐标。
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="空文件")
    buf = np.frombuffer(raw, dtype=np.uint8)
    im = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise HTTPException(status_code=400, detail="无法解码 PNG")
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGBA)
    elif im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGBA)
    elif im.shape[2] == 4:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
    else:
        raise HTTPException(status_code=400, detail="不支持的图片通道")

    h, w = im.shape[:2]
    cw, ch = int(canvas_w), int(canvas_h)
    if w != cw or h != ch:
        raise HTTPException(
            status_code=400,
            detail=f"PNG 尺寸与画布不一致：期望 {cw}x{ch}，实际 {w}x{h}",
        )

    polys = polygons_from_freedraw_rgba(
        im,
        (int(stroke_r), int(stroke_g), int(stroke_b)),
        color_tol=float(color_tol),
    )
    if not polys:
        raise HTTPException(
            status_code=400,
            detail="未识别到闭合描边：请沿房间边界画一圈并尽量闭合，或提高颜色容差",
        )
    best = polys[0]
    natural = stroke_polygon_to_natural(best, w, h, int(natural_w), int(natural_h))
    return {
        "polygon": natural,
        "polygon_canvas": best,
        "vertex_count": len(natural),
        "candidates": len(polys),
    }


@app.post("/rooms")
def upsert_room(
    body: RoomUpsertBody,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
):
    vn = _norm_video_name_key(body.video_name)
    rn = (body.room_name or "").strip()
    if not rn:
        raise HTTPException(status_code=400, detail="room_name 不能为空")
    poly = body.polygon
    if not isinstance(poly, list) or len(poly) < 3:
        raise HTTPException(status_code=400, detail="polygon 至少 3 个顶点")
    for p in poly:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            raise HTTPException(status_code=400, detail="polygon 每个点需为 [x,y]")
        try:
            float(p[0])
            float(p[1])
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="polygon 坐标必须为数字")

    poly_json = json.dumps(poly, ensure_ascii=False)
    with engine.begin() as conn:
        video_id = app_db.get_video_id(conn, vn)
        if video_id is None:
            raise HTTPException(status_code=404, detail="未找到该视频，请先上传或建模")
        conn.execute(
            text(
                """
                INSERT INTO rooms (video_id, name, polygon_json)
                VALUES (:video_id, :name, :polygon_json)
                ON DUPLICATE KEY UPDATE
                  polygon_json = VALUES(polygon_json),
                  updated_at = CURRENT_TIMESTAMP(3)
                """
            ),
            {"video_id": video_id, "name": rn, "polygon_json": poly_json},
        )
        rid = conn.execute(
            text("SELECT id FROM rooms WHERE video_id=:video_id AND name=:name LIMIT 1"),
            {"video_id": video_id, "name": rn},
        ).scalar()

    tracking_sync: dict | None = None
    try:
        tracking_sync = _sync_rooms_fast(vn)
        print(f"[rooms] 已写入多边形并算出停留: {tracking_sync}", flush=True)
    except Exception as exc:
        print(f"[rooms] 停留重算失败（房间已保存）: {exc}", flush=True)
        tracking_sync = {"error": str(exc)}
    _bump_room_recompute(vn)
    background_tasks.add_task(_recompute_rooms_background, vn)

    return {
        "ok": True,
        "id": int(rid) if rid is not None else None,
        "video_name": vn,
        "room_name": rn,
        "tracking_sync": tracking_sync,
        "recompute": "stays_ready",
    }


@app.delete("/rooms/by-id/{room_id}")
def delete_room(
    room_id: int,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
):
    video_name: str | None = None
    with engine.begin() as conn:
        row = conn.execute(
            text(
                """
                SELECT v.file_name AS video_name
                FROM rooms r
                JOIN videos v ON v.id = r.video_id
                WHERE r.id = :id
                """
            ),
            {"id": room_id},
        ).mappings().first()
        if not row:
            raise HTTPException(status_code=404, detail="未找到该房间记录")
        video_name = str(row["video_name"])
        r = conn.execute(text("DELETE FROM rooms WHERE id=:id"), {"id": room_id})
    if r.rowcount == 0:
        raise HTTPException(status_code=404, detail="未找到该房间记录")

    tracking_sync: dict | None = None
    if video_name:
        try:
            tracking_sync = _sync_rooms_fast(video_name)
            print(f"[rooms] 删除后已重算停留: {tracking_sync}", flush=True)
        except Exception as exc:
            print(f"[rooms] 删除后停留重算失败: {exc}", flush=True)
            tracking_sync = {"error": str(exc)}
        _bump_room_recompute(video_name)
        background_tasks.add_task(_recompute_rooms_background, video_name)

    return {
        "ok": True,
        "deleted_id": room_id,
        "tracking_sync": tracking_sync,
        "recompute": "stays_ready",
    }


# ---------- 视频裁切 ----------
@app.get("/get_video_clip")
def get_video_clip(
    video_name: str,
    timestamp: float,
    offset: int = 30,
    user: dict = Depends(get_current_user),
):
    try:
        with engine.connect() as conn:
            record = conn.execute(
                text(app_db.MEDIA_LOOKUP_SQL),
                {"v": video_name},
            ).mappings().first()

        if not record:
            raise HTTPException(status_code=404, detail="数据库中未找到该视频记录")

        input_source = record["raw_path"]
        if not input_source:
            input_source = os.path.join(VIDEO_DIR, video_name)
            print(f"⚠️ raw_path 缺失，尝试: {input_source}")

        if "NVR" in str(record["source_type"] or "") and str(input_source).startswith("rtsp://"):
            try:
                base_time_str = video_name.split("_")[1]
                base_dt = datetime.strptime(base_time_str, "%Y%m%dt%H%M%Sz")
                target_dt = base_dt + timedelta(seconds=timestamp)
                start_dt = target_dt - timedelta(seconds=offset)
                hk_start = start_dt.strftime("%Y%m%dt%H%M%Sz")
                base_url = input_source.split("?")[0]
                input_source = f"{base_url}?starttime={hk_start}"
            except Exception as e:
                print(f"⚠️ NVR 时间参数解析跳过: {e}")

        if not input_source.startswith("rtsp://") and not os.path.exists(input_source):
            raise HTTPException(
                status_code=404, detail=f"视频文件在硬盘上已丢失: {input_source}"
            )

        output_filename = f"clip_{uuid.uuid4().hex[:8]}.mp4"
        output_path = os.path.join(CLIP_DIR, output_filename)

        cmd = [FFMPEG_PATH, "-y"]
        if input_source.startswith("rtsp://"):
            cmd.extend(["-rtsp_transport", "tcp"])
        ss_time = max(0, timestamp - offset)
        cmd.extend(["-ss", str(ss_time), "-i", input_source])
        cmd.extend(
            [
                "-t",
                str(offset * 2),
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                output_path,
            ]
        )

        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        return FileResponse(output_path, media_type="video/mp4")

    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode("utf-8", errors="ignore")
        print(f"❌ FFmpeg 裁切失败:\n{err_msg}")
        raise HTTPException(status_code=500, detail="视频裁切过程失败")
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 系统内部报错: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------- 建模 ----------
@app.post("/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    skip_frames: int = Form(...),
    conf_val: float = Form(...),
    captured_at: Optional[str] = Form(None),
    user: dict = Depends(get_current_user),
):
    if skip_frames < 1:
        raise HTTPException(
            status_code=400,
            detail="抽帧步长必须大于等于1",
        )

    save_name, target_path = _allocate_upload_target(file.filename)
    try:
        captured_dt = parse_user_captured_at(captured_at)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # 同名再次上传：严格复用「视频源删除」逻辑。清理失败则中断上传，避免脏数据残留。
    try:
        removed = _delete_media_source_internal(file_name=save_name, error_if_missing=False)
        if removed:
            print(f"🧹 同名上传前已清理旧任务与建模产物: {save_name}", flush=True)
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 同名视频重新建模前清理失败: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"同名视频清理失败，请先删除旧视频源后重试: {e}")

    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    task_id = uuid.uuid4().hex

    try:
        with engine.begin() as conn:
            _upsert_processed_pending_upload(
                conn,
                file_name=save_name,
                task_id=task_id,
                raw_path=target_path,
                captured_at=captured_dt,
            )
    except Exception as e:
        try:
            os.remove(target_path)
        except OSError:
            pass
        print(f"❌ videos 写入失败: {e}")
        raise HTTPException(status_code=500, detail=f"媒体源记录写入失败（任务已撤销）: {e}")

    try:
        task = process_video_task.apply_async(
            args=(target_path, save_name, skip_frames, conf_val),
            task_id=task_id,
        )
    except Exception as e:
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM videos WHERE file_name=:v AND is_media_source=1"),
                {"v": save_name},
            )
        try:
            os.remove(target_path)
        except OSError:
            pass
        raise HTTPException(status_code=500, detail=f"任务入队失败: {e}")
    return {"task_id": task.id, "file_name": save_name}


@app.post("/analyze_stream")
async def analyze_stream(
    skip_frames: int = Form(...),
    conf_val: float = Form(...),
    url: Optional[str] = Form(None),
    start_time: Optional[str] = Form(None),
    duration: int = Form(3600),
    source_mode: str = Form("isapi"),
    host: Optional[str] = Form(None),
    username: Optional[str] = Form(None),
    password: Optional[str] = Form(None),
    channels: Optional[List[int]] = Form(None),
    isapi_start: Optional[str] = Form(None),
    isapi_end: Optional[str] = Form(None),
    user: dict = Depends(get_current_user),
):
    source_mode = (source_mode or "").lower().strip()
    source_type = "NVR流"
    captured_dt = None

    if source_mode == "isapi":
        if not host or not username or not isapi_start or not isapi_end:
            raise HTTPException(
                status_code=400,
                detail="ISAPI 参数不完整（需填写设备 IP、用户名、起止时间）",
            )
        if not (password and str(password).strip()):
            raise HTTPException(status_code=400, detail="ISAPI 须填写设备登录密码")
        if not channels:
            raise HTTPException(status_code=400, detail="请至少选择一个通道")
        try:
            start_dt = datetime.fromisoformat(isapi_start)
            end_dt = datetime.fromisoformat(isapi_end)
            captured_dt = parse_user_captured_at(isapi_start)
        except ValueError:
            raise HTTPException(status_code=400, detail="开始/结束时间格式无效")
        if end_dt <= start_dt:
            raise HTTPException(status_code=400, detail="结束时间必须晚于开始时间")
        duration = int((end_dt - start_dt).total_seconds())
        source_type = "NVR_ISAPI"
    elif source_mode == "rtsp":
        if not url or "rtsp://" not in url:
            raise HTTPException(status_code=400, detail="无效 RTSP 地址")
        user_url = url.strip()
        if start_time and "starttime=" not in user_url.lower():
            sep = "&" if "?" in user_url else "?"
            user_url = f"{user_url}{sep}starttime={start_time}"
        try:
            from box_tunnel import resolve_rtsp_playback_url

            url = resolve_rtsp_playback_url(user_url)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"RTSP 隧道中继失败: {e}") from e
        try:
            captured_dt = parse_user_captured_at(start_time)
        except ValueError:
            raise HTTPException(status_code=400, detail="开始/结束时间格式无效")
    else:
        raise HTTPException(status_code=400, detail="source_mode 必须是 isapi 或 rtsp")

    if source_mode == "isapi":
        task_ids: List[str] = []
        task_names: List[str] = []
        for channel in sorted(set(int(c) for c in channels or [])):
            task_name = _auto_isapi_task_name(channel, isapi_start or "", isapi_end or "")
            task_url = (
                f"isapi://{host}"
                f"?user={quote_plus(username)}"
                f"&password={quote_plus(password)}"
                f"&channel={channel}"
                f"&start={isapi_start}"
                f"&end={isapi_end}"
            )
            tid = uuid.uuid4().hex
            try:
                with engine.begin() as conn:
                    _upsert_processed_pending_row(
                        conn,
                        file_name=task_name,
                        source_type=source_type,
                        task_id=tid,
                        raw_path=task_url,
                        captured_at=captured_dt,
                    )
            except Exception as e:
                print(f"❌ ISAPI 媒体源记录写入失败 channel={channel}: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"媒体源记录写入失败（该通道任务已撤销）: {e}",
                )
            try:
                process_video_task.apply_async(
                    args=(task_url, task_name, skip_frames, conf_val, duration),
                    task_id=tid,
                )
            except Exception as e:
                with engine.begin() as conn:
                    conn.execute(
                        text("DELETE FROM videos WHERE file_name=:v AND is_media_source=1"),
                        {"v": task_name},
                    )
                raise HTTPException(status_code=500, detail=f"任务入队失败: {e}")
            task_ids.append(tid)
            task_names.append(task_name)
        return {"task_ids": task_ids, "task_names": task_names, "count": len(task_ids)}

    tid = uuid.uuid4().hex
    task_name = start_time or "RTSP任务"
    try:
        with engine.begin() as conn:
            _upsert_processed_pending_row(
                conn,
                file_name=task_name,
                source_type=source_type,
                task_id=tid,
                raw_path=url,
                captured_at=captured_dt,
            )
    except Exception as e:
        print(f"❌ RTSP 媒体源记录写入失败: {e}")
        raise HTTPException(status_code=500, detail=f"媒体源记录写入失败（任务已撤销）: {e}")
    try:
        task = process_video_task.apply_async(
            args=(url, task_name, skip_frames, conf_val, duration),
            task_id=tid,
        )
    except Exception as e:
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM videos WHERE file_name=:v AND is_media_source=1"),
                {"v": task_name},
            )
        raise HTTPException(status_code=500, detail=f"任务入队失败: {e}")
    return {"task_id": task.id}


@app.post("/stop_task")
def stop_task(task_id: str, user: dict = Depends(get_current_user)):
    celery_app.control.revoke(task_id, terminate=True, signal="SIGTERM")
    return {"status": "terminated"}


@app.get("/status/{task_id}")
def get_status(task_id: str, user: dict = Depends(get_current_user)):
    res = celery_app.AsyncResult(task_id)
    state = res.state
    progress = (
        100
        if state == "SUCCESS"
        else (
            res.info.get("current", 0)
            if isinstance(res.info, dict) and state == "PROGRESS"
            else 0
        )
    )
    return {"state": state, "progress": progress}


# ---------- 检索 ----------
@app.post("/search")
async def search_trajectory(
    algorithm: str = Form("SIGLIP"),
    threshold: float = Form(0.85),
    time_gap: float = Form(60.0),
    group_mode: str = Form("false"),
    co_time_threshold: float = Form(2.0),
    q_text: Optional[str] = Form(None),
    include_stay_segments: str = Form("true"),
    files: Optional[List[UploadFile]] = File(None),
    user: dict = Depends(get_current_user),
):
    image_bytes: List[bytes] = []
    names: List[str] = []
    for uf in files if files else []:
        if not uf.filename:
            continue
        image_bytes.append(await uf.read())
        names.append(uf.filename)
    gm = str(group_mode).lower() in ("1", "true", "yes", "on")
    inc_stay = str(include_stay_segments).lower() in ("1", "true", "yes", "on")
    try:
        payload = perform_search(
            engine=engine,
            algorithm=algorithm,
            threshold=threshold,
            time_gap=time_gap,
            group_mode=gm,
            co_time_threshold=co_time_threshold,
            q_text=q_text,
            image_bytes_list=image_bytes,
            image_names=names,
            include_stay_segments=inc_stay,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e) or e.__class__.__name__)
    return {
        "results": payload["results"],
        "stay_segments": payload.get("stay_segments") or {},
        "algorithm": payload["algorithm"],
        "query_image_names": payload["query_image_names"],
    }


@app.post("/search/room-copresence")
async def search_room_copresence(
    algorithm: str = Form("OSNet"),
    threshold: float = Form(0.85),
    time_gap: float = Form(60.0),
    role_a_label: str = Form("患者"),
    role_b_label: str = Form("医生"),
    role_a_files: List[UploadFile] = File(...),
    role_b_files: Optional[List[UploadFile]] = File(None),
    user: dict = Depends(get_current_user),
):
    """双人房间共现：患者/医生各可多图，检索与单人轨迹一致（分图检索后按 meta_id 并集取 max）。"""
    bytes_a: List[bytes] = []
    names_a: List[str] = []
    for uf in role_a_files:
        if not uf.filename:
            continue
        raw = await uf.read()
        if raw:
            bytes_a.append(raw)
            names_a.append(uf.filename)
    if not bytes_a:
        raise HTTPException(status_code=400, detail="请至少上传一张有效的患者查询图")

    bytes_b: List[bytes] = []
    names_b: List[str] = []
    for uf in role_b_files or []:
        if not uf.filename:
            continue
        raw = await uf.read()
        if raw:
            bytes_b.append(raw)
            names_b.append(uf.filename)

    try:
        payload = perform_room_copresence(
            engine=engine,
            algorithm=algorithm,
            threshold=threshold,
            time_gap=time_gap,
            images_a_bytes=bytes_a,
            images_a_names=names_a,
            images_b_bytes=bytes_b if bytes_b else None,
            images_b_names=names_b if names_b else None,
            role_a_label=role_a_label,
            role_b_label=role_b_label,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e) or e.__class__.__name__)
    return payload


class SaveSearchLogBody(BaseModel):
    query_image_paths: str = ""
    query_image_paths_b: str = ""
    search_query: Optional[str] = None
    algorithm: str
    threshold: float
    results: dict = {}
    stay_segments: Optional[dict] = None
    time_gap: Optional[float] = None
    log_kind: str = "single"
    copresence: Optional[dict] = None


def _serialize_search_log_results(body: SaveSearchLogBody) -> str:
    """v3 双人共现；v2 单人 stay_segments；旧版仅扁平 results。"""
    if (body.log_kind or "").strip() == "room_copresence" and body.copresence is not None:
        videos = body.copresence.get("videos", body.copresence)
        return json.dumps(
            {
                "version": 3,
                "kind": "room_copresence",
                "algorithm": body.algorithm,
                "threshold": float(body.threshold),
                "time_gap": float(body.time_gap) if body.time_gap is not None else 60.0,
                "query_image_a": body.query_image_paths or "",
                "query_image_b": body.query_image_paths_b or "",
                "videos": videos,
            },
            ensure_ascii=False,
        )
    if body.stay_segments is not None:
        return json.dumps(
            {
                "version": 2,
                "kind": "single",
                "results": body.results,
                "stay_segments": body.stay_segments,
                "time_gap": float(body.time_gap) if body.time_gap is not None else 60.0,
            },
            ensure_ascii=False,
        )
    return json.dumps(body.results, ensure_ascii=False)


@app.post("/search/logs")
def save_search_log(body: SaveSearchLogBody, user: dict = Depends(get_current_user)):
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO search_logs
                (query_image_paths, search_query, algorithm, threshold, results_json, created_at)
                VALUES (:imgs, :q, :algo, :th, :res, NOW())
            """
            ),
            {
                "imgs": body.query_image_paths,
                "q": body.search_query or "",
                "algo": body.algorithm,
                "th": body.threshold,
                "res": _serialize_search_log_results(body),
            },
        )
    return {"ok": True}


@app.get("/search/logs")
def list_search_logs(user: dict = Depends(get_current_user)):
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT * FROM search_logs ORDER BY created_at DESC")
        ).mappings().all()
    out = []
    for r in rows:
        d = dict(r)
        if d.get("created_at") is not None:
            d["created_at"] = serialize_datetime(d["created_at"])
        out.append(d)
    return out


def _reset_search_logs_auto_increment(conn) -> None:
    """
    删除后把自增起点设为 max(id)+1，表空时下一插入从 1 开始，避免删光 id=1 后新记录变成 id=2。
    """
    row = conn.execute(
        text("SELECT COALESCE(MAX(id), 0) AS m FROM search_logs")
    ).mappings().first()
    next_val = int(row["m"] if row else 0) + 1
    conn.execute(text(f"ALTER TABLE search_logs AUTO_INCREMENT = {next_val}"))


@app.delete("/search/logs/{log_id}")
def delete_search_log(log_id: int, user: dict = Depends(require_admin)):
    with engine.begin() as conn:
        res = conn.execute(text("DELETE FROM search_logs WHERE id=:id"), {"id": log_id})
        if res.rowcount == 0:
            raise HTTPException(status_code=404, detail="日志不存在")
        _reset_search_logs_auto_increment(conn)
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    # 关闭 access log，避免大量 /files/crop 请求刷屏掩盖关键运行信息
    uvicorn.run(app, host="0.0.0.0", port=8001, access_log=False)
