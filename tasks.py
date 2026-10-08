"""Celery视频分析任务入口。

FastAPI只负责登记任务并把消息放入Redis；Celery Worker加载本文件后执行
``process_video_task``。模型推理由 ``services.inference`` 提供，人物分析主流程
由 ``services.analysis.tracking.workflow`` 编排，本文件只保留任务状态和视频来源准备。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import requests
from requests import exceptions as requests_exc
from requests.auth import HTTPDigestAuth
from sqlalchemy import text

from services.config import TEMP_DIR, VIDEO_DIR, ensure_runtime_directories
from services.media.time_utils import db_datetime_to_utc_iso, parse_user_captured_at
from services.persistence import database as app_db
from services.persistence.cleanup import purge_video_modeling_artifacts
from services.tasks.celery_app import app
from services.tasks.support import (
    get_task_engine as _ensure_sqlalchemy_engine,
    mark_task_failed as _set_task_failed,
)

BASE_DIR = Path(__file__).resolve().parent
ensure_runtime_directories()


def _redact_source_url(value: str) -> str:
    """隐藏任务源地址中的密码，避免凭据进入 worker 日志。"""
    redacted = re.sub(r"([?&]password=)[^&]*", r"\1***", str(value), flags=re.I)
    return re.sub(
        r"(rtsp://[^:/@\s]+:)[^@/\s]+@",
        r"\1***@",
        redacted,
        flags=re.I,
    )


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
    from services.media.playback import schedule_finalize_archive

    schedule_finalize_archive(
        archive_path,
        on_done=lambda ok, err: _on_finalize_done(video_name, archive_path, ok, err),
    )


def _playback_finalize_enabled() -> bool:
    """归档可播化（faststart/转码）。默认关：设 VIDEO_PLAYBACK_FINALIZE=1 可恢复。"""
    raw = os.environ.get("VIDEO_PLAYBACK_FINALIZE", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


# 模型加载与特征提取已集中到 services/inference；任务入口不再承载推理实现。

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

    from services.integrations.nvr_tunnel import isapi_proxy_headers, isapi_proxy_url

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

    # 1. 取得数据库连接，并确认这仍是当前有效任务。
    db_engine = _ensure_sqlalchemy_engine()

    if not _media_source_row_exists(db_engine, video_name):
        print(
            f"[WORKER] 媒体源「{video_name}」在库中已不存在，跳过（可能已被用户删除）",
            flush=True,
        )
        return {"status": "CANCELLED", "reason": "deleted"}
    if not _task_still_current(db_engine, video_name, self.request.id):
        print(
            f"[WORKER] 任务已被终止或替换，跳过: {video_name}",
            flush=True,
        )
        return {"status": "CANCELLED", "reason": "stopped_or_superseded"}

    # 支持两类网络源：历史 RTSP 与 ISAPI 回放；均先下载为本地文件再处理
    temp_download_file = None
    is_isapi_source = raw_path.startswith("isapi://")
    download_weight = 35 if is_isapi_source else 0
    if raw_path.startswith("rtsp://") and "starttime=" in raw_path:
        try:
            from services.integrations.nvr_tunnel import resolve_rtsp_playback_url

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

    from services.analysis.tracking import analysis_enabled, run_video_analysis

    if not analysis_enabled():
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
            analysis_result = run_video_analysis(
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
            # 用户删除或替换任务时，分析流程主动抛出的取消异常。
            if track_exc.__class__.__name__ == "TrackingCancelled":
                print(
                    f"[WORKER] 跟踪已取消: {video_name} | {track_exc}",
                    flush=True,
                )
                return {"status": "CANCELLED", "reason": "tracking_cancelled"}
            raise
        if isinstance(analysis_result, dict) and analysis_result.get("skipped"):
            return {
                "status": "CANCELLED",
                "reason": str(analysis_result.get("skipped")),
            }
        if not _task_still_current(db_engine, video_name, self.request.id):
            print(
                f"[WORKER] 跟踪入库后任务已过期，不写回人数/完成态: {video_name}",
                flush=True,
            )
            return {"status": "CANCELLED", "reason": "superseded_after_track"}
        if isinstance(analysis_result, dict):
            person_count = int(
                (analysis_result.get("import") or {}).get("person_count") or 0
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
    except Exception as analysis_error:
        if not _task_still_current(db_engine, video_name, self.request.id):
            return {"status": "CANCELLED", "reason": "superseded"}
        reason = f"OSNet 跟踪入库失败: {analysis_error}"
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

    return {"status": "SUCCESS", "count": person_count, "mode": "analysis_pipeline"}
