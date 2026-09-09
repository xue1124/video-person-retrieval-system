"""
RTSP 实时拉流：单路 ffmpeg 拉流，split 后一路录 fMP4、一路 rawvideo 管道供 YOLO + SigLIP。
终止（stopped）保留录像与检索数据；删除时由调用方 request_stop 再 purge。
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from sqlalchemy import text

from video_playback import schedule_finalize_archive
from tasks import (
    GALLERY_DIR,
    VIDEO_DIR,
    _ensure_sqlalchemy_engine,
    _flush_buffer,
    _set_task_failed,
    live_yolo_predict,
)
import db as app_db
from video_time import utc_now_naive

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

FFMPEG_PATH = os.environ.get("FFMPEG_PATH") or shutil.which("ffmpeg") or "ffmpeg"
FFPROBE_PATH = os.environ.get("FFPROBE_PATH") or shutil.which("ffprobe") or "ffprobe"

SOURCE_TYPE_RTSP_LIVE = "RTSP实时"
STATUS_STREAMING = "streaming"
STATUS_STOPPED = "stopped"

_lock = threading.Lock()
_threads: dict[str, threading.Thread] = {}
_stop_events: dict[str, threading.Event] = {}
_recorders: dict[str, subprocess.Popen] = {}
_recorders_lock = threading.Lock()

_DB_TICK_SEC = 2.0
_FFMPEG_MOVFLAGS = "frag_keyframe+empty_moov+default_base_moof"
_FRAME_QUEUE_SIZE = 8
_PIPE_READ_FAIL_LIMIT = 5
_QUEUE_TIMEOUT = object()


def live_archive_path(file_name: str) -> Path:
    base = os.path.basename(str(file_name).replace("\\", "/"))
    if not base:
        raise ValueError("非法 file_name")
    if not base.lower().endswith(".mp4"):
        base = f"{base}.mp4"
    return VIDEO_DIR / base


def is_live_thread_running(file_name: str) -> bool:
    with _lock:
        t = _threads.get(file_name)
        return t is not None and t.is_alive()


def active_live_thread_count() -> int:
    with _lock:
        return sum(1 for t in _threads.values() if t.is_alive())


def _db_status(engine, file_name: str) -> Optional[str]:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT status FROM videos WHERE file_name=:v AND is_media_source=1 LIMIT 1"),
            {"v": file_name},
        ).first()
    return str(row[0]) if row else None


def _stop_ffmpeg_recorder(file_name: str) -> None:
    with _recorders_lock:
        proc = _recorders.pop(file_name, None)
    if proc is None:
        return
    if proc.poll() is None:
        try:
            if proc.stdin:
                proc.stdin.write(b"q")
                proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=3)
            except Exception:
                pass


def _parse_frame_rate(rate_str: str) -> float:
    rate_str = (rate_str or "").strip()
    if not rate_str or rate_str == "0/0":
        return 25.0
    if "/" in rate_str:
        num, den = rate_str.split("/", 1)
        try:
            n, d = float(num), float(den)
            if d > 0:
                return n / d
        except ValueError:
            pass
    try:
        return float(rate_str)
    except ValueError:
        return 25.0


def _normalize_fps(fps: float) -> float:
    if fps <= 0 or fps > 120:
        return 25.0
    return fps


def _probe_rtsp_video(rtsp_url: str, *, timeout: float = 15.0) -> tuple[int, int, float]:
    cmd = [
        FFPROBE_PATH,
        "-v",
        "error",
        "-rtsp_transport",
        "tcp",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-of",
        "json",
        rtsp_url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "ffprobe 失败").strip()[:500]
        raise RuntimeError(detail)
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError("ffprobe 未返回视频流")
    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    fps = _normalize_fps(_parse_frame_rate(str(stream.get("r_frame_rate") or "")))
    if width <= 0 or height <= 0:
        raise RuntimeError(f"无效分辨率: {width}x{height}")
    return width, height, fps


def _probe_rtsp_video_with_retry(
    rtsp_url: str,
    *,
    attempts: int = 6,
    delay: float = 2.0,
) -> tuple[int, int, float]:
    last_err: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            return _probe_rtsp_video(rtsp_url)
        except Exception as e:
            last_err = e
            if attempt < attempts - 1:
                time.sleep(delay)
    raise RuntimeError(f"无法探测 RTSP 流: {last_err}")


def _read_frame_bytes(proc: subprocess.Popen, frame_bytes: int) -> Optional[bytes]:
    if proc.poll() is not None:
        return None
    buf = bytearray()
    while len(buf) < frame_bytes:
        chunk = proc.stdout.read(frame_bytes - len(buf))  # type: ignore[union-attr]
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


class _PipeFrameReader:
    """后台持续读 ffmpeg 管道，避免 YOLO 推理阻塞导致 ffmpeg 死锁。"""

    def __init__(
        self,
        proc: subprocess.Popen,
        frame_bytes: int,
        stop_event: threading.Event,
    ) -> None:
        self._proc = proc
        self._frame_bytes = frame_bytes
        self._stop_event = stop_event
        self._queue: queue.Queue[Optional[bytes]] = queue.Queue(maxsize=_FRAME_QUEUE_SIZE)
        self._thread = threading.Thread(target=self._run, name="rtsp-pipe-reader", daemon=True)
        self._read_failures = 0

    def start(self) -> None:
        self._thread.start()

    def get(self, timeout: float = 0.5):
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return _QUEUE_TIMEOUT

    def _run(self) -> None:
        while not self._stop_event.is_set():
            raw = _read_frame_bytes(self._proc, self._frame_bytes)
            if raw is None:
                self._read_failures += 1
                if self._read_failures >= _PIPE_READ_FAIL_LIMIT:
                    try:
                        self._queue.put_nowait(None)
                    except queue.Full:
                        pass
                    break
                time.sleep(0.05)
                continue
            self._read_failures = 0
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._queue.put_nowait(raw)
            except queue.Full:
                pass


def _start_ffmpeg_pipeline(
    file_name: str,
    rtsp_url: str,
) -> tuple[subprocess.Popen, int, int, float, int]:
    out_path = live_archive_path(file_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        try:
            out_path.unlink()
        except OSError:
            pass

    width, height, fps = _probe_rtsp_video_with_retry(rtsp_url)
    frame_bytes = width * height * 3
    cmd = [
        FFMPEG_PATH,
        "-hide_banner",
        "-loglevel",
        "error",
        "-rtsp_transport",
        "tcp",
        "-i",
        rtsp_url,
        "-an",
        "-filter_complex",
        "[0:v]split=2[v1][v2]",
        "-map",
        "[v1]",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        _FFMPEG_MOVFLAGS,
        "-y",
        str(out_path),
        "-map",
        "[v2]",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(0.8)
    if proc.poll() is not None:
        err = ""
        try:
            err = (proc.stderr.read() or b"").decode("utf-8", errors="ignore")[:500]
        except Exception:
            pass
        raise RuntimeError(f"ffmpeg 启动失败: {err or '进程已退出'}")
    with _recorders_lock:
        _recorders[file_name] = proc
    return proc, width, height, fps, frame_bytes


def request_stop(file_name: str) -> None:
    """终止：通知分析线程退出、停录制、DB 标 stopped。"""
    with _lock:
        ev = _stop_events.get(file_name)
        if ev:
            ev.set()
    _stop_ffmpeg_recorder(file_name)

    db_engine = _ensure_sqlalchemy_engine()
    with db_engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE videos SET status=:st, completed_at=NOW(), "
                "duration='已停止' "
                "WHERE file_name=:v AND is_media_source=1 AND status IN ('streaming', 'pending')"
            ),
            {"v": file_name, "st": STATUS_STOPPED},
        )


def request_stop_and_wait(file_name: str, timeout: float = 8.0) -> None:
    request_stop(file_name)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not is_live_thread_running(file_name):
            return
        time.sleep(0.1)
    _stop_ffmpeg_recorder(file_name)


def stamp_live_captured_at(
    db_engine,
    file_name: str,
    *,
    now: Optional[datetime] = None,
) -> None:
    """直播真正开始拉流后写入 captured_at；已有值不覆盖，也不用文件时间猜测。"""
    captured = now or utc_now_naive()
    if captured.tzinfo is not None:
        captured = captured.astimezone(timezone.utc).replace(tzinfo=None)
    with db_engine.begin() as conn:
        app_db.stamp_captured_at_if_missing(conn, file_name, captured)


def _mark_live_phase(
    db_engine,
    file_name: str,
    archive_path: Path,
    *,
    duration: str,
) -> None:
    with db_engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE videos SET status=:st, duration=:d, source_path=:rp, "
                "progress=0, processing_started_at=COALESCE(processing_started_at, NOW(3)) "
                "WHERE file_name=:v AND is_media_source=1"
            ),
            {
                "v": file_name,
                "st": STATUS_STREAMING,
                "d": duration,
                "rp": str(archive_path),
            },
        )


def _run_stream(
    file_name: str,
    rtsp_url: str,
    skip_frames: int,
    conf_val: float,
    stop_event: threading.Event,
) -> None:
    proc: Optional[subprocess.Popen] = None
    reader: Optional[_PipeFrameReader] = None
    db_engine = _ensure_sqlalchemy_engine()
    archive_path = live_archive_path(file_name)
    user_stopped = False
    try:
        if stop_event.is_set():
            return

        skip_frames = max(1, int(skip_frames))
        conf_val = float(conf_val)

        frame_idx = 0
        count = 0
        crop_buffer = []
        full_frame_buffer = []
        meta_buffer = []
        BATCH_SIZE = 64
        fail_streak = 0
        last_db_tick = 0.0
        width = 0
        height = 0
        fps = 25.0

        def _should_continue() -> bool:
            return _should_continue_impl(stop_event, db_engine, file_name)

        def _update_db(**fields) -> None:
            if stop_event.is_set() and fields.get("status") == STATUS_STREAMING:
                return
            allowed = {
                "status",
                "fps",
                "duration",
                "progress",
                "target_count",
                "failure_reason",
                "source_path",
            }
            sets = []
            params = {"v": file_name}
            for k, val in fields.items():
                if k == "raw_path":
                    k = "source_path"
                    val = fields.get("raw_path")
                if k not in allowed:
                    continue
                sets.append(f"{k}=:{k}")
                params[k] = val
            if not sets:
                return
            sql = (
                f"UPDATE videos SET {', '.join(sets)} "
                "WHERE file_name=:v AND is_media_source=1"
            )
            with db_engine.begin() as conn:
                conn.execute(text(sql), params)

        def _maybe_tick_db() -> None:
            nonlocal last_db_tick
            now = time.time()
            if now - last_db_tick < _DB_TICK_SEC:
                return
            last_db_tick = now
            _update_db(target_count=count)

        _mark_live_phase(db_engine, file_name, archive_path, duration="启动中…")
        try:
            proc, width, height, fps, frame_bytes = _start_ffmpeg_pipeline(file_name, rtsp_url)
        except Exception as e:
            raise RuntimeError(f"启动拉流失败: {e}") from e
        print(f"[RTSP_LIVE] 单路拉流已开始 | {file_name} | {archive_path}", flush=True)
        stamp_live_captured_at(db_engine, file_name)

        reader = _PipeFrameReader(proc, frame_bytes, stop_event)
        reader.start()

        if stop_event.is_set() or not _should_continue():
            return

        _update_db(
            fps=round(fps, 1),
            duration="直播中",
            target_count=0,
            raw_path=str(archive_path),
        )
        print(
            f"[RTSP_LIVE] 分析已开始 | {file_name} | skip={skip_frames} | {width}x{height}@{fps:.1f}fps",
            flush=True,
        )

        while _should_continue():
            raw = reader.get(timeout=0.5)
            if raw is _QUEUE_TIMEOUT:
                fail_streak += 1
                if fail_streak > 300:
                    raise RuntimeError("RTSP 读帧连续失败，连接可能已断开")
                if not _should_continue():
                    user_stopped = True
                    break
                continue
            if raw is None:
                if proc is not None and proc.poll() is not None:
                    err = ""
                    try:
                        err = (proc.stderr.read() or b"").decode("utf-8", errors="ignore")[:300]
                    except Exception:
                        pass
                    raise RuntimeError(f"RTSP 管道已结束{(': ' + err) if err else ''}")
                fail_streak += 1
                if fail_streak > 300:
                    raise RuntimeError("RTSP 读帧连续失败，连接可能已断开")
                if not _should_continue():
                    user_stopped = True
                    break
                continue
            fail_streak = 0

            if frame_idx % skip_frames != 0:
                frame_idx += 1
                _maybe_tick_db()
                continue

            if not _should_continue():
                user_stopped = True
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            ts = frame_idx / fps
            res = live_yolo_predict(frame, conf_val)
            if not _should_continue():
                user_stopped = True
                break

            boxes = res[0].boxes
            if len(boxes) > 0:
                h, w, _ = frame.shape
                current_full = frame.copy()
                for b_idx, box in enumerate(boxes):
                    if not _should_continue():
                        user_stopped = True
                        break
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    p = GALLERY_DIR / f"{file_name}_{frame_idx}_{b_idx}.jpg"
                    crop_buffer.append(crop)
                    full_frame_buffer.append(current_full)
                    meta_buffer.append(
                        {
                            "v": file_name,
                            "t": ts,
                            "p": str(p),
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        }
                    )
                    if len(crop_buffer) >= BATCH_SIZE:
                        if not _should_continue():
                            user_stopped = True
                            break
                        count += _flush_buffer(
                            crop_buffer, full_frame_buffer, meta_buffer, db_engine
                        )
                        crop_buffer.clear()
                        full_frame_buffer.clear()
                        meta_buffer.clear()
                        _update_db(target_count=count)

            if user_stopped:
                break

            frame_idx += 1
            _maybe_tick_db()

        if crop_buffer:
            count += _flush_buffer(
                crop_buffer, full_frame_buffer, meta_buffer, db_engine
            )
            _update_db(target_count=count)

        st = _db_status(db_engine, file_name)
        if st == STATUS_STREAMING:
            _update_db(
                status=STATUS_STOPPED,
                duration="已停止",
                progress=100,
                target_count=count,
            )
            with db_engine.begin() as conn:
                conn.execute(
                    text(
                        "UPDATE videos SET completed_at=NOW(3) "
                        "WHERE file_name=:v AND is_media_source=1 AND completed_at IS NULL"
                    ),
                    {"v": file_name},
                )
        print(f"[RTSP_LIVE] 已结束 | {file_name} | count={count}", flush=True)

    except Exception as e:
        stopped = bool(
            stop_event.is_set()
            or (db_engine is not None and _db_status(db_engine, file_name) == STATUS_STOPPED)
        )
        if stopped or user_stopped:
            print(f"[RTSP_LIVE] 用户终止 | {file_name}", flush=True)
            if str(e).strip():
                print(
                    f"[RTSP_LIVE] 终止细节 | {file_name} | {type(e).__name__}: {e}",
                    flush=True,
                )
        elif db_engine is not None:
            print(f"[RTSP_LIVE] 失败 | {file_name} | {e}", flush=True)
            if _db_status(db_engine, file_name) is not None:
                _set_task_failed(db_engine, file_name, str(e))
        else:
            print(f"[RTSP_LIVE] 失败 | {file_name} | {e}", flush=True)
    finally:
        stop_event.set()
        _stop_ffmpeg_recorder(file_name)
        archive = live_archive_path(file_name)
        try:
            if archive.is_file() and archive.stat().st_size > 32768:
                schedule_finalize_archive(str(archive))
        except OSError:
            pass
        with _lock:
            _threads.pop(file_name, None)
            _stop_events.pop(file_name, None)


def _should_continue_impl(
    stop_event: threading.Event,
    db_engine,
    file_name: str,
) -> bool:
    if stop_event.is_set():
        return False
    st = _db_status(db_engine, file_name)
    if st is None:
        return False
    return st == STATUS_STREAMING


def start_live_stream(
    *,
    engine,
    file_name: str,
    rtsp_url: str,
    skip_frames: int,
    conf_val: float,
    task_id: str,
) -> None:
    url = (rtsp_url or "").strip()
    if not url.lower().startswith("rtsp://"):
        raise ValueError("RTSP 地址必须以 rtsp:// 开头")

    with _lock:
        if file_name in _threads and _threads[file_name].is_alive():
            raise RuntimeError("该任务已在直播中")

    stop_event = threading.Event()
    t = threading.Thread(
        target=_run_stream,
        args=(file_name, url, skip_frames, conf_val, stop_event),
        name=f"rtsp-live-{file_name[:32]}",
        daemon=True,
    )
    with _lock:
        _stop_events[file_name] = stop_event
        _threads[file_name] = t
    t.start()


def shutdown_all() -> None:
    with _lock:
        names = list(_threads.keys())
    for name in names:
        request_stop(name)
    for name in names:
        request_stop_and_wait(name, timeout=3.0)
