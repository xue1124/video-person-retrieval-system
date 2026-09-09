"""归档视频可播化：任务完成后就地 faststart / 转码，覆盖 video_archives 原文件。"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from typing import Callable, List, Optional

FFMPEG_PATH = os.environ.get("FFMPEG_PATH") or shutil.which("ffmpeg") or "ffmpeg"


def _ffprobe_bin() -> str:
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
    return shutil.which("ffprobe") or "ffprobe"


def probe_video_codec(path: str) -> Optional[str]:
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


def is_browser_friendly_codec(codec: Optional[str]) -> bool:
    return (codec or "") in ("h264", "avc1", "avc3")


def is_browser_friendly_mp4(path: str) -> bool:
    return is_browser_friendly_codec(probe_video_codec(path))


def playback_revision(source_path: str) -> str:
    """直播边录边播：文件变大时 revision 变化，供前端按需刷新 URL。"""
    try:
        st = os.stat(source_path)
        return f"{st.st_size}-{int(st.st_mtime)}"
    except OSError:
        return "0"


def finalize_lock_path(archive_path: str) -> str:
    return os.path.normpath(archive_path) + ".finalize.lock"


def finalize_part_path(archive_path: str) -> str:
    return os.path.normpath(archive_path) + ".part.mp4"


def is_finalize_in_progress(archive_path: str) -> bool:
    return os.path.isfile(finalize_lock_path(archive_path))


def _clear_stale_lock(lock_path: str, max_age_sec: int = 1800) -> None:
    if not os.path.isfile(lock_path):
        return
    try:
        if time.time() - os.path.getmtime(lock_path) > max_age_sec:
            os.remove(lock_path)
    except OSError:
        pass


def _has_audio_stream(path: str) -> bool:
    try:
        out = subprocess.run(
            [
                _ffprobe_bin(),
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "default=nw=1:nk=1",
                path,
            ],
            capture_output=True,
            timeout=30,
            check=False,
        )
        return out.returncode == 0 and bool((out.stdout or b"").strip())
    except Exception:
        return False


def _ffmpeg_transcode_cmd(source_path: str, output_target: str, *, use_gpu: bool) -> List[str]:
    cmd: List[str] = [
        FFMPEG_PATH,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        source_path,
        "-map",
        "0:v:0",
    ]
    if use_gpu:
        cmd[1:1] = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    if _has_audio_stream(source_path):
        cmd.extend(["-map", "0:a:0", "-c:a", "aac", "-b:a", "128k"])
    else:
        cmd.append("-an")
    if use_gpu:
        cmd.extend(
            [
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p4",
                "-cq",
                "23",
                "-rc",
                "vbr",
                "-b:v",
                "0",
            ]
        )
    else:
        cmd.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"])
    cmd.extend(["-movflags", "+faststart", "-pix_fmt", "yuv420p"])
    cmd.append(output_target)
    return cmd


def _remux_faststart_cmd(source_path: str, output_target: str) -> List[str]:
    cmd = [
        FFMPEG_PATH,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        source_path,
        "-map",
        "0:v:0",
        "-c:v",
        "copy",
    ]
    if _has_audio_stream(source_path):
        cmd.extend(["-map", "0:a:0", "-c:a", "aac", "-b:a", "128k"])
    else:
        cmd.append("-an")
    cmd.extend(["-movflags", "+faststart", output_target])
    return cmd


def finalize_archive_for_playback(archive_path: str, *, timeout: int = 3600) -> str:
    """
    将 video_archives 归档就地处理为浏览器友好 mp4（H.264 + faststart），覆盖原文件。
    H.264：流复制 + faststart；其它编码：转 H.264。
    """
    src = os.path.normpath(archive_path)
    if not os.path.isfile(src) or os.path.getsize(src) <= 0:
        raise FileNotFoundError("视频归档不存在或为空")

    lock_path = finalize_lock_path(src)
    part_path = finalize_part_path(src)
    _clear_stale_lock(lock_path)

    if os.path.isfile(lock_path):
        deadline = time.time() + min(timeout, 120)
        while time.time() < deadline and os.path.isfile(lock_path):
            time.sleep(0.25)
        if os.path.isfile(lock_path):
            raise RuntimeError("归档可播化正在进行中，请稍后重试")

    os.makedirs(os.path.dirname(src) or ".", exist_ok=True)
    try:
        with open(lock_path, "w", encoding="utf-8") as lf:
            lf.write(str(os.getpid()))
    except OSError:
        pass

    codec = probe_video_codec(src)
    use_remux = is_browser_friendly_codec(codec)
    mode = "faststart" if use_remux else "transcode"
    print(f"[PLAYBACK] 开始归档可播化 ({mode}): {src}", flush=True)

    try:
        if os.path.isfile(part_path):
            os.remove(part_path)

        if use_remux:
            subprocess.run(
                _remux_faststart_cmd(src, part_path),
                check=True,
                capture_output=True,
                timeout=timeout,
            )
        else:
            prefer_gpu = str(os.environ.get("VIDEO_TRANSCODE_USE_GPU", "1")).lower() not in (
                "0",
                "false",
                "no",
                "off",
            )
            if prefer_gpu:
                try:
                    subprocess.run(
                        _ffmpeg_transcode_cmd(src, part_path, use_gpu=True),
                        check=True,
                        capture_output=True,
                        timeout=timeout,
                    )
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    if os.path.isfile(part_path):
                        os.remove(part_path)
            if not os.path.isfile(part_path) or os.path.getsize(part_path) <= 0:
                subprocess.run(
                    _ffmpeg_transcode_cmd(src, part_path, use_gpu=False),
                    check=True,
                    capture_output=True,
                    timeout=timeout,
                )

        if not os.path.isfile(part_path) or os.path.getsize(part_path) <= 0:
            raise RuntimeError("归档可播化输出为空")

        os.replace(part_path, src)
        print(f"[PLAYBACK] 归档可播化完成 ({mode}): {src}", flush=True)
        return src
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"").decode("utf-8", errors="ignore")[-800:]
        raise RuntimeError(f"归档可播化失败: {err}") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("归档可播化超时") from e
    finally:
        try:
            if os.path.isfile(lock_path):
                os.remove(lock_path)
        except OSError:
            pass
        try:
            if os.path.isfile(part_path):
                os.remove(part_path)
        except OSError:
            pass


def schedule_finalize_archive(
    archive_path: str,
    *,
    on_done: Optional[Callable[[bool, Optional[str]], None]] = None,
) -> None:
    """后台就地可播化（用于 RTSP 终止后不阻塞线程退出）。"""

    def _run() -> None:
        err_msg: Optional[str] = None
        ok = False
        try:
            finalize_archive_for_playback(archive_path)
            ok = True
        except Exception as e:
            err_msg = str(e)
            print(f"[PLAYBACK] 后台归档可播化失败: {archive_path} | {e}", flush=True)
        if on_done:
            try:
                on_done(ok, err_msg)
            except Exception as cb_err:
                print(f"[PLAYBACK] on_done 回调失败: {archive_path} | {cb_err}", flush=True)

    threading.Thread(
        target=_run,
        daemon=True,
        name="playback-finalize-archive",
    ).start()


def cleanup_finalize_sidecars(archive_path: str) -> None:
    """删除遗留的 lock / part 及旧版 *_web.mp4。"""
    src = os.path.normpath(archive_path)
    for p in (finalize_lock_path(src), finalize_part_path(src)):
        try:
            if os.path.isfile(p):
                os.remove(p)
        except OSError:
            pass
    stem, _ext = os.path.splitext(src)
    legacy_web = f"{stem}_web.mp4"
    for p in (legacy_web, legacy_web + ".lock", legacy_web + ".part.mp4"):
        try:
            if os.path.isfile(p):
                os.remove(p)
        except OSError:
            pass
