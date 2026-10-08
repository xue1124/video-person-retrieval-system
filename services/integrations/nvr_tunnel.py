"""
AI 盒子 frp 隧道：ISAPI 经 isapi_proxy（8080 + X-Target-NVR），RTSP 经 rtsp_relay（18090/18554）。
云端填真实 NVR 地址；本模块负责转为隧道入口。
"""
from __future__ import annotations

import os
import time
from typing import Dict, Optional
import requests

ISAPI_PROXY_BASE = os.environ.get("ISAPI_PROXY_BASE", "http://127.0.0.1:8080").rstrip("/")
RTSP_RELAY_CTRL = os.environ.get("RTSP_RELAY_CTRL", "http://127.0.0.1:18090").rstrip("/")
RTSP_RELAY_PLAY_URL = os.environ.get("RTSP_RELAY_PLAY_URL", "rtsp://127.0.0.1:18554/relay").strip()
RTSP_RELAY_WARMUP_SEC = float(os.environ.get("RTSP_RELAY_WARMUP_SEC", "3"))


def normalize_nvr_host(host: str) -> str:
    """前端设备 IP，如 192.168.1.3 或 192.168.1.3:8080。"""
    h = (host or "").strip()
    if not h:
        raise ValueError("设备 IP 不能为空")
    return h


def isapi_proxy_headers(target_nvr: str) -> Dict[str, str]:
    return {"X-Target-NVR": normalize_nvr_host(target_nvr)}


def isapi_proxy_url(path: str) -> str:
    if not path.startswith("/"):
        path = "/" + path
    return f"{ISAPI_PROXY_BASE}{path}"


def _is_relay_play_url(rtsp_url: str) -> bool:
    u = (rtsp_url or "").strip().lower()
    if not u.startswith("rtsp://"):
        return False
    play = RTSP_RELAY_PLAY_URL.lower()
    return u == play or u.rstrip("/") == play.rstrip("/")


def start_rtsp_relay(source_rtsp_url: str, *, timeout: float = 30.0) -> None:
    """通知盒子 rtsp_relay 从 source_rtsp_url 拉流并推到 mediamtx /relay。"""
    url = (source_rtsp_url or "").strip()
    if not url.lower().startswith("rtsp://"):
        raise ValueError("RTSP 地址必须以 rtsp:// 开头")
    if _is_relay_play_url(url):
        return
    ctrl = f"{RTSP_RELAY_CTRL}/relay/start"
    resp = requests.post(
        ctrl,
        json={"rtsp_url": url},
        timeout=timeout,
    )
    if resp.status_code >= 400:
        detail = resp.text[:500] if resp.text else resp.reason
        raise RuntimeError(f"RTSP 中继启动失败 ({resp.status_code}): {detail}")
    time.sleep(RTSP_RELAY_WARMUP_SEC)


def stop_rtsp_relay(*, timeout: float = 10.0) -> None:
    try:
        requests.post(f"{RTSP_RELAY_CTRL}/relay/stop", timeout=timeout)
    except Exception:
        pass


def resolve_rtsp_playback_url(user_rtsp_url: str) -> str:
    """
    用户填真实 rtsp://NVR... → 启动盒子中继 → 返回固定隧道出口 rtsp://127.0.0.1:18554/relay。
    若已是中继出口地址则原样返回。
    """
    url = (user_rtsp_url or "").strip()
    if not url.lower().startswith("rtsp://"):
        raise ValueError("RTSP 地址必须以 rtsp:// 开头")
    if _is_relay_play_url(url):
        return url
    start_rtsp_relay(url)
    return RTSP_RELAY_PLAY_URL
