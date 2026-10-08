"""健康检查、模型运行状态与Celery诊断接口。"""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends

from services.auth.dependencies import get_current_user
from services.inference.runtime import get_runtime_backend_info
from services.tasks.celery_app import app as celery_app

router = APIRouter(tags=["system"])


@router.get("/health")
def health():
    """轻量存活检查，不主动连接数据库或加载模型。"""
    return {"ok": True}


@router.get("/runtime/backends")
def runtime_backends(user: dict = Depends(get_current_user)):
    del user
    return get_runtime_backend_info()


@router.get("/celery/inspect")
def celery_inspect(user: dict = Depends(get_current_user)):
    """返回Worker、执行中任务和Redis默认队列长度。"""
    del user
    broker = str(celery_app.conf.broker_url or "")
    inspector = celery_app.control.inspect(timeout=3.0)
    if not inspector:
        return {
            "ok": False,
            "broker": broker,
            "hint": "没有Worker响应，请检查Celery与API是否使用同一个Redis地址。",
        }
    result = {
        "ok": True,
        "broker": broker,
        "active": inspector.active() or {},
        "reserved": inspector.reserved() or {},
        "scheduled": inspector.scheduled() or {},
        "stats": inspector.stats() or {},
    }
    try:
        import redis

        client = redis.Redis.from_url(
            os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
        )
        result["redis_llen_celery_queue"] = int(client.llen("celery"))
    except Exception as exc:
        result["redis_llen_celery_queue"] = None
        result["redis_queue_note"] = str(exc)
    return result
