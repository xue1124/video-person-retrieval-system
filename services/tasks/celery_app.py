"""Celery应用与Worker生命周期日志。

根目录 ``tasks.py`` 负责定义视频任务；本模块只负责连接 Redis、创建 Celery
应用以及记录Worker事件。API查询任务状态时也复用同一个 ``app`` 对象。
"""

from __future__ import annotations

import os

from celery import Celery
from celery.signals import task_failure, task_postrun, task_prerun, worker_ready

from services.persistence import database as app_db

app_db.load_siglip_env()
BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", BROKER_URL)

app = Celery("reid_tasks", broker=BROKER_URL, backend=RESULT_BACKEND)


def log_worker(message: str) -> None:
    print(f"[WORKER][pid={os.getpid()}] {message}", flush=True)


@worker_ready.connect
def on_worker_ready(sender=None, **kwargs) -> None:
    del sender, kwargs
    log_worker(f"Celery worker就绪 | broker={app.conf.broker_url}")
    if os.environ.get("TRACKING_V3_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        try:
            from services.analysis.tracking.engines import warm_tracking_engines

            info = warm_tracking_engines()
            log_worker(f"人物分析模型预热完成 | {info}")
        except Exception as exc:
            log_worker(f"人物分析模型预热失败，将在首个任务中重试: {exc}")


@task_prerun.connect
def on_task_prerun(task_id=None, task=None, args=None, **kwargs) -> None:
    del kwargs
    line = f"TASK_START | id={task_id} | {getattr(task, 'name', '')}"
    if args and len(args) > 1:
        line += f" | video_name={args[1]!r}"
    log_worker(line)


@task_postrun.connect
def on_task_postrun(task_id=None, task=None, state=None, **kwargs) -> None:
    del kwargs
    log_worker(f"TASK_END | id={task_id} | {getattr(task, 'name', '')} | state={state}")


@task_failure.connect
def on_task_failure(task_id=None, exception=None, **kwargs) -> None:
    del kwargs
    log_worker(f"TASK_FAIL | id={task_id} | {exception!r}")
