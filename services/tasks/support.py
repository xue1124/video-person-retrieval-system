"""API、Celery任务与RTSP线程共享的任务状态辅助函数。"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from services.persistence import database as app_db


def get_task_engine() -> Any:
    """取得应用共享的SQLAlchemy引擎，不触发模型加载。"""
    return app_db.get_engine()


def mark_task_failed(db_engine: Any, video_name: str, reason: str) -> None:
    """记录可展示的失败原因，避免Celery失败后数据库仍显示processing。"""
    message = (reason or "")[:500]
    with db_engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE videos SET status='failed', failure_reason=:reason "
                "WHERE file_name=:video_name AND is_media_source=1"
            ),
            {"reason": message, "video_name": video_name},
        )
