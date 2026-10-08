"""Celery任务状态查询与人工终止接口。"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import text

from services.auth.dependencies import get_current_user
from services.tasks.celery_app import app as celery_app

router = APIRouter(tags=["tasks"])


@router.post("/stop_task")
def stop_task(
    request: Request,
    task_id: str = Query(..., min_length=1, max_length=100),
    user: dict = Depends(get_current_user),
):
    """撤销Celery消息并同步数据库，Worker也会通过task_id变化主动停止。"""
    del user
    engine = request.app.state.engine
    with engine.begin() as conn:
        result = conn.execute(
            text(
                "UPDATE videos SET status='stopped', task_id=NULL, "
                "failure_reason='用户终止任务' "
                "WHERE task_id=:task_id AND is_media_source=1 "
                "AND status IN ('pending','processing','transcoding')"
            ),
            {"task_id": task_id},
        )
    celery_app.control.revoke(task_id, terminate=False)
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="未找到可终止的运行中任务")
    return {"status": "stopped", "task_id": task_id}


@router.get("/status/{task_id}")
def get_status(task_id: str, user: dict = Depends(get_current_user)):
    del user
    result = celery_app.AsyncResult(task_id)
    state = result.state
    progress = 100 if state == "SUCCESS" else 0
    if state == "PROGRESS" and isinstance(result.info, dict):
        progress = int(result.info.get("current", 0) or 0)
    return {"state": state, "progress": max(0, min(100, progress))}
