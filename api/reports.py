"""AI分析报告HTTP接口。

本模块只负责参数、权限和HTTP错误转换；报告预览、Dify调用与数据库读写仍由
``services.reports`` 完成。
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from services.auth.dependencies import get_current_user
from services.persistence import database as app_db
from services.reports import store
from services.reports.dify_client import DifyError
from services.reports.generate import generate_report
from services.reports.models import (
    GeneratedReportResponse,
    ReportListResponse,
    ReportPreviewRequest,
    ReportPreviewResponse,
)
from services.reports.preview import build_report_preview

router = APIRouter(tags=["reports"])


def _engine(request: Request):
    return getattr(request.app.state, "engine", None) or app_db.get_engine()


def _http_from_dify(exc: DifyError, report_id: Optional[int] = None) -> HTTPException:
    rid = report_id if report_id is not None else getattr(exc, "report_id", None)
    return HTTPException(
        status_code=int(getattr(exc, "http_status", 502) or 502),
        detail={"error": exc.message, "report_id": rid},
    )


@router.post("/reports/preview", response_model=ReportPreviewResponse)
def reports_preview(
    body: ReportPreviewRequest,
    request: Request,
    user: dict = Depends(get_current_user),
):
    del user
    try:
        return build_report_preview(_engine(request), body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/reports/generate", response_model=GeneratedReportResponse)
def reports_generate(
    body: ReportPreviewRequest,
    request: Request,
    user: dict = Depends(get_current_user),
):
    try:
        payload = generate_report(_engine(request), body, user)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DifyError as exc:
        raise _http_from_dify(exc) from exc
    return GeneratedReportResponse.model_validate(payload)


@router.get("/reports", response_model=ReportListResponse)
def reports_list(
    request: Request,
    user: dict = Depends(get_current_user),
    status: Optional[str] = Query(None),
    report_date: Optional[date] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    if status is not None and status not in {"generating", "completed", "failed"}:
        raise HTTPException(status_code=400, detail="无效的status")
    items, total = store.list_reports(
        _engine(request),
        user_id=int(user["id"]),
        is_admin=user.get("role") == "admin",
        status=status,
        report_date=report_date.isoformat() if report_date else None,
        page=page,
        page_size=page_size,
    )
    return ReportListResponse(
        items=[store.to_public_dict(row, detail=False) for row in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/reports/{report_id}", response_model=GeneratedReportResponse)
def reports_detail(
    report_id: int,
    request: Request,
    user: dict = Depends(get_current_user),
):
    row = store.fetch_report(_engine(request), report_id)
    if row is None:
        raise HTTPException(status_code=404, detail="报告不存在")
    if int(row["created_by"]) != int(user["id"]) and user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="无权查看该报告")
    return GeneratedReportResponse.model_validate(store.to_public_dict(row, detail=True))


@router.delete("/reports/{report_id}")
def reports_delete(
    report_id: int,
    request: Request,
    user: dict = Depends(get_current_user),
):
    engine = _engine(request)
    row = store.fetch_report(engine, report_id, include_deleted=True)
    if row is None or row.get("deleted_at") is not None:
        raise HTTPException(status_code=404, detail="报告不存在")
    if int(row["created_by"]) != int(user["id"]) and user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="无权删除该报告")
    if str(row.get("status") or "") == "generating":
        raise HTTPException(status_code=409, detail="报告正在生成，暂不能删除")
    store.soft_delete(engine, report_id, int(user["id"]))
    return {"ok": True, "id": int(report_id)}
