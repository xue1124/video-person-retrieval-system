"""人物检索、房间共现检索与检索日志接口。"""

from __future__ import annotations

import json
from typing import List, Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import text

from services.analysis.room_copresence import perform_room_copresence
from services.auth.dependencies import get_current_user, require_admin
from services.search.service import perform_search

router = APIRouter(tags=["search"])
_UTC = ZoneInfo("UTC")
_SHANGHAI = ZoneInfo("Asia/Shanghai")


def _serialize_datetime(value) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        if value.tzinfo is None:
            value = value.replace(tzinfo=_UTC)
        return value.astimezone(_SHANGHAI).isoformat()
    return str(value)


@router.post("/search")
async def search_trajectory(
    request: Request,
    algorithm: str = Form("SIGLIP"),
    threshold: float = Form(0.85, ge=0.0, le=1.0),
    time_gap: float = Form(60.0, ge=0.0),
    group_mode: str = Form("false"),
    co_time_threshold: float = Form(2.0, ge=0.0),
    q_text: Optional[str] = Form(None),
    include_stay_segments: str = Form("true"),
    files: Optional[List[UploadFile]] = File(None),
    user: dict = Depends(get_current_user),
):
    """根据图片或文字召回相似人物，并整理为前端轨迹结果。"""
    del user
    image_bytes: List[bytes] = []
    names: List[str] = []
    for upload in files or []:
        if not upload.filename:
            continue
        image_bytes.append(await upload.read())
        names.append(upload.filename)
    group_enabled = str(group_mode).lower() in ("1", "true", "yes", "on")
    include_stays = str(include_stay_segments).lower() in ("1", "true", "yes", "on")
    try:
        payload = perform_search(
            engine=request.app.state.engine,
            algorithm=algorithm,
            threshold=threshold,
            time_gap=time_gap,
            group_mode=group_enabled,
            co_time_threshold=co_time_threshold,
            q_text=q_text,
            image_bytes_list=image_bytes,
            image_names=names,
            include_stay_segments=include_stays,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=str(exc) or exc.__class__.__name__
        ) from exc
    return {
        "results": payload["results"],
        "stay_segments": payload.get("stay_segments") or {},
        "algorithm": payload["algorithm"],
        "query_image_names": payload["query_image_names"],
    }


@router.post("/search/room-copresence")
async def search_room_copresence(
    request: Request,
    algorithm: str = Form("OSNet"),
    threshold: float = Form(0.85, ge=0.0, le=1.0),
    time_gap: float = Form(60.0, ge=0.0),
    role_a_label: str = Form("患者"),
    role_b_label: str = Form("医生"),
    role_a_files: List[UploadFile] = File(...),
    role_b_files: Optional[List[UploadFile]] = File(None),
    user: dict = Depends(get_current_user),
):
    """分别检索两组人物图片，再计算同房间、同时段的共现结果。"""
    del user
    bytes_a: List[bytes] = []
    names_a: List[str] = []
    for upload in role_a_files:
        if not upload.filename:
            continue
        raw = await upload.read()
        if raw:
            bytes_a.append(raw)
            names_a.append(upload.filename)
    if not bytes_a:
        raise HTTPException(status_code=400, detail="请至少上传一张有效的患者查询图")

    bytes_b: List[bytes] = []
    names_b: List[str] = []
    for upload in role_b_files or []:
        if not upload.filename:
            continue
        raw = await upload.read()
        if raw:
            bytes_b.append(raw)
            names_b.append(upload.filename)

    try:
        return perform_room_copresence(
            engine=request.app.state.engine,
            algorithm=algorithm,
            threshold=threshold,
            time_gap=time_gap,
            images_a_bytes=bytes_a,
            images_a_names=names_a,
            images_b_bytes=bytes_b or None,
            images_b_names=names_b or None,
            role_a_label=role_a_label,
            role_b_label=role_b_label,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=str(exc) or exc.__class__.__name__
        ) from exc


class SaveSearchLogBody(BaseModel):
    query_image_paths: str = ""
    query_image_paths_b: str = ""
    search_query: Optional[str] = None
    algorithm: str
    threshold: float = Field(ge=0.0, le=1.0)
    results: dict = Field(default_factory=dict)
    stay_segments: Optional[dict] = None
    time_gap: Optional[float] = Field(default=None, ge=0.0)
    log_kind: str = "single"
    copresence: Optional[dict] = None


def _serialize_search_log_results(body: SaveSearchLogBody) -> str:
    """兼容当前共现日志、单人停留日志与早期扁平结果。"""
    if (body.log_kind or "").strip() == "room_copresence" and body.copresence is not None:
        videos = body.copresence.get("videos", body.copresence)
        payload = {
            "version": 3,
            "kind": "room_copresence",
            "algorithm": body.algorithm,
            "threshold": float(body.threshold),
            "time_gap": float(body.time_gap) if body.time_gap is not None else 60.0,
            "query_image_a": body.query_image_paths or "",
            "query_image_b": body.query_image_paths_b or "",
            "videos": videos,
        }
    elif body.stay_segments is not None:
        payload = {
            "version": 2,
            "kind": "single",
            "results": body.results,
            "stay_segments": body.stay_segments,
            "time_gap": float(body.time_gap) if body.time_gap is not None else 60.0,
        }
    else:
        payload = body.results
    return json.dumps(payload, ensure_ascii=False)


@router.post("/search/logs")
def save_search_log(
    body: SaveSearchLogBody,
    request: Request,
    user: dict = Depends(get_current_user),
):
    del user
    with request.app.state.engine.begin() as conn:
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


@router.get("/search/logs")
def list_search_logs(request: Request, user: dict = Depends(get_current_user)):
    del user
    with request.app.state.engine.connect() as conn:
        rows = conn.execute(
            text("SELECT * FROM search_logs ORDER BY created_at DESC")
        ).mappings().all()
    output = []
    for row in rows:
        item = dict(row)
        if item.get("created_at") is not None:
            item["created_at"] = _serialize_datetime(item["created_at"])
        output.append(item)
    return output


@router.delete("/search/logs/{log_id}")
def delete_search_log(
    log_id: int,
    request: Request,
    user: dict = Depends(require_admin),
):
    del user
    with request.app.state.engine.begin() as conn:
        result = conn.execute(
            text("DELETE FROM search_logs WHERE id=:id"), {"id": log_id}
        )
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="日志不存在")
    return {"ok": True}
