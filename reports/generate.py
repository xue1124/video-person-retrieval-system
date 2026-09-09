"""调用 preview 统计后请求 Dify，并把结果写入 analysis_reports。"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy.engine import Engine

from reports.dify_client import DifyError, DifyUpstreamError, redact_secrets, run_report_workflow
from reports.models import ReportPreviewRequest
from reports.preview import build_report_preview
from reports import store


def dump_report_data(report_data: dict[str, Any]) -> str:
    return json.dumps(report_data, ensure_ascii=False)


def generate_report(
    engine: Engine,
    request: ReportPreviewRequest,
    user: dict[str, Any],
) -> dict[str, Any]:
    preview = build_report_preview(engine, request)
    report_data = preview.model_dump(mode="json")
    report_data_json = dump_report_data(report_data)
    report_date = request.date.isoformat() if request.date is not None else report_data.get("scope", {}).get("date")
    report_id = store.insert_generating(
        engine,
        user_id=int(user["id"]),
        report_type=str(report_data.get("report_type") or "daily_activity_report"),
        report_date=report_date,
        timezone_name=request.timezone,
        scope=report_data.get("scope") or {},
        report_data_json=report_data_json,
    )
    try:
        result = run_report_workflow(report_data_json, user_id=int(user["id"]))
        store.mark_completed(
            engine,
            report_id,
            title=result.outputs["title"],
            report_markdown=result.outputs["report_markdown"],
            warnings=list(result.outputs["warnings"]),
            workflow_run_id=result.workflow_run_id,
            dify_output=result.raw,
        )
    except DifyError as exc:
        store.mark_failed(engine, report_id, redact_secrets(exc.message))
        exc.report_id = report_id
        raise
    except Exception:
        store.mark_failed(engine, report_id, "报告生成失败")
        wrapped = DifyUpstreamError("报告生成失败")
        wrapped.report_id = report_id
        raise wrapped from None

    row = store.fetch_report(engine, report_id)
    if row is None:
        raise RuntimeError("报告写入后无法读取")
    return store.to_public_dict(row, detail=True)
