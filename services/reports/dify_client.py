"""Dify Workflow 客户端。API Key 只从环境变量读取，不写日志、不返回前端。"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Optional

import requests

from services.persistence.database import load_siglip_env

CONNECT_TIMEOUT_SECONDS = 10


class DifyError(Exception):
    http_status = 502

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class DifyConfigError(DifyError):
    http_status = 502


class DifyTimeoutError(DifyError):
    http_status = 504


class DifyUpstreamError(DifyError):
    http_status = 502


class DifyOutputError(DifyError):
    http_status = 502


@dataclass(frozen=True)
class DifyConfig:
    base_url: str
    api_key: str
    timeout_seconds: int


@dataclass(frozen=True)
class DifyWorkflowResult:
    workflow_run_id: Optional[str]
    outputs: dict[str, Any]
    raw: dict[str, Any]


def load_dify_config() -> DifyConfig:
    load_siglip_env()
    base_url = os.environ.get("DIFY_BASE_URL", "").strip().rstrip("/")
    api_key = os.environ.get("DIFY_WORKFLOW_API_KEY", "").strip()
    raw_timeout = os.environ.get("DIFY_REPORT_TIMEOUT_SECONDS", "120").strip() or "120"
    try:
        timeout_seconds = int(raw_timeout)
    except ValueError as exc:
        raise DifyConfigError("DIFY_REPORT_TIMEOUT_SECONDS 无效") from exc
    if timeout_seconds <= 0:
        raise DifyConfigError("DIFY_REPORT_TIMEOUT_SECONDS 无效")
    if not base_url:
        raise DifyConfigError("未配置 DIFY_BASE_URL")
    if not api_key:
        raise DifyConfigError("未配置 DIFY_WORKFLOW_API_KEY")
    return DifyConfig(
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )


def redact_secrets(text: str, api_key: str = "") -> str:
    cleaned = str(text or "")
    if api_key:
        cleaned = cleaned.replace(api_key, "***")
    cleaned = re.sub(r"Bearer\s+\S+", "Bearer ***", cleaned, flags=re.I)
    cleaned = re.sub(r"(api[_-]?key\s*[:=]\s*)\S+", r"\1***", cleaned, flags=re.I)
    return cleaned[:500]


def run_report_workflow(
    report_data_json: str,
    *,
    user_id: int,
    config: Optional[DifyConfig] = None,
) -> DifyWorkflowResult:
    cfg = config or load_dify_config()
    url = f"{cfg.base_url}/workflows/run"
    payload = {
        "inputs": {"report_data_json": report_data_json},
        "response_mode": "blocking",
        "user": f"medical-audit-user-{int(user_id)}",
    }
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=(CONNECT_TIMEOUT_SECONDS, cfg.timeout_seconds),
        )
    except requests.Timeout as exc:
        raise DifyTimeoutError("Dify 调用超时") from exc
    except requests.ConnectionError as exc:
        raise DifyUpstreamError("Dify 连接失败") from exc
    except requests.RequestException as exc:
        raise DifyUpstreamError(
            redact_secrets(f"Dify 调用失败: {exc}", cfg.api_key)
        ) from exc

    if not (200 <= response.status_code < 300):
        raise DifyUpstreamError(f"Dify 返回 HTTP {response.status_code}")

    try:
        body = response.json()
    except ValueError as exc:
        raise DifyUpstreamError("Dify 响应不是 JSON") from exc
    if not isinstance(body, dict):
        raise DifyUpstreamError("Dify 响应结构无效")
    return _parse_workflow_body(body)


def _parse_workflow_body(body: dict[str, Any]) -> DifyWorkflowResult:
    data = body.get("data")
    if not isinstance(data, dict):
        data = {}
    status = str(data.get("status") or body.get("status") or "").strip().lower()
    if status != "succeeded":
        raise DifyUpstreamError(
            f"Dify workflow 未成功: {status or 'unknown'}"
        )
    outputs = data.get("outputs")
    if not isinstance(outputs, dict):
        raise DifyOutputError("Dify data.outputs 不存在")

    title = outputs.get("title")
    markdown = outputs.get("report_markdown")
    warnings = outputs.get("warnings")
    if not isinstance(title, str) or not title.strip():
        raise DifyOutputError("Dify title 缺失或类型错误")
    if not isinstance(markdown, str) or not markdown.strip():
        raise DifyOutputError("Dify report_markdown 缺失或类型错误")
    if not _is_str_list(warnings):
        raise DifyOutputError("Dify warnings 不是字符串数组")

    run_id = body.get("workflow_run_id") or data.get("id")
    return DifyWorkflowResult(
        workflow_run_id=str(run_id) if run_id else None,
        outputs={
            "title": title.strip(),
            "report_markdown": markdown,
            "warnings": list(warnings),
        },
        raw=body,
    )


def _is_str_list(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    return all(isinstance(item, str) for item in value)
