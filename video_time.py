"""videos.captured_at 时区约定。

MySQL DATETIME 无时区。本项目与 serialize_datetime / 报告预览一致：
库中的 naive datetime 表示 UTC。

用户填写或 ISAPI 回放选择的无时区时间视为 Asia/Shanghai 墙钟，写入前换成 UTC naive。
已带偏移的时间按偏移换算到 UTC，不会再加 8 小时。

不要用文件 mtime、ctime 或上传时间猜测拍摄时间。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

UTC = timezone.utc
SHANGHAI = ZoneInfo("Asia/Shanghai")


def utc_now_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def parse_user_captured_at(value: Any) -> Optional[datetime]:
    """用户输入 → 库用 UTC naive。空值返回 None，不猜测。"""
    parsed = _parse_datetime(value)
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=SHANGHAI)
    return parsed.astimezone(UTC).replace(tzinfo=None)


def parse_iso_captured_at(value: Any) -> Optional[datetime]:
    """内部传递（ISO / 库值）→ UTC naive。无时区视为已经是 UTC。"""
    parsed = _parse_datetime(value)
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        return parsed
    return parsed.astimezone(UTC).replace(tzinfo=None)


def db_datetime_to_utc_iso(value: Any) -> Optional[str]:
    """把库中 captured_at 编成带 UTC 偏移的 ISO，供 tracking_v3 入库。"""
    parsed = parse_iso_captured_at(value)
    if parsed is None:
        return None
    return parsed.replace(tzinfo=UTC).isoformat()


def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(" ", "T", 1)
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"拍摄时间格式无效: {value}") from exc
