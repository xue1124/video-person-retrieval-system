#!/usr/bin/env python3
"""幂等应用 006_analysis_reports.sql。不操作 siglip_v2。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SQL_PATH = PROJECT_ROOT / "migrations" / "006_analysis_reports.sql"
TARGET = "medical_audit_v3"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db import load_siglip_env  # noqa: E402


def _statements(sql: str) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    for raw in sql.splitlines():
        line = raw.strip()
        if not line or line.startswith("--"):
            continue
        buf.append(raw)
        if line.endswith(";"):
            stmt = "\n".join(buf).strip().rstrip(";")
            if stmt:
                out.append(stmt)
            buf = []
    if buf:
        stmt = "\n".join(buf).strip().rstrip(";")
        if stmt:
            out.append(stmt)
    return out


def _ensure_index(conn, name: str, ddl: str) -> None:
    exists = conn.execute(
        text(
            """
            SELECT 1 FROM information_schema.statistics
            WHERE table_schema = DATABASE()
              AND table_name = 'analysis_reports'
              AND index_name = :name
            LIMIT 1
            """
        ),
        {"name": name},
    ).first()
    if exists is None:
        conn.execute(text(ddl))


def main() -> int:
    load_siglip_env()
    url = os.environ.get("TRACKING_DB_URL", "").strip()
    if not url:
        print("未配置 TRACKING_DB_URL", file=sys.stderr)
        return 1
    if (make_url(url).database or "").lower() != TARGET:
        print("TRACKING_DB_URL 必须指向 medical_audit_v3", file=sys.stderr)
        return 1
    engine = create_engine(url, pool_pre_ping=True)
    stmts = _statements(SQL_PATH.read_text(encoding="utf-8"))
    with engine.begin() as conn:
        conn.execute(text(f"USE {TARGET}"))
        for stmt in stmts:
            upper = stmt.lstrip().upper()
            if upper.startswith("USE "):
                continue
            conn.execute(text(stmt))
        _ensure_index(
            conn,
            "idx_analysis_reports_user_created",
            "ALTER TABLE analysis_reports ADD INDEX idx_analysis_reports_user_created (created_by, created_at)",
        )
        _ensure_index(
            conn,
            "idx_analysis_reports_status",
            "ALTER TABLE analysis_reports ADD INDEX idx_analysis_reports_status (status)",
        )
        _ensure_index(
            conn,
            "idx_analysis_reports_date",
            "ALTER TABLE analysis_reports ADD INDEX idx_analysis_reports_date (report_date)",
        )
        _ensure_index(
            conn,
            "idx_analysis_reports_created",
            "ALTER TABLE analysis_reports ADD INDEX idx_analysis_reports_created (created_at)",
        )
    engine.dispose()
    print("006_analysis_reports applied (idempotent)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
