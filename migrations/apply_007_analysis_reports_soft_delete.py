#!/usr/bin/env python3
"""幂等为 analysis_reports 增加软删除字段。不操作 siglip_v2。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = "medical_audit_v3"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db import load_siglip_env  # noqa: E402


def _column_exists(conn, name: str) -> bool:
    return (
        conn.execute(
            text(
                """
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = DATABASE()
                  AND table_name = 'analysis_reports'
                  AND column_name = :name
                LIMIT 1
                """
            ),
            {"name": name},
        ).first()
        is not None
    )


def _index_exists(conn, name: str) -> bool:
    return (
        conn.execute(
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
        is not None
    )


def _fk_exists(conn, name: str) -> bool:
    return (
        conn.execute(
            text(
                """
                SELECT 1 FROM information_schema.table_constraints
                WHERE table_schema = DATABASE()
                  AND table_name = 'analysis_reports'
                  AND constraint_name = :name
                  AND constraint_type = 'FOREIGN KEY'
                LIMIT 1
                """
            ),
            {"name": name},
        ).first()
        is not None
    )


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
    with engine.begin() as conn:
        conn.execute(text(f"USE {TARGET}"))
        exists = conn.execute(
            text(
                """
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = DATABASE()
                  AND table_name = 'analysis_reports'
                LIMIT 1
                """
            )
        ).first()
        if exists is None:
            print("analysis_reports 不存在，请先执行 006", file=sys.stderr)
            return 1
        if not _column_exists(conn, "deleted_at"):
            conn.execute(
                text(
                    "ALTER TABLE analysis_reports "
                    "ADD COLUMN deleted_at DATETIME(3) NULL "
                    "COMMENT '软删除时间（UTC naive）' AFTER updated_at"
                )
            )
        if not _column_exists(conn, "deleted_by"):
            conn.execute(
                text(
                    "ALTER TABLE analysis_reports "
                    "ADD COLUMN deleted_by INT UNSIGNED NULL "
                    "COMMENT '删除人 users.id' AFTER deleted_at"
                )
            )
        if not _index_exists(conn, "idx_analysis_reports_deleted"):
            conn.execute(
                text(
                    "ALTER TABLE analysis_reports "
                    "ADD INDEX idx_analysis_reports_deleted (deleted_at)"
                )
            )
        if not _fk_exists(conn, "fk_analysis_reports_deleted_by"):
            conn.execute(
                text(
                    """
                    ALTER TABLE analysis_reports
                    ADD CONSTRAINT fk_analysis_reports_deleted_by
                      FOREIGN KEY (deleted_by) REFERENCES users(id)
                      ON DELETE RESTRICT
                    """
                )
            )
    engine.dispose()
    print("007_analysis_reports_soft_delete applied (idempotent)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
