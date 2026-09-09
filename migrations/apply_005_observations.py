#!/usr/bin/env python3
"""幂等应用 005_observation_embeddings.sql。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SQL_PATH = PROJECT_ROOT / "migrations" / "005_observation_embeddings.sql"
TARGET = "medical_audit_v3"


def _load_env() -> None:
    env_file = PROJECT_ROOT / "deploy" / "env" / "siglip.env"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


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


def main() -> int:
    _load_env()
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
    engine.dispose()
    print("005_observation_embeddings applied (idempotent)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
