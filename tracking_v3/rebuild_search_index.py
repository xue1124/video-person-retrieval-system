#!/usr/bin/env python3
"""从 medical_audit_v3.observation_embeddings 重建 FAISS 索引。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="从数据库重建 OSNet / SigLIP FAISS 索引")
    parser.add_argument(
        "--modality",
        choices=("osnet", "siglip_image", "all"),
        default="all",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="只打印当前索引与数据库数量，不重建",
    )
    args = parser.parse_args()

    env_file = PROJECT_ROOT / "deploy" / "env" / "siglip.env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

    from sqlalchemy import create_engine

    from tracking_v3.search_index import ALL_MODALITIES, index_status, rebuild_from_db

    url = os.environ.get("TRACKING_DB_URL", "").strip()
    if not url:
        print("未配置 TRACKING_DB_URL", file=sys.stderr)
        return 1
    engine = create_engine(url, pool_pre_ping=True)
    try:
        if args.status_only:
            print(json.dumps(index_status(engine), ensure_ascii=False, indent=2, default=str))
            return 0
        modality = None if args.modality == "all" else args.modality
        report = rebuild_from_db(engine, modality)
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        status = index_status(engine)
        ok = all(
            status.get(m, {}).get("consistent")
            for m in (ALL_MODALITIES if modality is None else (modality,))
        )
        return 0 if ok else 2
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
