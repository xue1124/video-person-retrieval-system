#!/usr/bin/env python3
"""Read-only verification for the formal database schema and relational data."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TARGET = "medical_audit_v3"

REQUIRED_TABLES = {
    "analysis_reports",
    "cameras",
    "gallery_meta",
    "global_people",
    "observation_embeddings",
    "observations",
    "person_identity_assignments",
    "person_snapshots",
    "processing_runs",
    "rooms",
    "search_logs",
    "stay_segments",
    "track_points",
    "tracks",
    "users",
    "video_people",
    "videos",
}

REQUIRED_COLUMNS = {
    "videos": {
        "source_key", "file_name", "source_type", "origin_label", "task_id",
        "status", "failure_reason", "progress", "source_path", "is_media_source",
        "captured_at", "fps", "duration_sec", "duration", "target_count",
        "completed_at", "processing_started_at", "created_at", "updated_at",
    },
    "users": {"username", "password_hash", "role", "is_active"},
    "rooms": {"camera_id", "video_id", "name", "polygon_json"},
    "track_points": {
        "track_id", "room_id", "frame_index", "timestamp_sec", "bbox_x1",
        "bbox_y1", "bbox_x2", "bbox_y2", "foot_x", "foot_y", "confidence",
        "pre_merge_id", "assign_score", "merge_score",
    },
    "gallery_meta": {
        "video_id", "video_name", "timestamp", "image_path", "feature_vector",
        "clip_feature",
    },
    "analysis_reports": {
        "created_by", "scope_json", "report_data_json", "status", "deleted_at",
        "deleted_by",
    },
}

REQUIRED_OBS_COLS = {
    "video_id",
    "processing_run_id",
    "video_person_id",
    "global_person_id",
    "track_id",
    "timestamp_sec",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "room_id",
    "crop_path",
}
REQUIRED_EMB_COLS = {
    "observation_id",
    "modality",
    "embedding",
    "model_name",
    "model_version",
    "embedding_dim",
}
REQUIRED_INDEXES = [
    ("observations", "PRIMARY"),
    ("observations", "uq_obs_run_track_frame"),
    ("observations", "idx_obs_video_time"),
    ("observations", "idx_obs_global"),
    ("observation_embeddings", "PRIMARY"),
    ("observation_embeddings", "uq_obs_emb_modality"),
]
REQUIRED_FKS = [
    ("observations", "fk_obs_video"),
    ("observations", "fk_obs_run"),
    ("observations", "fk_obs_video_person"),
    ("observations", "fk_obs_global"),
    ("observations", "fk_obs_track"),
    ("observation_embeddings", "fk_obs_emb_observation"),
]


def _columns(conn, table: str) -> set[str]:
    return {str(r[0]) for r in conn.execute(text(f"SHOW COLUMNS FROM {table}"))}


def _indexes(conn, table: str) -> set[str]:
    return {str(r[2]) for r in conn.execute(text(f"SHOW INDEX FROM {table}"))}


def _fks(conn) -> set[tuple[str, str]]:
    rows = conn.execute(
        text(
            """
            SELECT TABLE_NAME, CONSTRAINT_NAME
            FROM information_schema.TABLE_CONSTRAINTS
            WHERE CONSTRAINT_SCHEMA = :db AND CONSTRAINT_TYPE = 'FOREIGN KEY'
            """
        ),
        {"db": TARGET},
    )
    return {(str(r[0]), str(r[1])) for r in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-faiss",
        action="store_true",
        help="同时检查 FAISS 数量；该过程可能同步索引元数据文件",
    )
    args = parser.parse_args()
    from services.persistence.database import load_siglip_env

    load_siglip_env()
    url = os.environ.get("TRACKING_DB_URL", "").strip()
    if not url:
        print("FAIL: 未配置 TRACKING_DB_URL")
        return 1
    if (make_url(url).database or "").lower() != TARGET:
        print("FAIL: TRACKING_DB_URL 必须指向 medical_audit_v3")
        return 1

    engine = create_engine(url, pool_pre_ping=True)
    errors: list[str] = []
    with engine.connect() as conn:
        tables = set(conn.execute(text("SHOW TABLES")).scalars())
        missing_tables = REQUIRED_TABLES - tables
        if missing_tables:
            errors.append(f"缺少表: {sorted(missing_tables)}")
        if errors:
            print("FAIL")
            for e in errors:
                print(" -", e)
            return 1

        for table, required in REQUIRED_COLUMNS.items():
            missing = required - _columns(conn, table)
            if missing:
                errors.append(f"{table} 缺列: {sorted(missing)}")

        obs_cols = _columns(conn, "observations")
        missing = REQUIRED_OBS_COLS - obs_cols
        if missing:
            errors.append(f"observations 缺列: {sorted(missing)}")
        emb_cols = _columns(conn, "observation_embeddings")
        missing = REQUIRED_EMB_COLS - emb_cols
        if missing:
            errors.append(f"observation_embeddings 缺列: {sorted(missing)}")

        obs_idx = _indexes(conn, "observations")
        emb_idx = _indexes(conn, "observation_embeddings")
        for table, name in REQUIRED_INDEXES:
            have = obs_idx if table == "observations" else emb_idx
            if name not in have:
                errors.append(f"缺少索引 {table}.{name}")

        fks = _fks(conn)
        for item in REQUIRED_FKS:
            if item not in fks:
                errors.append(f"缺少外键 {item[0]}.{item[1]}")

        n_obs = int(conn.execute(text("SELECT COUNT(*) FROM observations")).scalar() or 0)
        n_osnet = int(
            conn.execute(
                text(
                    "SELECT COUNT(*) FROM observation_embeddings WHERE modality='osnet'"
                )
            ).scalar()
            or 0
        )
        n_siglip = int(
            conn.execute(
                text(
                    "SELECT COUNT(*) FROM observation_embeddings WHERE modality='siglip_image'"
                )
            ).scalar()
            or 0
        )
        orphan = int(
            conn.execute(
                text(
                    """
                    SELECT COUNT(*) FROM observation_embeddings e
                    LEFT JOIN observations o ON o.id = e.observation_id
                    WHERE o.id IS NULL
                    """
                )
            ).scalar()
            or 0
        )
        if orphan:
            errors.append(f"孤立 embedding {orphan} 条")
        if n_obs and n_osnet and n_osnet != n_obs:
            errors.append(f"OSNet 向量 {n_osnet} 与观测 {n_obs} 不一致")
        if n_obs and n_siglip and n_siglip != n_obs:
            errors.append(f"SigLIP 向量 {n_siglip} 与观测 {n_obs} 不一致")

        print(
            f"observations={n_obs} osnet={n_osnet} siglip={n_siglip} "
            f"videos={int(conn.execute(text('SELECT COUNT(*) FROM videos')).scalar() or 0)}"
        )

    if args.check_faiss:
        try:
            from services.analysis.tracking.search_index import index_status

            status = index_status(engine)
            for mod, row in status.items():
                db_n = row.get("db_count")
                ntotal = row.get("ntotal")
                consistent = row.get("consistent")
                print(f"faiss {mod}: db={db_n} index={ntotal} consistent={consistent}")
                if db_n and ntotal is not None and not consistent:
                    errors.append(
                        f"FAISS {mod} 与数据库不一致（可运行 tools/rebuild_search_index.py）"
                    )
        except Exception as exc:
            errors.append(f"FAISS 检查失败: {exc}")

    engine.dispose()

    if errors:
        print("FAIL")
        for e in errors:
            print(" -", e)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
