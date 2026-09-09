#!/usr/bin/env python3
"""第二阶段验收：观测表、双特征、FAISS 与检索字段。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TARGET = "medical_audit_v3"

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
    _load_env()
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
        if "observations" not in tables:
            errors.append("缺少表 observations")
        if "observation_embeddings" not in tables:
            errors.append("缺少表 observation_embeddings")
        if errors:
            print("FAIL")
            for e in errors:
                print(" -", e)
            return 1

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

    from tracking_v3.search_index import index_status

    status = index_status(engine)
    for mod, row in status.items():
        db_n = row.get("db_count")
        ntotal = row.get("ntotal")
        consistent = row.get("consistent")
        print(f"faiss {mod}: db={db_n} index={ntotal} consistent={consistent}")
        if db_n and ntotal is not None and not consistent:
            errors.append(f"FAISS {mod} 与数据库不一致（可运行 rebuild_search_index.py）")

    engine.dispose()

    cluster_src = (
        PROJECT_ROOT / "tracking_experiments" / "detect_reid_video.py"
    ).read_text(encoding="utf-8")
    if 'row["feature"]' not in cluster_src or "siglip_feature" not in cluster_src:
        errors.append("detect_reid_video 未同时保留 OSNet feature 与 siglip_feature")
    if 'np.stack([row["feature"]' not in cluster_src.replace(" ", ""):
        # looser check
        if 'row["feature"] for row in observations' not in cluster_src:
            errors.append("聚类未明确只使用 OSNet feature")
    if "siglip_feature" in cluster_src and 'row["siglip_feature"] for row in observations' in cluster_src:
        errors.append("聚类错误地使用了 SigLIP 特征")

    if errors:
        print("FAIL")
        for e in errors:
            print(" -", e)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
