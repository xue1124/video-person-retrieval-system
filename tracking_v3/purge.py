"""清理 medical_audit_v3 中按视频文件名关联的跟踪数据。"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

TARGET_DATABASE = "medical_audit_v3"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT_ROOT = PROJECT_ROOT / "tracking_snapshots"


def _tracking_url() -> str | None:
    url = os.environ.get("TRACKING_DB_URL", "").strip()
    if not url:
        return None
    database = (make_url(url).database or "").lower()
    if database != TARGET_DATABASE:
        return None
    return url


def snapshot_root() -> Path:
    return Path(
        os.environ.get("TRACKING_SNAPSHOT_ROOT", str(DEFAULT_SNAPSHOT_ROOT))
    ).expanduser().resolve()


def _observation_ids_for_videos(conn: Any, video_ids: list[int]) -> list[int]:
    if not video_ids:
        return []
    from sqlalchemy import bindparam

    stmt = text(
        "SELECT id FROM observations WHERE video_id IN :ids"
    ).bindparams(bindparam("ids", expanding=True))
    try:
        return [int(x) for x in conn.execute(stmt, {"ids": video_ids}).scalars()]
    except Exception:
        return []


def _drop_faiss_ids(obs_ids: list[int], engine: Any | None = None) -> dict[str, Any]:
    try:
        from tracking_v3.search_index import reconcile_index, remove_observation_ids

        report: dict[str, Any] = {}
        if obs_ids:
            report["remove"] = remove_observation_ids(obs_ids, engine=engine)
        if engine is not None:
            report["reconcile"] = reconcile_index(engine)
        else:
            from tracking_v3.search_index import sync_metadata

            report["sync"] = sync_metadata()
        return report
    except Exception as exc:
        try:
            from tracking_v3.search_index import mark_dirty, sync_metadata

            mark_dirty()
            sync_metadata(engine)
        except Exception:
            pass
        return {"error": str(exc), "dirty": True}


def purge_tracking_by_file_name(file_name: str) -> dict[str, Any]:
    """删除 medical_audit_v3 中 file_name 对应的全部 videos（级联清理轨迹/房间/快照）。"""
    name = Path(str(file_name or "").replace("\\", "/")).name.strip()
    if not name:
        return {"skipped": "empty_file_name"}

    url = _tracking_url()
    if not url:
        return {"skipped": "TRACKING_DB_URL 未配置"}

    root = snapshot_root()
    engine = create_engine(url, pool_pre_ping=True)
    try:
        result: dict[str, Any] | None = None
        obs_ids: list[int] = []
        with engine.begin() as conn:
            video_ids = [
                int(v)
                for v in conn.execute(
                    text("SELECT id FROM videos WHERE file_name=:n ORDER BY id"),
                    {"n": name},
                ).scalars()
            ]
            if not video_ids:
                return {"file_name": name, "deleted_videos": [], "orphan_global_people": 0}

            obs_ids = _observation_ids_for_videos(conn, video_ids)

            # 先删快照 / 观测 crop 文件
            for vid in video_ids:
                snap_dir = root / f"v{vid}"
                if snap_dir.is_dir():
                    shutil.rmtree(snap_dir, ignore_errors=True)
                rows = conn.execute(
                    text("SELECT image_path FROM person_snapshots WHERE video_id=:v"),
                    {"v": vid},
                ).scalars().all()
                for rel in rows:
                    path = root / str(rel)
                    if path.is_file():
                        try:
                            path.unlink()
                        except OSError:
                            pass

            # 记录将受影响的 G，删视频后清理孤立 G
            affected_g = [
                int(g)
                for g in conn.execute(
                    text(
                        """
                        SELECT DISTINCT vp.global_person_id
                        FROM video_people vp
                        JOIN processing_runs pr ON pr.id = vp.processing_run_id
                        JOIN videos v ON v.id = pr.video_id
                        WHERE v.file_name = :n
                          AND vp.global_person_id IS NOT NULL
                        """
                    ),
                    {"n": name},
                ).scalars()
                if g is not None
            ]

            conn.execute(text("DELETE FROM videos WHERE file_name=:n"), {"n": name})

            orphan_ids: list[int] = []
            if affected_g:
                still = {
                    int(g)
                    for g in conn.execute(
                        text(
                            """
                            SELECT DISTINCT global_person_id
                            FROM video_people
                            WHERE global_person_id IN :ids
                            """
                        ).bindparams(ids=tuple(affected_g))
                    ).scalars()
                    if g is not None
                }
                orphan_ids = [g for g in affected_g if g not in still]
                if orphan_ids:
                    conn.execute(
                        text(
                            "DELETE FROM person_identity_assignments WHERE global_person_id IN :ids"
                        ).bindparams(ids=tuple(orphan_ids))
                    )
                    conn.execute(
                        text(
                            "DELETE FROM person_snapshots WHERE global_person_id IN :ids"
                        ).bindparams(ids=tuple(orphan_ids))
                    )
                    conn.execute(
                        text("DELETE FROM global_people WHERE id IN :ids").bindparams(
                            ids=tuple(orphan_ids)
                        )
                    )

            result = {
                "file_name": name,
                "deleted_videos": video_ids,
                "orphan_global_people": len(orphan_ids),
                "deleted_observations": len(obs_ids),
            }
        result["faiss"] = _drop_faiss_ids(obs_ids, engine)
        return result
    finally:
        engine.dispose()


def purge_processing_keep_video(video_id: int) -> dict[str, Any]:
    """删除某 video_id 的算法结果（processing_runs 级联轨迹），保留 videos 与 rooms。"""
    vid = int(video_id)
    url = _tracking_url()
    if not url:
        return {"skipped": "TRACKING_DB_URL 未配置"}

    root = snapshot_root()
    engine = create_engine(url, pool_pre_ping=True)
    try:
        result: dict[str, Any] | None = None
        obs_ids: list[int] = []
        with engine.begin() as conn:
            exists = conn.execute(
                text("SELECT id FROM videos WHERE id=:id"),
                {"id": vid},
            ).scalar()
            if exists is None:
                return {"video_id": vid, "skipped": "video_not_found"}

            obs_ids = _observation_ids_for_videos(conn, [vid])

            snap_dir = root / f"v{vid}"
            if snap_dir.is_dir():
                shutil.rmtree(snap_dir, ignore_errors=True)
            for rel in conn.execute(
                text("SELECT image_path FROM person_snapshots WHERE video_id=:v"),
                {"v": vid},
            ).scalars():
                path = root / str(rel)
                if path.is_file():
                    try:
                        path.unlink()
                    except OSError:
                        pass

            affected_g = [
                int(g)
                for g in conn.execute(
                    text(
                        """
                        SELECT DISTINCT vp.global_person_id
                        FROM video_people vp
                        JOIN processing_runs pr ON pr.id = vp.processing_run_id
                        WHERE pr.video_id = :vid
                          AND vp.global_person_id IS NOT NULL
                        """
                    ),
                    {"vid": vid},
                ).scalars()
                if g is not None
            ]

            conn.execute(
                text("DELETE FROM processing_runs WHERE video_id=:vid"),
                {"vid": vid},
            )

            orphan_ids: list[int] = []
            if affected_g:
                still = {
                    int(g)
                    for g in conn.execute(
                        text(
                            """
                            SELECT DISTINCT global_person_id
                            FROM video_people
                            WHERE global_person_id IN :ids
                            """
                        ).bindparams(ids=tuple(affected_g))
                    ).scalars()
                    if g is not None
                }
                orphan_ids = [g for g in affected_g if g not in still]
                if orphan_ids:
                    conn.execute(
                        text(
                            "DELETE FROM person_identity_assignments WHERE global_person_id IN :ids"
                        ).bindparams(ids=tuple(orphan_ids))
                    )
                    conn.execute(
                        text(
                            "DELETE FROM person_snapshots WHERE global_person_id IN :ids"
                        ).bindparams(ids=tuple(orphan_ids))
                    )
                    conn.execute(
                        text("DELETE FROM global_people WHERE id IN :ids").bindparams(
                            ids=tuple(orphan_ids)
                        )
                    )

            result = {
                "video_id": vid,
                "kept_video": True,
                "orphan_global_people": len(orphan_ids),
                "deleted_observations": len(obs_ids),
            }
        result["faiss"] = _drop_faiss_ids(obs_ids, engine)
        return result
    finally:
        engine.dispose()
