"""以 medical_audit_v3.observation_embeddings 为事实来源的 FAISS 索引。

IndexIDMap2 + IndexFlatIP，id = observation_id。
索引丢失、数量不一致或显式 dirty 时从数据库重建。
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_ROOT = PROJECT_ROOT / "tracking_faiss"
MODALITY_OSNET = "osnet"
MODALITY_SIGLIP = "siglip_image"
ALL_MODALITIES = (MODALITY_OSNET, MODALITY_SIGLIP)

_lock = threading.RLock()
_indexes: dict[str, faiss.Index] = {}
_meta: dict[str, dict[str, Any]] = {}
_MODEL_KEYS = ("model_name", "model_version", "embedding_dim")


def index_root() -> Path:
    raw = os.environ.get("TRACKING_FAISS_ROOT", "").strip()
    path = Path(raw) if raw else DEFAULT_INDEX_ROOT
    path.mkdir(parents=True, exist_ok=True)
    return path.expanduser().resolve()


def _index_path(modality: str) -> Path:
    return index_root() / f"{modality}.index"


def _meta_path(modality: str) -> Path:
    return index_root() / f"{modality}.meta.json"


def _write_meta(modality: str, payload: dict[str, Any]) -> None:
    path = _meta_path(modality)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    _meta[modality] = payload


def _read_meta(modality: str) -> dict[str, Any]:
    path = _meta_path(modality)
    if not path.is_file():
        return {
            "modality": modality,
            "ntotal": 0,
            "dim": 0,
            "db_count": 0,
            "dirty": True,
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            "modality": modality,
            "ntotal": 0,
            "dim": 0,
            "db_count": 0,
            "dirty": True,
        }
    _meta[modality] = data
    return data


def _kept_model_fields(meta: dict[str, Any] | None) -> dict[str, Any]:
    """保留模型名/版本/维度；占位 dim=1 或 embedding_dim=0 不覆盖已有维度。"""
    out: dict[str, Any] = {}
    if not meta:
        return out
    for key in _MODEL_KEYS:
        value = meta.get(key)
        if value in (None, ""):
            continue
        if key == "embedding_dim" and int(value) <= 0:
            continue
        out[key] = value
    dim = meta.get("dim")
    if out.get("embedding_dim") in (None, "", 0) and dim and int(dim) > 1:
        out["embedding_dim"] = int(dim)
    return out


def _fallback_embedding_dim(modality: str, model_name: str | None = None) -> int:
    blob = f"{modality} {model_name or ''}".lower()
    if "osnet" in blob:
        return 512
    if "siglip" in blob:
        return 768
    return 0


def _synced_payload(
    modality: str,
    *,
    ntotal: int,
    dim: int,
    db_count: int,
    dirty: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _kept_model_fields(_read_meta(modality))
    if extra:
        for key in _MODEL_KEYS:
            value = extra.get(key)
            if value in (None, ""):
                continue
            if key == "embedding_dim" and int(value) <= 0:
                continue
            payload[key] = value
        if extra.get("dim") and int(extra["dim"]) > 1 and not payload.get("embedding_dim"):
            payload["embedding_dim"] = int(extra["dim"])
    stored_dim = int(dim or 0)
    embed_dim = int(payload.get("embedding_dim") or 0)
    if embed_dim <= 0:
        embed_dim = _fallback_embedding_dim(modality, str(payload.get("model_name") or ""))
        if embed_dim:
            payload["embedding_dim"] = embed_dim
    if stored_dim <= 1 and embed_dim > 1:
        stored_dim = embed_dim
    elif stored_dim > 1:
        payload["embedding_dim"] = stored_dim
    payload["modality"] = modality
    payload["ntotal"] = int(ntotal)
    payload["dim"] = stored_dim
    payload["db_count"] = int(db_count)
    payload["dirty"] = bool(dirty)
    return payload


def mark_dirty(modality: str | None = None) -> None:
    targets = ALL_MODALITIES if modality is None else (modality,)
    with _lock:
        for mod in targets:
            meta = _read_meta(mod)
            ntotal = int(meta.get("ntotal") or 0)
            try:
                index = _load_index(mod)
                if index is not None:
                    ntotal = int(index.ntotal)
                    meta["dim"] = int(index.d)
            except Exception:
                index = None
            db_count = int(meta.get("db_count") or 0)
            _write_meta(
                mod,
                _synced_payload(
                    mod,
                    ntotal=ntotal,
                    dim=int(meta.get("dim") or 0),
                    db_count=db_count,
                    dirty=True,
                    extra=meta,
                ),
            )
            _indexes.pop(mod, None)


def _new_index(dim: int) -> faiss.Index:
    return faiss.IndexIDMap2(faiss.IndexFlatIP(int(dim)))


def _save_index(
    modality: str,
    index: faiss.Index,
    extra: dict[str, Any] | None = None,
    db_count: int | None = None,
) -> None:
    path = _index_path(modality)
    tmp = path.with_suffix(".tmp")
    faiss.write_index(index, str(tmp))
    tmp.replace(path)
    ntotal = int(index.ntotal)
    if db_count is None and extra is not None and extra.get("db_count") is not None:
        db_count = int(extra["db_count"])
    if db_count is None:
        db_count = ntotal
    payload = _synced_payload(
        modality,
        ntotal=ntotal,
        dim=int(index.d),
        db_count=int(db_count),
        dirty=int(db_count) != ntotal,
        extra=extra,
    )
    _write_meta(modality, payload)
    _indexes[modality] = index


def _load_index(modality: str) -> faiss.Index | None:
    if modality in _indexes:
        return _indexes[modality]
    path = _index_path(modality)
    if not path.is_file():
        return None
    index = faiss.read_index(str(path))
    _indexes[modality] = index
    return index


def _db_embedding_rows(engine: Engine, modality: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT observation_id, embedding, embedding_dim, model_name, model_version
                FROM observation_embeddings
                WHERE modality = :m
                ORDER BY observation_id
                """
            ),
            {"m": modality},
        ).mappings().all()
    if not rows:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 0), dtype=np.float32),
            {"model_name": "", "model_version": ""},
        )
    dim = int(rows[0]["embedding_dim"] or 0)
    ids = np.asarray([int(r["observation_id"]) for r in rows], dtype=np.int64)
    vecs = np.vstack(
        [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
    ).astype(np.float32)
    if dim and vecs.shape[1] != dim:
        raise ValueError(
            f"{modality} 向量维度不一致：blob={vecs.shape[1]} 声明={dim}"
        )
    info = {
        "model_name": str(rows[0].get("model_name") or ""),
        "model_version": str(rows[0].get("model_version") or ""),
        "embedding_dim": int(vecs.shape[1]),
        "db_count": int(len(rows)),
    }
    return ids, vecs, info


def db_embedding_count(engine: Engine, modality: str) -> int:
    with engine.connect() as conn:
        n = conn.execute(
            text(
                "SELECT COUNT(*) FROM observation_embeddings WHERE modality=:m"
            ),
            {"m": modality},
        ).scalar()
    return int(n or 0)


def rebuild_from_db(engine: Engine, modality: str | None = None) -> dict[str, Any]:
    """从 observation_embeddings 全量重建 FAISS。数据库是事实来源。"""
    targets = ALL_MODALITIES if modality is None else (modality,)
    report: dict[str, Any] = {}
    with _lock:
        for mod in targets:
            ids, vecs, info = _db_embedding_rows(engine, mod)
            if vecs.size == 0:
                kept = _kept_model_fields(_read_meta(mod))
                kept.update(_kept_model_fields(info))
                dim = int(kept.get("embedding_dim") or 0)
                if dim <= 1:
                    dim = _fallback_embedding_dim(mod, str(kept.get("model_name") or ""))
                    if dim:
                        kept["embedding_dim"] = dim
                index = _new_index(dim or 1)
                extra = {**kept, "db_count": 0}
                _save_index(mod, index, extra, db_count=0)
                report[mod] = {"ntotal": 0, "db_count": 0, "dirty": False, "rebuilt": True, **kept}
                continue
            index = _new_index(int(vecs.shape[1]))
            index.add_with_ids(np.ascontiguousarray(vecs), ids)
            extra = {**info, "db_count": int(info.get("db_count") or len(ids))}
            _save_index(mod, index, extra, db_count=int(extra["db_count"]))
            report[mod] = {
                "ntotal": int(index.ntotal),
                "db_count": int(extra["db_count"]),
                "dirty": False,
                "rebuilt": True,
                **info,
            }
    return report


def _index_needs_rebuild(engine: Engine, modality: str) -> bool:
    meta = _read_meta(modality)
    if meta.get("dirty"):
        return True
    if not _index_path(modality).is_file():
        return True
    try:
        index = _load_index(modality)
    except Exception:
        return True
    if index is None:
        return True
    db_n = db_embedding_count(engine, modality)
    if int(index.ntotal) != db_n:
        return True
    if int(meta.get("ntotal", -1)) != int(index.ntotal):
        return True
    if int(meta.get("db_count", -1)) != db_n:
        return True
    return False


def ensure_index(engine: Engine, modality: str) -> faiss.Index:
    with _lock:
        if _index_needs_rebuild(engine, modality):
            rebuild_from_db(engine, modality)
        index = _load_index(modality)
        if index is None:
            rebuild_from_db(engine, modality)
            index = _load_index(modality)
        if index is None:
            raise RuntimeError(f"无法加载 {modality} FAISS 索引")
        return index


def add_observation_vectors(
    engine: Engine,
    modality: str,
    observation_ids: list[int],
    vectors: np.ndarray,
) -> dict[str, Any]:
    if not observation_ids:
        return {"added": 0, "modality": modality}
    vecs = np.asarray(vectors, dtype=np.float32)
    if vecs.ndim == 1:
        vecs = vecs.reshape(1, -1)
    if vecs.shape[0] != len(observation_ids):
        raise ValueError("observation_ids 与向量行数不一致")
    ids = np.asarray(observation_ids, dtype=np.int64)
    with _lock:
        if _index_needs_rebuild(engine, modality):
            rebuild_from_db(engine, modality)
            return {"added": len(observation_ids), "modality": modality, "via": "rebuild"}
        index = _load_index(modality)
        if index is None or index.ntotal == 0:
            rebuild_from_db(engine, modality)
            return {"added": len(observation_ids), "modality": modality, "via": "rebuild"}
        if int(index.d) != int(vecs.shape[1]):
            mark_dirty(modality)
            rebuild_from_db(engine, modality)
            return {"added": len(observation_ids), "modality": modality, "via": "rebuild_dim"}
        index.add_with_ids(np.ascontiguousarray(vecs), ids)
        extra = dict(_read_meta(modality))
        extra.pop("dirty", None)
        extra.pop("ntotal", None)
        extra.pop("db_count", None)
        _save_index(modality, index, extra, db_count=int(index.ntotal))
        return {"added": int(len(observation_ids)), "modality": modality, "ntotal": int(index.ntotal)}


def remove_observation_ids(
    observation_ids: Iterable[int],
    engine: Engine | None = None,
) -> dict[str, Any]:
    ids = [int(x) for x in observation_ids if int(x) > 0]
    report: dict[str, Any] = {}
    with _lock:
        selector = (
            faiss.IDSelectorBatch(np.asarray(ids, dtype=np.int64)) if ids else None
        )
        for mod in ALL_MODALITIES:
            index = _load_index(mod)
            db_count = (
                db_embedding_count(engine, mod)
                if engine is not None
                else None
            )
            if index is None:
                if engine is not None:
                    _write_meta(
                        mod,
                        _synced_payload(
                            mod,
                            ntotal=0,
                            dim=int(_read_meta(mod).get("dim") or 0),
                            db_count=int(db_count or 0),
                            dirty=int(db_count or 0) != 0,
                        ),
                    )
                else:
                    mark_dirty(mod)
                report[mod] = {"removed": 0, "dirty": True, "ntotal": 0, "db_count": db_count}
                continue
            before = int(index.ntotal)
            if selector is not None:
                try:
                    index.remove_ids(selector)
                except Exception as exc:
                    dbn = int(db_count) if db_count is not None else before
                    _save_index(mod, index, db_count=dbn)
                    # _save_index sets dirty from mismatch; force dirty on remove error
                    meta = _read_meta(mod)
                    _write_meta(
                        mod,
                        _synced_payload(
                            mod,
                            ntotal=int(index.ntotal),
                            dim=int(index.d),
                            db_count=dbn,
                            dirty=True,
                            extra=meta,
                        ),
                    )
                    report[mod] = {
                        "removed": 0,
                        "error": str(exc),
                        "dirty": True,
                        "ntotal": int(index.ntotal),
                        "db_count": dbn,
                    }
                    continue
            extra = _kept_model_fields(_read_meta(mod))
            dbn = int(db_count) if db_count is not None else int(index.ntotal)
            _save_index(mod, index, extra, db_count=dbn)
            meta = _read_meta(mod)
            report[mod] = {
                "removed": before - int(index.ntotal),
                "ntotal": int(meta["ntotal"]),
                "db_count": int(meta["db_count"]),
                "dirty": bool(meta["dirty"]),
            }
    return report


def add_video_observations(engine: Engine, video_id: int) -> dict[str, Any]:
    """导入完成后从数据库重建索引，避免增量 add 与库内新行重复。"""
    _ = video_id
    return rebuild_from_db(engine)


def sync_metadata(engine: Engine | None = None) -> dict[str, Any]:
    """按当前索引与数据库回写 ntotal / db_count / dirty；保留模型名、版本、维度。"""
    report: dict[str, Any] = {}
    with _lock:
        for mod in ALL_MODALITIES:
            meta = _read_meta(mod)
            try:
                index = _load_index(mod)
            except Exception:
                index = None
            ntotal = int(index.ntotal) if index is not None else 0
            dim = int(index.d) if index is not None else int(meta.get("dim") or 0)
            if engine is not None:
                db_count = db_embedding_count(engine, mod)
            elif meta.get("db_count") is not None:
                db_count = int(meta["db_count"])
            else:
                db_count = ntotal
            dirty = index is None or int(db_count) != ntotal
            payload = _synced_payload(
                mod,
                ntotal=ntotal,
                dim=dim,
                db_count=db_count,
                dirty=dirty,
                extra=meta,
            )
            _write_meta(mod, payload)
            report[mod] = {
                "ntotal": payload["ntotal"],
                "db_count": payload["db_count"],
                "dirty": payload["dirty"],
                "dim": payload.get("dim"),
                "model_name": payload.get("model_name"),
                "model_version": payload.get("model_version"),
                "embedding_dim": payload.get("embedding_dim"),
            }
    return report


def reconcile_index(engine: Engine) -> dict[str, Any]:
    """删除/清空后：向量与库不一致则重建，否则只同步 metadata。"""
    needs_rebuild = False
    with _lock:
        for mod in ALL_MODALITIES:
            db_n = db_embedding_count(engine, mod)
            try:
                index = _load_index(mod)
            except Exception:
                index = None
            ntotal = int(index.ntotal) if index is not None else -1
            if ntotal != db_n:
                needs_rebuild = True
                break
    if needs_rebuild:
        return {"action": "rebuild", **rebuild_from_db(engine)}
    return {"action": "sync", **sync_metadata(engine)}


def observation_ids_for_video(engine: Engine, video_id: int) -> list[int]:
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT id FROM observations WHERE video_id=:v ORDER BY id"),
            {"v": int(video_id)},
        ).scalars().all()
    return [int(x) for x in rows]


def search(
    engine: Engine,
    modality: str,
    query: np.ndarray,
    *,
    candidate_k: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (scores, observation_ids)，长度 <= candidate_k。"""
    feat = np.asarray(query, dtype=np.float32).reshape(1, -1)
    feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
    index = ensure_index(engine, modality)
    if index.ntotal == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    if int(feat.shape[1]) != int(index.d):
        raise ValueError(
            f"维度不匹配：查询特征 {feat.shape[1]} 维，底库 {index.d} 维。"
            "请切换与建模一致的算法。"
        )
    k = min(int(candidate_k), int(index.ntotal))
    scores, ids = index.search(np.ascontiguousarray(feat), k)
    return scores[0], ids[0]


def load_embeddings_by_ids(
    engine: Engine, observation_ids: list[int], modality: str
) -> dict[int, np.ndarray]:
    if not observation_ids:
        return {}
    stmt = text(
        """
        SELECT observation_id, embedding
        FROM observation_embeddings
        WHERE modality = :m AND observation_id IN :ids
        """
    ).bindparams(bindparam("ids", expanding=True))
    with engine.connect() as conn:
        rows = conn.execute(
            stmt, {"m": modality, "ids": [int(x) for x in observation_ids]}
        ).mappings().all()
    out: dict[int, np.ndarray] = {}
    for r in rows:
        out[int(r["observation_id"])] = np.frombuffer(
            r["embedding"], dtype=np.float32
        ).copy()
    return out


def hydrate_observations(engine: Engine, observation_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not observation_ids:
        return {}
    uniq = sorted({int(x) for x in observation_ids if int(x) > 0})
    if not uniq:
        return {}
    stmt = text(
        """
        SELECT
          o.id,
          o.video_id,
          o.processing_run_id,
          o.video_person_id,
          o.global_person_id,
          o.track_id,
          o.timestamp_sec,
          o.frame_index,
          o.bbox_x1, o.bbox_y1, o.bbox_x2, o.bbox_y2,
          o.room_id,
          o.crop_path,
          v.file_name AS video_name,
          r.name AS room_name
        FROM observations o
        JOIN videos v ON v.id = o.video_id
        LEFT JOIN rooms r ON r.id = o.room_id
        WHERE o.id IN :ids
        """
    ).bindparams(bindparam("ids", expanding=True))
    with engine.connect() as conn:
        rows = conn.execute(stmt, {"ids": uniq}).mappings().all()
    out: dict[int, dict[str, Any]] = {}
    for r in rows:
        bbox = None
        if all(r.get(k) is not None for k in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")):
            bbox = [
                int(float(r["bbox_x1"])),
                int(float(r["bbox_y1"])),
                int(float(r["bbox_x2"])),
                int(float(r["bbox_y2"])),
            ]
        oid = int(r["id"])
        out[oid] = {
            "id": oid,
            "video_id": int(r["video_id"]),
            "processing_run_id": int(r["processing_run_id"]),
            "video_person_id": int(r["video_person_id"]),
            "global_person_id": (
                int(r["global_person_id"]) if r["global_person_id"] is not None else None
            ),
            "track_id": int(r["track_id"]),
            "timestamp_sec": float(r["timestamp_sec"]),
            "frame_index": int(r["frame_index"]),
            "bbox": bbox,
            "room_id": int(r["room_id"]) if r["room_id"] is not None else None,
            "room_name": (str(r["room_name"]).strip() if r.get("room_name") else ""),
            "crop_path": str(r["crop_path"] or ""),
            "video_name": str(r["video_name"]),
        }
    return out


def index_status(engine: Engine | None = None) -> dict[str, Any]:
    if engine is not None:
        sync_metadata(engine)
    out: dict[str, Any] = {}
    for mod in ALL_MODALITIES:
        meta = _read_meta(mod)
        ntotal = None
        try:
            index = _load_index(mod)
            ntotal = int(index.ntotal) if index is not None else None
        except Exception as exc:
            meta = {**meta, "load_error": str(exc)}
        row = {"meta": meta, "ntotal": ntotal, "path": str(_index_path(mod))}
        db_count = meta.get("db_count")
        if engine is not None:
            db_count = db_embedding_count(engine, mod)
            row["db_count"] = db_count
            row["consistent"] = (
                ntotal is not None
                and int(ntotal) == int(db_count)
                and int(meta.get("ntotal", -1)) == int(ntotal)
                and int(meta.get("db_count", -1)) == int(db_count)
                and not meta.get("dirty")
            )
        elif db_count is not None:
            row["db_count"] = db_count
            row["consistent"] = (
                ntotal is not None
                and int(meta.get("ntotal", -1)) == int(ntotal)
                and not meta.get("dirty")
            )
        out[mod] = row
    return out
