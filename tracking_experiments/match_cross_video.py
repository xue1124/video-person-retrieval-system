#!/usr/bin/env python3
"""将新视频中的 video_people 关联到已有 global_people。"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from reid_match import (
    DEFAULT_MAX_SAMPLES,
    DEFAULT_MIN_MATCH_PAIRS,
    DEFAULT_SIMILARITY_THRESHOLD,
    identity_match,
    merge_feature_galleries,
    pack_embedding_matrix,
    select_feature_matrix,
    unpack_embedding_matrix,
)


TARGET_DATABASE = "medical_audit_v3"
SYSTEM_DATABASES = {"mysql", "information_schema", "performance_schema", "sys"}


@dataclass
class PersonRow:
    video_person_id: int
    processing_run_id: int
    video_id: int
    file_name: str
    local_person_no: int
    global_person_id: int
    embeddings: np.ndarray
    sample_count: int
    start_sec: float
    end_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="跨视频人物匹配：把目标 processing_run 的 P 关联到已有 G。"
    )
    parser.add_argument(
        "--db-url",
        default=os.getenv("TRACKING_DB_URL"),
        help="也可通过 TRACKING_DB_URL 提供",
    )
    parser.add_argument(
        "--target-run-id",
        type=int,
        required=True,
        help="待匹配的 processing_run.id，例如后半段视频",
    )
    parser.add_argument(
        "--gallery-run-id",
        type=int,
        help="用作底库的 processing_run.id；默认使用其他所有已完成任务",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="跨视频独立匹配对阈值（单图余弦）",
    )
    parser.add_argument(
        "--reid-max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="每人最多参与匹配的单图特征数",
    )
    parser.add_argument(
        "--reid-min-pairs",
        type=int,
        default=DEFAULT_MIN_MATCH_PAIRS,
        help="至少需要多少组独立匹配对；短轨迹则要求全部样本达标",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只计算匹配结果，不写库",
    )
    return parser.parse_args()


def load_people(conn: Any, run_id: int | None = None, exclude_run_id: int | None = None) -> list[PersonRow]:
    sql = """
        SELECT
          vp.id AS video_person_id,
          vp.processing_run_id,
          pr.video_id,
          v.file_name,
          vp.local_person_no,
          vp.global_person_id,
          vp.representative_embedding,
          vp.embedding_dim,
          vp.profile_updates,
          vp.start_sec,
          vp.end_sec
        FROM video_people vp
        JOIN processing_runs pr ON pr.id = vp.processing_run_id
        JOIN videos v ON v.id = pr.video_id
        WHERE vp.representative_embedding IS NOT NULL
    """
    params: dict[str, Any] = {}
    if run_id is not None:
        sql += " AND vp.processing_run_id = :run_id"
        params["run_id"] = run_id
    if exclude_run_id is not None:
        sql += " AND vp.processing_run_id <> :exclude_run_id"
        params["exclude_run_id"] = exclude_run_id
    sql += " ORDER BY vp.processing_run_id, vp.local_person_no"

    rows: list[PersonRow] = []
    for row in conn.execute(text(sql), params):
        embeddings = unpack_embedding_matrix(
            row.representative_embedding,
            embedding_dim=int(row.embedding_dim) if row.embedding_dim else None,
        )
        if embeddings is None or row.global_person_id is None:
            continue
        rows.append(
            PersonRow(
                video_person_id=int(row.video_person_id),
                processing_run_id=int(row.processing_run_id),
                video_id=int(row.video_id),
                file_name=str(row.file_name),
                local_person_no=int(row.local_person_no),
                global_person_id=int(row.global_person_id),
                embeddings=embeddings,
                sample_count=max(1, int(embeddings.shape[0])),
                start_sec=float(row.start_sec),
                end_sec=float(row.end_sec),
            )
        )
    return rows


def person_time_overlap(first: PersonRow, second: PersonRow) -> bool:
    """同一视频内时间重叠则禁止合并；跨视频默认允许。"""
    if first.video_id != second.video_id:
        return False
    return max(first.start_sec, second.start_sec) <= min(first.end_sec, second.end_sec)


def greedy_match(
    queries: list[PersonRow],
    gallery: list[PersonRow],
    threshold: float,
    *,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    min_pairs: int = DEFAULT_MIN_MATCH_PAIRS,
) -> list[dict[str, Any]]:
    """一对一贪婪匹配：独立匹配对达标后，按最弱达标对从高到低分配。"""
    if not queries or not gallery:
        return []

    query_mats = [
        select_feature_matrix(person.embeddings, max_samples=max_samples)
        for person in queries
    ]
    gallery_mats = [
        select_feature_matrix(person.embeddings, max_samples=max_samples)
        for person in gallery
    ]

    pairs: list[tuple[float, int, int]] = []
    for qi, query in enumerate(queries):
        for gi, gallery_person in enumerate(gallery):
            if person_time_overlap(query, gallery_person):
                continue
            matched, score = identity_match(
                query_mats[qi],
                gallery_mats[gi],
                threshold=threshold,
                min_pairs=min_pairs,
            )
            if matched:
                pairs.append((score, qi, gi))
    pairs.sort(reverse=True)

    used_queries: set[int] = set()
    used_gallery: set[int] = set()
    used_global_ids: set[int] = set()
    matches: list[dict[str, Any]] = []

    for score, qi, gi in pairs:
        if qi in used_queries or gi in used_gallery:
            continue
        query = queries[qi]
        gallery_person = gallery[gi]
        if gallery_person.global_person_id in used_global_ids:
            continue
        used_queries.add(qi)
        used_gallery.add(gi)
        used_global_ids.add(gallery_person.global_person_id)
        matches.append(
            {
                "query_video_person_id": query.video_person_id,
                "query_local_person_no": query.local_person_no,
                "query_file": query.file_name,
                "query_old_global_person_id": query.global_person_id,
                "gallery_video_person_id": gallery_person.video_person_id,
                "gallery_local_person_no": gallery_person.local_person_no,
                "gallery_file": gallery_person.file_name,
                "global_person_id": gallery_person.global_person_id,
                "similarity": round(score, 6),
            }
        )
    return matches


def apply_matches(
    conn: Any,
    matches: list[dict[str, Any]],
    queries: list[PersonRow],
    *,
    max_samples: int = DEFAULT_MAX_SAMPLES,
) -> None:
    query_by_id = {person.video_person_id: person for person in queries}
    for match in matches:
        query = query_by_id[match["query_video_person_id"]]
        target_global_id = int(match["global_person_id"])
        old_global_id = int(match["query_old_global_person_id"])
        if target_global_id == old_global_id:
            continue

        gallery_row = conn.execute(
            text(
                """
                SELECT representative_embedding, embedding_dim, sample_count,
                       first_seen_at, last_seen_at
                FROM global_people
                WHERE id = :id
                """
            ),
            {"id": target_global_id},
        ).one()
        gallery_embeddings = unpack_embedding_matrix(
            gallery_row.representative_embedding,
            embedding_dim=int(gallery_row.embedding_dim)
            if gallery_row.embedding_dim
            else None,
        )
        if gallery_embeddings is None:
            gallery_embeddings = query.embeddings
        merged = merge_feature_galleries(
            gallery_embeddings,
            query.embeddings,
            max_samples=max_samples,
        )
        packed, dim, samples = pack_embedding_matrix(merged)

        conn.execute(
            text(
                """
                UPDATE global_people
                SET representative_embedding = :embedding,
                    embedding_dim = :dim,
                    sample_count = :samples,
                    updated_at = NOW(3)
                WHERE id = :id
                """
            ),
            {
                "embedding": packed,
                "dim": dim,
                "samples": samples,
                "id": target_global_id,
            },
        )
        conn.execute(
            text(
                """
                UPDATE video_people
                SET global_person_id = :global_id,
                    assignment_method = 'auto',
                    assignment_score = :score,
                    is_confirmed = FALSE
                WHERE id = :video_person_id
                """
            ),
            {
                "global_id": target_global_id,
                "score": match["similarity"],
                "video_person_id": query.video_person_id,
            },
        )
        conn.execute(
            text(
                """
                UPDATE observations
                SET global_person_id = :global_id
                WHERE video_person_id = :video_person_id
                """
            ),
            {
                "global_id": target_global_id,
                "video_person_id": query.video_person_id,
            },
        )
        conn.execute(
            text(
                """
                INSERT INTO person_identity_assignments
                  (video_person_id, global_person_id, method, score, note)
                VALUES
                  (:video_person_id, :global_person_id, 'auto', :score, :note)
                """
            ),
            {
                "video_person_id": query.video_person_id,
                "global_person_id": target_global_id,
                "score": match["similarity"],
                "note": (
                    f"跨视频自动匹配：视频内P{query.local_person_no} "
                    f"从 G{old_global_id} 归入 G{target_global_id}"
                ),
            },
        )

        remaining = conn.execute(
            text(
                """
                SELECT COUNT(*) FROM video_people
                WHERE global_person_id = :global_id
                """
            ),
            {"global_id": old_global_id},
        ).scalar_one()
        if int(remaining) == 0:
            conn.execute(
                text(
                    """
                    UPDATE global_people
                    SET status = 'rejected', updated_at = NOW(3)
                    WHERE id = :id
                    """
                ),
                {"id": old_global_id},
            )


def main() -> int:
    args = parse_args()
    if not args.db_url:
        raise ValueError("请通过 --db-url 或 TRACKING_DB_URL 提供新数据库地址")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold 必须在 0 到 1 之间")
    if args.reid_max_samples <= 0:
        raise ValueError("--reid-max-samples 必须大于 0")
    if args.reid_min_pairs <= 0:
        raise ValueError("--reid-min-pairs 必须大于 0")

    url = make_url(args.db_url)
    database = (url.database or "").lower()
    if database in SYSTEM_DATABASES or database != TARGET_DATABASE:
        raise ValueError(
            f"安全拒绝：目标必须是 {TARGET_DATABASE}，当前是 {database or '(空)'}"
        )

    engine = create_engine(args.db_url, pool_pre_ping=True)
    with engine.begin() as conn:
        queries = load_people(conn, run_id=args.target_run_id)
        if args.gallery_run_id is not None:
            gallery = load_people(conn, run_id=args.gallery_run_id)
        else:
            gallery = load_people(conn, exclude_run_id=args.target_run_id)

        if not queries:
            raise ValueError(f"目标任务 {args.target_run_id} 没有可匹配人物")
        if not gallery:
            raise ValueError("底库中没有可匹配人物")

        matches = greedy_match(
            queries,
            gallery,
            args.threshold,
            max_samples=args.reid_max_samples,
            min_pairs=args.reid_min_pairs,
        )
        unmatched = [
            person
            for person in queries
            if person.video_person_id
            not in {match["query_video_person_id"] for match in matches}
        ]

        report = {
            "target_run_id": args.target_run_id,
            "gallery_run_id": args.gallery_run_id,
            "threshold": args.threshold,
            "reid_max_samples": args.reid_max_samples,
            "reid_min_pairs": args.reid_min_pairs,
            "reid_similarity_mode": "pairwise_independent_pairs",
            "query_count": len(queries),
            "gallery_count": len(gallery),
            "match_count": len(matches),
            "unmatched_count": len(unmatched),
            "matches": matches,
            "unmatched": [
                {
                    "video_person_id": person.video_person_id,
                    "local_person_no": person.local_person_no,
                    "file_name": person.file_name,
                    "global_person_id": person.global_person_id,
                }
                for person in unmatched
            ],
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))

        if args.dry_run:
            conn.rollback()
            return 0

        apply_matches(
            conn,
            matches,
            queries,
            max_samples=args.reid_max_samples,
        )
        print(
            f"\n已写入：匹配 {len(matches)} 人，未匹配 {len(unmatched)} 人",
            flush=True,
        )
    engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
