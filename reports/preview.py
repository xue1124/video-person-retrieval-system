"""从 medical_audit_v3 批量读取并组装日报预览 JSON。只读，不写库。"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Iterable, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

from reports.models import (
    PersonVideo,
    PersonVisibleZone,
    ReportDataQuality,
    ReportPerson,
    ReportPreviewRequest,
    ReportPreviewResponse,
    ReportScope,
    ReportStatistics,
    TimeBasis,
    VisibleZoneSegment,
    ZoneStatistics,
)

SCHEMA_VERSION = "1.0"
REPORT_TYPE = "daily_activity_report"
VISIBLE_DURATION_WARNING = (
    "区域时长仅表示人物在摄像头画面内的可见时长，不表示诊室内部停留时间"
)
MISSING_ZONES_WARNING = "该范围缺少可见区域信息"


def build_report_preview(
    engine: Engine, request: ReportPreviewRequest
) -> ReportPreviewResponse:
    tz = resolve_timezone(request.timezone)
    generated_at = _to_iso(datetime.now(tz))

    with engine.connect() as conn:
        videos = _fetch_scoped_videos(conn, request, tz)
        camera_names = _lookup_names(
            conn,
            "SELECT id, name FROM cameras WHERE id IN :ids",
            "ids",
            request.camera_ids,
        )
        zone_names = _lookup_names(
            conn,
            "SELECT id, name FROM rooms WHERE id IN :ids",
            "ids",
            request.zone_ids,
        )
        if not videos:
            return _empty_response(
                request,
                tz,
                generated_at,
                camera_names=camera_names,
                zone_names=zone_names,
            )

        video_ids = [int(v["id"]) for v in videos]
        camera_ids_present = [
            int(v["camera_id"]) for v in videos if v.get("camera_id") is not None
        ]
        facts = _fetch_facts(conn, video_ids, camera_ids_present)

    return _assemble(
        request=request,
        tz=tz,
        generated_at=generated_at,
        videos=videos,
        facts=facts,
        camera_names=camera_names,
        zone_names=zone_names,
    )


def resolve_timezone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"无效的 timezone: {name}") from exc


def _empty_response(
    request: ReportPreviewRequest,
    tz: ZoneInfo,
    generated_at: str,
    *,
    camera_names: list[Optional[str]],
    zone_names: list[Optional[str]],
) -> ReportPreviewResponse:
    return ReportPreviewResponse(
        schema_version=SCHEMA_VERSION,
        report_type=REPORT_TYPE,
        scope=_build_scope(
            request,
            tz,
            generated_at,
            [],
            "absolute",
            camera_names,
            zone_names,
        ),
        statistics=ReportStatistics(),
        zone_statistics=[],
        people=[],
        data_quality=ReportDataQuality(
            processed_video_count=0,
            failed_video_count=0,
            unassigned_zone_point_count=0,
            people_without_zone_count=0,
            warnings=_base_warnings([]),
        ),
    )


def _assemble(
    *,
    request: ReportPreviewRequest,
    tz: ZoneInfo,
    generated_at: str,
    videos: list[dict[str, Any]],
    facts: dict[str, Any],
    camera_names: list[Optional[str]],
    zone_names: list[Optional[str]],
) -> ReportPreviewResponse:
    video_by_id = {int(v["id"]): v for v in videos}
    zone_filter = set(int(z) for z in request.zone_ids) if request.zone_ids else None

    observations = list(facts["observations"])
    if zone_filter is not None:
        observations = [
            row
            for row in observations
            if row.get("room_id") is not None and int(row["room_id"]) in zone_filter
        ]

    people_ids: set[int] = set()
    for row in observations:
        gid = row.get("global_person_id")
        if gid is not None:
            people_ids.add(int(gid))

    stay_rows = [
        row
        for row in facts["stay_segments"]
        if row.get("room_id") is not None
        and (zone_filter is None or int(row["room_id"]) in zone_filter)
    ]
    if zone_filter is not None:
        for row in stay_rows:
            gid = row.get("global_person_id")
            if gid is not None:
                people_ids.add(int(gid))

    obs_by_person: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        gid = row.get("global_person_id")
        if gid is None:
            continue
        gid = int(gid)
        if gid in people_ids:
            obs_by_person[gid].append(row)

    stays_by_person: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in stay_rows:
        gid = row.get("global_person_id")
        if gid is None:
            continue
        gid = int(gid)
        if gid in people_ids:
            stays_by_person[gid].append(row)

    people: list[ReportPerson] = []
    used_zone_ids: set[int] = set()
    total_visible = 0.0
    for gid in people_ids:
        person = _build_person(
            gid,
            obs_by_person.get(gid, []),
            stays_by_person.get(gid, []),
            video_by_id,
            tz,
        )
        people.append(person)
        total_visible += float(person.total_visible_duration_seconds or 0)
        for zone in person.visible_zones:
            used_zone_ids.add(int(zone.zone_id))

    people.sort(key=_person_sort_key)
    zone_statistics = _build_zone_statistics(stays_by_person)

    time_basis = _time_basis(videos)
    extra_warnings: list[str] = []
    if time_basis != "absolute":
        extra_warnings.append(_relative_time_warning(videos))
    rooms_in_scope = int(facts["room_count"] or 0)
    if rooms_in_scope == 0:
        extra_warnings.append(MISSING_ZONES_WARNING)

    people_without_zone = sum(1 for p in people if not p.visible_zones)
    track_ids: set[int] = set()
    observation_count = 0
    for rows in obs_by_person.values():
        observation_count += len(rows)
        for row in rows:
            if row.get("track_id") is not None:
                track_ids.add(int(row["track_id"]))

    return ReportPreviewResponse(
        schema_version=SCHEMA_VERSION,
        report_type=REPORT_TYPE,
        scope=_build_scope(
            request, tz, generated_at, videos, time_basis, camera_names, zone_names
        ),
        statistics=ReportStatistics(
            unique_person_count=len(people),
            track_count=len(track_ids),
            observation_count=observation_count,
            visible_zone_count=len(used_zone_ids),
            total_visible_duration_seconds=_json_sec(total_visible) or 0,
        ),
        zone_statistics=zone_statistics,
        people=people,
        data_quality=ReportDataQuality(
            processed_video_count=sum(1 for v in videos if v.get("status") == "completed"),
            failed_video_count=sum(1 for v in videos if v.get("status") == "failed"),
            unassigned_zone_point_count=int(facts["unassigned_zone_point_count"] or 0),
            people_without_zone_count=people_without_zone,
            warnings=_base_warnings(extra_warnings),
        ),
    )


def _build_zone_statistics(
    stays_by_person: dict[int, list[dict[str, Any]]],
) -> list[ZoneStatistics]:
    buckets: dict[int, dict[str, Any]] = {}
    for gid, stays in stays_by_person.items():
        for row in stays:
            if row.get("room_id") is None:
                continue
            zone_id = int(row["room_id"])
            bucket = buckets.get(zone_id)
            if bucket is None:
                bucket = {
                    "zone_id": zone_id,
                    "zone_name": row.get("zone_name"),
                    "people": set(),
                    "segment_count": 0,
                    "duration": 0.0,
                }
                buckets[zone_id] = bucket
            if bucket["zone_name"] is None:
                bucket["zone_name"] = row.get("zone_name")
            bucket["people"].add(int(gid))
            bucket["segment_count"] += 1
            bucket["duration"] += float(row.get("duration_sec") or 0)
    result: list[ZoneStatistics] = []
    for zone_id in sorted(buckets):
        bucket = buckets[zone_id]
        result.append(
            ZoneStatistics(
                zone_id=zone_id,
                zone_name=bucket["zone_name"],
                unique_person_count=len(bucket["people"]),
                segment_count=int(bucket["segment_count"]),
                total_visible_duration_seconds=_json_sec(bucket["duration"]) or 0,
            )
        )
    return result


def _build_person(
    gid: int,
    observations: list[dict[str, Any]],
    stays: list[dict[str, Any]],
    video_by_id: dict[int, dict[str, Any]],
    tz: ZoneInfo,
) -> ReportPerson:
    first_obs, last_obs = _first_last_observation(observations, video_by_id)
    first_stay, last_stay = _first_last_from_stays(stays) if stays else (None, None)
    if first_obs is None:
        first_obs, last_obs = first_stay, last_stay

    first_video = (
        video_by_id.get(int(first_obs["video_id"])) if first_obs is not None else None
    )
    last_video = (
        video_by_id.get(int(last_obs["video_id"])) if last_obs is not None else None
    )
    track_ids = {
        int(row["track_id"]) for row in observations if row.get("track_id") is not None
    }

    videos_payload = _build_person_videos(observations, stays, video_by_id, tz)
    zones_payload, total_visible = _build_visible_zones(stays, video_by_id, tz)

    return ReportPerson(
        global_person_id=gid,
        first_seen_at=_obs_absolute(first_obs, first_video, tz),
        last_seen_at=_obs_absolute(last_obs, last_video, tz),
        first_seen_video_second=_json_sec(
            first_obs["timestamp_sec"] if first_obs is not None else None
        ),
        last_seen_video_second=_json_sec(
            last_obs["timestamp_sec"] if last_obs is not None else None
        ),
        track_count=len(track_ids),
        observation_count=len(observations),
        total_visible_duration_seconds=_json_sec(total_visible) or 0,
        videos=videos_payload,
        visible_zones=zones_payload,
    )


def _build_person_videos(
    observations: list[dict[str, Any]],
    stays: list[dict[str, Any]],
    video_by_id: dict[int, dict[str, Any]],
    tz: ZoneInfo,
) -> list[PersonVideo]:
    by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        by_video[int(row["video_id"])].append(row)
    if not by_video:
        for row in stays:
            by_video[int(row["video_id"])].append(
                {
                    "id": row.get("id"),
                    "video_id": row["video_id"],
                    "timestamp_sec": row.get("start_sec"),
                    "end_sec": row.get("end_sec"),
                }
            )
    result: list[PersonVideo] = []
    for video_id in sorted(by_video):
        rows = by_video[video_id]
        video = video_by_id.get(video_id) or {}
        first_obs, last_obs = _first_last_observation(rows, video_by_id)
        result.append(
            PersonVideo(
                video_id=video_id,
                video_name=video.get("file_name"),
                camera_id=_opt_int(video.get("camera_id")),
                camera_name=video.get("camera_name"),
                first_seen_at=_obs_absolute(first_obs, video, tz),
                last_seen_at=_obs_absolute(last_obs, video, tz),
                first_seen_video_second=_json_sec(
                    first_obs["timestamp_sec"] if first_obs is not None else None
                ),
                last_seen_video_second=_json_sec(
                    last_obs.get("end_sec", last_obs.get("timestamp_sec"))
                    if last_obs is not None
                    else None
                ),
            )
        )
    return result


def _build_visible_zones(
    stays: list[dict[str, Any]],
    video_by_id: dict[int, dict[str, Any]],
    tz: ZoneInfo,
) -> tuple[list[PersonVisibleZone], float]:
    by_zone: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in stays:
        if row.get("room_id") is None:
            continue
        by_zone[int(row["room_id"])].append(row)

    zones: list[PersonVisibleZone] = []
    total = 0.0
    for zone_id in sorted(by_zone):
        rows = sorted(
            by_zone[zone_id],
            key=lambda r: (
                float(r.get("start_sec") or 0),
                int(r.get("id") or 0),
            ),
        )
        segments: list[VisibleZoneSegment] = []
        zone_duration = 0.0
        zone_name = None
        for row in rows:
            video_id = int(row["video_id"])
            video = video_by_id.get(video_id) or {}
            duration = float(row.get("duration_sec") or 0)
            zone_duration += duration
            zone_name = row.get("zone_name") if zone_name is None else zone_name
            start_sec = _json_sec(row.get("start_sec"))
            end_sec = _json_sec(row.get("end_sec"))
            segments.append(
                VisibleZoneSegment(
                    stay_segment_id=int(row["id"]),
                    video_id=video_id,
                    video_name=video.get("file_name"),
                    start_at=_absolute_iso(video.get("captured_at"), start_sec, tz),
                    end_at=_absolute_iso(video.get("captured_at"), end_sec, tz),
                    start_video_second=start_sec,
                    end_video_second=end_sec,
                    visible_duration_seconds=_json_sec(duration),
                )
            )
        total += zone_duration
        zones.append(
            PersonVisibleZone(
                zone_id=zone_id,
                zone_name=zone_name,
                visible_duration_seconds=_json_sec(zone_duration),
                segment_count=len(segments),
                segments=segments,
            )
        )
    return zones, total


def _first_last_observation(
    observations: list[dict[str, Any]],
    video_by_id: dict[int, dict[str, Any]],
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    if not observations:
        return None, None

    def sort_key(row: dict[str, Any]):
        video = video_by_id.get(int(row["video_id"])) or {}
        captured = _as_utc_naive(video.get("captured_at"))
        ts = float(row.get("timestamp_sec") or 0)
        abs_ts = None
        if captured is not None:
            abs_ts = captured + timedelta(seconds=ts)
        return (
            0 if abs_ts is not None else 1,
            abs_ts or datetime.min,
            int(row["video_id"]),
            ts,
            int(row.get("id") or 0),
        )

    ordered = sorted(observations, key=sort_key)
    return ordered[0], ordered[-1]


def _first_last_from_stays(
    stays: list[dict[str, Any]],
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    if not stays:
        return None, None
    first = min(
        stays,
        key=lambda r: (
            int(r["video_id"]),
            float(r.get("start_sec") or 0),
            int(r.get("id") or 0),
        ),
    )
    last = max(
        stays,
        key=lambda r: (
            int(r["video_id"]),
            float(r.get("end_sec") or 0),
            int(r.get("id") or 0),
        ),
    )
    return (
        {
            "id": first.get("id"),
            "video_id": first["video_id"],
            "timestamp_sec": first.get("start_sec"),
        },
        {
            "id": last.get("id"),
            "video_id": last["video_id"],
            "timestamp_sec": last.get("end_sec"),
        },
    )


def _person_sort_key(person: ReportPerson):
    first_video_id = person.videos[0].video_id if person.videos else 0
    return (
        0 if person.first_seen_at is not None else 1,
        person.first_seen_at or "",
        first_video_id,
        person.first_seen_video_second if person.first_seen_video_second is not None else 0,
        person.global_person_id,
    )


def _obs_absolute(
    obs: Optional[dict[str, Any]],
    video: Optional[dict[str, Any]],
    tz: ZoneInfo,
) -> Optional[str]:
    if obs is None or not video:
        return None
    return _absolute_iso(video.get("captured_at"), obs.get("timestamp_sec"), tz)


def _fetch_scoped_videos(
    conn: Any, request: ReportPreviewRequest, tz: ZoneInfo
) -> list[dict[str, Any]]:
    sql = """
        SELECT
          v.id,
          v.file_name,
          v.camera_id,
          v.captured_at,
          v.status,
          c.name AS camera_name
        FROM videos v
        LEFT JOIN cameras c ON c.id = v.camera_id
        WHERE 1=1
    """
    params: dict[str, Any] = {}
    expanding: list[str] = []
    if request.video_ids:
        sql += " AND v.id IN :video_ids"
        params["video_ids"] = [int(v) for v in request.video_ids]
        expanding.append("video_ids")
    if request.camera_ids:
        sql += " AND v.camera_id IN :camera_ids"
        params["camera_ids"] = [int(v) for v in request.camera_ids]
        expanding.append("camera_ids")
    if request.date is not None:
        start_utc, end_utc = _utc_day_bounds(request.date, tz)
        params["day_start"] = start_utc
        params["day_end"] = end_utc
        if request.video_ids:
            sql += """
                AND (
                  (v.captured_at IS NOT NULL AND v.captured_at >= :day_start AND v.captured_at < :day_end)
                  OR (v.captured_at IS NULL AND v.id IN :video_ids)
                )
            """
        else:
            sql += """
                AND v.captured_at IS NOT NULL
                AND v.captured_at >= :day_start
                AND v.captured_at < :day_end
            """
    sql += " ORDER BY v.id"
    rows = conn.execute(_stmt(sql, expanding), params).mappings().all()
    return [dict(row) for row in rows]


def _fetch_facts(
    conn: Any, video_ids: list[int], camera_ids: list[int]
) -> dict[str, Any]:
    params: dict[str, Any] = {"video_ids": video_ids}
    expanding = ["video_ids"]
    observations = [
        dict(row)
        for row in conn.execute(
            _stmt(
                """
                SELECT
                  o.id,
                  o.video_id,
                  o.global_person_id,
                  o.track_id,
                  o.video_person_id,
                  o.timestamp_sec,
                  o.room_id
                FROM observations o
                WHERE o.video_id IN :video_ids
                  AND o.global_person_id IS NOT NULL
                """,
                expanding,
            ),
            params,
        )
        .mappings()
        .all()
    ]
    stay_segments = [
        dict(row)
        for row in conn.execute(
            _stmt(
                """
                SELECT
                  ss.id,
                  ss.video_person_id,
                  ss.room_id,
                  ss.start_sec,
                  ss.end_sec,
                  ss.duration_sec,
                  r.name AS zone_name,
                  pr.video_id,
                  vp.global_person_id
                FROM stay_segments ss
                JOIN video_people vp ON vp.id = ss.video_person_id
                JOIN processing_runs pr ON pr.id = vp.processing_run_id
                LEFT JOIN rooms r ON r.id = ss.room_id
                WHERE pr.video_id IN :video_ids
                """,
                expanding,
            ),
            params,
        )
        .mappings()
        .all()
    ]
    unassigned = conn.execute(
        _stmt(
            """
            SELECT COUNT(*) AS n
            FROM track_points tp
            JOIN tracks t ON t.id = tp.track_id
            JOIN processing_runs pr ON pr.id = t.processing_run_id
            WHERE pr.video_id IN :video_ids
              AND tp.room_id IS NULL
            """,
            expanding,
        ),
        params,
    ).scalar()
    room_sql = "SELECT COUNT(*) AS n FROM rooms WHERE video_id IN :video_ids"
    room_params = dict(params)
    room_expanding = list(expanding)
    if camera_ids:
        room_sql = """
            SELECT COUNT(*) AS n
            FROM rooms
            WHERE video_id IN :video_ids
               OR camera_id IN :camera_ids
        """
        room_params["camera_ids"] = camera_ids
        room_expanding.append("camera_ids")
    room_count = conn.execute(_stmt(room_sql, room_expanding), room_params).scalar()
    return {
        "observations": observations,
        "stay_segments": stay_segments,
        "unassigned_zone_point_count": int(unassigned or 0),
        "room_count": int(room_count or 0),
    }


def _lookup_names(
    conn: Any, sql: str, param: str, ids: Iterable[int]
) -> list[Optional[str]]:
    id_list = [int(i) for i in ids]
    if not id_list:
        return []
    rows = conn.execute(_stmt(sql, [param]), {param: id_list}).mappings().all()
    by_id = {int(row["id"]): row.get("name") for row in rows}
    return [by_id.get(i) for i in id_list]


def _build_scope(
    request: ReportPreviewRequest,
    tz: ZoneInfo,
    generated_at: str,
    videos: list[dict[str, Any]],
    time_basis: TimeBasis,
    camera_names: list[Optional[str]],
    zone_names: list[Optional[str]],
) -> ReportScope:
    return ReportScope(
        date=request.date.isoformat() if request.date is not None else None,
        timezone=str(tz),
        time_basis=time_basis,
        video_ids=[int(v["id"]) for v in videos],
        video_names=[v.get("file_name") or "" for v in videos],
        camera_ids=list(request.camera_ids),
        camera_names=camera_names,
        zone_ids=list(request.zone_ids),
        zone_names=zone_names,
        generated_at=generated_at,
    )


def _time_basis(videos: list[dict[str, Any]]) -> TimeBasis:
    if not videos:
        return "absolute"
    has_abs = any(_as_utc_naive(v.get("captured_at")) is not None for v in videos)
    has_rel = any(_as_utc_naive(v.get("captured_at")) is None for v in videos)
    if has_abs and has_rel:
        return "mixed"
    if has_rel:
        return "video_relative"
    return "absolute"


def _relative_time_warning(videos: list[dict[str, Any]]) -> str:
    names = [
        f"{v.get('file_name') or ''}(id={int(v['id'])})".strip()
        for v in videos
        if _as_utc_naive(v.get("captured_at")) is None
    ]
    joined = "、".join(names) if names else "所选视频"
    return f"{joined} 缺少拍摄时间 captured_at，只能提供视频相对时间"


def _base_warnings(extra: list[str]) -> list[str]:
    warnings = [VISIBLE_DURATION_WARNING]
    for item in extra:
        if item and item not in warnings:
            warnings.append(item)
    return warnings


def _utc_day_bounds(day: date, tz: ZoneInfo) -> tuple[datetime, datetime]:
    start_local = datetime.combine(day, time.min, tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    start_utc = start_local.astimezone(timezone.utc).replace(tzinfo=None)
    end_utc = end_local.astimezone(timezone.utc).replace(tzinfo=None)
    return start_utc, end_utc


def _absolute_iso(
    captured_at: Any, video_second: Any, tz: ZoneInfo
) -> Optional[str]:
    captured = _as_utc_naive(captured_at)
    if captured is None or video_second is None:
        return None
    instant = captured.replace(tzinfo=timezone.utc) + timedelta(
        seconds=float(video_second)
    )
    return _to_iso(instant.astimezone(tz))


def _as_utc_naive(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    return None


def _to_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    micro = dt.microsecond
    if micro == 0:
        return dt.isoformat()
    millisecond = (micro + 500) // 1000
    if millisecond >= 1000:
        dt = dt.replace(microsecond=0) + timedelta(seconds=1)
        return dt.isoformat()
    return dt.replace(microsecond=millisecond * 1000).isoformat(timespec="milliseconds")


def _json_sec(value: Any) -> Optional[float]:
    if value is None:
        return None
    rounded = round(float(value), 3)
    if rounded == int(rounded):
        return int(rounded)
    return rounded


def _opt_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _stmt(sql: str, expanding: Optional[list[str]] = None):
    stmt = text(sql)
    for name in expanding or []:
        stmt = stmt.bindparams(bindparam(name, expanding=True))
    return stmt
