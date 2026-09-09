"""报告预览的固定请求 / 响应结构。字段名不可更改。"""

from __future__ import annotations

from datetime import date as date_type
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


TimeBasis = Literal["absolute", "video_relative", "mixed"]


class ReportPreviewRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    date: Optional[date_type] = None
    timezone: str = "Asia/Shanghai"
    video_ids: list[int] = Field(default_factory=list)
    camera_ids: list[int] = Field(default_factory=list)
    zone_ids: list[int] = Field(default_factory=list)

    @field_validator("date", mode="before")
    @classmethod
    def _empty_date(cls, value):
        if value is None or value == "":
            return None
        return value

    @field_validator("timezone", mode="before")
    @classmethod
    def _default_timezone(cls, value):
        if value is None or str(value).strip() == "":
            return "Asia/Shanghai"
        return str(value).strip()

    @field_validator("video_ids", "camera_ids", "zone_ids", mode="before")
    @classmethod
    def _empty_list(cls, value):
        if value is None:
            return []
        return value

    @model_validator(mode="after")
    def _date_or_video_ids(self):
        if self.date is None and not self.video_ids:
            raise ValueError("date 为空时 video_ids 至少传一个")
        return self


class VisibleZoneSegment(BaseModel):
    stay_segment_id: int
    video_id: int
    video_name: Optional[str] = None
    start_at: Optional[str] = None
    end_at: Optional[str] = None
    start_video_second: Optional[float] = None
    end_video_second: Optional[float] = None
    visible_duration_seconds: Optional[float] = None


class PersonVisibleZone(BaseModel):
    zone_id: int
    zone_name: Optional[str] = None
    visible_duration_seconds: Optional[float] = None
    segment_count: int = 0
    segments: list[VisibleZoneSegment] = Field(default_factory=list)


class PersonVideo(BaseModel):
    video_id: int
    video_name: Optional[str] = None
    camera_id: Optional[int] = None
    camera_name: Optional[str] = None
    first_seen_at: Optional[str] = None
    last_seen_at: Optional[str] = None
    first_seen_video_second: Optional[float] = None
    last_seen_video_second: Optional[float] = None


class ReportPerson(BaseModel):
    global_person_id: int
    first_seen_at: Optional[str] = None
    last_seen_at: Optional[str] = None
    first_seen_video_second: Optional[float] = None
    last_seen_video_second: Optional[float] = None
    track_count: int = 0
    observation_count: int = 0
    total_visible_duration_seconds: float = 0
    videos: list[PersonVideo] = Field(default_factory=list)
    visible_zones: list[PersonVisibleZone] = Field(default_factory=list)


class ReportScope(BaseModel):
    date: Optional[str] = None
    timezone: str
    time_basis: TimeBasis
    video_ids: list[int] = Field(default_factory=list)
    video_names: list[str] = Field(default_factory=list)
    camera_ids: list[int] = Field(default_factory=list)
    camera_names: list[Optional[str]] = Field(default_factory=list)
    zone_ids: list[int] = Field(default_factory=list)
    zone_names: list[Optional[str]] = Field(default_factory=list)
    generated_at: str


class ReportStatistics(BaseModel):
    unique_person_count: int = 0
    track_count: int = 0
    observation_count: int = 0
    visible_zone_count: int = 0
    total_visible_duration_seconds: float = 0


class ZoneStatistics(BaseModel):
    zone_id: int
    zone_name: Optional[str] = None
    unique_person_count: int = 0
    segment_count: int = 0
    total_visible_duration_seconds: float = 0


class ReportDataQuality(BaseModel):
    processed_video_count: int = 0
    failed_video_count: int = 0
    unassigned_zone_point_count: int = 0
    people_without_zone_count: int = 0
    warnings: list[str] = Field(default_factory=list)


class ReportPreviewResponse(BaseModel):
    schema_version: str = "1.0"
    report_type: str = "daily_activity_report"
    scope: ReportScope
    statistics: ReportStatistics
    zone_statistics: list[ZoneStatistics] = Field(default_factory=list)
    people: list[ReportPerson] = Field(default_factory=list)
    data_quality: ReportDataQuality


class GeneratedReportResponse(BaseModel):
    id: int
    status: str
    report_type: Optional[str] = None
    report_date: Optional[str] = None
    timezone: Optional[str] = None
    title: Optional[str] = None
    report_markdown: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    scope: dict = Field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    error_message: Optional[str] = None
    dify_workflow_run_id: Optional[str] = None
    report_data_json: Optional[dict] = None


class ReportListItem(BaseModel):
    id: int
    status: str
    report_type: Optional[str] = None
    report_date: Optional[str] = None
    timezone: Optional[str] = None
    title: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    scope: dict = Field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    error_message: Optional[str] = None
    dify_workflow_run_id: Optional[str] = None


class ReportListResponse(BaseModel):
    items: list[ReportListItem] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    page_size: int = 20
