import http from '@/api/http'
import { apiErrorMessage } from '@/utils/apiError'

export type TimeBasis = 'absolute' | 'video_relative' | 'mixed'
export type ReportStatus = 'generating' | 'completed' | 'failed'
export type ReportMode = 'date' | 'video'

export interface ReportPreviewRequest {
  date?: string | null
  timezone?: string
  video_ids?: number[]
  camera_ids?: number[]
  zone_ids?: number[]
}

export interface ReportScope {
  date: string | null
  timezone: string
  time_basis: TimeBasis
  video_ids: number[]
  video_names: string[]
  camera_ids: number[]
  camera_names: Array<string | null>
  zone_ids: number[]
  zone_names: Array<string | null>
  generated_at: string
}

export interface ReportStatistics {
  unique_person_count: number
  track_count: number
  observation_count: number
  visible_zone_count: number
  total_visible_duration_seconds: number
}

export interface ZoneStatistics {
  zone_id: number
  zone_name: string | null
  unique_person_count: number
  segment_count: number
  total_visible_duration_seconds: number
}

export interface ReportDataQuality {
  processed_video_count: number
  failed_video_count: number
  unassigned_zone_point_count: number
  people_without_zone_count: number
  warnings: string[]
}

export interface ReportPreview {
  schema_version: string
  report_type: string
  scope: ReportScope
  statistics: ReportStatistics
  zone_statistics: ZoneStatistics[]
  data_quality: ReportDataQuality
}

export interface GeneratedReport {
  id: number
  status: ReportStatus | string
  report_type: string | null
  report_date: string | null
  timezone: string | null
  title: string | null
  report_markdown: string | null
  warnings: string[]
  scope: Partial<ReportScope>
  created_at: string | null
  updated_at: string | null
  error_message: string | null
  dify_workflow_run_id: string | null
}

export interface ReportListItem {
  id: number
  status: ReportStatus | string
  report_type: string | null
  report_date: string | null
  timezone: string | null
  title: string | null
  warnings: string[]
  scope: Partial<ReportScope>
  created_at: string | null
  updated_at: string | null
  error_message: string | null
  dify_workflow_run_id: string | null
}

export interface ReportListResponse {
  items: ReportListItem[]
  total: number
  page: number
  page_size: number
}

export interface ReportListQuery {
  status?: ReportStatus | ''
  report_date?: string | null
  page?: number
  page_size?: number
}

export interface ReportRequestError {
  message: string
  reportId: number | null
  status: number | null
}

const GENERATE_TIMEOUT_MS = 180000

let inflightGenerate: Promise<GeneratedReport> | null = null

export function generateReportInFlight(): boolean {
  return inflightGenerate != null
}

export function currentGeneratePromise(): Promise<GeneratedReport> | null {
  return inflightGenerate
}

export function previewReport(body: ReportPreviewRequest) {
  return http.post<ReportPreview>('/reports/preview', body).then((r) => r.data)
}

export function generateReport(body: ReportPreviewRequest): Promise<GeneratedReport> {
  if (inflightGenerate) {
    return inflightGenerate
  }
  inflightGenerate = http
    .post<GeneratedReport>('/reports/generate', body, { timeout: GENERATE_TIMEOUT_MS })
    .then((r) => r.data)
    .finally(() => {
      inflightGenerate = null
    })
  return inflightGenerate
}

export function fetchReports(query: ReportListQuery = {}) {
  const params: Record<string, string | number> = {
    page: query.page ?? 1,
    page_size: query.page_size ?? 20,
  }
  if (query.status) params.status = query.status
  if (query.report_date) params.report_date = query.report_date
  return http.get<ReportListResponse>('/reports', { params }).then((r) => r.data)
}

export function fetchReport(reportId: number) {
  return http.get<GeneratedReport>(`/reports/${reportId}`).then((r) => r.data)
}

export function deleteReport(reportId: number) {
  return http.delete<{ ok: boolean; id: number }>(`/reports/${reportId}`).then((r) => r.data)
}

function redactClientText(text: string): string {
  return text
    .replace(/Bearer\s+\S+/gi, 'Bearer ***')
    .replace(/app-[A-Za-z0-9]+/g, 'app-***')
    .replace(/DIFY_[A-Z0-9_]+/gi, 'DIFY_***')
}

export function parseReportRequestError(err: unknown, fallback: string): ReportRequestError {
  const axiosErr = err as {
    response?: { status?: number; data?: { detail?: unknown } }
  }
  const status = axiosErr.response?.status ?? null
  const detail = axiosErr.response?.data?.detail
  let reportId: number | null = null
  let message = fallback
  if (detail && typeof detail === 'object') {
    const rec = detail as { error?: unknown; report_id?: unknown; message?: unknown }
    if (typeof rec.report_id === 'number') reportId = rec.report_id
    else if (typeof rec.report_id === 'string' && rec.report_id.trim()) {
      const parsed = Number(rec.report_id)
      reportId = Number.isFinite(parsed) ? parsed : null
    }
    if (typeof rec.error === 'string' && rec.error.trim()) message = rec.error.trim()
    else if (typeof rec.message === 'string' && rec.message.trim()) message = rec.message.trim()
    else message = apiErrorMessage(err, fallback)
  } else {
    message = apiErrorMessage(err, fallback)
  }
  return {
    message: redactClientText(message),
    reportId,
    status,
  }
}

export function previewHasActivity(preview: ReportPreview | null): boolean {
  if (!preview) return false
  return preview.statistics.unique_person_count > 0 || preview.statistics.observation_count > 0
}
