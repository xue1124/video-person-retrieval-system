/** 轨迹出现段：与检索 API stay_segments 及日志 v2 结构一致 */

export interface StayCrop {
  meta_id?: number
  observation_id?: number
  global_person_id?: number | null
  video_person_id?: number
  track_id?: number
  vid?: string
  time: number
  score: number
  path: string
  imageUrl?: string
  bbox?: number[]
  room_id?: number | null
  room_name?: string
  room_method?: string
  room_dist_px?: number
}

export interface StaySegment {
  segment_idx: number
  global_person_id?: number | null
  room_name: string
  entry_room: string
  exit_room: string
  start_sec: number
  end_sec: number
  duration_sec: number
  start_time: string
  end_time: string
  duration_time: string
  hit_count: number
  best_similarity: number
  entry_foot_x: number
  entry_foot_y: number
  exit_foot_x: number
  exit_foot_y: number
  crops: StayCrop[]
}

export type SegmentRow = StaySegment & { video_name: string; stay_duration_display: string }

export interface Hit {
  vid: string
  time: number
  path: string
  score: number
  imageUrl?: string
  bbox?: number[]
  meta_id?: number
  observation_id?: number
  global_person_id?: number | null
  room_name?: string
}

export interface LogHit {
  vid?: string
  time: number
  path?: string
  imageUrl?: string
  score: number
}

export interface CopresenceVideoPayload {
  video_name: string
  room_annotated?: boolean
  stay_segments: StaySegment[]
  stay_segments_doctor: StaySegment[]
  stay_room_names: string[]
}

export interface ParsedLogPayload {
  kind: 'single' | 'room_copresence'
  results: Record<string, LogHit[]>
  stay_segments?: Record<string, StaySegment[]>
  time_gap?: number
  legacyOnly: boolean
  copresence_videos?: CopresenceVideoPayload[]
  query_image_a?: string
  query_image_b?: string
}

export function fmtHms(sec: number): string {
  const total = Math.max(0, Math.floor(sec))
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

/** 文搜图为 sigmoid 映射分（通常 65~95）；图搜图仍为 0~1 映射分 */
export function formatSimilarityScore(score: number): string {
  if (!Number.isFinite(score)) return '—'
  if (score > 1.5) return score.toFixed(1)
  return score.toFixed(2)
}

export function resolveCropPath(c: StayCrop): string {
  if (c.imageUrl) return c.imageUrl
  const b = (c.path || '').split(/[/\\]/).pop() || ''
  return `/files/crop/${b}`
}

export function resolveHitPath(hit: LogHit): string {
  if (hit.imageUrl) return hit.imageUrl
  if (hit.path) {
    const b = hit.path.split(/[/\\]/).pop() || ''
    return `/files/crop/${b}`
  }
  return ''
}

/** 解析日志 results_json：v2 含 stay_segments，旧版为 video -> hits */
export function parseLogResultsJson(json: string): ParsedLogPayload {
  try {
    const raw = JSON.parse(json) as Record<string, unknown>
    if (raw && typeof raw === 'object' && (raw.kind === 'room_copresence' || raw.version === 3)) {
      return {
        kind: 'room_copresence',
        results: {},
        legacyOnly: false,
        time_gap: typeof raw.time_gap === 'number' ? raw.time_gap : undefined,
        copresence_videos: (raw.videos as CopresenceVideoPayload[]) || [],
        query_image_a: String(raw.query_image_a || ''),
        query_image_b: String(raw.query_image_b || ''),
      }
    }
    if (raw && typeof raw === 'object' && ('stay_segments' in raw || raw.version === 2)) {
      return {
        kind: 'single',
        results: (raw.results as Record<string, LogHit[]>) || {},
        stay_segments: (raw.stay_segments as Record<string, StaySegment[]>) || {},
        time_gap: typeof raw.time_gap === 'number' ? raw.time_gap : undefined,
        legacyOnly: false,
      }
    }
    return {
      kind: 'single',
      results: raw as Record<string, LogHit[]>,
      stay_segments: undefined,
      legacyOnly: true,
    }
  } catch {
    return { kind: 'single', results: {}, stay_segments: undefined, legacyOnly: true }
  }
}

export function logKindLabel(json: string, searchQuery?: string | null): string {
  const p = parseLogResultsJson(json)
  if (p.kind === 'room_copresence') return '双人房间共现'
  if ((searchQuery || '').includes('双人房间共现')) return '双人房间共现'
  return '单人轨迹检索'
}

function splitQueryNames(raw: string | undefined): string[] {
  if (!raw?.trim()) return []
  return raw
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
}

/** 日志列表预览图路径 */
export function previewPathsFromLog(json: string, limit = 3): string[] {
  const parsed = parseLogResultsJson(json)
  const paths: string[] = []
  if (parsed.kind === 'room_copresence' && parsed.copresence_videos?.length) {
    for (const v of parsed.copresence_videos) {
      for (const seg of v.stay_segments || []) {
        for (const c of seg.crops || []) {
          const p = resolveCropPath(c)
          if (p) paths.push(p)
          if (paths.length >= limit) return paths
        }
      }
    }
  }
  if (hasStaySegments(parsed.stay_segments)) {
    for (const vn of Object.keys(parsed.stay_segments!).sort()) {
      for (const seg of parsed.stay_segments![vn] || []) {
        for (const c of seg.crops || []) {
          const p = resolveCropPath(c)
          if (p) paths.push(p)
          if (paths.length >= limit) return paths
        }
      }
    }
  }
  for (const items of Object.values(parsed.results)) {
    for (const it of items) {
      const p = resolveHitPath(it)
      if (p) paths.push(p)
      if (paths.length >= limit) return paths
    }
  }
  return paths
}

export function queryImageNamesFromLog(parsed: ParsedLogPayload, role: 'a' | 'b'): string[] {
  if (parsed.kind === 'room_copresence') {
    return role === 'a'
      ? splitQueryNames(parsed.query_image_a)
      : splitQueryNames(parsed.query_image_b)
  }
  return role === 'a' ? splitQueryNames(parsed.query_image_a) : []
}

export function hasStaySegments(m: Record<string, StaySegment[]> | null | undefined): boolean {
  return !!m && Object.keys(m).some((k) => (m[k]?.length ?? 0) > 0)
}

export function buildSegmentsFlat(staySegments: Record<string, StaySegment[]> | null): SegmentRow[] {
  if (!staySegments) return []
  const rows: SegmentRow[] = []
  const videoNames = Object.keys(staySegments).sort()
  for (const vn of videoNames) {
    const sorted = [...(staySegments[vn] || [])].sort((a, b) => a.start_sec - b.start_sec)
    for (let i = 0; i < sorted.length; i++) {
      const s = sorted[i]
      const next = sorted[i + 1]
      let stay_duration_display = '—'
      if (next) {
        const gapSec = Math.max(0, next.start_sec - s.end_sec)
        stay_duration_display = fmtHms(Math.floor(gapSec))
      }
      rows.push({ ...s, video_name: vn, stay_duration_display })
    }
  }
  return rows
}

export function buildSegmentsByVideo(
  staySegments: Record<string, StaySegment[]> | null,
): Array<{ video_name: string; rows: SegmentRow[] }> {
  const flat = buildSegmentsFlat(staySegments)
  if (!flat.length) return []
  const m = new Map<string, SegmentRow[]>()
  for (const r of flat) {
    if (!m.has(r.video_name)) m.set(r.video_name, [])
    m.get(r.video_name)!.push(r)
  }
  const out: Array<{ video_name: string; rows: SegmentRow[] }> = []
  for (const [video_name, rows] of m.entries()) {
    out.push({ video_name, rows })
  }
  out.sort((a, b) => a.video_name.localeCompare(b.video_name))
  return out
}

export function segmentRowKey(row: StaySegment & { video_name: string }): string {
  return `${row.video_name}|${row.segment_idx}`
}

export function cropStableKey(c: StayCrop, row: StaySegment & { video_name: string }): string {
  const id = c.meta_id != null ? String(c.meta_id) : 'n'
  const base = (c.path || '').split(/[/\\]/).pop() || ''
  return `${row.video_name}|${row.segment_idx}|${id}|${c.time}|${base}`
}

function segmentPrimaryRoom(cropNames: string[], entryRoom: string, exitRoom: string): string {
  const trimmed = cropNames.map((x) => (x || '').trim()).filter(Boolean)
  if (trimmed.length) {
    const counts = new Map<string, number>()
    for (const n of trimmed) counts.set(n, (counts.get(n) || 0) + 1)
    let top = trimmed[0]
    let max = 0
    for (const [name, c] of counts) {
      if (c > max) {
        max = c
        top = name
      }
    }
    return top
  }
  if (entryRoom && exitRoom && entryRoom !== exitRoom) return `${entryRoom}→${exitRoom}`
  return entryRoom || exitRoom || ''
}

function bboxFootXy(bbox: number[]): [number, number] {
  const [x1, , x2, y2] = bbox
  return [(x1 + x2) * 0.5, y2]
}

/** 删帧后重算段级摘要（检索页可编辑时使用） */
export function recomputeStaySegmentSummary(seg: StaySegment): void {
  const crops = [...seg.crops].sort((a, b) => a.time - b.time)
  seg.crops = crops
  if (!crops.length) return

  const first = crops[0]
  const last = crops[crops.length - 1]
  const cropNames = crops.map((c) => (c.room_name || '').trim())

  seg.entry_room = (first.room_name || '').trim()
  seg.exit_room = (last.room_name || '').trim()
  seg.room_name = segmentPrimaryRoom(cropNames, seg.entry_room, seg.exit_room)
  seg.hit_count = crops.length
  seg.best_similarity = Math.round(Math.max(...crops.map((c) => c.score)) * 10000) / 10000

  const t0 = Math.min(...crops.map((c) => c.time))
  const t1 = Math.max(...crops.map((c) => c.time))
  seg.start_sec = Math.round(t0 * 1000) / 1000
  seg.end_sec = Math.round(t1 * 1000) / 1000
  seg.duration_sec = Math.max(0, Math.round((t1 - t0) * 1000) / 1000)
  seg.start_time = fmtHms(Math.floor(t0))
  seg.end_time = fmtHms(Math.floor(t1))
  seg.duration_time = fmtHms(Math.floor(Math.max(0, t1 - t0)))

  if (first.bbox && first.bbox.length >= 4) {
    const [fx, fy] = bboxFootXy(first.bbox)
    seg.entry_foot_x = Math.round(fx * 10) / 10
    seg.entry_foot_y = Math.round(fy * 10) / 10
  } else {
    seg.entry_foot_x = 0
    seg.entry_foot_y = 0
  }
  if (last.bbox && last.bbox.length >= 4) {
    const [lx, ly] = bboxFootXy(last.bbox)
    seg.exit_foot_x = Math.round(lx * 10) / 10
    seg.exit_foot_y = Math.round(ly * 10) / 10
  } else {
    seg.exit_foot_x = 0
    seg.exit_foot_y = 0
  }
}
