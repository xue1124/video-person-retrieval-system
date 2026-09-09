import http from '@/api/http'

export interface TrackingVideo {
  id: number
  file_name: string
  source_type: string | null
  captured_at: string | null
  fps: number | null
  duration_sec: number | null
  camera_name: string | null
  channel_no: string | null
  person_count: number
  run_count: number
}

export interface TrackingRoom {
  id: number
  name: string
  video_id: number | null
  file_name: string | null
}

export interface TrackingSnapshot {
  id: number
  url: string | null
  snapshot_type: string
  is_primary: boolean
  timestamp_sec: number | null
  timestamp_hms: string
  quality_score: number | null
  video_person_id: number
  stay_segment_id: number | null
  video_id: number
}

export interface TrackingPerson {
  id: number
  label: string
  status: string
  sample_count: number
  first_seen_at: string | null
  last_seen_at: string | null
  video_person_count: number
  video_count: number
  track_count: number
  stay_sec_total: number
  members: string | null
  local_person_no: number | null
  thumb_url: string | null
}

export interface StaySegmentRow {
  id: number
  video_id: number
  file_name: string
  local_person_no: number
  room_id: number | null
  room_name: string | null
  start_sec: number | null
  end_sec: number | null
  duration_sec: number | null
  start_hms: string
  end_hms: string
  duration_hms: string
  entered_at: string | null
  exited_at: string | null
  point_count: number
  snapshot_url: string | null
}

export interface PersonArchive {
  overview: {
    id: number
    status: string
    sample_count: number
    first_seen_at: string | null
    last_seen_at: string | null
    video_count: number
    video_person_count: number
    track_count: number
    stay_segment_count: number
    stay_sec_total: number
    stay_hms_total: string
    primary_snapshot: TrackingSnapshot | null
    candidate_snapshots: TrackingSnapshot[]
    rooms_visited: Array<{ name: string; stay_sec: number; stay_hms: string }>
    members: Array<{
      video_person_id: number
      video_id: number
      file_name: string
      local_person_no: number
      label: string
      start_sec: number | null
      end_sec: number | null
      camera_name: string | null
      assignment_method: string | null
      assignment_score: number | null
      thumb_url: string | null
    }>
  }
  stays: StaySegmentRow[]
  timeline_by_video: Array<{ file_name: string; segments: StaySegmentRow[] }>
  points: Array<{
    file_name: string
    local_person_no: number
    local_track_no: number
    timestamp_sec: number | null
    timestamp_hms: string
    occurred_at: string | null
    room_name: string | null
    confidence: number | null
    pre_merge_id: number | null
    assign_score: number | null
    merge_score: number | null
    bbox: Array<number | null>
    foot: Array<number | null>
  }>
  points_truncated: boolean
  points_limit: number
}

export async function fetchTrackingStatus() {
  const { data } = await http.get('/tracking/status')
  return data as {
    enabled: boolean
    ready: boolean
    detail: string
    counts: Record<string, number>
  }
}

export async function fetchTrackingVideos() {
  const { data } = await http.get('/tracking/videos')
  return (data.items || []) as TrackingVideo[]
}

export async function fetchTrackingRooms(videoId?: number | null) {
  const { data } = await http.get('/tracking/rooms', {
    params: videoId != null ? { video_id: videoId } : undefined,
  })
  return (data.items || []) as TrackingRoom[]
}

export async function fetchTrackingPeople(params?: {
  video_id?: number | null
  room_id?: number | null
}) {
  const q: Record<string, number> = {}
  if (params?.video_id != null) q.video_id = params.video_id
  if (params?.room_id != null) q.room_id = params.room_id
  const { data } = await http.get('/tracking/people', { params: q })
  return (data.items || []) as TrackingPerson[]
}

export async function fetchPersonArchive(
  globalPersonId: number,
  params?: {
    video_id?: number | null
    room_id?: number | null
    include_points?: boolean
    points_limit?: number
  },
) {
  const q: Record<string, number | boolean> = {}
  if (params?.video_id != null) q.video_id = params.video_id
  if (params?.room_id != null) q.room_id = params.room_id
  if (params?.include_points) q.include_points = true
  if (params?.points_limit != null) q.points_limit = params.points_limit
  const { data } = await http.get(`/tracking/people/${globalPersonId}`, { params: q })
  return data as PersonArchive
}
