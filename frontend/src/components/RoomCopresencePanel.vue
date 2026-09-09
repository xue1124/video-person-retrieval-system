<script setup lang="ts">
import AuthImage from '@/components/AuthImage.vue'
import http from '@/api/http'
import type { UploadUserFile } from 'element-plus'
import { ElMessage } from 'element-plus'
import { computed, reactive, ref, watch } from 'vue'
import { apiErrorMessage } from '@/utils/apiError'

const props = withDefaults(
  defineProps<{
    /** 日志回看：隐藏上传表单，直接展示已保存的分析结果 */
    hideForm?: boolean
    replayPayload?: CopresencePayload | null
  }>(),
  { hideForm: false, replayPayload: null },
)

const emit = defineEmits<{
  playVideo: [vid: string, time: number]
  preview: [path: string]
}>()

interface StayCrop {
  meta_id?: number
  vid?: string
  time: number
  score: number
  path: string
  imageUrl?: string
  bbox?: number[]
  room_name?: string
}

interface StaySegment {
  segment_idx: number
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
  crops: StayCrop[]
}

interface VideoPayload {
  video_name: string
  room_annotated?: boolean
  stay_segments: StaySegment[]
  stay_segments_doctor: StaySegment[]
  stay_room_names: string[]
}

interface CopresencePayload {
  algorithm: string
  videos: VideoPayload[]
  query_image_a?: string
  query_image_b?: string
  threshold?: number
  time_gap?: number
}

interface TimeIv {
  start_sec: number
  end_sec: number
  start_time: string
  end_time: string
  duration_display: string
}

interface DoctorRefLine {
  line_key: string
  line_type: 'room_stay' | 'monitor' | 'empty'
  summary: string
  enter_crop: StayCrop | null
  leave_crop: StayCrop | null
  enter_label: string
  leave_label: string
}

interface CoPresenceResult {
  label: string
  together_sec: number
  solo_sec: number
  intervals: TimeIv[]
  intervals_display: string
  doctor_same_room_lines: DoctorRefLine[]
  doctor_other_lines: DoctorRefLine[]
}

interface RoomStayRow {
  episode_idx: number
  video_name: string
  room_name: string
  room_stay_duration_display: string
  enter_room_time: string
  leave_room_time: string
  enter_room_crops: StayCrop[]
  leave_room_crops: StayCrop[]
  /** 对应患者 stay_segments 中触发本次「停留房间」的那一段 */
  suspect_segment_idx: number
  co_presence: CoPresenceResult
}

const algorithm = ref<'OSNet' | 'SIGLIP'>('OSNet')
const threshold = ref(0.85)
const timeGap = ref(60)
const uploadA = ref<UploadUserFile[]>([])
const uploadB = ref<UploadUserFile[]>([])
const loading = ref(false)
const result = ref<CopresencePayload | null>(null)
const selectedVideo = ref('')
const selectedStayRoom = ref('')
const lastMeta = reactive({
  query_image_a: '',
  query_image_b: '',
  algorithm: '',
  threshold: 0,
  time_gap: 60,
})
const savingLog = ref(false)

const canAnalyze = computed(
  () => uploadA.value.some((u) => u.raw) && uploadB.value.some((u) => u.raw),
)

/** 患者出现段已全部移除的视频不再出现在下拉中 */
const videosWithSuspectSegments = computed(() => {
  if (!result.value?.videos?.length) return []
  return result.value.videos.filter((v) => (v.stay_segments?.length ?? 0) > 0)
})

const currentVideo = computed(() =>
  videosWithSuspectSegments.value.find((v) => v.video_name === selectedVideo.value),
)

const stayRoomOptions = computed(() => {
  const vid = currentVideo.value
  if (!vid) return []
  const names = new Set<string>()
  for (const n of vid.stay_room_names ?? []) {
    const t = (n || '').trim()
    if (t) names.add(t)
  }
  if (!names.size) {
    for (const s of vid.stay_segments ?? []) {
      for (const r of [s.exit_room, s.entry_room, s.room_name]) {
        const t = (r || '').trim()
        if (t && !t.includes('→')) names.add(t)
      }
    }
  }
  return [...names].sort().map((n) => ({ label: n, value: n }))
})

function fmtTime(sec: number) {
  const total = Math.max(0, Math.floor(sec))
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

function secToHmsStr(sec: number) {
  return fmtTime(sec)
}

function tailCrop(crops: StayCrop[]): StayCrop | null {
  return crops.length ? crops[crops.length - 1] : null
}

function headCrop(crops: StayCrop[]): StayCrop | null {
  return crops.length ? crops[0] : null
}

function intervalOverlap(
  a0: number,
  a1: number,
  b0: number,
  b1: number,
): { start: number; end: number; dur: number } | null {
  const start = Math.max(a0, b0)
  const end = Math.min(a1, b1)
  if (end <= start) return null
  return { start, end, dur: end - start }
}

function mergeSecIntervals(raw: Array<{ start: number; end: number }>) {
  if (!raw.length) return []
  const sorted = [...raw].sort((a, b) => a.start - b.start)
  const out: Array<{ start: number; end: number }> = [{ ...sorted[0] }]
  for (let i = 1; i < sorted.length; i++) {
    const last = out[out.length - 1]
    if (sorted[i].start <= last.end) {
      last.end = Math.max(last.end, sorted[i].end)
    } else {
      out.push({ ...sorted[i] })
    }
  }
  return out
}

function buildDoctorOtherLines(
  doctorSegs: StaySegment[],
  w0: number,
  w1: number,
  suspectRoom: string,
): DoctorRefLine[] {
  const other: DoctorRefLine[] = []

  if (!doctorSegs.length) {
    return other
  }

  const sorted = [...doctorSegs].sort((a, b) => a.start_sec - b.start_sec)
  const usedSegIdx = new Set<number>()

  for (let i = 0; i < sorted.length - 1; i++) {
    const s = sorted[i]
    const next = sorted[i + 1]
    const rn = (s.exit_room || '').trim()
    if (!rn) continue
    const enterSec = s.end_sec
    const leaveSec = next.start_sec
    if (!intervalOverlap(enterSec, leaveSec, w0, w1)) continue

    usedSegIdx.add(s.segment_idx)
    usedSegIdx.add(next.segment_idx)
    if (rn === suspectRoom) continue

    other.push({
      line_key: `room-${s.segment_idx}`,
      line_type: 'room_stay',
      summary: `房内停留 · ${rn} · ${s.end_time} — ${next.start_time}（${fmtTime(leaveSec - enterSec)}）`,
      enter_crop: tailCrop(s.crops),
      leave_crop: headCrop(next.crops),
      enter_label: '进入房间',
      leave_label: '离开房间',
    })
  }

  for (const seg of sorted) {
    if (usedSegIdx.has(seg.segment_idx)) continue
    if (seg.end_sec <= w0 || seg.start_sec >= w1) continue
    if (!intervalOverlap(seg.start_sec, seg.end_sec, w0, w1)) continue

    other.push({
      line_key: `mon-${seg.segment_idx}`,
      line_type: 'monitor',
      summary: `监控内出现 · ${seg.start_time} — ${seg.end_time}（${seg.duration_time}）`,
      enter_crop: headCrop(seg.crops),
      leave_crop: tailCrop(seg.crops),
      enter_label: '进入监控',
      leave_label: '离开监控',
    })
  }

  return other
}

function analyzeCoPresence(
  doctorSegs: StaySegment[],
  w0: number,
  w1: number,
  suspectRoom: string,
): CoPresenceResult {
  const total = Math.max(0, w1 - w0)
  const empty: CoPresenceResult = {
    label: '—',
    together_sec: 0,
    solo_sec: total,
    intervals: [],
    intervals_display: '—',
    doctor_same_room_lines: [],
    doctor_other_lines: [],
  }

  if (w1 <= w0) return empty

  const other = buildDoctorOtherLines(doctorSegs, w0, w1, suspectRoom)

  const rawIv: Array<{ start: number; end: number }> = []
  const doctorSameRoomLines: DoctorRefLine[] = []

  if (!doctorSegs.length) {
    return {
      ...empty,
      label: '患者单独在房',
      doctor_other_lines: [
        {
          line_key: 'empty',
          line_type: 'empty',
          summary: '本时段无医生检索命中',
          enter_crop: null,
          leave_crop: null,
          enter_label: '',
          leave_label: '',
        },
      ],
    }
  }

  const sorted = [...doctorSegs].sort((a, b) => a.start_sec - b.start_sec)
  for (let i = 0; i < sorted.length - 1; i++) {
    const s = sorted[i]
    const next = sorted[i + 1]
    const rn = (s.exit_room || '').trim()
    if (rn !== suspectRoom) continue
    const ov = intervalOverlap(s.end_sec, next.start_sec, w0, w1)
    if (!ov) continue
    rawIv.push({ start: ov.start, end: ov.end })
    const key = `room-${s.segment_idx}`
    doctorSameRoomLines.push({
      line_key: key,
      line_type: 'room_stay',
      summary: `医生同在 ${rn} · ${secToHmsStr(ov.start)} — ${secToHmsStr(ov.end)}`,
      enter_crop: tailCrop(s.crops),
      leave_crop: headCrop(next.crops),
      enter_label: '进入房间',
      leave_label: '离开房间',
    })
  }

  const merged = mergeSecIntervals(rawIv)
  let together = 0
  const intervals: TimeIv[] = merged.map((iv) => {
    const dur = iv.end - iv.start
    together += dur
    return {
      start_sec: iv.start,
      end_sec: iv.end,
      start_time: secToHmsStr(iv.start),
      end_time: secToHmsStr(iv.end),
      duration_display: fmtTime(Math.floor(dur)),
    }
  })

  const solo = Math.max(0, total - together)
  const intervals_display =
    intervals.length > 0
      ? intervals.map((iv) => `${iv.start_time} — ${iv.end_time}`).join('；')
      : '—'

  let label: string
  if (together <= 0) {
    label = '患者单独在房'
  } else if (solo <= 0) {
    label = '与医生同时在房'
  } else {
    label = `部分同房（共现 ${fmtTime(Math.floor(together))}，单独 ${fmtTime(Math.floor(solo))}）`
  }

  let doctor_other_lines = other
  if (together <= 0 && !doctor_other_lines.length && doctorSegs.length) {
    doctor_other_lines.push({
      line_key: 'hint-no-same',
      line_type: 'empty',
      summary: '本时段有医生活动，但未检出进入该停留房间',
      enter_crop: null,
      leave_crop: null,
      enter_label: '',
      leave_label: '',
    })
  }

  return {
    label,
    together_sec: together,
    solo_sec: solo,
    intervals,
    intervals_display,
    doctor_same_room_lines: doctorSameRoomLines,
    doctor_other_lines,
  }
}

const roomStayRows = computed((): RoomStayRow[] => {
  const vid = currentVideo.value
  const room = selectedStayRoom.value
  if (!vid || !room) return []

  const segs = [...vid.stay_segments].sort((a, b) => a.start_sec - b.start_sec)
  const doctorSegs = vid.stay_segments_doctor ?? []
  const rows: RoomStayRow[] = []

  for (let i = 0; i < segs.length; i++) {
    const s = segs[i]
    if ((s.exit_room || '').trim() !== room) continue

    const next = segs[i + 1]
    const w0 = s.end_sec
    const w1 = next ? next.start_sec : s.end_sec

    let roomStayDisplay = '—'
    let leaveTime = '—'
    if (next) {
      roomStayDisplay = fmtTime(Math.floor(Math.max(0, w1 - w0)))
      leaveTime = next.start_time
    }

    const co = w1 > w0 ? analyzeCoPresence(doctorSegs, w0, w1, room) : analyzeCoPresence([], 0, 0, room)

    rows.push({
      episode_idx: rows.length + 1,
      video_name: vid.video_name,
      room_name: room,
      room_stay_duration_display: roomStayDisplay,
      enter_room_time: s.end_time,
      leave_room_time: leaveTime,
      enter_room_crops: (() => {
        const c = tailCrop(s.crops)
        return c ? [c] : []
      })(),
      leave_room_crops: (() => {
        const c = next ? headCrop(next.crops) : null
        return c ? [c] : []
      })(),
      suspect_segment_idx: s.segment_idx,
      co_presence: co,
    })
  }
  return rows
})

const tableGroup = computed(() => {
  if (!roomStayRows.value.length || !selectedVideo.value) return null
  return { video_name: selectedVideo.value, rows: roomStayRows.value }
})

watch(
  videosWithSuspectSegments,
  (list) => {
    if (!list.length) {
      selectedVideo.value = ''
      selectedStayRoom.value = ''
      return
    }
    if (!list.some((v) => v.video_name === selectedVideo.value)) {
      selectedVideo.value = list[0].video_name
    }
  },
)

watch(selectedVideo, () => {
  selectedStayRoom.value = ''
})

watch(
  () => props.replayPayload,
  (p) => {
    if (p?.videos?.length) {
      result.value = p
      const withSegs = p.videos.filter((v) => (v.stay_segments?.length ?? 0) > 0)
      selectedVideo.value = withSegs[0]?.video_name ?? ''
      selectedStayRoom.value = ''
    }
  },
  { immediate: true },
)

function resolveCropPath(c: StayCrop): string {
  if (c.imageUrl) return c.imageUrl
  const b = c.path.split(/[/\\]/).pop() || ''
  return `/files/crop/${b}`
}

function rowKey(row: RoomStayRow) {
  return `${row.video_name}|${row.episode_idx}`
}

function coPresenceTagType(label: string): 'success' | 'warning' | 'info' {
  if (label.includes('同时在房')) return 'success'
  if (label.includes('部分')) return 'warning'
  return 'info'
}

function playCrop(vid: string, crop: StayCrop | null | undefined) {
  if (crop == null) return
  emit('playVideo', vid, crop.time)
}

/** 移除主表中「此次停留」对应整条患者出现段（非单帧） */
function removeRoomStayEpisode(videoName: string, suspectSegmentIdx: number) {
  const payload = result.value
  if (!payload) return
  const vid = payload.videos.find((v) => v.video_name === videoName)
  if (!vid) return
  const before = vid.stay_segments.length
  vid.stay_segments = vid.stay_segments.filter((s) => s.segment_idx !== suspectSegmentIdx)
  if (vid.stay_segments.length === before) {
    ElMessage.warning('未找到对应停留段')
    return
  }
  if (!vid.stay_segments.length && selectedVideo.value === videoName) {
    const rest = payload.videos.filter((v) => (v.stay_segments?.length ?? 0) > 0)
    selectedVideo.value = rest[0]?.video_name ?? ''
    selectedStayRoom.value = ''
  }
  ElMessage.success('已移除此次停留')
}

async function runAnalyze() {
  const filesA = uploadA.value.filter((u) => u.raw).map((u) => u.raw!)
  const filesB = uploadB.value.filter((u) => u.raw).map((u) => u.raw!)
  if (!filesA.length || !filesB.length) {
    ElMessage.warning('请为患者、医生各至少上传一张查询图')
    return
  }
  const fd = new FormData()
  fd.append('algorithm', algorithm.value)
  fd.append('threshold', String(threshold.value))
  fd.append('time_gap', String(timeGap.value))
  fd.append('role_a_label', '患者')
  fd.append('role_b_label', '医生')
  for (const f of filesA) fd.append('role_a_files', f)
  for (const f of filesB) fd.append('role_b_files', f)

  loading.value = true
  selectedVideo.value = ''
  selectedStayRoom.value = ''
  try {
    const { data } = await http.post<CopresencePayload>('/search/room-copresence', fd)
    result.value = data
    lastMeta.query_image_a = data.query_image_a ?? ''
    lastMeta.query_image_b = data.query_image_b ?? ''
    lastMeta.algorithm = data.algorithm
    lastMeta.threshold = threshold.value
    lastMeta.time_gap = timeGap.value
    const withSegs = data.videos.filter((v) => (v.stay_segments?.length ?? 0) > 0)
    if (withSegs.length) {
      selectedVideo.value = withSegs[0].video_name
    }
    ElMessage.success('分析完成，请选择停留房间')
  } catch (e: unknown) {
    ElMessage.error(apiErrorMessage(e, '房间共现分析请求失败'))
  } finally {
    loading.value = false
  }
}

async function saveLog() {
  if (!result.value?.videos?.length) {
    ElMessage.warning('请先完成分析再保存')
    return
  }
  savingLog.value = true
  try {
    await http.post('/search/logs', {
      log_kind: 'room_copresence',
      query_image_paths: lastMeta.query_image_a,
      query_image_paths_b: lastMeta.query_image_b,
      search_query: null,
      algorithm: lastMeta.algorithm || result.value.algorithm,
      threshold: lastMeta.threshold || threshold.value,
      time_gap: lastMeta.time_gap || timeGap.value,
      results: {},
      copresence: { videos: result.value.videos },
    })
    ElMessage.success('已保存到检索日志')
  } catch (e: unknown) {
    ElMessage.error(apiErrorMessage(e, '保存检索日志失败'))
  } finally {
    savingLog.value = false
  }
}
</script>

<template>
  <div class="space-y-6">
    <el-card v-if="!hideForm" shadow="never" class="rounded-xl border border-slate-200/70">
      <el-form label-width="120px" class="max-w-4xl">
        <el-form-item label="检索方式">
          <el-radio-group v-model="algorithm">
            <el-radio-button value="OSNet">图像检索</el-radio-button>
            <el-radio-button value="SIGLIP">多模态检索</el-radio-button>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="相似度阈值">
          <el-slider v-model="threshold" :min="0.1" :max="0.95" :step="0.01" style="max-width: 400px" />
        </el-form-item>
        <el-form-item label="轨迹聚合间隔(秒)">
          <el-input-number v-model="timeGap" :min="1" :max="60" />
        </el-form-item>
        <el-form-item label="患者">
          <el-upload v-model:file-list="uploadA" multiple :auto-upload="false" accept=".jpg,.jpeg,.png">
            <el-button type="primary">选择图片</el-button>
          </el-upload>
          <p v-if="uploadA.length > 1" class="mt-1 text-xs text-slate-500">
            多张图将分别检索，同一监控帧保留最高相似度后合并轨迹，适合同一人多角度。
          </p>
        </el-form-item>
        <el-form-item label="医生">
          <el-upload v-model:file-list="uploadB" multiple :auto-upload="false" accept=".jpg,.jpeg,.png">
            <el-button type="primary">选择图片</el-button>
          </el-upload>
          <p v-if="uploadB.length > 1" class="mt-1 text-xs text-slate-500">
            多张图将分别检索，同一监控帧保留最高相似度后合并轨迹，适合同一人多角度。
          </p>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :loading="loading" :disabled="!canAnalyze" @click="runAnalyze">
            开始分析
          </el-button>
          <el-button
            v-if="result?.videos?.length"
            :loading="savingLog"
            @click="saveLog"
          >
            保存到日志
          </el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <template v-if="result">
      <el-card v-if="videosWithSuspectSegments.length" shadow="never" class="rounded-xl border border-slate-200/70">
        <el-form inline class="flex flex-wrap gap-4">
          <el-form-item label="视频源" class="!mb-0">
            <el-select v-model="selectedVideo" style="min-width: 280px">
              <el-option
                v-for="v in videosWithSuspectSegments"
                :key="v.video_name"
                :label="v.video_name"
                :value="v.video_name"
              />
            </el-select>
          </el-form-item>
          <el-form-item label="停留房间" class="!mb-0">
            <el-select
              v-model="selectedStayRoom"
              placeholder="选择停留房间"
              style="min-width: 200px"
              filterable
              :disabled="!stayRoomOptions.length"
            >
              <el-option v-for="o in stayRoomOptions" :key="o.value" :label="o.label" :value="o.value" />
            </el-select>
          </el-form-item>
        </el-form>
      </el-card>

      <el-card v-if="tableGroup" shadow="never" class="rounded-xl border border-emerald-200/70">
        <template #header>
          <div class="text-base font-semibold text-slate-800">房内停留 · {{ selectedStayRoom }}</div>
        </template>
        <p class="mb-3 text-sm text-slate-500">
          主表直接给出<strong>同房结论</strong>与<strong>同房时段</strong>（仅统计医生进入<strong>同一停留房间</strong>的时间重叠，不含监控内路过）。
          展开可看患者进/离房画面；若有同房再显示医生同房片段；其它医生活动默认折叠。误检可点「移除此次停留」整段去掉。
        </p>
        <el-card shadow="never" class="rounded-lg border border-slate-200/80">
          <template #header>
            <div class="flex items-center justify-between text-sm">
              <span class="font-semibold text-slate-800">{{ tableGroup.video_name }}</span>
              <span class="text-slate-500">{{ tableGroup.rows.length }} 次停留</span>
            </div>
          </template>
          <el-table :data="tableGroup.rows" :row-key="rowKey" border stripe size="small" class="w-full">
            <el-table-column type="expand">
              <template #default="{ row }">
                <div class="space-y-4 p-3">
                  <div v-if="!hideForm" class="flex justify-end">
                    <el-button
                      size="small"
                      type="danger"
                      plain
                      @click.stop="removeRoomStayEpisode(row.video_name, row.suspect_segment_idx)"
                    >
                      移除此次停留
                    </el-button>
                  </div>
                  <div>
                    <div class="mb-2 text-sm font-semibold text-sky-800">患者进/离房</div>
                    <div class="flex flex-wrap gap-6">
                      <div v-if="row.enter_room_crops[0]" class="w-28">
                        <div class="mb-1 text-xs text-slate-500">进入 {{ row.enter_room_time }}</div>
                        <AuthImage
                          :path="resolveCropPath(row.enter_room_crops[0])"
                          class="aspect-video max-h-32 w-full rounded border object-cover"
                        />
                        <el-button
                          class="mt-1"
                          size="small"
                          type="primary"
                          link
                          @click.stop="playCrop(row.video_name, row.enter_room_crops[0])"
                        >
                          播放
                        </el-button>
                      </div>
                      <div v-if="row.leave_room_crops[0]" class="w-28">
                        <div class="mb-1 text-xs text-slate-500">离开 {{ row.leave_room_time }}</div>
                        <AuthImage
                          :path="resolveCropPath(row.leave_room_crops[0])"
                          class="aspect-video max-h-32 w-full rounded border object-cover"
                        />
                        <el-button
                          class="mt-1"
                          size="small"
                          type="primary"
                          link
                          @click.stop="playCrop(row.video_name, row.leave_room_crops[0])"
                        >
                          播放
                        </el-button>
                      </div>
                    </div>
                  </div>

                  <div class="rounded-lg border border-emerald-100 bg-emerald-50/60 p-3">
                    <div class="mb-1 text-sm font-semibold text-emerald-800">同房结论</div>
                    <p class="text-sm text-slate-700">{{ row.co_presence.label }}</p>
                    <p v-if="row.co_presence.intervals.length" class="mt-1 text-xs text-emerald-700">
                      同房时段：{{ row.co_presence.intervals_display }}
                    </p>
                    <template v-if="row.co_presence.doctor_same_room_lines.length">
                      <div class="mt-3 text-xs font-medium text-slate-600">医生在该房内的片段（同房）</div>
                      <div
                        v-for="dl in row.co_presence.doctor_same_room_lines"
                        :key="dl.line_key"
                        class="mt-2 flex flex-wrap gap-4"
                      >
                        <p class="w-full text-xs text-slate-500">{{ dl.summary }}</p>
                        <div v-if="dl.enter_crop" class="w-28">
                          <div class="mb-1 text-xs text-slate-500">{{ dl.enter_label }}</div>
                          <AuthImage
                            :path="resolveCropPath(dl.enter_crop)"
                            class="aspect-video max-h-32 w-full rounded border object-cover"
                          />
                          <el-button
                            class="mt-1"
                            size="small"
                            type="primary"
                            link
                            @click.stop="playCrop(row.video_name, dl.enter_crop)"
                          >
                            播放
                          </el-button>
                        </div>
                        <div v-if="dl.leave_crop" class="w-28">
                          <div class="mb-1 text-xs text-slate-500">{{ dl.leave_label }}</div>
                          <AuthImage
                            :path="resolveCropPath(dl.leave_crop)"
                            class="aspect-video max-h-32 w-full rounded border object-cover"
                          />
                          <el-button
                            class="mt-1"
                            size="small"
                            type="primary"
                            link
                            @click.stop="playCrop(row.video_name, dl.leave_crop)"
                          >
                            播放
                          </el-button>
                        </div>
                      </div>
                    </template>
                  </div>

                  <el-collapse v-if="row.co_presence.doctor_other_lines.length">
                    <el-collapse-item
                      :title="`本时段其它医生活动（${row.co_presence.doctor_other_lines.length} 条，仅供参考）`"
                      name="other"
                    >
                      <ul class="space-y-2 text-xs text-slate-600">
                        <li
                          v-for="dl in row.co_presence.doctor_other_lines"
                          :key="dl.line_key"
                          class="rounded border border-slate-100 bg-white p-2"
                        >
                          <p>{{ dl.summary }}</p>
                          <div v-if="dl.enter_crop || dl.leave_crop" class="mt-2 flex flex-wrap gap-6">
                            <div v-if="dl.enter_crop" class="w-28">
                              <div class="mb-1 text-xs text-slate-500">{{ dl.enter_label }}</div>
                              <AuthImage
                                :path="resolveCropPath(dl.enter_crop)"
                                class="aspect-video max-h-32 w-full rounded border object-cover"
                              />
                              <el-button
                                class="mt-1"
                                size="small"
                                type="primary"
                                link
                                @click.stop="playCrop(row.video_name, dl.enter_crop)"
                              >
                                播放
                              </el-button>
                            </div>
                            <div v-if="dl.leave_crop" class="w-28">
                              <div class="mb-1 text-xs text-slate-500">{{ dl.leave_label }}</div>
                              <AuthImage
                                :path="resolveCropPath(dl.leave_crop)"
                                class="aspect-video max-h-32 w-full rounded border object-cover"
                              />
                              <el-button
                                class="mt-1"
                                size="small"
                                type="primary"
                                link
                                @click.stop="playCrop(row.video_name, dl.leave_crop)"
                              >
                                播放
                              </el-button>
                            </div>
                          </div>
                        </li>
                      </ul>
                    </el-collapse-item>
                  </el-collapse>
                </div>
              </template>
            </el-table-column>
            <el-table-column prop="episode_idx" label="#" width="44" />
            <el-table-column prop="room_name" label="停留房间" width="88" />
            <el-table-column prop="room_stay_duration_display" label="房间停留" width="96" />
            <el-table-column prop="enter_room_time" label="进入房间" width="96" />
            <el-table-column prop="leave_room_time" label="离开房间" width="96" />
            <el-table-column label="同房结论" min-width="160">
              <template #default="{ row }">
                <el-tag :type="coPresenceTagType(row.co_presence.label)" size="small">
                  {{ row.co_presence.label }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column label="同房时段" min-width="140">
              <template #default="{ row }">
                <span class="text-sm">{{ row.co_presence.intervals_display }}</span>
              </template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-card>

      <el-empty
        v-else-if="videosWithSuspectSegments.length && selectedVideo && !selectedStayRoom"
        description="请选择停留房间"
      />
      <el-empty
        v-else-if="selectedStayRoom && !roomStayRows.length"
        :description="`无停留房间为「${selectedStayRoom}」的记录`"
      />
      <el-empty
        v-else-if="result.videos.length && !videosWithSuspectSegments.length"
        description="当前结果中的视频源已无患者停留段（可能已全部移除）"
      />
      <el-empty v-else-if="!result.videos.length" description="无满足阈值的命中" />
    </template>
  </div>
</template>
