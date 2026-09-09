<script setup lang="ts">
import RoomAnnotator from '@/components/RoomAnnotator.vue'
import http from '@/api/http'
import { formatServerDateTime, parseServerDate } from '@/utils/datetime'
import { useAuthStore } from '@/stores/auth'
import { ElMessage, ElMessageBox } from 'element-plus'
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'

interface Row {
  file_name: string
  status: string
  failure_reason?: string | null
  source_type: string
  task_id: string | null
  raw_path: string | null
  fps: number | null
  duration: string | null
  progress: number | null
  completed_at: string | null
  target_count: number | null
  created_at: string | null
  updated_at?: string | null
  processing_started_at?: string | null
  elapsed_seconds?: number | null
  room_count?: number
}

const auth = useAuthStore()
const rows = ref<Row[]>([])
let loadTimer: ReturnType<typeof setInterval> | null = null
let liveUiTimer: ReturnType<typeof setInterval> | null = null
/** 每秒递增，驱动直播中总耗时本地计时 */
const liveTick = ref(Date.now())
const roomCountCache = ref<Record<string, number>>({})

const roomDrawerOpen = ref(false)
const roomVideoName = ref('')

const totalCount = computed(() => rows.value.length)
const processingCount = computed(() => rows.value.filter((r) => r.status === 'processing').length)
const streamingCount = computed(() => rows.value.filter((r) => r.status === 'streaming').length)
const pendingCount = computed(() => rows.value.filter((r) => r.status === 'pending').length)
const transcodingCount = computed(() => rows.value.filter((r) => r.status === 'transcoding').length)
const failedCount = computed(() => rows.value.filter((r) => r.status === 'failed').length)

const hasActiveLive = computed(() =>
  rows.value.some((r) => isLiveRow(r) && (r.status === 'streaming' || r.status === 'pending')),
)

const hasActiveWork = computed(
  () =>
    hasActiveLive.value ||
    rows.value.some(
      (r) => r.status === 'processing' || r.status === 'pending' || r.status === 'transcoding',
    ),
)

function isLiveRow(row: Row) {
  return row.source_type === 'RTSP实时'
}

function hasGalleryCrops(row: Row) {
  return (row.target_count ?? 0) > 0
}

function canAnnotateRooms(row: Row) {
  if (row.status === 'completed' || row.status === 'transcoding') return true
  if (isLiveRow(row) && row.status === 'stopped') return true
  // 固定机位：直播中有入库 crop 即可用全画面底图标房间
  if (isLiveRow(row) && row.status === 'streaming' && hasGalleryCrops(row)) return true
  return false
}

function roomAnnotateHint(row: Row): string {
  if (row.status === 'pending') return '任务启动中，请稍候'
  if (row.status === 'processing') return '建模进行中，请完成后标注'
  if (row.status === 'transcoding') return '视频转码中，入库数据已可用'
  if (isLiveRow(row) && row.status === 'streaming' && !hasGalleryCrops(row)) {
    return '实时流需先有入库 crop 后再标注（固定机位）'
  }
  return '离线任务需建模完成后再标注'
}

function isProgressPercent(row: Row) {
  if (isLiveRow(row)) return false
  return row.status === 'processing' || row.status === 'pending' || row.status === 'transcoding'
}

function liveProgressLabel(row: Row) {
  if (!isLiveRow(row)) return ''
  if (row.status === 'pending' || row.duration === '启动中…' || row.duration === '排队中…') {
    return '启动中…'
  }
  if (row.status === 'streaming') {
    return `已入库 ${row.target_count ?? 0} 条`
  }
  if (row.status === 'stopped') {
    return `已入库 ${row.target_count ?? 0} 条`
  }
  return ''
}

function fmtElapsedLive(
  created: string | null,
  processingStarted: string | null | undefined,
  status: string,
) {
  void liveTick.value
  const startSrc = processingStarted || created
  if (!startSrc) return status === 'pending' ? '启动中…' : '直播中'
  const start = parseServerDate(startSrc)
  const sec = Math.max(0, Math.floor((Date.now() - start) / 1000))
  const suffix =
    status === 'pending' ? '(启动中)' : status === 'streaming' ? '(直播中)' : ''
  return suffix ? `${fmtDuration(sec)} ${suffix}` : fmtDuration(sec)
}

function canStopLive(row: Row) {
  return isLiveRow(row) && (row.status === 'streaming' || row.status === 'pending')
}

async function load() {
  const { data } = await http.get<Row[]>('/media/sources')
  rows.value = data.map((r) => {
    const cached = roomCountCache.value[r.file_name]
    return {
      ...r,
      room_count: typeof cached === 'number' ? cached : (r.room_count ?? 0),
    }
  })
}

function setupLoadTimer() {
  if (loadTimer) clearInterval(loadTimer)
  const ms = hasActiveLive.value ? 1000 : hasActiveWork.value ? 2000 : 3000
  loadTimer = setInterval(() => {
    load().catch(() => {})
  }, ms)
}

watch(hasActiveWork, () => setupLoadTimer())

function openRoomDrawer(row: Row) {
  if (!canAnnotateRooms(row)) return
  roomVideoName.value = row.file_name
  roomDrawerOpen.value = true
}

async function stopLiveRow(r: Row) {
  await ElMessageBox.confirm(
    `确定终止「${r.file_name}」？将停止后续拉流分析，已入库的检索数据会保留。`,
    '终止实时流',
    { type: 'warning' },
  )
  const fd = new URLSearchParams()
  fd.append('file_name', r.file_name)
  try {
    await http.post('/stream/live/stop', fd, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    })
    ElMessage.success('已终止，可继续检索与标注')
    await load()
  } catch (err: any) {
    ElMessage.error(err?.response?.data?.detail || '终止失败')
    await load()
  }
}

function onRoomsChanged(count: number) {
  roomCountCache.value[roomVideoName.value] = count
  const r = rows.value.find((x) => x.file_name === roomVideoName.value)
  if (r) r.room_count = count
}

function roomLabel(row: Row): string {
  const n = row.room_count ?? 0
  if (n > 0) return `已标 ${n} 间`
  return '标注'
}

async function removeRow(r: Row) {
  await ElMessageBox.confirm(`确定删除「${r.file_name}」及关联底库？`, '确认', { type: 'warning' })
  await http.delete('/media/sources', { params: { file_name: r.file_name } })
  ElMessage.success('已删除')
  if (roomVideoName.value === r.file_name) {
    roomDrawerOpen.value = false
    roomVideoName.value = ''
  }
  delete roomCountCache.value[r.file_name]
  await load()
}

/** 完成时间等：按东八区墙钟显示 */
function fmtDateTime(iso: string | null | undefined): string {
  return formatServerDateTime(iso)
}

function fmtDuration(totalSeconds: number): string {
  let sec = Math.max(0, Math.floor(totalSeconds))
  const h = Math.floor(sec / 3600)
  sec %= 3600
  const m = Math.floor(sec / 60)
  const s = sec % 60
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

/** 直播中：用 processing_started_at + 本地时钟，不依赖后端写库 */
function fmtElapsedStreaming(
  processingStarted: string | null | undefined,
  created: string | null,
) {
  void liveTick.value
  const startSrc = processingStarted || created
  if (!startSrc) return '直播中'
  const start = parseServerDate(startSrc)
  const sec = Math.max(0, Math.floor((Date.now() - start) / 1000))
  return `${fmtDuration(sec)} (直播中)`
}

function fmtDurationFps(row: Row) {
  if (isLiveRow(row) && (row.status === 'streaming' || row.status === 'pending')) {
    if (row.duration === '启动中…' || row.duration === '排队中…') return '启动中…'
    return fmtElapsedStreaming(row.processing_started_at, row.created_at)
  }
  const fps =
    row.fps != null && row.fps > 0 && row.fps <= 120 ? row.fps : null
  return `${row.duration || '--'} / ${fps ?? '--'}`
}

function fmtElapsed(
  created: string | null,
  completed: string | null,
  updated: string | null | undefined,
  status: string,
  processingStarted: string | null | undefined,
  elapsedSeconds?: number | null,
  sourceType?: string,
) {
  void liveTick.value
  if (sourceType === 'RTSP实时' && (status === 'pending' || status === 'streaming')) {
    return fmtElapsedLive(created, processingStarted, status)
  }
  if (status === 'pending') return '--'
  if (status === 'streaming') {
    return fmtElapsedStreaming(processingStarted, created)
  }
  if (status === 'processing' || status === 'transcoding') {
    const startSrc = processingStarted || created
    if (!startSrc) return '--'
    const start = parseServerDate(startSrc)
    const sec = Math.max(0, Math.floor((Date.now() - start) / 1000))
    return `${fmtDuration(sec)} (进行中)`
  }
  if (typeof elapsedSeconds === 'number' && Number.isFinite(elapsedSeconds)) {
    const base = fmtDuration(elapsedSeconds)
    if (status === 'stopped') return `${base} (已停止)`
    return base
  }
  if (!created) return '--'
  const end = parseServerDate(completed || updated || created)
  const start = processingStarted
    ? parseServerDate(processingStarted)
    : parseServerDate(created)
  const sec = Math.max(0, Math.floor((end - start) / 1000))
  const base = fmtDuration(sec)
  if (status === 'stopped') return `${base} (已停止)`
  return status === 'completed' || status === 'failed' ? base : `${base} (进行中)`
}

onMounted(() => {
  load().catch(() => ElMessage.error('加载失败'))
  liveUiTimer = setInterval(() => {
    liveTick.value = Date.now()
  }, 1000)
  setupLoadTimer()
})
onBeforeUnmount(() => {
  if (loadTimer) clearInterval(loadTimer)
  if (liveUiTimer) clearInterval(liveUiTimer)
})
</script>

<template>
  <div class="space-y-5">
    <div class="rounded-xl border border-slate-200/70 bg-white/85 p-4 shadow-sm backdrop-blur">
      <div class="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 class="text-xl font-semibold text-slate-800">视频源列表</h2>
          <p class="mt-1 text-sm text-slate-500">
            统一查看任务状态、处理进度与失败原因；建模有数据后可标注房间（实时流直播中亦可），支持一键删除清理。
          </p>
        </div>
        <div class="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
          <div class="rounded-lg bg-slate-50 px-3 py-2 text-center">
            <div class="text-xs text-slate-500">任务总数</div>
            <div class="text-lg font-semibold text-slate-800">{{ totalCount }}</div>
          </div>
          <div class="rounded-lg bg-violet-50 px-3 py-2 text-center">
            <div class="text-xs text-violet-600">直播中</div>
            <div class="text-lg font-semibold text-violet-700">{{ streamingCount }}</div>
          </div>
          <div class="rounded-lg bg-sky-50 px-3 py-2 text-center">
            <div class="text-xs text-sky-600">处理中</div>
            <div class="text-lg font-semibold text-sky-700">{{ processingCount }}</div>
          </div>
          <div class="rounded-lg bg-indigo-50 px-3 py-2 text-center">
            <div class="text-xs text-indigo-600">转码中</div>
            <div class="text-lg font-semibold text-indigo-700">{{ transcodingCount }}</div>
          </div>
          <div class="rounded-lg bg-amber-50 px-3 py-2 text-center">
            <div class="text-xs text-amber-600">等待中</div>
            <div class="text-lg font-semibold text-amber-700">{{ pendingCount }}</div>
          </div>
          <div class="rounded-lg bg-rose-50 px-3 py-2 text-center">
            <div class="text-xs text-rose-600">失败</div>
            <div class="text-lg font-semibold text-rose-700">{{ failedCount }}</div>
          </div>
        </div>
      </div>
    </div>

    <el-card shadow="never" class="rounded-xl border border-slate-200/70">
      <el-table :data="rows" stripe border style="width: 100%" max-height="70vh">
        <el-table-column label="视频名称" min-width="440">
          <template #default="{ row }">
            <span class="inline-block w-full whitespace-normal break-all">{{ row.file_name }}</span>
          </template>
        </el-table-column>
        <el-table-column label="时长/帧率" width="136">
          <template #default="{ row }">{{ fmtDurationFps(row) }}</template>
        </el-table-column>
        <el-table-column label="进度 / 入库" width="200">
          <template #default="{ row }">
            <el-progress v-if="isProgressPercent(row)" :percentage="row.progress ?? 0" :stroke-width="10" />
            <span v-else-if="liveProgressLabel(row)" class="text-sm text-violet-700">
              {{ liveProgressLabel(row) }}
            </span>
            <span v-else-if="row.status === 'stopped'" class="text-sm text-violet-700">
              已入库 {{ row.target_count ?? 0 }} 条
            </span>
            <span v-else class="text-sm text-slate-500">—</span>
          </template>
        </el-table-column>
        <el-table-column label="总耗时" width="126" show-overflow-tooltip>
          <template #default="{ row }">
            {{
              fmtElapsed(
                row.created_at,
                row.completed_at,
                row.updated_at,
                row.status,
                row.processing_started_at,
                row.elapsed_seconds,
                row.source_type,
              )
            }}
          </template>
        </el-table-column>
        <el-table-column label="完成时间" width="168" show-overflow-tooltip>
          <template #default="{ row }">
            {{ fmtDateTime(row.completed_at) }}
          </template>
        </el-table-column>
        <el-table-column label="检测到人数" width="100">
          <template #default="{ row }">
            {{ row.target_count == null ? '--' : `${row.target_count} 人` }}
          </template>
        </el-table-column>
        <el-table-column prop="status" label="状态" width="108">
          <template #default="{ row }">
            <el-tag v-if="row.status === 'completed'" type="success">完成</el-tag>
            <el-tag v-else-if="row.status === 'streaming'" type="primary" effect="dark">
              {{ row.duration === '启动中…' ? '启动中' : '直播中' }}
            </el-tag>
            <el-tag v-else-if="row.status === 'stopped'" type="info">已停止</el-tag>
            <el-tag v-else-if="row.status === 'processing'" type="primary" effect="dark">处理中</el-tag>
            <el-tag v-else-if="row.status === 'transcoding'" type="info" effect="dark">转码中</el-tag>
            <el-tag v-else-if="row.status === 'failed'" type="danger">失败</el-tag>
            <el-tag v-else-if="row.status === 'pending' && isLiveRow(row)" type="warning">启动中</el-tag>
            <el-tag v-else-if="row.status === 'pending'" type="warning">等待中</el-tag>
            <el-tag v-else type="warning">等待</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="失败说明" min-width="180" show-overflow-tooltip>
          <template #default="{ row }">
            <span v-if="row.status === 'failed' && row.failure_reason" class="text-rose-700 text-sm">
              {{ row.failure_reason }}
            </span>
            <span v-else-if="row.status === 'failed'" class="text-slate-400 text-sm">—</span>
          </template>
        </el-table-column>
        <el-table-column label="房间标注" width="112" fixed="right">
          <template #default="{ row }">
            <el-button
              v-if="canAnnotateRooms(row)"
              type="primary"
              link
              @click="openRoomDrawer(row)"
            >
              {{ roomLabel(row) }}
            </el-button>
            <el-tooltip v-else :content="roomAnnotateHint(row)" placement="top">
              <span class="text-slate-400 text-sm">—</span>
            </el-tooltip>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="140" fixed="right">
          <template #default="{ row }">
            <el-button
              v-if="canStopLive(row)"
              type="warning"
              link
              @click="stopLiveRow(row)"
            >
              终止
            </el-button>
            <el-button v-if="auth.isAdmin" type="danger" link @click="removeRow(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-drawer
      v-model="roomDrawerOpen"
      :title="roomVideoName ? `房间标注 · ${roomVideoName}` : '房间标注'"
      direction="rtl"
      size="920px"
      destroy-on-close
    >
      <RoomAnnotator
        v-if="roomDrawerOpen && roomVideoName"
        :video-name="roomVideoName"
        embedded
        @rooms-changed="onRoomsChanged"
      />
    </el-drawer>
  </div>
</template>
