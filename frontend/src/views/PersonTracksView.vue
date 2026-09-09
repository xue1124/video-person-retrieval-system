<script setup lang="ts">
import AuthImage from '@/components/AuthImage.vue'
import {
  fetchPersonArchive,
  fetchTrackingPeople,
  fetchTrackingRooms,
  fetchTrackingStatus,
  fetchTrackingVideos,
  type PersonArchive,
  type StaySegmentRow,
  type TrackingPerson,
  type TrackingRoom,
  type TrackingVideo,
} from '@/api/tracking'
import { ElMessage } from 'element-plus'
import { computed, onMounted, ref, watch } from 'vue'

defineOptions({
  name: 'PersonTracksView',
})

const statusReady = ref(false)
const statusDetail = ref('')
const loadingFilters = ref(false)
const loadingArchive = ref(false)
const loadingPoints = ref(false)

const videos = ref<TrackingVideo[]>([])
const rooms = ref<TrackingRoom[]>([])
const people = ref<TrackingPerson[]>([])

const selectedVideoId = ref<number | null>(null)
const selectedRoomId = ref<number | null>(null)
const selectedPersonId = ref<number | null>(null)

const archive = ref<PersonArchive | null>(null)
const showPoints = ref(false)
const previewOpen = ref(false)
const previewPath = ref('')

const roomOptions = computed(() => {
  if (selectedVideoId.value == null) return rooms.value
  return rooms.value.filter(
    (r) => r.video_id == null || r.video_id === selectedVideoId.value,
  )
})

const roomColorMap = computed(() => {
  const names = archive.value?.overview.rooms_visited.map((r) => r.name) || []
  const palette = [
    '#0ea5e9',
    '#10b981',
    '#f59e0b',
    '#8b5cf6',
    '#ef4444',
    '#14b8a6',
    '#f97316',
    '#6366f1',
  ]
  const map: Record<string, string> = {}
  names.forEach((name, i) => {
    map[name] = palette[i % palette.length]
  })
  map['未标注区域'] = '#94a3b8'
  return map
})

function fmtScore(value: number | null | undefined) {
  if (value == null || Number.isNaN(Number(value))) return '—'
  return Number(value).toFixed(3)
}

function roomColor(name: string | null | undefined) {
  if (!name) return '#94a3b8'
  return roomColorMap.value[name] || '#64748b'
}

function openPreview(path: string | null | undefined) {
  if (!path) return
  previewPath.value = path
  previewOpen.value = true
}

function errMsg(e: unknown) {
  return (
    (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
    (e as Error)?.message ||
    '请求失败'
  )
}

async function loadStatus() {
  try {
    const st = await fetchTrackingStatus()
    statusReady.value = st.ready
    statusDetail.value = st.detail
    if (!st.ready) {
      ElMessage.warning(st.detail || '轨迹档案库未就绪')
    }
  } catch (e) {
    statusReady.value = false
    statusDetail.value = errMsg(e)
  }
}

async function loadVideosAndRooms() {
  loadingFilters.value = true
  try {
    const [v, r] = await Promise.all([
      fetchTrackingVideos(),
      fetchTrackingRooms(selectedVideoId.value),
    ])
    videos.value = v
    rooms.value = r
  } catch (e) {
    ElMessage.error(errMsg(e))
  } finally {
    loadingFilters.value = false
  }
}

async function loadPeople() {
  if (selectedVideoId.value == null) {
    people.value = []
    selectedPersonId.value = null
    archive.value = null
    return
  }
  loadingFilters.value = true
  try {
    people.value = await fetchTrackingPeople({
      video_id: selectedVideoId.value,
      room_id: selectedRoomId.value,
    })
    if (
      selectedPersonId.value != null &&
      !people.value.some((p) => p.id === selectedPersonId.value)
    ) {
      selectedPersonId.value = null
      archive.value = null
    }
  } catch (e) {
    ElMessage.error(errMsg(e))
  } finally {
    loadingFilters.value = false
  }
}

async function loadArchive(includePoints = false) {
  if (selectedPersonId.value == null) {
    archive.value = null
    return
  }
  if (includePoints) loadingPoints.value = true
  else loadingArchive.value = true
  try {
    archive.value = await fetchPersonArchive(selectedPersonId.value, {
      video_id: selectedVideoId.value,
      room_id: selectedRoomId.value,
      include_points: includePoints,
      points_limit: 500,
    })
    if (!includePoints) showPoints.value = false
  } catch (e) {
    ElMessage.error(errMsg(e))
    archive.value = null
  } finally {
    loadingArchive.value = false
    loadingPoints.value = false
  }
}

async function onTogglePoints(open: boolean) {
  showPoints.value = open
  if (open && archive.value && !archive.value.points_truncated) {
    await loadArchive(true)
  }
}

function maxEndSec(segments: StaySegmentRow[]) {
  return Math.max(...segments.map((s) => Number(s.end_sec || 0)), 1)
}

function barLeft(seg: StaySegmentRow, maxSec: number) {
  return `${(Number(seg.start_sec || 0) / maxSec) * 100}%`
}

function barWidth(seg: StaySegmentRow, maxSec: number) {
  const dur = Math.max(Number(seg.duration_sec || 0), 0.5)
  return `${(dur / maxSec) * 100}%`
}

watch(selectedVideoId, async () => {
  selectedRoomId.value = null
  await loadVideosAndRooms()
  await loadPeople()
  if (selectedPersonId.value != null) await loadArchive()
})

watch(selectedRoomId, async () => {
  await loadPeople()
  if (selectedPersonId.value != null) await loadArchive()
})

watch(selectedPersonId, async () => {
  await loadArchive()
})

onMounted(async () => {
  await loadStatus()
  if (!statusReady.value) return
  await loadVideosAndRooms()
})
</script>

<template>
  <div class="space-y-4 p-1">
    <el-alert
      v-if="!statusReady"
      type="warning"
      :closable="false"
      title="轨迹档案库未就绪"
      :description="statusDetail || '请确认 TRACKING_DB_URL 已配置且 medical_audit_v3 可访问'"
    />

    <el-card shadow="never" class="rounded-xl border border-slate-200/80">
      <div class="flex flex-wrap items-end gap-4">
        <div class="min-w-[220px] flex-1">
          <div class="mb-1 text-xs text-slate-500">视频</div>
          <el-select
            v-model="selectedVideoId"
            clearable
            filterable
            placeholder="全部视频"
            class="w-full"
            :disabled="!statusReady"
            :loading="loadingFilters"
          >
            <el-option
              v-for="v in videos"
              :key="v.id"
              :label="`${v.file_name}（G×${v.person_count}）`"
              :value="v.id"
            />
          </el-select>
        </div>
        <div class="min-w-[180px] flex-1">
          <div class="mb-1 text-xs text-slate-500">房间 / 区域（可选）</div>
          <el-select
            v-model="selectedRoomId"
            clearable
            filterable
            placeholder="全部区域"
            class="w-full"
            :disabled="!statusReady"
          >
            <el-option
              v-for="r in roomOptions"
              :key="r.id"
              :label="r.name"
              :value="r.id"
            />
          </el-select>
        </div>
        <div class="min-w-[240px] flex-[1.2]">
          <div class="mb-1 text-xs text-slate-500">人物 G（先选视频）</div>
          <el-select
            v-model="selectedPersonId"
            clearable
            filterable
            placeholder="请先选择视频"
            class="w-full"
            :disabled="!statusReady || selectedVideoId == null"
            :loading="loadingFilters"
          >
            <el-option
              v-for="p in people"
              :key="p.id"
              :label="p.label"
              :value="p.id"
            />
          </el-select>
        </div>
      </div>
    </el-card>

    <el-skeleton :loading="loadingArchive" animated :rows="8">
      <template #default>
        <template v-if="archive">
          <el-card shadow="never" class="rounded-xl border border-slate-200/80">
            <template #header>
              <div class="flex flex-wrap items-center justify-between gap-2">
                <div class="text-base font-semibold text-slate-800">
                  人物档案 · G{{ archive.overview.id }}
                </div>
                <el-tag size="small" type="info">{{ archive.overview.status }}</el-tag>
              </div>
            </template>
            <div class="mb-4 flex flex-wrap gap-4">
              <div
                class="w-28 shrink-0 cursor-pointer overflow-hidden rounded-lg border border-slate-200 bg-slate-50"
                @click="openPreview(archive.overview.primary_snapshot?.url)"
              >
                <AuthImage
                  v-if="archive.overview.primary_snapshot?.url"
                  :path="archive.overview.primary_snapshot.url"
                  class="aspect-[3/4] w-full"
                />
                <div
                  v-else
                  class="flex aspect-[3/4] items-center justify-center text-xs text-slate-400"
                >
                  暂无主图
                </div>
                <div class="border-t border-slate-100 px-2 py-1 text-center text-xs text-slate-500">
                  主头像
                </div>
              </div>
              <div class="min-w-0 flex-1">
                <el-descriptions :column="3" border size="small">
                  <el-descriptions-item label="首次出现">
                    {{ archive.overview.first_seen_at || '—' }}
                  </el-descriptions-item>
                  <el-descriptions-item label="末次出现">
                    {{ archive.overview.last_seen_at || '—' }}
                  </el-descriptions-item>
                  <el-descriptions-item label="累计停留">
                    {{ archive.overview.stay_hms_total }}
                  </el-descriptions-item>
                  <el-descriptions-item label="跨视频数">
                    {{ archive.overview.video_count }}
                  </el-descriptions-item>
                  <el-descriptions-item label="视频内人物数 P">
                    {{ archive.overview.video_person_count }}
                  </el-descriptions-item>
                  <el-descriptions-item label="轨迹数 T">
                    {{ archive.overview.track_count }}
                  </el-descriptions-item>
                  <el-descriptions-item label="停留段数">
                    {{ archive.overview.stay_segment_count }}
                  </el-descriptions-item>
                  <el-descriptions-item label="出现房间" :span="2">
                    <div class="flex flex-wrap gap-1.5">
                      <el-tag
                        v-for="r in archive.overview.rooms_visited"
                        :key="r.name"
                        size="small"
                        effect="plain"
                        :style="{ borderColor: roomColor(r.name), color: roomColor(r.name) }"
                      >
                        {{ r.name }} · {{ r.stay_hms }}
                      </el-tag>
                      <span v-if="!archive.overview.rooms_visited.length" class="text-slate-400">
                        暂无停留数据
                      </span>
                    </div>
                  </el-descriptions-item>
                </el-descriptions>
              </div>
            </div>

            <div class="mb-2 text-sm font-medium text-slate-700">成员映射</div>
            <div class="mb-3 flex flex-wrap gap-3">
              <div
                v-for="m in archive.overview.members"
                :key="m.video_person_id"
                class="w-36 overflow-hidden rounded-lg border border-slate-200 bg-white"
              >
                <div class="cursor-pointer bg-slate-50" @click="openPreview(m.thumb_url)">
                  <AuthImage v-if="m.thumb_url" :path="m.thumb_url" class="aspect-[3/4] w-full" />
                  <div
                    v-else
                    class="flex aspect-[3/4] items-center justify-center text-xs text-slate-400"
                  >
                    无图
                  </div>
                </div>
                <div class="space-y-0.5 px-2 py-1.5 text-xs">
                  <div class="font-medium text-slate-700">{{ m.label }}</div>
                  <div class="truncate text-slate-500" :title="m.file_name">{{ m.file_name }}</div>
                </div>
              </div>
            </div>

            <div v-if="archive.overview.candidate_snapshots?.length">
              <div class="mb-2 text-sm font-medium text-slate-700">候选代表图</div>
              <div class="flex flex-wrap gap-2">
                <div
                  v-for="s in archive.overview.candidate_snapshots"
                  :key="s.id"
                  class="w-16 cursor-pointer overflow-hidden rounded border border-slate-200"
                  :class="s.is_primary ? 'ring-2 ring-sky-400' : ''"
                  @click="openPreview(s.url)"
                >
                  <AuthImage v-if="s.url" :path="s.url" class="aspect-[3/4] w-full" />
                </div>
              </div>
            </div>
          </el-card>

          <el-card shadow="never" class="rounded-xl border border-slate-200/80">
            <template #header>
              <div class="text-base font-semibold text-slate-800">停留时间线</div>
            </template>
            <div v-if="archive.timeline_by_video.length" class="space-y-5">
              <div
                v-for="group in archive.timeline_by_video"
                :key="group.file_name"
                class="space-y-2"
              >
                <div class="flex items-center justify-between text-sm">
                  <span class="font-medium text-slate-700">{{ group.file_name }}</span>
                  <span class="text-slate-400">{{ group.segments.length }} 段</span>
                </div>
                <div class="relative h-9 overflow-hidden rounded-md bg-slate-100">
                  <div
                    v-for="seg in group.segments"
                    :key="seg.id"
                    class="absolute top-1.5 h-6 rounded-sm opacity-90 transition hover:opacity-100"
                    :style="{
                      left: barLeft(seg, maxEndSec(group.segments)),
                      width: barWidth(seg, maxEndSec(group.segments)),
                      background: roomColor(seg.room_name),
                      minWidth: '4px',
                    }"
                    :title="`${seg.room_name || '未标注'} ${seg.start_hms}-${seg.end_hms} (${seg.duration_hms})`"
                  />
                </div>
                <div class="flex flex-wrap gap-2 text-xs text-slate-500">
                  <span
                    v-for="seg in group.segments"
                    :key="'leg-' + seg.id"
                    class="inline-flex items-center gap-1"
                  >
                    <i
                      class="inline-block h-2.5 w-2.5 rounded-sm"
                      :style="{ background: roomColor(seg.room_name) }"
                    />
                    {{ seg.room_name || '未标注' }} {{ seg.start_hms }}–{{ seg.end_hms }}
                  </span>
                </div>
              </div>
            </div>
            <el-empty v-else description="暂无停留时间线（可能尚未标注房间或未计算停留）" :image-size="64" />
          </el-card>

          <el-card shadow="never" class="rounded-xl border border-slate-200/80">
            <template #header>
              <div class="text-base font-semibold text-slate-800">停留明细</div>
            </template>
            <el-table :data="archive.stays" stripe border size="small" max-height="420">
              <el-table-column label="现场图" width="78">
                <template #default="{ row }">
                  <div
                    class="h-12 w-10 cursor-pointer overflow-hidden rounded border border-slate-200 bg-slate-50"
                    @click="openPreview(row.snapshot_url)"
                  >
                    <AuthImage
                      v-if="row.snapshot_url"
                      :path="row.snapshot_url"
                      class="h-12 w-10"
                    />
                  </div>
                </template>
              </el-table-column>
              <el-table-column prop="file_name" label="视频" min-width="160" show-overflow-tooltip />
              <el-table-column label="视频内人物" width="100">
                <template #default="{ row }">P{{ row.local_person_no }}</template>
              </el-table-column>
              <el-table-column label="房间" min-width="120">
                <template #default="{ row }">
                  <span :style="{ color: roomColor(row.room_name) }">
                    {{ row.room_name || '—' }}
                  </span>
                </template>
              </el-table-column>
              <el-table-column label="起止" min-width="140">
                <template #default="{ row }">{{ row.start_hms }} — {{ row.end_hms }}</template>
              </el-table-column>
              <el-table-column prop="duration_hms" label="停留" width="90" />
              <el-table-column prop="point_count" label="轨迹点数" width="90" />
              <el-table-column label="进入时间" min-width="150" show-overflow-tooltip>
                <template #default="{ row }">{{ row.entered_at || '—' }}</template>
              </el-table-column>
              <el-table-column label="离开时间" min-width="150" show-overflow-tooltip>
                <template #default="{ row }">{{ row.exited_at || '—' }}</template>
              </el-table-column>
            </el-table>
          </el-card>

          <el-card shadow="never" class="rounded-xl border border-slate-200/80">
            <el-collapse>
              <el-collapse-item name="points">
                <template #title>
                  <div class="flex w-full items-center justify-between pr-4">
                    <span class="font-semibold text-slate-800">原始轨迹点（默认折叠）</span>
                    <el-button
                      size="small"
                      type="primary"
                      link
                      :loading="loadingPoints"
                      @click.stop="onTogglePoints(!showPoints)"
                    >
                      {{ archive.points_truncated ? '已加载' : '点击加载' }}
                    </el-button>
                  </div>
                </template>
                <el-table
                  v-if="archive.points_truncated"
                  :data="archive.points"
                  stripe
                  border
                  size="small"
                  max-height="360"
                >
                  <el-table-column prop="file_name" label="视频" min-width="140" show-overflow-tooltip />
                  <el-table-column label="合并前 ID" width="100">
                    <template #default="{ row }">
                      {{ row.pre_merge_id != null ? row.pre_merge_id : '—' }}
                    </template>
                  </el-table-column>
                  <el-table-column label="合并后 ID" width="100">
                    <template #default="{ row }">P{{ row.local_person_no }}</template>
                  </el-table-column>
                  <el-table-column prop="timestamp_hms" label="时刻" width="90" />
                  <el-table-column prop="room_name" label="房间" min-width="110" />
                  <el-table-column label="脚底点" min-width="120">
                    <template #default="{ row }">
                      {{ row.foot?.[0] ?? '—' }}, {{ row.foot?.[1] ?? '—' }}
                    </template>
                  </el-table-column>
                  <el-table-column label="YOLO得分" width="100">
                    <template #default="{ row }">{{ fmtScore(row.confidence) }}</template>
                  </el-table-column>
                  <el-table-column label="分配相似度" width="110">
                    <template #default="{ row }">{{ fmtScore(row.assign_score) }}</template>
                  </el-table-column>
                  <el-table-column label="合并相似度" width="110">
                    <template #default="{ row }">{{ fmtScore(row.merge_score) }}</template>
                  </el-table-column>
                </el-table>
                <el-empty
                  v-else
                  description="展开后点击「点击加载」拉取轨迹点"
                  :image-size="48"
                />
              </el-collapse-item>
            </el-collapse>
          </el-card>
        </template>

        <el-empty
          v-else-if="statusReady"
          description="请选择人物 G 查看轨迹档案"
          :image-size="80"
        />
      </template>
    </el-skeleton>

    <el-dialog v-model="previewOpen" title="人物快照" width="min(520px, 92vw)" destroy-on-close>
      <AuthImage v-if="previewPath" :path="previewPath" fit="contain" class="w-full" />
    </el-dialog>
  </div>
</template>
