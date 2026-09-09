<script setup lang="ts">
import AuthImage from '@/components/AuthImage.vue'
import RoomCopresencePanel from '@/components/RoomCopresencePanel.vue'
import StaySegmentsPanel from '@/components/StaySegmentsPanel.vue'
import http from '@/api/http'
import { buildPlaybackUrl, checkFullVideo, videoLoadingHint } from '@/utils/videoPlay'
import {
  buildSegmentsFlat,
  fmtHms,
  hasStaySegments,
  logKindLabel,
  parseLogResultsJson,
  previewPathsFromLog,
  queryImageNamesFromLog,
  resolveHitPath,
} from '@/utils/staySegments'
import { useAuthStore } from '@/stores/auth'
import { ZoomIn } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

interface LogRow {
  id: number
  query_image_paths: string | null
  search_query: string | null
  algorithm: string
  threshold: number
  results_json: string
  created_at: string
}

const auth = useAuthStore()
const rows = ref<LogRow[]>([])
const detailOpen = ref(false)
const currentLog = ref<LogRow | null>(null)

const previewOpen = ref(false)
const previewPath = ref('')
const clipOpen = ref(false)
const clipUrl = ref<string | null>(null)
const clipSeekSec = ref(0)
const clipPlaybackVid = ref('')
const videoPlayerRef = ref<HTMLVideoElement | null>(null)
const videoLoading = ref(false)
const clipIsLive = ref(false)
const clipNeedsTranscode = ref(false)
const clipNeedsFaststart = ref(false)
const autoPlayHintShown = ref(false)
const initialSeekDone = ref(false)

const detailPayload = computed(() => {
  if (!currentLog.value) {
    return parseLogResultsJson('{}')
  }
  return parseLogResultsJson(currentLog.value.results_json)
})

const detailIsCopresence = computed(() => detailPayload.value.kind === 'room_copresence')
const detailStaySegments = computed(() => detailPayload.value.stay_segments ?? null)
const detailHasSegments = computed(
  () => !detailIsCopresence.value && hasStaySegments(detailStaySegments.value),
)
const detailSegmentsFlat = computed(() => buildSegmentsFlat(detailStaySegments.value))

const copresenceReplay = computed(() => {
  if (!detailIsCopresence.value || !currentLog.value) return null
  const videos = detailPayload.value.copresence_videos ?? []
  if (!videos.length) return null
  return {
    algorithm: currentLog.value.algorithm,
    videos,
  }
})

const queryImageNames = computed(() => {
  if (!currentLog.value) return [] as string[]
  if (detailIsCopresence.value) return []
  const raw = currentLog.value.query_image_paths
  if (!raw?.trim()) return []
  return raw
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
})

const queryImageNamesA = computed(() => {
  if (!currentLog.value) return []
  if (detailIsCopresence.value) {
    const names = queryImageNamesFromLog(detailPayload.value, 'a')
    if (names.length) return names
    const raw = currentLog.value.query_image_paths
    if (!raw?.trim()) return []
    return raw.split(',').map((s) => s.trim()).filter(Boolean)
  }
  return queryImageNames.value
})

const queryImageNamesB = computed(() => {
  if (!detailIsCopresence.value || !currentLog.value) return []
  return queryImageNamesFromLog(detailPayload.value, 'b')
})

function openPreview(path: string) {
  if (!path) return
  previewPath.value = path
  previewOpen.value = true
}

function closeClip() {
  clipOpen.value = false
  clipUrl.value = null
  clipPlaybackVid.value = ''
  clipSeekSec.value = 0
  videoLoading.value = false
  clipIsLive.value = false
  clipNeedsTranscode.value = false
  clipNeedsFaststart.value = false
  autoPlayHintShown.value = false
  initialSeekDone.value = false
}

async function playClip(vid: string | undefined, t: number) {
  if (!vid) {
    ElMessage.warning('该条命中缺少视频信息，无法播放')
    return
  }
  const target = Math.max(0, t)
  let chk
  try {
    chk = await checkFullVideo(vid)
    if (chk.is_live_recording) {
      ElMessage.info('直播中：将播放截至目前已录制的视频片段')
    } else if (chk.needs_transcode) {
      ElMessage.info('首次播放需转码为浏览器可播格式，请稍候（完成后可正常拖动进度条）')
    }
  } catch (e: unknown) {
    const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
    ElMessage.error(typeof msg === 'string' ? msg : '未找到可播放的完整视频')
    return
  }

  const nextUrl = buildPlaybackUrl(chk, vid)
  const reusePlayer =
    clipOpen.value && clipPlaybackVid.value === vid && clipUrl.value === nextUrl

  clipIsLive.value = !!chk.is_live_recording
  clipNeedsTranscode.value = !!chk.needs_transcode
  clipNeedsFaststart.value = !!chk.needs_faststart

  if (reusePlayer) {
    clipSeekSec.value = target
    initialSeekDone.value = false
    seekVideoToHit()
    return
  }

  closeClip()
  clipPlaybackVid.value = vid
  clipSeekSec.value = target
  videoLoading.value = true
  initialSeekDone.value = false
  clipOpen.value = true
  clipUrl.value = nextUrl
}

function seekVideoToHit() {
  const el = videoPlayerRef.value
  if (!el) return
  const target = clipSeekSec.value

  const apply = () => {
    if (!Number.isFinite(target) || target <= 0) {
      void el.play().catch(() => {
        if (!autoPlayHintShown.value) {
          ElMessage.info('请点击播放按钮开始播放')
          autoPlayHintShown.value = true
        }
      })
      return
    }
    const dur = el.duration
    if (!Number.isFinite(dur) || dur <= 0) return
    const safe = Math.min(Math.max(0, target), Math.max(0, dur - 0.05))
    const onSeeked = () => {
      el.removeEventListener('seeked', onSeeked)
      void el.play().catch(() => {
        if (!autoPlayHintShown.value) {
          ElMessage.info('请点击播放按钮开始播放')
          autoPlayHintShown.value = true
        }
      })
    }
    el.addEventListener('seeked', onSeeked)
    try {
      el.currentTime = safe
    } catch {
      el.removeEventListener('seeked', onSeeked)
    }
  }

  if (el.readyState >= 1 && Number.isFinite(el.duration) && el.duration > 0) {
    apply()
    return
  }
  el.addEventListener(
    'loadedmetadata',
    () => {
      apply()
    },
    { once: true },
  )
}

function onVideoCanPlay() {
  videoLoading.value = false
  if (initialSeekDone.value) return
  initialSeekDone.value = true
  seekVideoToHit()
}

function onVideoPlayError() {
  videoLoading.value = false
  ElMessage.error('完整视频加载失败，请确认 ffmpeg 可用且归档文件未损坏')
  closeClip()
}

async function load() {
  const { data } = await http.get<LogRow[]>('/search/logs')
  rows.value = data
}

function queryPhotoPath(fileName: string) {
  return `/files/query/${encodeURIComponent(fileName)}`
}

function openDetail(row: LogRow) {
  currentLog.value = row
  detailOpen.value = true
}

function fmtDateTime(iso: string | null | undefined): string {
  if (!iso?.trim()) return '--'
  const m = iso.trim().match(/^(\d{4}-\d{2}-\d{2})[T\s](\d{2}:\d{2}:\d{2})/)
  if (m) return `${m[1]} ${m[2]}`
  return iso.replace('T', ' ')
}

function formatAlgorithm(raw: string | null | undefined): string {
  if (!raw) return '图像检索'
  if (raw.toUpperCase() === 'SIGLIP') return '多模态检索'
  return '图像检索'
}

function rowPreviewPaths(row: LogRow): string[] {
  return previewPathsFromLog(row.results_json, 3)
}

/** 文字查询列：仅展示 SIGLIP 文搜图文案；双人共现等无文字检索的记为 — */
function displaySearchQuery(row: LogRow): string {
  const p = parseLogResultsJson(row.results_json)
  if (p.kind === 'room_copresence') return '—'
  const q = (row.search_query || '').trim()
  if (q === '双人房间共现') return '—'
  return q || '—'
}

async function remove(id: number) {
  await ElMessageBox.confirm('确定删除该日志？', '确认', { type: 'warning' })
  await http.delete(`/search/logs/${id}`)
  ElMessage.success('已删除')
  if (currentLog.value?.id === id) {
    detailOpen.value = false
    currentLog.value = null
  }
  await load()
}

onMounted(() => load().catch(() => ElMessage.error('加载失败')))
onBeforeUnmount(() => closeClip())
</script>

<template>
  <div class="space-y-5">
    <div class="rounded-xl border border-slate-200/70 bg-white/85 p-4 shadow-sm backdrop-blur">
      <div class="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 class="text-xl font-semibold text-slate-800">检索日志</h2>
          <p class="mt-1 text-sm text-slate-500">沉淀检索历史，支持快速回看命中画面并对结果进行复盘。</p>
        </div>
        <div class="rounded-lg bg-slate-50 px-3 py-2 text-center">
          <div class="text-xs text-slate-500">日志总数</div>
          <div class="text-lg font-semibold text-slate-700">{{ rows.length }}</div>
        </div>
      </div>
    </div>

    <el-card shadow="never" class="rounded-xl border border-slate-200/70">
      <el-table :data="rows" stripe border max-height="70vh">
        <el-table-column prop="id" label="ID" width="70" />
        <el-table-column label="时间" width="180">
          <template #default="{ row }">{{ fmtDateTime(row.created_at) }}</template>
        </el-table-column>
        <el-table-column label="类型" width="120">
          <template #default="{ row }">{{ logKindLabel(row.results_json, row.search_query) }}</template>
        </el-table-column>
        <el-table-column label="检索方式" width="120">
          <template #default="{ row }">{{ formatAlgorithm(row.algorithm) }}</template>
        </el-table-column>
        <el-table-column prop="threshold" label="阈值" width="80" />
        <el-table-column label="文字查询" min-width="120" show-overflow-tooltip>
          <template #default="{ row }">{{ displaySearchQuery(row) }}</template>
        </el-table-column>
        <el-table-column label="预览" min-width="160">
          <template #default="{ row }">
            <div class="flex flex-wrap gap-1">
              <template v-for="(p, i) in rowPreviewPaths(row)" :key="`${row.id}-${i}`">
                <div class="group relative h-14 w-14 overflow-hidden rounded border border-slate-200">
                  <AuthImage :path="p" class="h-full w-full" />
                  <el-button
                    type="primary"
                    circle
                    size="small"
                    class="!absolute bottom-0.5 right-0.5 !h-6 !w-6 !p-0 opacity-90 shadow group-hover:opacity-100"
                    title="放大"
                    @click.stop="openPreview(p)"
                  >
                    <el-icon class="text-xs"><ZoomIn /></el-icon>
                  </el-button>
                </div>
              </template>
            </div>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="140" fixed="right">
          <template #default="{ row }">
            <el-button type="primary" link @click="openDetail(row)">查看详情</el-button>
            <el-button v-if="auth.isAdmin" type="danger" link @click="remove(row.id)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog
      v-model="detailOpen"
      title="检索记录详情"
      width="min(1100px, 96vw)"
      destroy-on-close
      class="log-detail-dialog"
      @closed="currentLog = null"
    >
      <el-scrollbar max-height="75vh">
        <div v-if="currentLog" class="space-y-6 pr-2">
          <el-descriptions :column="2" border size="small">
            <el-descriptions-item label="ID">{{ currentLog.id }}</el-descriptions-item>
            <el-descriptions-item label="记录时间">{{ fmtDateTime(currentLog.created_at) }}</el-descriptions-item>
            <el-descriptions-item label="类型">
              {{ logKindLabel(currentLog.results_json, currentLog.search_query) }}
            </el-descriptions-item>
            <el-descriptions-item label="检索方式">{{ formatAlgorithm(currentLog.algorithm) }}</el-descriptions-item>
            <el-descriptions-item label="相似度阈值">{{ currentLog.threshold }}</el-descriptions-item>
            <el-descriptions-item v-if="detailPayload.time_gap != null" label="轨迹聚合间隔">
              {{ detailPayload.time_gap }} 秒
            </el-descriptions-item>
            <el-descriptions-item
              v-if="!detailIsCopresence"
              label="文字描述"
              :span="detailPayload.time_gap != null ? 1 : 2"
            >
              {{ currentLog.search_query || '—' }}
            </el-descriptions-item>
          </el-descriptions>

          <section v-if="detailIsCopresence">
            <h3 class="mb-2 text-base font-semibold text-slate-800">查询目标 · 患者</h3>
            <div v-if="queryImageNamesA.length" class="mb-4 flex flex-wrap gap-4">
              <div
                v-for="name in queryImageNamesA"
                :key="'a-' + name"
                class="w-40 shrink-0 overflow-hidden rounded-lg border border-slate-200 bg-slate-50"
              >
                <AuthImage :path="queryPhotoPath(name)" class="aspect-square" />
                <div class="flex flex-col gap-1 border-t border-slate-100 bg-white px-2 py-1.5">
                  <el-button size="small" @click="openPreview(queryPhotoPath(name))">放大</el-button>
                </div>
              </div>
            </div>
            <el-empty v-else class="mb-4" description="无患者查询图记录" :image-size="48" />
            <h3 class="mb-2 text-base font-semibold text-slate-800">查询目标 · 医生</h3>
            <div v-if="queryImageNamesB.length" class="flex flex-wrap gap-4">
              <div
                v-for="name in queryImageNamesB"
                :key="'b-' + name"
                class="w-40 shrink-0 overflow-hidden rounded-lg border border-slate-200 bg-slate-50"
              >
                <AuthImage :path="queryPhotoPath(name)" class="aspect-square" />
                <div class="flex flex-col gap-1 border-t border-slate-100 bg-white px-2 py-1.5">
                  <el-button size="small" @click="openPreview(queryPhotoPath(name))">放大</el-button>
                </div>
              </div>
            </div>
            <el-empty v-else description="无医生查询图记录" :image-size="48" />
          </section>

          <section v-else>
            <h3 class="mb-3 text-base font-semibold text-slate-800">查询目标（用户上传）</h3>
            <div v-if="queryImageNames.length" class="flex flex-wrap gap-4">
              <div
                v-for="name in queryImageNames"
                :key="name"
                class="w-40 shrink-0 overflow-hidden rounded-lg border border-slate-200 bg-slate-50"
              >
                <AuthImage :path="queryPhotoPath(name)" class="aspect-square" />
                <div class="flex flex-col gap-1 border-t border-slate-100 bg-white px-2 py-1.5">
                  <el-button size="small" @click="openPreview(queryPhotoPath(name))">放大</el-button>
                  <p class="truncate text-center text-xs text-slate-500" :title="name">{{ name }}</p>
                </div>
              </div>
            </div>
            <el-empty v-else description="本次检索未上传查询图片（可能为纯文字检索）" :image-size="64" />
          </section>

          <section v-if="detailIsCopresence && copresenceReplay">
            <h3 class="mb-3 text-base font-semibold text-slate-800">房内停留与共现分析</h3>
            <RoomCopresencePanel
              hide-form
              :replay-payload="copresenceReplay"
              @play-video="(vid: string, t: number) => playClip(vid, t)"
              @preview="openPreview"
            />
          </section>

          <section v-else-if="detailHasSegments">
            <h3 class="mb-3 text-base font-semibold text-slate-800">监控中出现段</h3>
            <div class="mb-3 flex flex-wrap gap-2 text-sm text-slate-600">
              <span>涉及视频 {{ new Set(detailSegmentsFlat.map((r) => r.video_name)).size }}</span>
              <span>·</span>
              <span>出现段 {{ detailSegmentsFlat.length }}</span>
              <span>·</span>
              <span>段内总帧数 {{ detailSegmentsFlat.reduce((a, r) => a + r.hit_count, 0) }}</span>
            </div>
            <StaySegmentsPanel
              :stay-segments="detailStaySegments"
              :show-hint="true"
              @preview="openPreview"
              @play="(p) => playClip(p.vid, p.time)"
            />
          </section>

          <section v-else-if="!detailIsCopresence">
            <h3 class="mb-3 text-base font-semibold text-slate-800">命中结果（按视频源划分）</h3>
            <el-alert
              v-if="detailPayload.legacyOnly"
              type="info"
              :closable="false"
              show-icon
              class="mb-3"
              title="该记录为旧版日志，仅保存了扁平命中列表；重新检索并保存后可显示「出现段」表格。"
            />
            <div v-if="Object.keys(detailPayload.results).length" class="space-y-4">
              <el-card
                v-for="(items, videoName) in detailPayload.results"
                :key="String(videoName)"
                shadow="never"
                class="border border-slate-200"
              >
                <template #header>
                  <span class="font-medium text-slate-700">视频源：{{ videoName }}</span>
                  <el-tag size="small" class="ml-2">{{ items.length }} 条</el-tag>
                </template>
                <div class="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4">
                  <div
                    v-for="(it, idx) in items"
                    :key="idx"
                    class="overflow-hidden rounded-lg border border-slate-100 bg-slate-50/80"
                  >
                    <AuthImage v-if="resolveHitPath(it)" :path="resolveHitPath(it)" class="aspect-video" />
                    <div class="flex flex-col gap-1.5 border-t border-slate-100 bg-white px-2 py-1.5">
                      <div class="flex justify-between text-xs text-slate-600">
                        <span>相似度 {{ it.score?.toFixed?.(2) ?? '—' }}</span>
                        <span>{{ fmtHms(it.time) }}</span>
                      </div>
                      <el-button
                        v-if="resolveHitPath(it)"
                        size="small"
                        @click="openPreview(resolveHitPath(it))"
                      >
                        放大
                      </el-button>
                      <el-button v-if="it.vid" size="small" type="primary" @click="playClip(it.vid, it.time)">
                        播放视频
                      </el-button>
                    </div>
                  </div>
                </div>
              </el-card>
            </div>
            <el-empty v-else description="无结构化命中结果" :image-size="64" />
          </section>
        </div>
      </el-scrollbar>
    </el-dialog>

    <el-dialog
      v-model="previewOpen"
      title="图片预览"
      width="min(1240px, 98vw)"
      destroy-on-close
      append-to-body
      @closed="previewPath = ''"
    >
      <div
        v-if="previewPath"
        class="flex max-h-[90vh] min-h-[240px] items-center justify-center overflow-auto rounded bg-slate-900/90 p-3"
      >
        <AuthImage :path="previewPath" fit="contain" class="w-full max-w-full" />
      </div>
    </el-dialog>

    <el-dialog
      v-model="clipOpen"
      :title="clipPlaybackVid ? `视频播放 · ${clipPlaybackVid}` : '视频播放'"
      width="min(1240px, 98vw)"
      destroy-on-close
      append-to-body
      @closed="closeClip"
    >
      <div
        v-if="clipUrl"
        class="flex max-h-[88vh] min-h-[240px] flex-col items-center justify-center rounded bg-black/95 p-2"
      >
        <p v-if="videoLoading" class="mb-2 text-sm text-slate-300">
          {{
            videoLoadingHint({
              isLive: clipIsLive,
              needsTranscode: clipNeedsTranscode,
              needsFaststart: clipNeedsFaststart,
            })
          }}
        </p>
        <p v-else-if="clipSeekSec > 0" class="mb-2 text-xs text-slate-300">
          已从 {{ fmtHms(Math.floor(clipSeekSec)) }} 处开始播放
        </p>
        <video
          ref="videoPlayerRef"
          :src="clipUrl"
          controls
          playsinline
          preload="auto"
          class="max-h-[84vh] w-full rounded bg-black"
          @canplay="onVideoCanPlay"
          @error="onVideoPlayError"
        />
      </div>
    </el-dialog>
  </div>
</template>
