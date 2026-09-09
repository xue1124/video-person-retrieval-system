<script setup lang="ts">
import AuthImage from '@/components/AuthImage.vue'
import RoomCopresencePanel from '@/components/RoomCopresencePanel.vue'
import StaySegmentsPanel from '@/components/StaySegmentsPanel.vue'
import http from '@/api/http'
import {
  buildSegmentsFlat,
  fmtHms,
  recomputeStaySegmentSummary,
  type Hit,
  type StaySegment,
} from '@/utils/staySegments'
import type { UploadUserFile } from 'element-plus'
import { ElMessage } from 'element-plus'
import { buildPlaybackUrl, checkFullVideo, videoLoadingHint } from '@/utils/videoPlay'
import { apiErrorMessage } from '@/utils/apiError'
import { computed, onBeforeUnmount, reactive, ref, watch } from 'vue'

const previewOpen = ref(false)
const previewPath = ref('')
const clipOpen = ref(false)

const searchTab = ref<'single' | 'dual'>('single')
const algorithm = ref<'OSNet' | 'SIGLIP'>('OSNet')
const threshold = ref(0.85)
const timeGap = ref(60)
const groupMode = ref(false)
const coTime = ref(0)
const qText = ref('')
const uploadFileList = ref<UploadUserFile[]>([])

const results = ref<Record<string, Hit[]> | null>(null)
/** 与 yolo_osnet_query_search_streamlit 一致：阈值内全量命中按 time_gap 合并的停留段及段内 crops */
const staySegments = ref<Record<string, StaySegment[]> | null>(null)
const lastMeta = reactive({
  algorithm: '',
  query_image_names: '',
  threshold: 0,
  qText: '',
})

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

const fmtTime = fmtHms

function openPreview(path: string) {
  if (!path) return
  previewPath.value = path
  previewOpen.value = true
}

/** 从结构化段中删除一帧（仅前端展示；保存日志时会写入当前 stay_segments） */
function removeStayCrop(videoName: string, segmentIdx: number, cropIndex: number) {
  const m = staySegments.value
  if (!m?.[videoName]) return
  const seg = m[videoName].find((s) => s.segment_idx === segmentIdx)
  if (!seg?.crops?.length) return
  seg.crops.splice(cropIndex, 1)
  if (!seg.crops.length) {
    m[videoName] = m[videoName].filter((s) => s.segment_idx !== segmentIdx)
    if (!m[videoName].length) delete m[videoName]
    if (!Object.keys(m).length) staySegments.value = null
    ElMessage.success('已删除该帧，本段已无画面已移除')
    return
  }
  recomputeStaySegmentSummary(seg)
  ElMessage.success('已移除该帧')
}

const canSearch = computed(() => {
  if (algorithm.value === 'OSNet') return uploadFileList.value.some((u) => u.raw)
  return uploadFileList.value.some((u) => u.raw) || !!qText.value.trim()
})
const hasQueryImage = computed(() => uploadFileList.value.some((u) => u.raw))
const hasQueryText = computed(() => !!qText.value.trim())
const siglipMixedSearch = computed(
  () => algorithm.value === 'SIGLIP' && hasQueryText.value && hasQueryImage.value,
)
/** 纯文搜图：展示分 0~100 */
const useTextPercentScore = computed(
  () => algorithm.value === 'SIGLIP' && hasQueryText.value && !hasQueryImage.value,
)
const thresholdMin = computed(() => (useTextPercentScore.value ? 65 : 0.1))
const thresholdMax = computed(() => (useTextPercentScore.value ? 92 : 0.95))
const thresholdStep = computed(() => (useTextPercentScore.value ? 1 : 0.01))

watch(useTextPercentScore, (pct, prev) => {
  if (pct === prev) return
  threshold.value = pct ? 80 : 0.85
})

const thresholdGuides = computed(() => {
  if (algorithm.value === 'OSNet') {
    return [
      {
      title: '图搜人阈值建议',
      range: '经验范围：相似度通常集中在 0.7 - 1.0。',
      start: '建议先设为 0.9，先拿到高精度结果，尽量避免混入不相关图片。',
      },
    ]
  }
  const imageGuide = {
    title: '多模态图搜图阈值建议',
    range: '经验范围：相似度通常集中在 0.7 - 1.0。',
    start: '建议先设为 0.9，先拿到高精度结果，尽量避免混入不相关图片。',
  }
  const textGuide = {
    title: '文搜图',
    range: '只填文字描述，不上传图片。',
    start: '阈值建议从 80 起试：结果太少就调低，误报多就调高。',
  }
  const mixedGuide = {
    title: '图文混搜',
    range: '同时上传参考图并填写文字描述。',
    start: '需同时像图又像文才会命中；图像阈值用下方滑块（建议 0.85）。',
  }
  if (siglipMixedSearch.value) return [mixedGuide]
  if (hasQueryText.value && !hasQueryImage.value) return [textGuide]
  if (hasQueryImage.value && !hasQueryText.value) return [imageGuide]
  return [imageGuide, textGuide]
})

const thresholdGuideNote = computed(() => {
  if (algorithm.value !== 'SIGLIP') return ''
  if (siglipMixedSearch.value) return '当前：图文混搜。'
  if (hasQueryText.value && !hasQueryImage.value) return '当前：文搜图。'
  if (hasQueryImage.value && !hasQueryText.value) return '当前：图搜图。'
  return ''
})

const segmentsFlat = computed(() => buildSegmentsFlat(staySegments.value))

const hitVideoCount = computed(() => {
  if (segmentsFlat.value.length) {
    return new Set(segmentsFlat.value.map((r) => r.video_name)).size
  }
  return results.value ? Object.keys(results.value).length : 0
})
const hitCount = computed(() => {
  if (segmentsFlat.value.length) {
    return segmentsFlat.value.reduce((a, r) => a + r.hit_count, 0)
  }
  return results.value ? Object.values(results.value).reduce((acc, cur) => acc + cur.length, 0) : 0
})

function algorithmLabel(raw: string): string {
  if (raw === 'SIGLIP') return '多模态检索'
  return '图像检索'
}

async function runSearch() {
  // 新检索开始前，关闭上一轮的视频片段预览，避免旧内容残留
  closeClip()
  const fd = new FormData()
  fd.append('algorithm', algorithm.value)
  fd.append('threshold', String(threshold.value))
  fd.append('time_gap', String(timeGap.value))
  fd.append('group_mode', groupMode.value ? 'true' : 'false')
  fd.append('co_time_threshold', String(coTime.value))
  if (qText.value.trim()) fd.append('q_text', qText.value.trim())
  for (const uf of uploadFileList.value) {
    if (uf.raw) fd.append('files', uf.raw)
  }

  const { data } = await http.post<{
    results: Record<string, Hit[]>
    stay_segments?: Record<string, StaySegment[]>
    algorithm: string
    query_image_names: string
  }>('/search', fd)
  results.value = data.results
  staySegments.value = data.stay_segments && Object.keys(data.stay_segments).length ? data.stay_segments : null
  lastMeta.algorithm = data.algorithm
  lastMeta.query_image_names = data.query_image_names
  lastMeta.threshold = threshold.value
  lastMeta.qText = qText.value.trim()
  ElMessage.success('检索完成')
}

async function saveLog() {
  if (results.value == null && !staySegments.value) return
  await http.post('/search/logs', {
    log_kind: 'single',
    query_image_paths: lastMeta.query_image_names,
    search_query: lastMeta.qText || null,
    algorithm: lastMeta.algorithm,
    threshold: lastMeta.threshold,
    results: results.value ?? {},
    stay_segments: staySegments.value ?? {},
    time_gap: timeGap.value,
  })
  ElMessage.success('已保存到检索日志')
}

async function playClip(vid: string, t: number) {
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
  const code = videoPlayerRef.value?.error?.code
  const hint =
    code === 4
      ? '视频格式无法解码（需 ffmpeg 转码或安装 ffprobe）'
      : '完整视频加载失败，请确认 ffmpeg 可用且归档文件未损坏'
  ElMessage.error(hint)
  closeClip()
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

onBeforeUnmount(() => {
  closeClip()
})
</script>

<template>
  <div class="space-y-6">
    <el-tabs v-model="searchTab" class="search-tabs">
      <el-tab-pane label="单人轨迹检索" name="single">
    <div class="space-y-6">
    <div class="rounded-xl border border-slate-200/70 bg-white/85 p-4 shadow-sm backdrop-blur">
      <div class="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 class="text-xl font-semibold text-slate-800">轨迹检索</h2>
          <p class="mt-1 text-sm text-slate-500">支持图搜图、文图混合检索和轨迹聚合，结果可手动筛选后入库日志。</p>
        </div>
        <div class="grid grid-cols-3 gap-2">
          <div class="rounded-lg bg-slate-50 px-3 py-2 text-center">
            <div class="text-xs text-slate-500">检索模式</div>
            <div class="text-sm font-semibold text-slate-700">{{ algorithmLabel(algorithm) }}</div>
          </div>
          <div class="rounded-lg bg-sky-50 px-3 py-2 text-center">
            <div class="text-xs text-sky-600">涉及视频</div>
            <div class="text-sm font-semibold text-sky-700">{{ hitVideoCount }}</div>
          </div>
          <div class="rounded-lg bg-indigo-50 px-3 py-2 text-center">
            <div class="text-xs text-indigo-600">段内总帧数</div>
            <div class="text-sm font-semibold text-indigo-700">{{ hitCount }}</div>
          </div>
        </div>
      </div>
    </div>

    <el-card shadow="never" class="rounded-xl border border-slate-200/70">
      <el-form label-width="120px" class="max-w-4xl">
        <el-form-item label="检索方式">
          <el-radio-group v-model="algorithm">
            <el-radio-button value="OSNet">图像检索</el-radio-button>
            <el-radio-button value="SIGLIP">多模态检索</el-radio-button>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="相似度阈值">
          <div class="w-full max-w-4xl">
            <el-slider
              v-model="threshold"
              :min="thresholdMin"
              :max="thresholdMax"
              :step="thresholdStep"
              style="max-width: 400px"
            />
            <div class="mt-2 rounded-lg border border-sky-100 bg-sky-50/70 px-3 py-2 text-xs leading-5 text-slate-600">
              <div v-for="guide in thresholdGuides" :key="guide.title" class="mt-1 first:mt-0">
                <p class="font-semibold text-sky-700">{{ guide.title }}</p>
                <p class="mt-1">{{ guide.range }}</p>
                <p class="mt-1">{{ guide.start }}</p>
              </div>
              <p v-if="thresholdGuideNote" class="mt-1 text-sky-700">{{ thresholdGuideNote }}</p>
              <p class="mt-1">
                展开下方「出现段」可逐帧放大、删除或播放完整视频（自动跳到对应时刻）；保存日志仍保存服务端返回的摘要结果。
              </p>
            </div>
          </div>
        </el-form-item>
        <el-form-item label="轨迹聚合间隔(秒)">
          <el-input-number v-model="timeGap" :min="1" :max="60" />
          <span class="ml-2 text-xs text-slate-500">相邻命中间隔小于此值时合并为同一段「连续出现」</span>
        </el-form-item>
        <el-form-item label="团伙共现模式">
          <el-switch v-model="groupMode" />
        </el-form-item>
        <el-form-item v-if="groupMode" label="同行时间差(秒)">
          <el-input-number v-model="coTime" :min="0" :max="30" :step="0.5" />
        </el-form-item>
        <el-form-item v-if="algorithm === 'SIGLIP'" label="文字描述">
          <el-input
            v-model="qText"
            type="textarea"
            rows="2"
            placeholder="例如：穿红衣服的人"
          />
        </el-form-item>
        <el-form-item label="查询图片">
          <el-upload v-model:file-list="uploadFileList" multiple :auto-upload="false" accept=".jpg,.jpeg,.png">
            <el-button type="primary">选择图片</el-button>
          </el-upload>
          <p v-if="uploadFileList.length > 1" class="mt-1 text-xs text-slate-500">
            多张图将分别检索，同一监控帧（meta_id）保留最高相似度后合并为轨迹段；适合同一人多角度查询。
          </p>
        </el-form-item>
        <el-form-item>
          <el-button
            type="primary"
            :disabled="!canSearch"
            @click="runSearch().catch((e) => ElMessage.error(apiErrorMessage(e, '检索请求失败')))"
          >
            开始检索
          </el-button>
          <el-button v-if="results != null" @click="saveLog">保存到日志</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card v-if="segmentsFlat.length" shadow="never" class="rounded-xl border border-emerald-200/70">
      <template #header>
        <div class="text-base font-semibold text-slate-800">监控中出现段</div>
      </template>
      <StaySegmentsPanel
        :stay-segments="staySegments"
        editable
        @preview="openPreview"
        @play="(p) => playClip(p.vid, p.time)"
        @remove-crop="(p) => removeStayCrop(p.videoName, p.segmentIdx, p.cropIndex)"
      />
    </el-card>

    <el-empty
      v-if="results != null && !segmentsFlat.length && (!results || !Object.keys(results).length)"
      description="无匹配结果"
    />
    </div>
      </el-tab-pane>

      <el-tab-pane label="双人房间共现" name="dual">
        <RoomCopresencePanel
          @play-video="(vid: string, t: number) => playClip(vid, t)"
          @preview="(path: string) => openPreview(path)"
        />
      </el-tab-pane>
    </el-tabs>

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
      <div v-if="clipUrl" class="flex max-h-[88vh] min-h-[240px] flex-col items-center justify-center rounded bg-black/95 p-2">
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
          已从 {{ fmtTime(Math.floor(clipSeekSec)) }} 处开始播放
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
