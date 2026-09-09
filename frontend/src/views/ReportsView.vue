<script setup lang="ts">
import SafeMarkdown from '@/components/SafeMarkdown.vue'
import {
  fetchReport,
  fetchReports,
  currentGeneratePromise,
  deleteReport,
  generateReport,
  generateReportInFlight,
  parseReportRequestError,
  previewHasActivity,
  previewReport,
  type GeneratedReport,
  type ReportListItem,
  type ReportMode,
  type ReportPreview,
  type ReportPreviewRequest,
  type ReportRequestError,
  type ReportStatus,
} from '@/api/reports'
import {
  fetchTrackingRooms,
  fetchTrackingVideos,
  type TrackingRoom,
  type TrackingVideo,
} from '@/api/tracking'
import { formatServerDateTime } from '@/utils/datetime'
import { DocumentCopy, Download, Notebook } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { computed, onMounted, ref } from 'vue'

defineOptions({
  name: 'ReportsView',
})

const TIMEZONE = 'Asia/Shanghai'

const mode = ref<ReportMode>('date')
const reportDate = ref('')
const selectedVideoIds = ref<number[]>([])
const selectedZoneIds = ref<number[]>([])

const videos = ref<TrackingVideo[]>([])
const rooms = ref<TrackingRoom[]>([])
const loadingFilters = ref(false)

const preview = ref<ReportPreview | null>(null)
const previewing = ref(false)
const previewEmpty = ref(false)
const lastPreviewKey = ref('')

const generating = ref(false)
const displayedReport = ref<GeneratedReport | null>(null)
const generateError = ref<ReportRequestError | null>(null)
const viewingHistory = ref(false)

const history = ref<ReportListItem[]>([])
const historyTotal = ref(0)
const historyPage = ref(1)
const historyPageSize = ref(20)
const historyStatus = ref<ReportStatus | ''>('')
const historyDate = ref('')
const historyLoading = ref(false)
const detailLoading = ref(false)
const deletingId = ref<number | null>(null)

const generateBlockedByEmpty = computed(() => {
  const body = buildRequest()
  if (!body) return false
  return previewEmpty.value && lastPreviewKey.value === requestKey(body)
})

const roomOptions = computed(() => {
  if (mode.value === 'video' && selectedVideoIds.value.length) {
    return rooms.value.filter(
      (r) => r.video_id == null || selectedVideoIds.value.includes(r.video_id),
    )
  }
  return rooms.value
})

const selectedVideoLabels = computed(() => {
  const map = new Map(videos.value.map((v) => [v.id, v.file_name]))
  return selectedVideoIds.value.map((id) => map.get(id) || `视频 ${id}`)
})

const selectedZoneLabels = computed(() => {
  const map = new Map(rooms.value.map((r) => [r.id, r.name]))
  return selectedZoneIds.value.map((id) => map.get(id) || `区域 ${id}`)
})

function requestKey(body: ReportPreviewRequest): string {
  return JSON.stringify({
    date: body.date ?? null,
    timezone: body.timezone ?? TIMEZONE,
    video_ids: body.video_ids ?? [],
    zone_ids: body.zone_ids ?? [],
  })
}

function buildRequest(): ReportPreviewRequest | null {
  const zoneIds = [...selectedZoneIds.value]
  if (mode.value === 'date') {
    if (!reportDate.value) return null
    return {
      date: reportDate.value,
      timezone: TIMEZONE,
      video_ids: [],
      zone_ids: zoneIds,
    }
  }
  if (!selectedVideoIds.value.length) return null
  return {
    date: null,
    timezone: TIMEZONE,
    video_ids: [...selectedVideoIds.value],
    zone_ids: zoneIds,
  }
}

function timeBasisLabel(value: string | undefined): string {
  if (value === 'absolute') return 'absolute'
  if (value === 'video_relative') return 'video_relative'
  if (value === 'mixed') return 'mixed'
  return value || '—'
}

function statusType(status: string): 'success' | 'warning' | 'danger' | 'info' {
  if (status === 'completed') return 'success'
  if (status === 'generating') return 'warning'
  if (status === 'failed') return 'danger'
  return 'info'
}

function statusLabel(status: string): string {
  if (status === 'completed') return '已完成'
  if (status === 'generating') return '生成中'
  if (status === 'failed') return '失败'
  return status || '—'
}

function reportRangeText(report: GeneratedReport | ReportListItem): string {
  if (report.report_date) return report.report_date
  const names = report.scope?.video_names
  if (names && names.length) return names.join('、')
  const ids = report.scope?.video_ids
  if (ids && ids.length) return `视频 ${ids.join(', ')}`
  return '—'
}

function warningCount(report: GeneratedReport | ReportListItem): number {
  return Array.isArray(report.warnings) ? report.warnings.length : 0
}

async function loadFilters() {
  loadingFilters.value = true
  try {
    const [videoRows, roomRows] = await Promise.all([
      fetchTrackingVideos(),
      fetchTrackingRooms(),
    ])
    videos.value = videoRows
    rooms.value = roomRows
  } catch {
    ElMessage.error('无法加载视频或区域列表')
  } finally {
    loadingFilters.value = false
  }
}

async function onHistoryFilterChange() {
  historyPage.value = 1
  await loadHistory()
}

async function loadHistory() {
  historyLoading.value = true
  try {
    const data = await fetchReports({
      status: historyStatus.value,
      report_date: historyDate.value || null,
      page: historyPage.value,
      page_size: historyPageSize.value,
    })
    history.value = data.items
    historyTotal.value = data.total
    historyPage.value = data.page
    historyPageSize.value = data.page_size
  } catch (err) {
    ElMessage.error(parseReportRequestError(err, '无法加载历史报告').message)
  } finally {
    historyLoading.value = false
  }
}

async function runPreview() {
  const body = buildRequest()
  if (!body) {
    ElMessage.warning(mode.value === 'date' ? '请选择日期' : '请至少选择一个视频')
    return
  }
  previewing.value = true
  previewEmpty.value = false
  generateError.value = null
  try {
    const data = await previewReport(body)
    preview.value = data
    lastPreviewKey.value = requestKey(body)
    previewEmpty.value = !previewHasActivity(data)
  } catch (err) {
    preview.value = null
    lastPreviewKey.value = ''
    ElMessage.error(parseReportRequestError(err, '预览统计失败').message)
  } finally {
    previewing.value = false
  }
}

async function runGenerate() {
  if (generating.value || generateReportInFlight()) return
  const body = buildRequest()
  if (!body) {
    ElMessage.warning(mode.value === 'date' ? '请选择日期' : '请至少选择一个视频')
    return
  }
  if (lastPreviewKey.value !== requestKey(body) || !preview.value) {
    await runPreview()
  }
  if (!previewHasActivity(preview.value)) {
    previewEmpty.value = true
    ElMessage.warning('当前范围暂无可生成的活动数据')
    return
  }
  generating.value = true
  generateError.value = null
  viewingHistory.value = false
  try {
    displayedReport.value = await generateReport(body)
    await loadHistory()
  } catch (err) {
    generateError.value = parseReportRequestError(err, '报告生成失败')
    displayedReport.value = null
  } finally {
    generating.value = false
  }
}

async function openHistoryDetail(row: ReportListItem) {
  detailLoading.value = true
  generateError.value = null
  try {
    displayedReport.value = await fetchReport(row.id)
    viewingHistory.value = true
  } catch (err) {
    ElMessage.error(parseReportRequestError(err, '无法读取报告详情').message)
  } finally {
    detailLoading.value = false
  }
}

async function removeHistoryReport(row: ReportListItem) {
  const title = row.title?.trim() || `报告 #${row.id}`
  try {
    await ElMessageBox.confirm(`确定删除报告「${title}」？删除后列表中不再显示。`, '删除确认', {
      type: 'warning',
      confirmButtonText: '删除',
      cancelButtonText: '取消',
      confirmButtonClass: 'el-button--danger',
    })
  } catch {
    return
  }
  if (deletingId.value != null) return
  deletingId.value = row.id
  try {
    await deleteReport(row.id)
    ElMessage.success('报告已删除')
    if (displayedReport.value?.id === row.id) {
      viewingHistory.value = false
      displayedReport.value = null
    }
    await loadHistory()
  } catch (err) {
    ElMessage.error(parseReportRequestError(err, '删除报告失败').message)
  } finally {
    deletingId.value = null
  }
}

function backToHistory() {
  viewingHistory.value = false
  displayedReport.value = null
}

async function copyMarkdown() {
  const text = displayedReport.value?.report_markdown || ''
  if (!text) {
    ElMessage.warning('没有可复制的报告正文')
    return
  }
  try {
    await navigator.clipboard.writeText(text)
    ElMessage.success('报告正文已复制')
  } catch {
    ElMessage.error('复制失败')
  }
}

function downloadMarkdown() {
  const report = displayedReport.value
  if (!report?.report_markdown) {
    ElMessage.warning('没有可下载的报告正文')
    return
  }
  const title = (report.title || '分析报告').replace(/[\\/:*?"<>|]/g, '_')
  const datePart = report.report_date || (report.created_at || '').slice(0, 10) || 'undated'
  const filename = `${title}-${datePart}.md`
  const content = `# ${report.title || '分析报告'}\n\n${report.report_markdown}`
  const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}

onMounted(async () => {
  await Promise.all([loadFilters(), loadHistory()])
  const pending = currentGeneratePromise()
  if (pending) {
    generating.value = true
    try {
      displayedReport.value = await pending
      await loadHistory()
    } catch (err) {
      generateError.value = parseReportRequestError(err, '报告生成失败')
    } finally {
      generating.value = false
    }
  }
})
</script>

<template>
  <div class="space-y-4 p-1">
    <el-card shadow="never" class="rounded-xl border border-slate-200/80">
      <template #header>
        <div class="flex items-center gap-2 text-sm font-semibold">
          <el-icon><Notebook /></el-icon>
          统计范围
        </div>
      </template>
      <div class="flex flex-wrap items-end gap-4">
        <div>
          <div class="mb-1 text-xs text-slate-500">生成方式</div>
          <el-radio-group v-model="mode">
            <el-radio-button value="date">按日期</el-radio-button>
            <el-radio-button value="video">按视频</el-radio-button>
          </el-radio-group>
        </div>
        <div v-if="mode === 'date'" class="min-w-[220px]">
          <div class="mb-1 text-xs text-slate-500">日期</div>
          <el-date-picker
            v-model="reportDate"
            type="date"
            value-format="YYYY-MM-DD"
            placeholder="选择日期"
            class="w-full"
          />
        </div>
        <div v-else class="min-w-[280px] flex-1">
          <div class="mb-1 text-xs text-slate-500">视频（可多选，含无拍摄时间的视频）</div>
          <el-select
            v-model="selectedVideoIds"
            multiple
            filterable
            collapse-tags
            collapse-tags-tooltip
            placeholder="选择视频"
            class="w-full"
            :loading="loadingFilters"
          >
            <el-option
              v-for="video in videos"
              :key="video.id"
              :label="video.file_name"
              :value="video.id"
            >
              <span>{{ video.file_name }}</span>
              <span class="ml-2 text-xs text-slate-400">G×{{ video.person_count }}</span>
              <span v-if="!video.captured_at" class="ml-2 text-xs text-amber-600">无 captured_at</span>
            </el-option>
          </el-select>
        </div>
        <div class="min-w-[220px] flex-1">
          <div class="mb-1 text-xs text-slate-500">可见区域（可选）</div>
          <el-select
            v-model="selectedZoneIds"
            multiple
            clearable
            filterable
            collapse-tags
            placeholder="全部区域"
            class="w-full"
            :loading="loadingFilters"
          >
            <el-option
              v-for="room in roomOptions"
              :key="room.id"
              :label="room.name"
              :value="room.id"
            />
          </el-select>
        </div>
      </div>
      <div class="mt-4 flex flex-wrap items-center gap-3">
        <el-button type="primary" plain :loading="previewing" @click="runPreview">预览统计</el-button>
        <el-button
          type="primary"
          :loading="generating"
          :disabled="generating || generateBlockedByEmpty"
          @click="runGenerate"
        >
          生成 AI 报告
        </el-button>
        <span class="text-xs text-slate-500">时区默认 Asia/Shanghai</span>
      </div>
      <el-alert
        v-if="generating"
        class="mt-4"
        type="info"
        :closable="false"
        title="AI 报告生成中，通常需要约 20–60 秒"
        description="请勿重复点击。切换页面不会再次提交，可稍后回到本页查看结果。"
      />
    </el-card>

    <el-card shadow="never" class="rounded-xl border border-slate-200/80">
      <template #header>
        <div class="text-sm font-semibold">预览统计</div>
      </template>
      <el-empty
        v-if="generateBlockedByEmpty"
        description="当前范围暂无可生成的活动数据"
        :image-size="72"
      />
      <div v-else-if="preview" class="space-y-4">
        <div class="grid grid-cols-2 gap-3 md:grid-cols-5">
          <div class="rounded-lg bg-slate-50 px-3 py-2">
            <div class="text-xs text-slate-500">去重人物数</div>
            <div class="text-lg font-semibold">{{ preview.statistics.unique_person_count }}</div>
          </div>
          <div class="rounded-lg bg-slate-50 px-3 py-2">
            <div class="text-xs text-slate-500">轨迹数</div>
            <div class="text-lg font-semibold">{{ preview.statistics.track_count }}</div>
          </div>
          <div class="rounded-lg bg-slate-50 px-3 py-2">
            <div class="text-xs text-slate-500">观测数</div>
            <div class="text-lg font-semibold">{{ preview.statistics.observation_count }}</div>
          </div>
          <div class="rounded-lg bg-slate-50 px-3 py-2">
            <div class="text-xs text-slate-500">可见区域数</div>
            <div class="text-lg font-semibold">{{ preview.statistics.visible_zone_count }}</div>
          </div>
          <div class="rounded-lg bg-slate-50 px-3 py-2">
            <div class="text-xs text-slate-500">总画面内可见时长</div>
            <div class="text-lg font-semibold">
              {{ preview.statistics.total_visible_duration_seconds }} 秒
            </div>
          </div>
        </div>
        <el-descriptions :column="2" border size="small">
          <el-descriptions-item label="时间口径">
            {{ timeBasisLabel(preview.scope.time_basis) }}
          </el-descriptions-item>
          <el-descriptions-item label="时区">{{ preview.scope.timezone }}</el-descriptions-item>
          <el-descriptions-item label="视频">
            {{
              preview.scope.video_names.length
                ? preview.scope.video_names.join('、')
                : selectedVideoLabels.join('、') || '—'
            }}
          </el-descriptions-item>
          <el-descriptions-item label="摄像头">
            {{
              preview.scope.camera_names.filter(Boolean).join('、') || '未筛选'
            }}
          </el-descriptions-item>
          <el-descriptions-item label="可见区域" :span="2">
            {{
              preview.scope.zone_names.filter(Boolean).join('、') ||
              selectedZoneLabels.join('、') ||
              '全部区域'
            }}
          </el-descriptions-item>
        </el-descriptions>
        <el-alert
          v-if="preview.data_quality.warnings.length"
          type="warning"
          :closable="false"
          title="数据质量警告"
        >
          <ul class="list-disc pl-4 text-sm">
            <li v-for="(warning, wi) in preview.data_quality.warnings" :key="wi">{{ warning }}</li>
          </ul>
        </el-alert>
      </div>
      <el-empty v-else description="请先选择范围并点击预览统计" :image-size="72" />
    </el-card>

    <el-card v-if="generateError" shadow="never" class="rounded-xl border border-rose-200/80">
      <el-alert type="error" :closable="false" title="报告生成失败">
        <div class="space-y-1 text-sm">
          <div>{{ generateError.message }}</div>
          <div v-if="generateError.reportId != null">report_id：{{ generateError.reportId }}</div>
          <div v-if="generateError.status != null">状态码：{{ generateError.status }}</div>
        </div>
      </el-alert>
      <div class="mt-3">
        <el-button type="primary" :disabled="generating" @click="runGenerate">重新尝试</el-button>
      </div>
    </el-card>

    <el-card
      v-if="displayedReport && displayedReport.status === 'completed'"
      shadow="never"
      class="rounded-xl border border-slate-200/80"
    >
      <template #header>
        <div class="flex flex-wrap items-center justify-between gap-2">
          <div class="text-sm font-semibold">{{ viewingHistory ? '历史报告详情' : 'AI 分析报告' }}</div>
          <div class="flex flex-wrap gap-2">
            <el-button size="small" :icon="DocumentCopy" @click="copyMarkdown">复制报告正文</el-button>
            <el-button size="small" :icon="Download" @click="downloadMarkdown">下载 Markdown</el-button>
            <el-button v-if="viewingHistory" size="small" @click="backToHistory">返回历史列表</el-button>
          </div>
        </div>
      </template>
      <el-descriptions :column="2" border size="small" class="mb-4">
        <el-descriptions-item label="标题" :span="2">
          {{ displayedReport.title || '—' }}
        </el-descriptions-item>
        <el-descriptions-item label="报告日期 / 视频范围">
          {{ reportRangeText(displayedReport) }}
        </el-descriptions-item>
        <el-descriptions-item label="生成时间">
          {{ formatServerDateTime(displayedReport.created_at) }}
        </el-descriptions-item>
        <el-descriptions-item label="状态">
          <el-tag :type="statusType(displayedReport.status)" size="small">
            {{ statusLabel(displayedReport.status) }}
          </el-tag>
        </el-descriptions-item>
        <el-descriptions-item label="数据提示数量">
          {{ warningCount(displayedReport) }}
        </el-descriptions-item>
      </el-descriptions>
      <SafeMarkdown :markdown="displayedReport.report_markdown || ''" />
    </el-card>

    <el-card shadow="never" class="rounded-xl border border-slate-200/80">
      <template #header>
        <div class="text-sm font-semibold">历史报告</div>
      </template>
      <div class="mb-4 flex flex-wrap items-end gap-3">
        <div>
          <div class="mb-1 text-xs text-slate-500">状态</div>
          <el-select v-model="historyStatus" clearable placeholder="全部状态" class="w-40" @change="onHistoryFilterChange">
            <el-option label="已完成" value="completed" />
            <el-option label="生成中" value="generating" />
            <el-option label="失败" value="failed" />
          </el-select>
        </div>
        <div>
          <div class="mb-1 text-xs text-slate-500">报告日期</div>
          <el-date-picker
            v-model="historyDate"
            type="date"
            value-format="YYYY-MM-DD"
            placeholder="全部日期"
            @change="onHistoryFilterChange"
          />
        </div>
      </div>
      <el-table :data="history" v-loading="historyLoading || detailLoading" stripe empty-text="暂无历史报告">
        <el-table-column prop="title" label="标题" min-width="220" show-overflow-tooltip />
        <el-table-column label="报告日期" width="130">
          <template #default="{ row }">{{ row.report_date || '—' }}</template>
        </el-table-column>
        <el-table-column label="状态" width="110">
          <template #default="{ row }">
            <el-tag :type="statusType(row.status)" size="small">{{ statusLabel(row.status) }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="生成时间" width="180">
          <template #default="{ row }">{{ formatServerDateTime(row.created_at) }}</template>
        </el-table-column>
        <el-table-column label="数据提示数量" width="120">
          <template #default="{ row }">{{ warningCount(row) }}</template>
        </el-table-column>
        <el-table-column label="失败原因" min-width="180">
          <template #default="{ row }">
            <span v-if="row.status === 'failed'" class="text-rose-600">{{ row.error_message || '生成失败' }}</span>
            <span v-else class="text-slate-400">—</span>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="160" fixed="right">
          <template #default="{ row }">
            <el-button type="primary" link @click="openHistoryDetail(row)">查看详情</el-button>
            <el-button
              type="danger"
              link
              :disabled="deletingId === row.id || row.status === 'generating'"
              :loading="deletingId === row.id"
              @click="removeHistoryReport(row)"
            >
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
      <div class="mt-4 flex justify-end">
        <el-pagination
          v-model:current-page="historyPage"
          v-model:page-size="historyPageSize"
          :total="historyTotal"
          :page-sizes="[10, 20, 50]"
          layout="total, sizes, prev, pager, next"
          @current-change="loadHistory"
          @size-change="loadHistory"
        />
      </div>
    </el-card>
  </div>
</template>
