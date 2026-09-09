<script setup lang="ts">
import http from '@/api/http'
import { ElMessage, ElMessageBox } from 'element-plus'
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'

const props = withDefaults(
  defineProps<{
    /** 嵌入 Drawer 时由父组件传入，与 videos.file_name 一致 */
    videoName?: string
    /** 为 true 时隐藏卡片标题与视频下拉，用于视频源列表侧栏 */
    embedded?: boolean
    /** embedded 且 videoName 变化时自动加载底图 */
    autoLoad?: boolean
  }>(),
  { embedded: false, autoLoad: true },
)

const emit = defineEmits<{
  'rooms-changed': [count: number]
}>()

interface MediaRow {
  file_name: string
  status: string
}

interface RefMeta {
  video_name: string
  filename: string
  image_url: string
}

interface RoomRow {
  id: number
  video_name: string
  room_name: string
  polygon: number[][]
  created_at?: string
  updated_at?: string
}

/** 与后端 /rooms/extract_freedraw 默认 stroke_r/g/b 一致，便于 OpenCV 颜色掩膜 */
const STROKE_R = 255
const STROKE_G = 0
const STROKE_B = 128
const STROKE_CSS = `rgb(${STROKE_R}, ${STROKE_G}, ${STROKE_B})`

const MAX_CANVAS_W = 960

const sourceNames = ref<string[]>([])
const videoNameLocal = ref('')
const loadingSources = ref(false)
const loadingRef = ref(false)
const saving = ref(false)
const extracting = ref(false)

const refMeta = ref<RefMeta | null>(null)
const naturalW = ref(0)
const naturalH = ref(0)
const canvasW = ref(0)
const canvasH = ref(0)

const bgCanvasEl = ref<HTMLCanvasElement | null>(null)
const drawCanvasEl = ref<HTMLCanvasElement | null>(null)
const bgImage = ref<HTMLImageElement | null>(null)
const blobUrl = ref<string | null>(null)

/** 提取后的多边形（原图像素，与 gallery_meta bbox 一致） */
const draftPolygon = ref<number[][] | null>(null)
const rooms = ref<RoomRow[]>([])
const roomNameInput = ref('房间1')

const strokeWidthPx = ref(10)
const colorTol = ref(60)

const drawing = ref(false)

const activeVideoName = computed(() => {
  if (props.embedded && props.videoName) return props.videoName.trim()
  return videoNameLocal.value.trim()
})

const canLoadRef = computed(() => !!activeVideoName.value)
const canSave = computed(() => (draftPolygon.value?.length ?? 0) >= 3)

function emitRoomCount() {
  emit('rooms-changed', rooms.value.length)
}

function revokeBlob() {
  if (blobUrl.value) {
    URL.revokeObjectURL(blobUrl.value)
    blobUrl.value = null
  }
  bgImage.value = null
}

function resetCanvasState() {
  revokeBlob()
  draftPolygon.value = null
  refMeta.value = null
  naturalW.value = 0
  naturalH.value = 0
  rooms.value = []
  roomNameInput.value = '房间1'
}

async function loadSources() {
  loadingSources.value = true
  try {
    const { data } = await http.get<MediaRow[]>('/media/sources')
    sourceNames.value = data
      .filter((r) => (r.status === 'completed' || r.status === 'transcoding') && r.file_name)
      .map((r) => r.file_name)
  } catch {
    ElMessage.error('加载视频源列表失败')
  } finally {
    loadingSources.value = false
  }
}

function syncCanvasBitmapSize() {
  const w = canvasW.value
  const h = canvasH.value
  for (const el of [bgCanvasEl.value, drawCanvasEl.value]) {
    if (!el) continue
    el.width = w
    el.height = h
  }
}

function bitmapFromEvent(e: MouseEvent, c: HTMLCanvasElement): [number, number] {
  const rect = c.getBoundingClientRect()
  const x = ((e.clientX - rect.left) / Math.max(rect.width, 1)) * c.width
  const y = ((e.clientY - rect.top) / Math.max(rect.height, 1)) * c.height
  return [x, y]
}

function redrawBackground() {
  const c = bgCanvasEl.value
  const img = bgImage.value
  if (!c || !img) return
  const ctx = c.getContext('2d')
  if (!ctx) return
  ctx.clearRect(0, 0, c.width, c.height)
  ctx.drawImage(img, 0, 0, c.width, c.height)

  const nw = naturalW.value
  const nh = naturalH.value
  const toBmp = (nx: number, ny: number) => {
    return [(nx / Math.max(nw, 1)) * c.width, (ny / Math.max(nh, 1)) * c.height] as const
  }

  ctx.lineWidth = 2
  for (const r of rooms.value) {
    const poly = r.polygon
    if (!poly || poly.length < 2) continue
    ctx.strokeStyle = 'rgba(34, 197, 94, 0.85)'
    ctx.beginPath()
    const [x0, y0] = toBmp(poly[0][0], poly[0][1])
    ctx.moveTo(x0, y0)
    for (let i = 1; i < poly.length; i++) {
      const [xi, yi] = toBmp(poly[i][0], poly[i][1])
      ctx.lineTo(xi, yi)
    }
    ctx.closePath()
    ctx.stroke()
  }

  const d = draftPolygon.value
  if (d && d.length >= 2) {
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.95)'
    ctx.fillStyle = 'rgba(59, 130, 246, 0.22)'
    ctx.beginPath()
    const [bx0, by0] = toBmp(d[0][0], d[0][1])
    ctx.moveTo(bx0, by0)
    for (let i = 1; i < d.length; i++) {
      const [bxi, byi] = toBmp(d[i][0], d[i][1])
      ctx.lineTo(bxi, byi)
    }
    ctx.closePath()
    ctx.stroke()
    if (d.length >= 3) {
      ctx.fill()
    }
  }
}

function clearDrawLayer() {
  const c = drawCanvasEl.value
  if (!c) return
  const ctx = c.getContext('2d')
  if (!ctx) return
  ctx.clearRect(0, 0, c.width, c.height)
}

function clearStrokeOnly() {
  clearDrawLayer()
}

function clearDraftPolygon() {
  draftPolygon.value = null
  redrawBackground()
}

async function loadReferenceAndRooms() {
  const vn = activeVideoName.value
  if (!vn) {
    ElMessage.warning('请选择或输入视频名称')
    return
  }
  revokeBlob()
  draftPolygon.value = null
  refMeta.value = null
  naturalW.value = 0
  naturalH.value = 0

  loadingRef.value = true
  try {
    const { data: refData } = await http.get<RefMeta>('/rooms/reference', { params: { video_name: vn } })
    refMeta.value = refData

    const imgRes = await http.get(refData.image_url, { responseType: 'blob' })
    revokeBlob()
    blobUrl.value = URL.createObjectURL(imgRes.data)

    await new Promise<void>((resolve, reject) => {
      const img = new Image()
      img.onload = () => {
        bgImage.value = img
        naturalW.value = img.naturalWidth || img.width
        naturalH.value = img.naturalHeight || img.height
        const nw = naturalW.value
        const nh = naturalH.value
        const scale = Math.min(1, MAX_CANVAS_W / Math.max(nw, 1))
        canvasW.value = Math.max(1, Math.round(nw * scale))
        canvasH.value = Math.max(1, Math.round(nh * scale))
        resolve()
      }
      img.onerror = () => reject(new Error('图片加载失败'))
      img.src = blobUrl.value!
    })

    const { data: listData } = await http.get<{ rooms: RoomRow[] }>('/rooms', { params: { video_name: vn } })
    rooms.value = listData.rooms || []
    roomNameInput.value = `房间${rooms.value.length + 1}`

    await nextTick()
    syncCanvasBitmapSize()
    redrawBackground()
    clearDrawLayer()
    emitRoomCount()
    if (!props.embedded) {
      ElMessage.success('底图与房间列表已加载')
    }
  } catch (e: unknown) {
    const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
    ElMessage.error(typeof msg === 'string' ? msg : '加载底图失败（请确认已建模且名称与任务一致）')
  } finally {
    loadingRef.value = false
  }
}

function onDrawMouseDown(e: MouseEvent) {
  if (!bgImage.value || !naturalW.value) return
  const c = drawCanvasEl.value
  if (!c) return
  const [x, y] = bitmapFromEvent(e, c)
  if (draftPolygon.value) {
    draftPolygon.value = null
    redrawBackground()
  }
  drawing.value = true
  const ctx = c.getContext('2d')
  if (!ctx) return
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  ctx.strokeStyle = STROKE_CSS
  ctx.globalAlpha = 1
  ctx.lineWidth = strokeWidthPx.value
  ctx.beginPath()
  ctx.moveTo(x, y)
}

function onDrawMouseMove(e: MouseEvent) {
  if (!drawing.value) return
  const c = drawCanvasEl.value
  if (!c) return
  const [x, y] = bitmapFromEvent(e, c)
  const ctx = c.getContext('2d')
  if (!ctx) return
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  ctx.strokeStyle = STROKE_CSS
  ctx.lineWidth = strokeWidthPx.value
  ctx.lineTo(x, y)
  ctx.stroke()
  ctx.beginPath()
  ctx.moveTo(x, y)
}

function onDrawMouseUp() {
  drawing.value = false
}

function blobFromDrawCanvas(): Promise<Blob | null> {
  const c = drawCanvasEl.value
  if (!c) return Promise.resolve(null)
  return new Promise((resolve) => {
    c.toBlob((b) => resolve(b), 'image/png')
  })
}

async function extractFromStroke() {
  const vn = activeVideoName.value
  if (!vn || !bgImage.value) return
  const blob = await blobFromDrawCanvas()
  if (!blob || blob.size < 80) {
    ElMessage.warning('请先在底图上用鼠标沿房间边界画一圈（尽量闭合）')
    return
  }
  const fd = new FormData()
  fd.append('file', blob, 'stroke.png')
  fd.append('natural_w', String(naturalW.value))
  fd.append('natural_h', String(naturalH.value))
  fd.append('canvas_w', String(canvasW.value))
  fd.append('canvas_h', String(canvasH.value))
  fd.append('stroke_r', String(STROKE_R))
  fd.append('stroke_g', String(STROKE_G))
  fd.append('stroke_b', String(STROKE_B))
  fd.append('color_tol', String(colorTol.value))

  extracting.value = true
  try {
    const { data } = await http.post<{
      polygon: number[][]
      vertex_count: number
    }>('/rooms/extract_freedraw', fd)
    draftPolygon.value = data.polygon || null
    clearDrawLayer()
    redrawBackground()
    ElMessage.success(`已提取轮廓（${data.vertex_count ?? draftPolygon.value?.length ?? 0} 顶点）`)
  } catch (e: unknown) {
    const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
    ElMessage.error(typeof msg === 'string' ? msg : '提取失败')
  } finally {
    extracting.value = false
  }
}

async function saveRoom() {
  const vn = activeVideoName.value
  if (!vn) return
  if (!draftPolygon.value || draftPolygon.value.length < 3) {
    ElMessage.warning('请先手绘并点击「提取轮廓」得到多边形')
    return
  }
  const name = roomNameInput.value.trim() || `房间${rooms.value.length + 1}`
  saving.value = true
  try {
    await http.post('/rooms', {
      video_name: vn,
      room_name: name,
      polygon: draftPolygon.value,
    })
    ElMessage.success(`已保存「${name}」，停留图稍后自动更新`)
    draftPolygon.value = null
    const { data } = await http.get<{ rooms: RoomRow[] }>('/rooms', { params: { video_name: vn } })
    rooms.value = data.rooms || []
    roomNameInput.value = `房间${rooms.value.length + 1}`
    redrawBackground()
    clearDrawLayer()
    emitRoomCount()
  } catch (e: unknown) {
    const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
    ElMessage.error(typeof msg === 'string' ? msg : '保存失败')
  } finally {
    saving.value = false
  }
}

async function deleteRoom(row: RoomRow) {
  try {
    await ElMessageBox.confirm(`删除房间「${row.room_name}」？`, '确认', { type: 'warning' })
    await http.delete(`/rooms/by-id/${row.id}`)
    ElMessage.success('已删除')
    const vn = activeVideoName.value
    const { data } = await http.get<{ rooms: RoomRow[] }>('/rooms', { params: { video_name: vn } })
    rooms.value = data.rooms || []
    redrawBackground()
    emitRoomCount()
  } catch {
    /* cancel */
  }
}

watch(
  () => props.videoName,
  (v, oldV) => {
    if (!props.embedded) return
    const name = (v || '').trim()
    if (!name) {
      resetCanvasState()
      return
    }
    if (name === (oldV || '').trim() && refMeta.value) return
    resetCanvasState()
    if (props.autoLoad) void loadReferenceAndRooms()
  },
  { immediate: true },
)

onBeforeUnmount(() => {
  revokeBlob()
})

if (!props.embedded) {
  void loadSources()
}
</script>

<template>
  <el-card v-if="!embedded" shadow="never" class="rounded-xl border border-slate-200/70">
    <template #header>
      <div class="text-base font-semibold text-slate-800">房间标注</div>
    </template>
    <p class="mb-4 text-sm text-slate-500">
      选择与「视频源」里建模完成一致的<strong>视频名称</strong>，加载底图后<strong>按住鼠标沿房间边界画一圈</strong>（尽量闭合），再点「提取轮廓」，确认蓝色预览无误后填写房间名并保存。坐标与底库 bbox
      同为原图像素系；描边颜色固定为洋红以便与 Streamlit 侧算法一致。
    </p>

    <div class="mb-4 flex flex-wrap items-end gap-3">
      <div class="min-w-[280px] flex-1">
        <div class="mb-1 text-xs text-slate-500">视频名称</div>
        <el-select
          v-model="videoNameLocal"
          filterable
          allow-create
          default-first-option
          placeholder="选择已完成任务，或手动输入如 test3.mp4"
          class="w-full"
          :loading="loadingSources"
        >
          <el-option v-for="n in sourceNames" :key="n" :label="n" :value="n" />
        </el-select>
      </div>
      <el-button type="primary" :loading="loadingRef" :disabled="!canLoadRef" @click="loadReferenceAndRooms">
        加载底图与房间
      </el-button>
      <el-button @click="loadSources">刷新任务列表</el-button>
    </div>

    <div v-if="refMeta && bgImage" class="annotator-workspace space-y-3">
      <div class="overflow-auto rounded-lg border border-slate-200 bg-slate-900/5 p-2">
        <div
          class="relative mx-auto block rounded shadow"
          :style="{ width: canvasW + 'px', height: canvasH + 'px' }"
        >
          <canvas
            ref="bgCanvasEl"
            class="absolute left-0 top-0 block"
            :style="{ width: canvasW + 'px', height: canvasH + 'px' }"
          />
          <canvas
            ref="drawCanvasEl"
            class="absolute left-0 top-0 block cursor-crosshair"
            :style="{ width: canvasW + 'px', height: canvasH + 'px' }"
            @mousedown="onDrawMouseDown"
            @mousemove="onDrawMouseMove"
            @mouseup="onDrawMouseUp"
            @mouseleave="onDrawMouseUp"
          />
        </div>
      </div>

      <div class="flex flex-wrap items-center gap-4">
        <div class="min-w-[200px]">
          <div class="mb-1 text-xs text-slate-500">笔刷粗细（像素）</div>
          <el-slider v-model="strokeWidthPx" :min="4" :max="24" :step="1" show-input :show-input-controls="false" />
        </div>
        <div class="min-w-[200px]">
          <div class="mb-1 text-xs text-slate-500">颜色容差（提取用，越大越宽松）</div>
          <el-slider v-model="colorTol" :min="25" :max="120" :step="1" show-input :show-input-controls="false" />
        </div>
      </div>

      <div class="flex flex-wrap items-center gap-2">
        <el-button size="small" @click="clearStrokeOnly">清空笔迹</el-button>
        <el-button size="small" @click="clearDraftPolygon" :disabled="!draftPolygon">清除已提取预览</el-button>
        <el-button type="primary" size="small" :loading="extracting" @click="extractFromStroke">提取轮廓</el-button>
        <span v-if="draftPolygon" class="text-sm text-slate-600">预览顶点：{{ draftPolygon.length }}</span>
        <span class="text-xs text-slate-400">原图约 {{ naturalW }}×{{ naturalH }} px</span>
      </div>

      <div class="flex flex-wrap items-end gap-3">
        <el-input v-model="roomNameInput" placeholder="房间名称" class="max-w-xs" clearable />
        <el-button type="success" :loading="saving" :disabled="!canSave" @click="saveRoom">保存当前多边形为房间</el-button>
      </div>

      <div v-if="rooms.length" class="mt-4">
        <div class="mb-2 text-sm font-medium text-slate-700">已保存房间</div>
        <el-table :data="rooms" size="small" border stripe>
          <el-table-column prop="room_name" label="名称" min-width="120" />
          <el-table-column label="顶点数" width="90">
            <template #default="{ row }">{{ row.polygon?.length ?? 0 }}</template>
          </el-table-column>
          <el-table-column label="操作" width="100">
            <template #default="{ row }">
              <el-button type="danger" link size="small" @click="deleteRoom(row)">删除</el-button>
            </template>
          </el-table-column>
        </el-table>
      </div>
      <el-empty v-else description="该视频尚未保存房间" />
    </div>
  </el-card>

  <div v-else class="annotator-embedded space-y-3">
    <p class="text-sm text-slate-500">
      在底图上<strong>按住鼠标沿房间边界画一圈</strong>（尽量闭合），点「提取轮廓」后填写房间名保存。未标注房间的视频在轨迹检索结果中「进入 / 离开」显示为 —。
    </p>
    <div class="flex flex-wrap items-center gap-2">
      <el-button type="primary" size="small" :loading="loadingRef" :disabled="!canLoadRef" @click="loadReferenceAndRooms">
        重新加载底图
      </el-button>
      <span v-if="loadingRef" class="text-xs text-slate-400">正在加载…</span>
    </div>

    <div v-if="refMeta && bgImage" class="annotator-workspace space-y-3">
      <div class="overflow-auto rounded-lg border border-slate-200 bg-slate-900/5 p-2">
        <div
          class="relative mx-auto block rounded shadow"
          :style="{ width: canvasW + 'px', height: canvasH + 'px' }"
        >
          <canvas
            ref="bgCanvasEl"
            class="absolute left-0 top-0 block"
            :style="{ width: canvasW + 'px', height: canvasH + 'px' }"
          />
          <canvas
            ref="drawCanvasEl"
            class="absolute left-0 top-0 block cursor-crosshair"
            :style="{ width: canvasW + 'px', height: canvasH + 'px' }"
            @mousedown="onDrawMouseDown"
            @mousemove="onDrawMouseMove"
            @mouseup="onDrawMouseUp"
            @mouseleave="onDrawMouseUp"
          />
        </div>
      </div>

      <div class="flex flex-wrap items-center gap-4">
        <div class="min-w-[200px]">
          <div class="mb-1 text-xs text-slate-500">笔刷粗细（像素）</div>
          <el-slider v-model="strokeWidthPx" :min="4" :max="24" :step="1" show-input :show-input-controls="false" />
        </div>
        <div class="min-w-[200px]">
          <div class="mb-1 text-xs text-slate-500">颜色容差（提取用，越大越宽松）</div>
          <el-slider v-model="colorTol" :min="25" :max="120" :step="1" show-input :show-input-controls="false" />
        </div>
      </div>

      <div class="flex flex-wrap items-center gap-2">
        <el-button size="small" @click="clearStrokeOnly">清空笔迹</el-button>
        <el-button size="small" @click="clearDraftPolygon" :disabled="!draftPolygon">清除已提取预览</el-button>
        <el-button type="primary" size="small" :loading="extracting" @click="extractFromStroke">提取轮廓</el-button>
        <span v-if="draftPolygon" class="text-sm text-slate-600">预览顶点：{{ draftPolygon.length }}</span>
        <span class="text-xs text-slate-400">原图约 {{ naturalW }}×{{ naturalH }} px</span>
      </div>

      <div class="flex flex-wrap items-end gap-3">
        <el-input v-model="roomNameInput" placeholder="房间名称" class="max-w-xs" clearable />
        <el-button type="success" :loading="saving" :disabled="!canSave" @click="saveRoom">保存当前多边形为房间</el-button>
      </div>

      <div v-if="rooms.length" class="mt-4">
        <div class="mb-2 text-sm font-medium text-slate-700">已保存房间</div>
        <el-table :data="rooms" size="small" border stripe>
          <el-table-column prop="room_name" label="名称" min-width="120" />
          <el-table-column label="顶点数" width="90">
            <template #default="{ row }">{{ row.polygon?.length ?? 0 }}</template>
          </el-table-column>
          <el-table-column label="操作" width="100">
            <template #default="{ row }">
              <el-button type="danger" link size="small" @click="deleteRoom(row)">删除</el-button>
            </template>
          </el-table-column>
        </el-table>
      </div>
      <el-empty v-else description="该视频尚未保存房间" />
    </div>
    <el-empty v-else-if="!loadingRef" description="加载底图失败或尚无建模 crop" />
  </div>
</template>
