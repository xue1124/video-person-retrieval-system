<script setup lang="ts">
import AuthImage from '@/components/AuthImage.vue'
import {
  buildSegmentsByVideo,
  cropStableKey,
  fmtHms,
  formatSimilarityScore,
  hasStaySegments,
  resolveCropPath,
  segmentRowKey,
  type StaySegment,
} from '@/utils/staySegments'
import { computed } from 'vue'

const props = withDefaults(
  defineProps<{
    staySegments: Record<string, StaySegment[]> | null
    editable?: boolean
    showHint?: boolean
  }>(),
  { editable: false, showHint: true },
)

const emit = defineEmits<{
  preview: [path: string]
  play: [payload: { vid: string; time: number }]
  'remove-crop': [payload: { videoName: string; segmentIdx: number; cropIndex: number }]
}>()

const segmentsByVideo = computed(() => buildSegmentsByVideo(props.staySegments))
const hasData = computed(() => hasStaySegments(props.staySegments))
</script>

<template>
  <div v-if="hasData" class="space-y-4">
    <p v-if="showHint" class="text-sm text-slate-500">
      在相似度阈值内的命中按「轨迹聚合间隔」合并为<strong>连续出现时间段</strong>；展开可查看该时段内每一帧 crop。
      「进入 / 离开」= 本段在监控画面中<strong>首次、末次</strong>出现时，行人底边中点落在哪个房间；未标注或无 bbox 时显示 —。
      「停留房间」= 本段「离开」所示房间；「停留时间」= 本段结束至<strong>同视频下一段出现</strong>的间隔，最后一段为 —。
    </p>
    <el-card
      v-for="group in segmentsByVideo"
      :key="group.video_name"
      shadow="never"
      class="rounded-lg border border-slate-200/80"
    >
      <template #header>
        <div class="flex items-center justify-between text-sm">
          <span class="font-semibold text-slate-800">视频源：{{ group.video_name }}</span>
          <span class="text-slate-500">出现段：{{ group.rows.length }}</span>
        </div>
      </template>
      <el-table :data="group.rows" :row-key="segmentRowKey" border stripe size="small" class="w-full">
        <el-table-column type="expand">
          <template #default="{ row }">
            <div class="grid grid-cols-1 gap-3 p-2 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              <div
                v-for="(c, ci) in row.crops"
                :key="cropStableKey(c, row)"
                class="overflow-hidden rounded-lg border border-slate-200 bg-white p-2 shadow-sm"
              >
                <AuthImage :path="resolveCropPath(c)" class="aspect-video max-h-36 w-full object-cover" />
                <div class="mt-1.5 truncate text-xs text-slate-600">
                  {{ fmtHms(c.time) }} · 相似 {{ formatSimilarityScore(c.score) }}
                  <span v-if="c.global_person_id != null"> · G{{ c.global_person_id }}</span>
                </div>
                <div v-if="c.room_name" class="truncate text-xs text-emerald-700">帧判房间：{{ c.room_name }}</div>
                <div class="mt-2 flex flex-wrap gap-1">
                  <el-button size="small" @click.stop="emit('preview', resolveCropPath(c))">放大</el-button>
                  <el-button
                    v-if="editable"
                    size="small"
                    type="danger"
                    plain
                    @click.stop="
                      emit('remove-crop', {
                        videoName: row.video_name,
                        segmentIdx: row.segment_idx,
                        cropIndex: ci,
                      })
                    "
                  >
                    删除
                  </el-button>
                  <el-button
                    size="small"
                    type="primary"
                    @click.stop="emit('play', { vid: c.vid || row.video_name, time: c.time })"
                  >
                    播放视频
                  </el-button>
                </div>
              </div>
            </div>
          </template>
        </el-table-column>
        <el-table-column prop="segment_idx" label="#" width="48" />
        <el-table-column label="人物" width="80">
          <template #default="{ row }">{{ row.global_person_id != null ? `G${row.global_person_id}` : '—' }}</template>
        </el-table-column>
        <el-table-column label="出现起止" min-width="200">
          <template #default="{ row }">{{ row.start_time }} — {{ row.end_time }}</template>
        </el-table-column>
        <el-table-column prop="duration_time" label="出现时长" width="100" />
        <el-table-column label="进入" min-width="100">
          <template #default="{ row }">{{ row.entry_room || '—' }}</template>
        </el-table-column>
        <el-table-column label="离开" min-width="100">
          <template #default="{ row }">{{ row.exit_room || '—' }}</template>
        </el-table-column>
        <el-table-column label="停留房间" min-width="100">
          <template #default="{ row }">{{ row.exit_room || '—' }}</template>
        </el-table-column>
        <el-table-column label="停留时间" width="100">
          <template #default="{ row }">{{ row.stay_duration_display }}</template>
        </el-table-column>
        <el-table-column prop="hit_count" label="段内帧数" width="90" />
        <el-table-column prop="best_similarity" label="最高相似" width="100">
          <template #default="{ row }">{{ formatSimilarityScore(row.best_similarity) }}</template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>
