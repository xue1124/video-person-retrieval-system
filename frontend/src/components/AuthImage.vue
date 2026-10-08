<script setup lang="ts">
import http from '@/api/http'
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'

const props = withDefaults(
  defineProps<{
    path: string
    alt?: string
    /** 预览大图等场景用 contain，列表缩略图默认 cover */
    fit?: 'cover' | 'contain'
    /** 列表中的紧凑缩略图，不使用大图预览的最小高度 */
    compact?: boolean
  }>(),
  { fit: 'cover', compact: false },
)

const isContain = computed(() => props.fit === 'contain')

const blobUrl = ref<string | null>(null)
const loading = ref(false)

async function load() {
  if (!props.path) return
  loading.value = true
  try {
    if (blobUrl.value) {
      URL.revokeObjectURL(blobUrl.value)
      blobUrl.value = null
    }
    const { data } = await http.get(props.path, { responseType: 'blob' })
    blobUrl.value = URL.createObjectURL(data)
  } finally {
    loading.value = false
  }
}

onMounted(load)
watch(() => props.path, load)
onBeforeUnmount(() => {
  if (blobUrl.value) URL.revokeObjectURL(blobUrl.value)
})
</script>

<template>
  <div
    class="relative w-full bg-slate-800/40"
    :class="
      isContain
        ? compact
          ? 'flex h-32 items-center justify-center overflow-hidden rounded-lg bg-slate-100 p-1'
          : 'flex min-h-[200px] max-h-[85vh] items-center justify-center overflow-visible rounded-lg p-2'
        : 'min-h-[80px] overflow-hidden rounded-lg'
    "
  >
    <div v-if="loading" class="flex h-32 w-full items-center justify-center text-slate-400">加载中…</div>
    <img
      v-else-if="blobUrl"
      :src="blobUrl"
      :alt="alt || ''"
      :class="
        isContain
          ? compact
            ? 'max-h-full max-w-full object-contain'
            : 'max-h-[82vh] max-w-full object-contain'
          : 'h-full w-full object-cover'
      "
    />
  </div>
</template>
