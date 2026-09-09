<script setup lang="ts">
import { useAuthStore } from '@/stores/auth'
import { useUploadStore } from '@/stores/upload'
import {
  DataAnalysis,
  Document,
  Fold,
  Expand,
  Moon,
  Notebook,
  Search,
  Sunny,
  User,
  VideoCamera,
} from '@element-plus/icons-vue'
import { computed, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'

const router = useRouter()
const route = useRoute()
const auth = useAuthStore()
const upload = useUploadStore()
const collapsed = ref(false)
const isDark = ref(localStorage.getItem('ui_theme') === 'dark')

const pageTitle = computed(() => {
  const name = route.name?.toString() || ''
  const titleMap: Record<string, string> = {
    model: '视频导入及建模',
    media: '视频源列表',
    search: '轨迹检索',
    tracks: '人物轨迹档案',
    reports: 'AI 分析报告',
    logs: '检索日志',
  }
  return titleMap[name] || '工作台'
})

function go(name: string) {
  router.push({ name })
}

function logout() {
  auth.logout()
  router.push({ name: 'login' })
}

function toggleTheme() {
  isDark.value = !isDark.value
  localStorage.setItem('ui_theme', isDark.value ? 'dark' : 'light')
}
</script>

<template>
  <div
    class="relative flex h-full overflow-hidden transition-colors duration-300"
    :class="isDark ? 'bg-slate-950 text-slate-200' : 'bg-slate-100 text-slate-700'"
  >
    <div class="pointer-events-none absolute inset-0">
      <div
        class="absolute -top-24 -left-20 h-72 w-72 rounded-full blur-3xl"
        :class="isDark ? 'bg-sky-900/20' : 'bg-sky-200/35'"
      />
      <div
        class="absolute right-0 bottom-0 h-96 w-96 rounded-full blur-3xl"
        :class="isDark ? 'bg-indigo-900/20' : 'bg-indigo-200/30'"
      />
    </div>
    <aside
      class="z-10 flex shrink-0 flex-col text-white transition-[width] duration-200"
      :class="collapsed ? 'w-[72px]' : 'w-[220px]'"
      style="
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        box-shadow: 4px 0 26px rgba(15, 23, 42, 0.22);
      "
    >
      <div class="flex items-center gap-3 border-b border-white/10 px-4 py-5">
        <el-icon class="text-2xl text-sky-400"><VideoCamera /></el-icon>
        <div v-show="!collapsed" class="leading-tight">
          <div class="text-[15px] font-semibold text-slate-100">医保智能稽查分析系统</div>
          <div class="text-xs text-slate-400">多模态轨迹追踪</div>
        </div>
      </div>
      <el-menu
        :default-active="route.name?.toString()"
        class="sidebar-menu flex-1 border-0 bg-transparent"
        :collapse="collapsed"
        :router="false"
        text-color="#cbd5e1"
        active-text-color="#ffffff"
        style="background-color: transparent; --el-menu-bg-color: transparent"
      >
        <el-menu-item index="model" @click="go('model')">
          <el-icon><VideoCamera /></el-icon>
          <span>视频导入及建模</span>
        </el-menu-item>
        <el-menu-item index="media" @click="go('media')">
          <el-icon><DataAnalysis /></el-icon>
          <span>视频源列表</span>
        </el-menu-item>
        <el-menu-item index="search" @click="go('search')">
          <el-icon><Search /></el-icon>
          <span>轨迹检索</span>
        </el-menu-item>
        <el-menu-item index="tracks" @click="go('tracks')">
          <el-icon><User /></el-icon>
          <span>人物轨迹档案</span>
        </el-menu-item>
        <el-menu-item index="reports" @click="go('reports')">
          <el-icon><Notebook /></el-icon>
          <span>AI 分析报告</span>
        </el-menu-item>
        <el-menu-item index="logs" @click="go('logs')">
          <el-icon><Document /></el-icon>
          <span>检索日志</span>
        </el-menu-item>
      </el-menu>
      <div class="border-t border-white/10 p-2">
        <el-button class="w-full !text-slate-300 hover:!text-sky-300" text type="primary" @click="collapsed = !collapsed">
          <el-icon><Fold v-if="!collapsed" /><Expand v-else /></el-icon>
        </el-button>
      </div>
    </aside>

    <div class="z-10 flex min-w-0 flex-1 flex-col">
      <header
        class="flex h-16 shrink-0 items-center justify-between border-b px-6 backdrop-blur transition-colors duration-300"
        :class="
          isDark
            ? 'border-slate-800/80 bg-slate-900/90'
            : 'border-slate-200/60 bg-white/90'
        "
      >
        <div class="flex flex-col">
          <span class="text-base font-semibold" :class="isDark ? 'text-slate-100' : 'text-slate-800'">{{ pageTitle }}</span>
        </div>
        <div class="flex items-center gap-3">
          <div
            v-if="upload.inProgress && route.name !== 'model'"
            class="flex max-w-md items-center gap-2 rounded-full border px-3 py-1"
            :class="isDark ? 'border-sky-800 bg-sky-950/60 text-sky-200' : 'border-sky-200 bg-sky-50 text-sky-700'"
          >
            <span class="truncate text-xs">
              正在上传 {{ upload.uploadFileName || '视频' }}
              <span v-if="upload.queuedCount > 0"> · 队列 {{ upload.queuedCount }}</span>
            </span>
            <span class="shrink-0 text-xs font-semibold">{{ upload.uploadPercent }}%</span>
          </div>
          <el-button circle size="small" @click="toggleTheme">
            <el-icon><Moon v-if="!isDark" /><Sunny v-else /></el-icon>
          </el-button>
          <span
            class="rounded-full px-3 py-1 text-sm"
            :class="isDark ? 'bg-slate-800 text-slate-200' : 'bg-slate-100 text-slate-600'"
          >
            {{ auth.username }}
          </span>
          <el-tag v-if="auth.isAdmin" type="warning" size="small">管理员</el-tag>
          <el-button type="danger" plain size="small" round @click="logout">退出</el-button>
        </div>
      </header>
      <main class="min-h-0 flex-1 overflow-auto p-6">
        <router-view v-slot="{ Component }">
          <keep-alive include="VideoModelView,PersonTracksView,ReportsView">
            <component :is="Component" />
          </keep-alive>
        </router-view>
      </main>
    </div>
  </div>
</template>

<style scoped>
.sidebar-menu {
  padding-top: 10px;
  background: transparent !important;
  --el-menu-bg-color: transparent;
  --el-menu-hover-bg-color: transparent;
  border-right: none !important;
}
.sidebar-menu :deep(.el-menu) {
  background: transparent !important;
  --el-menu-bg-color: transparent;
  --el-menu-hover-bg-color: transparent;
  border-right: none !important;
}
.sidebar-menu :deep(.el-menu-item) {
  margin: 4px 10px;
  border-radius: 10px;
  color: #f1f5f9 !important;
  font-weight: 600;
  opacity: 1 !important;
}
.sidebar-menu :deep(.el-menu-item .el-icon) {
  color: #e2e8f0 !important;
}
.sidebar-menu :deep(.el-menu-item.is-active) {
  background: linear-gradient(90deg, rgba(59, 130, 246, 0.45) 0%, rgba(59, 130, 246, 0.2) 100%);
  border-left: 3px solid #3b82f6;
  color: #ffffff !important;
  font-weight: 600;
}
.sidebar-menu :deep(.el-menu-item.is-active .el-icon) {
  color: #ffffff !important;
}
.sidebar-menu :deep(.el-menu-item:hover) {
  background: linear-gradient(90deg, rgba(64, 158, 255, 0.2) 0%, transparent 100%);
  color: #ffffff !important;
}
.sidebar-menu :deep(.el-menu-item:hover .el-icon) {
  color: #ffffff !important;
}

/* 暗色主题下，让内容模块也随主题切换 */
.bg-slate-950 :deep(.el-card) {
  background: #111827 !important;
  border-color: #334155 !important;
  color: #e5e7eb !important;
}

.bg-slate-950 :deep(.el-card__header) {
  border-bottom-color: #334155 !important;
}

.bg-slate-950 :deep(.el-form-item__label),
.bg-slate-950 :deep(.el-descriptions__label) {
  color: #cbd5e1 !important;
}

.bg-slate-950 :deep(.el-input__wrapper),
.bg-slate-950 :deep(.el-textarea__inner),
.bg-slate-950 :deep(.el-select__wrapper),
.bg-slate-950 :deep(.el-input-number__decrease),
.bg-slate-950 :deep(.el-input-number__increase) {
  background: #0f172a !important;
  border-color: #334155 !important;
  box-shadow: 0 0 0 1px #334155 inset !important;
  color: #e2e8f0 !important;
}

.bg-slate-950 :deep(.el-table) {
  --el-table-bg-color: #0f172a;
  --el-table-tr-bg-color: #0f172a;
  --el-table-header-bg-color: #111827;
  --el-table-border-color: #334155;
  --el-table-text-color: #e2e8f0;
  --el-fill-color-lighter: #111827;
  --el-fill-color-blank: #0f172a;
}

.bg-slate-950 :deep(.el-table th.el-table__cell),
.bg-slate-950 :deep(.el-table td.el-table__cell) {
  background: transparent !important;
  border-bottom-color: #334155 !important;
}

.bg-slate-950 :deep(.el-radio-button__inner) {
  background: #0f172a;
  border-color: #334155;
  color: #cbd5e1;
}

.bg-slate-950 :deep(.el-radio-button__original-radio:checked + .el-radio-button__inner) {
  background: #2563eb;
  border-color: #2563eb;
  color: #fff;
}

.bg-slate-950 :deep(.el-upload-dragger) {
  background: #0b1220 !important;
  border-color: #334155 !important;
  color: #cbd5e1 !important;
}
</style>
