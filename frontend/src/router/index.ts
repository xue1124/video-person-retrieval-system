import { useAuthStore } from '@/stores/auth'
import MainLayout from '@/layouts/MainLayout.vue'
import LoginView from '@/views/LoginView.vue'
import MediaSourcesView from '@/views/MediaSourcesView.vue'
import PersonTracksView from '@/views/PersonTracksView.vue'
import ReportsView from '@/views/ReportsView.vue'
import SearchLogsView from '@/views/SearchLogsView.vue'
import SearchView from '@/views/SearchView.vue'
import VideoModelView from '@/views/VideoModelView.vue'
import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: '/login', name: 'login', component: LoginView, meta: { public: true } },
    {
      path: '/',
      component: MainLayout,
      meta: { requiresAuth: true },
      children: [
        { path: '', redirect: { name: 'model' } },
        { path: 'model', name: 'model', component: VideoModelView },
        { path: 'media', name: 'media', component: MediaSourcesView },
        { path: 'search', name: 'search', component: SearchView },
        { path: 'tracks', name: 'tracks', component: PersonTracksView },
        { path: 'reports', name: 'reports', component: ReportsView },
        { path: 'logs', name: 'logs', component: SearchLogsView },
      ],
    },
  ],
})

router.beforeEach(async (to) => {
  const auth = useAuthStore()
  if (to.meta.public) {
    if (auth.token && to.name === 'login') {
      return { name: 'model' }
    }
    return true
  }
  if (!auth.token) {
    return { name: 'login', query: { redirect: to.fullPath } }
  }
  if (!auth.username) {
    try {
      await auth.fetchMe()
    } catch {
      auth.logout()
      return { name: 'login' }
    }
  }
  return true
})

export default router
