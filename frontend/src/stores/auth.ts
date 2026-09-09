import http from '@/api/http'
import { defineStore } from 'pinia'

export const useAuthStore = defineStore('auth', {
  state: () => ({
    token: localStorage.getItem('token') || '',
    username: '',
    role: '',
  }),
  getters: {
    isAdmin: (s) => s.role === 'admin',
  },
  actions: {
    async login(username: string, password: string) {
      const { data } = await http.post('/auth/login', { username, password })
      this.token = data.access_token
      this.username = data.username
      this.role = data.role
      localStorage.setItem('token', this.token)
      http.defaults.headers.common.Authorization = `Bearer ${this.token}`
    },
    async fetchMe() {
      if (!this.token) return
      const { data } = await http.get('/auth/me')
      this.username = data.username
      this.role = data.role
    },
    logout() {
      this.token = ''
      this.username = ''
      this.role = ''
      localStorage.removeItem('token')
      delete http.defaults.headers.common.Authorization
    },
  },
})
