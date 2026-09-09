import axios from 'axios'

const http = axios.create({
  baseURL: '/api',
  timeout: 0,
})

http.interceptors.request.use((config) => {
  const t = localStorage.getItem('token')
  if (t) {
    config.headers.Authorization = `Bearer ${t}`
  }
  return config
})

http.interceptors.response.use(
  (r) => r,
  (err) => {
    if (err.response?.status === 401) {
      const reqUrl = err.config?.url || ''
      if (reqUrl.includes('/auth/login')) {
        return Promise.reject(err)
      }
      localStorage.removeItem('token')
      if (!window.location.pathname.includes('/login')) {
        window.location.href = '/login'
      }
    }
    return Promise.reject(err)
  },
)

export default http
