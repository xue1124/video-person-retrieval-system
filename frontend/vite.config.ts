import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import vue from '@vitejs/plugin-vue'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [vue(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    // 监听 0.0.0.0，局域网内可通过 http://<本机IP>:5174 访问（v2 与 v1 5173 区分）
    host: true,
    port: 5174,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8002',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ''),
        timeout: 0,
      },
    },
  },
})
