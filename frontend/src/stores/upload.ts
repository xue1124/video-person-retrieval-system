import http from '@/api/http'
import type { UploadRequestOptions } from 'element-plus'
import { ElMessage } from 'element-plus'
import { defineStore } from 'pinia'

type QueuedUpload = {
  opt: UploadRequestOptions
  skipFrames: number
  confVal: number
  capturedAt: string | null
}

export const useUploadStore = defineStore('upload', {
  state: () => ({
    uploading: false,
    uploadPercent: 0,
    uploadFileName: '',
    uploadController: null as AbortController | null,
    queue: [] as QueuedUpload[],
  }),
  getters: {
    queuedCount: (s) => s.queue.length,
    inProgress: (s) => s.uploading || s.queue.length > 0,
  },
  actions: {
    enqueue(opt: UploadRequestOptions, skipFrames: number, confVal: number, capturedAt: string | null = null) {
      this.queue.push({ opt, skipFrames, confVal, capturedAt })
      void this.processQueue()
    },
    async processQueue() {
      if (this.uploading) return
      const item = this.queue.shift()
      if (!item) return

      const fd = new FormData()
      const file = item.opt.file as File
      fd.append('file', file)
      fd.append('skip_frames', String(item.skipFrames))
      fd.append('conf_val', String(item.confVal))
      if (item.capturedAt) {
        fd.append('captured_at', item.capturedAt)
      }
      this.uploading = true
      this.uploadPercent = 0
      this.uploadFileName = file.name
      this.uploadController = new AbortController()
      try {
        const { data } = await http.post<{ task_id: string }>('/analyze', fd, {
          signal: this.uploadController.signal,
          onUploadProgress: (evt) => {
            if (!evt.total) return
            const percent = Math.min(100, Math.round((evt.loaded * 100) / evt.total))
            this.uploadPercent = percent
            item.opt.onProgress?.({ percent } as any)
          },
        })
        this.uploadPercent = 100
        ElMessage({
          type: 'success',
          message: `本地上传已提交，task_id=${data.task_id}。请到「视频源」看进度；多任务在 solo 池会排队。`,
          duration: 6000,
        })
        item.opt.onSuccess?.({})
      } catch (err: any) {
        this.uploadPercent = 0
        item.opt.onError?.(err as any)
        if (err?.code === 'ERR_CANCELED' || err?.name === 'CanceledError') {
          ElMessage.info('已取消本地视频上传')
        } else {
          ElMessage.error('本地视频上传失败，请检查网络或服务器限制')
        }
      } finally {
        this.uploading = false
        this.uploadController = null
        void this.processQueue()
      }
    },
    cancelCurrent() {
      this.uploadController?.abort()
    },
    cancelAll() {
      this.queue = []
      this.uploadController?.abort()
    },
  },
})
