/** 从 axios / FastAPI 错误中取出后端返回的具体信息。 */
export function apiErrorMessage(err: unknown, fallback: string): string {
  const detail = (err as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail
  if (typeof detail === 'string' && detail.trim()) return detail.trim()
  if (Array.isArray(detail)) {
    const parts = detail.map((item) => {
      if (typeof item === 'string') return item
      if (item && typeof item === 'object' && 'msg' in item) {
        return String((item as { msg: unknown }).msg)
      }
      try {
        return JSON.stringify(item)
      } catch {
        return String(item)
      }
    })
    const joined = parts.filter(Boolean).join('；')
    if (joined) return joined
  }
  if (detail && typeof detail === 'object' && detail !== null) {
    if ('error' in detail) {
      const msg = String((detail as { error: unknown }).error || '').trim()
      if (msg) return msg
    }
    if ('message' in detail) {
      const msg = String((detail as { message: unknown }).message || '').trim()
      if (msg) return msg
    }
  }
  const message = err instanceof Error ? err.message.trim() : ''
  if (message && message !== 'Network Error') return message
  if (message === 'Network Error') return '网络错误，无法连接检索服务'
  return fallback
}
