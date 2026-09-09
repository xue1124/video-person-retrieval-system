import http from '@/api/http'

export interface FullVideoUrlOptions {
  /** 直播边录边播：仅文件变大时变更，避免同片段反复全量重下 */
  revision?: string
}

/** 完整视频 URL（供 <video src>，query 携带 JWT） */
export function fullVideoUrl(videoName: string, options?: FullVideoUrlOptions): string {
  const params = new URLSearchParams({ video_name: videoName })
  const token = localStorage.getItem('token')
  if (token) params.set('token', token)
  if (options?.revision) params.set('v', options.revision)
  return `/api/files/video?${params.toString()}`
}

export interface VideoCheckResult {
  ok: boolean
  needs_transcode?: boolean
  browser_friendly?: boolean
  is_live_recording?: boolean
  needs_faststart?: boolean
  playback_revision?: string | null
}

/** 播放前检查归档是否存在 */
export async function checkFullVideo(videoName: string): Promise<VideoCheckResult> {
  const { data } = await http.get<VideoCheckResult>('/files/video/check', {
    params: { video_name: videoName },
  })
  return data
}

export function buildPlaybackUrl(chk: VideoCheckResult, videoName: string): string {
  if (chk.is_live_recording) {
    return fullVideoUrl(videoName, {
      revision: chk.playback_revision || String(Date.now()),
    })
  }
  return fullVideoUrl(videoName)
}

export function videoLoadingHint(opts: {
  isLive?: boolean
  needsTranscode?: boolean
  needsFaststart?: boolean
}): string {
  if (opts.isLive) return '正在加载已录制片段…'
  if (opts.needsTranscode) return '正在加载视频（首次需转码，请稍候）…'
  if (opts.needsFaststart) return '正在优化播放格式（请稍候）…'
  return '正在加载视频…'
}
