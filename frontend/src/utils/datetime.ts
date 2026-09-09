/** 解析 API 时间；带时区按标准解析，无时区 legacy 视为 UTC（与后端 serialize_datetime 一致）。 */
export function parseServerDate(iso: string | null | undefined): number {
  if (iso == null || String(iso).trim() === "") return NaN
  const s = String(iso).trim().replace(" ", "T")
  if (/[zZ]$/.test(s) || /[+-]\d{2}:?\d{2}$/.test(s)) {
    return new Date(s).getTime()
  }
  return new Date(`${s}Z`).getTime()
}

/** 格式化为东八区墙钟显示：YYYY-MM-DD HH:mm:ss */
export function formatServerDateTime(iso: string | null | undefined): string {
  const t = parseServerDate(iso)
  if (!Number.isFinite(t)) return "--"
  const parts = new Intl.DateTimeFormat("zh-CN", {
    timeZone: "Asia/Shanghai",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).formatToParts(new Date(t))
  const pick = (type: string) => parts.find((p) => p.type === type)?.value ?? ""
  return `${pick("year")}-${pick("month")}-${pick("day")} ${pick("hour")}:${pick("minute")}:${pick("second")}`
}
