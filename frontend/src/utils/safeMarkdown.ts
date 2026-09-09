export type MarkdownBlock =
  | { kind: 'heading'; level: 1 | 2 | 3 | 4; text: string }
  | { kind: 'paragraph'; text: string }
  | { kind: 'list'; ordered: boolean; items: string[] }
  | { kind: 'table'; headers: string[]; rows: string[][] }
  | { kind: 'hr' }

export interface InlinePart {
  text: string
  bold: boolean
}

function stripHtmlTags(input: string): string {
  return input.replace(/<\/?[a-zA-Z][^>]*>/g, '')
}

function isTableSeparator(line: string): boolean {
  const trimmed = line.trim()
  if (!trimmed.includes('|')) return false
  return /^\|?[\s:|-]+\|[\s:|-]*\|?$/.test(trimmed)
}

function splitTableRow(line: string): string[] {
  let raw = line.trim()
  if (raw.startsWith('|')) raw = raw.slice(1)
  if (raw.endsWith('|')) raw = raw.slice(0, -1)
  return raw.split('|').map((cell) => cell.trim())
}

function headingLevel(line: string): 1 | 2 | 3 | 4 | null {
  const match = /^(#{1,4})\s+(.+)$/.exec(line)
  if (!match) return null
  const level = match[1].length as 1 | 2 | 3 | 4
  return level
}

export function parseMarkdownBlocks(source: string): MarkdownBlock[] {
  const lines = stripHtmlTags(source).replace(/\r\n/g, '\n').split('\n')
  const blocks: MarkdownBlock[] = []
  let i = 0
  while (i < lines.length) {
    const trimmed = lines[i].trim()
    if (!trimmed) {
      i += 1
      continue
    }
    if (/^---+$/.test(trimmed) || /^\*\*\*+$/.test(trimmed)) {
      blocks.push({ kind: 'hr' })
      i += 1
      continue
    }
    const heading = headingLevel(trimmed)
    if (heading) {
      blocks.push({
        kind: 'heading',
        level: heading,
        text: trimmed.replace(/^#{1,4}\s+/, ''),
      })
      i += 1
      continue
    }
    if (trimmed.includes('|') && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
      const headers = splitTableRow(trimmed)
      const rows: string[][] = []
      i += 2
      while (i < lines.length && lines[i].includes('|') && lines[i].trim()) {
        if (!isTableSeparator(lines[i])) {
          rows.push(splitTableRow(lines[i]))
        }
        i += 1
      }
      blocks.push({ kind: 'table', headers, rows })
      continue
    }
    if (/^[-*+]\s+/.test(trimmed)) {
      const items: string[] = []
      while (i < lines.length && /^[-*+]\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^[-*+]\s+/, ''))
        i += 1
      }
      blocks.push({ kind: 'list', ordered: false, items })
      continue
    }
    if (/^\d+\.\s+/.test(trimmed)) {
      const items: string[] = []
      while (i < lines.length && /^\d+\.\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^\d+\.\s+/, ''))
        i += 1
      }
      blocks.push({ kind: 'list', ordered: true, items })
      continue
    }
    const para: string[] = [trimmed]
    i += 1
    while (
      i < lines.length &&
      lines[i].trim() &&
      !headingLevel(lines[i].trim()) &&
      !/^[-*+]\s+/.test(lines[i].trim()) &&
      !/^\d+\.\s+/.test(lines[i].trim()) &&
      !lines[i].includes('|')
    ) {
      para.push(lines[i].trim())
      i += 1
    }
    blocks.push({ kind: 'paragraph', text: para.join(' ') })
  }
  return blocks
}

export function splitInline(text: string): InlinePart[] {
  const parts: InlinePart[] = []
  const re = /\*\*(.+?)\*\*/g
  let last = 0
  let match = re.exec(text)
  while (match) {
    if (match.index > last) {
      parts.push({ text: text.slice(last, match.index), bold: false })
    }
    parts.push({ text: match[1], bold: true })
    last = match.index + match[0].length
    match = re.exec(text)
  }
  if (last < text.length) parts.push({ text: text.slice(last), bold: false })
  return parts.length ? parts : [{ text: '', bold: false }]
}
