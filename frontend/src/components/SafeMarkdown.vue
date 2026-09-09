<script setup lang="ts">
import { parseMarkdownBlocks, splitInline } from '@/utils/safeMarkdown'

defineProps<{
  markdown: string
}>()
</script>

<template>
  <div class="safe-md">
    <template v-for="(block, index) in parseMarkdownBlocks(markdown)" :key="index">
      <h1 v-if="block.kind === 'heading' && block.level === 1" class="safe-md-h1">
        <template v-for="(part, pi) in splitInline(block.text)" :key="pi">
          <strong v-if="part.bold">{{ part.text }}</strong>
          <template v-else>{{ part.text }}</template>
        </template>
      </h1>
      <h2 v-else-if="block.kind === 'heading' && block.level === 2" class="safe-md-h2">
        <template v-for="(part, pi) in splitInline(block.text)" :key="pi">
          <strong v-if="part.bold">{{ part.text }}</strong>
          <template v-else>{{ part.text }}</template>
        </template>
      </h2>
      <h3 v-else-if="block.kind === 'heading' && block.level === 3" class="safe-md-h3">
        <template v-for="(part, pi) in splitInline(block.text)" :key="pi">
          <strong v-if="part.bold">{{ part.text }}</strong>
          <template v-else>{{ part.text }}</template>
        </template>
      </h3>
      <h4 v-else-if="block.kind === 'heading' && block.level === 4" class="safe-md-h4">
        <template v-for="(part, pi) in splitInline(block.text)" :key="pi">
          <strong v-if="part.bold">{{ part.text }}</strong>
          <template v-else>{{ part.text }}</template>
        </template>
      </h4>
      <p v-else-if="block.kind === 'paragraph'" class="safe-md-p">
        <template v-for="(part, pi) in splitInline(block.text)" :key="pi">
          <strong v-if="part.bold">{{ part.text }}</strong>
          <template v-else>{{ part.text }}</template>
        </template>
      </p>
      <ul v-else-if="block.kind === 'list' && !block.ordered" class="safe-md-ul">
        <li v-for="(item, li) in block.items" :key="li">
          <template v-for="(part, pi) in splitInline(item)" :key="pi">
            <strong v-if="part.bold">{{ part.text }}</strong>
            <template v-else>{{ part.text }}</template>
          </template>
        </li>
      </ul>
      <ol v-else-if="block.kind === 'list' && block.ordered" class="safe-md-ol">
        <li v-for="(item, li) in block.items" :key="li">
          <template v-for="(part, pi) in splitInline(item)" :key="pi">
            <strong v-if="part.bold">{{ part.text }}</strong>
            <template v-else>{{ part.text }}</template>
          </template>
        </li>
      </ol>
      <div v-else-if="block.kind === 'table'" class="safe-md-table-wrap">
        <table class="safe-md-table">
          <thead>
            <tr>
              <th v-for="(head, hi) in block.headers" :key="hi">
                <template v-for="(part, pi) in splitInline(head)" :key="pi">
                  <strong v-if="part.bold">{{ part.text }}</strong>
                  <template v-else>{{ part.text }}</template>
                </template>
              </th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(row, ri) in block.rows" :key="ri">
              <td v-for="(cell, ci) in row" :key="ci">
                <template v-for="(part, pi) in splitInline(cell)" :key="pi">
                  <strong v-if="part.bold">{{ part.text }}</strong>
                  <template v-else>{{ part.text }}</template>
                </template>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <hr v-else-if="block.kind === 'hr'" class="safe-md-hr" />
    </template>
  </div>
</template>

<style scoped>
.safe-md {
  color: inherit;
  line-height: 1.7;
  word-break: break-word;
}
.safe-md-h1 {
  margin: 0.4em 0 0.6em;
  font-size: 1.35rem;
  font-weight: 700;
}
.safe-md-h2 {
  margin: 1.2em 0 0.55em;
  font-size: 1.15rem;
  font-weight: 700;
}
.safe-md-h3 {
  margin: 1em 0 0.4em;
  font-size: 1.02rem;
  font-weight: 650;
}
.safe-md-h4 {
  margin: 0.85em 0 0.35em;
  font-size: 0.95rem;
  font-weight: 650;
}
.safe-md-p {
  margin: 0.4em 0 0.7em;
}
.safe-md-ul,
.safe-md-ol {
  margin: 0.3em 0 0.8em;
  padding-left: 1.4em;
}
.safe-md-ul li,
.safe-md-ol li {
  margin: 0.2em 0;
}
.safe-md-table-wrap {
  margin: 0.6em 0 1em;
  overflow-x: auto;
}
.safe-md-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
}
.safe-md-table th,
.safe-md-table td {
  border: 1px solid #cbd5e1;
  padding: 0.45em 0.65em;
  text-align: left;
}
.safe-md-table th {
  background: #f8fafc;
  font-weight: 650;
}
.safe-md-hr {
  margin: 1em 0;
  border: 0;
  border-top: 1px solid #e2e8f0;
}
</style>
