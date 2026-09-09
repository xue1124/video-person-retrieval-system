<script setup lang="ts">
import { useAuthStore } from '@/stores/auth'
import { ElMessage } from 'element-plus'
import { reactive, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'

const router = useRouter()
const route = useRoute()
const auth = useAuthStore()
const loading = ref(false)
const form = reactive({ username: '', password: '' })

async function submit() {
  if (!form.username.trim() || !form.password) {
    ElMessage.warning('请输入用户名和密码')
    return
  }
  loading.value = true
  try {
    await auth.login(form.username, form.password)
    ElMessage.success('登录成功')
    const redir = (route.query.redirect as string) || '/'
    router.replace(redir)
  } catch (e: unknown) {
    ElMessage.error('登录失败，请检查账号密码')
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div
    class="flex min-h-full items-center justify-center bg-gradient-to-br from-[#1a1a2e] via-[#16162a] to-[#0f172a] p-6"
  >
    <el-card class="w-full max-w-md shadow-2xl" shadow="always">
      <template #header>
        <div class="text-center text-lg font-semibold text-slate-800">医保智能稽查分析系统</div>
      </template>
      <el-form label-position="top" @submit.prevent="submit">
        <el-form-item label="用户名">
          <el-input v-model="form.username" autocomplete="username" />
        </el-form-item>
        <el-form-item label="密码">
          <el-input v-model="form.password" type="password" autocomplete="current-password" />
        </el-form-item>
        <el-button type="primary" class="w-full" :loading="loading" native-type="submit">登录</el-button>
      </el-form>
      <p class="mt-4 text-center text-xs text-slate-400"> </p>
    </el-card>
  </div>
</template>
