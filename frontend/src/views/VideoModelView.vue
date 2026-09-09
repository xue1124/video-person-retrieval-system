<script setup lang="ts">
import http from "@/api/http";
import { useUploadStore } from "@/stores/upload";
import { UploadFilled } from "@element-plus/icons-vue";
import type { UploadRequestOptions } from "element-plus";
import { ElMessage } from "element-plus";
import { computed, reactive, ref } from "vue";

defineOptions({
  name: "VideoModelView",
});

interface IsapiChannel {
  id: number;
  name: string;
  raw_id: string;
}

const upload = useUploadStore();
const skip = ref(75);
const conf = ref(0.6);
const capturedAt = ref("");
const probingChannels = ref(false);
const channelOptions = ref<IsapiChannel[]>([]);
const modelSubPage = ref<"local" | "isapi" | "rtsp">("local");
const rtspSubmitting = ref(false);
const rtspLive = reactive({
  url: "",
  taskName: "",
});
const vendorOptions = [{ label: "海康NVR", value: "hikvision" }];
const selectedVendor = ref(vendorOptions[0].value);

const nvr = reactive({
  host: "",
  username: "admin",
  password: "",
  channels: [] as number[],
  startAt: "2025-04-16 10:00",
  endAt: "2025-04-16 11:00",
});

const canProbeChannels = computed(() => !!nvr.host.trim() && !!nvr.username.trim() && !!nvr.password.trim());
const canSubmitIsapi = computed(
  () =>
    !!nvr.password.trim() &&
    !!nvr.startAt &&
    !!nvr.endAt &&
    Array.isArray(nvr.channels) &&
    nvr.channels.length > 0,
);

const autoTaskNames = computed(() =>
  {
    const vendorLabel = vendorOptions.find((item) => item.value === selectedVendor.value)?.label || "海康NVR";
    return [...nvr.channels]
      .sort((a, b) => a - b)
      .map((channel) => `${vendorLabel}回放_通道${String(channel).padStart(2, "0")}_${nvr.startAt}_${nvr.endAt}`);
  },
);

const canSubmitRtsp = computed(
  () => !!rtspLive.url.trim().startsWith("rtsp://") && !!rtspLive.taskName.trim(),
);

function doLocalUpload(opt: UploadRequestOptions) {
  upload.enqueue(opt, skip.value, conf.value, capturedAt.value || null);
}

function cancelLocalUpload() {
  upload.cancelCurrent();
}

function cancelAllLocalUploads() {
  upload.cancelAll();
}

async function detectChannels() {
  if (!canProbeChannels.value) {
    ElMessage.warning("请先填写设备 IP、用户名和密码");
    return;
  }
  probingChannels.value = true;
  try {
    const fd = new URLSearchParams();
    fd.append("host", nvr.host.trim());
    fd.append("username", nvr.username.trim());
    fd.append("password", nvr.password);
    const { data } = await http.post<{ channels: IsapiChannel[] }>("/isapi/channels", fd, {
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
    });
    channelOptions.value = data.channels || [];
    nvr.channels = channelOptions.value.map((it) => it.id);
    ElMessage.success(`已识别 ${channelOptions.value.length} 个通道`);
  } catch (err: any) {
    ElMessage.error(err?.response?.data?.detail || "识别通道失败");
  } finally {
    probingChannels.value = false;
  }
}

async function submitIsapi() {
  if (!nvr.password.trim()) {
    ElMessage.warning("请填写设备登录密码");
    return;
  }
  if (!nvr.startAt || !nvr.endAt) {
    ElMessage.warning("请选择开始时间和结束时间");
    return;
  }
  if (!nvr.channels.length) {
    ElMessage.warning("请至少选择一个通道");
    return;
  }
  const start = nvr.startAt.replace(" ", "T") + ":00";
  const end = nvr.endAt.replace(" ", "T") + ":00";
  const fd = new URLSearchParams();
  fd.append("source_mode", "isapi");
  fd.append("host", nvr.host.trim());
  fd.append("username", nvr.username.trim());
  fd.append("password", nvr.password);
  fd.append("isapi_start", start);
  fd.append("isapi_end", end);
  fd.append("skip_frames", String(skip.value));
  fd.append("conf_val", String(conf.value));
  for (const ch of nvr.channels) {
    fd.append("channels", String(ch));
  }
  const { data } = await http.post<{ task_ids: string[]; task_names: string[]; count: number }>(
    "/analyze_stream",
    fd,
    { headers: { "Content-Type": "application/x-www-form-urlencoded" } },
  );
  ElMessage({
    type: "success",
    message: `ISAPI 已提交 ${data.count} 个任务。请到「视频源列表」查看各通道进度。`,
    duration: 7000,
  });
}

async function submitRtspLive() {
  const url = rtspLive.url.trim();
  const name = rtspLive.taskName.trim();
  if (!url.startsWith("rtsp://")) {
    ElMessage.warning("请填写以 rtsp:// 开头的地址");
    return;
  }
  if (!name) {
    ElMessage.warning("请填写任务名称");
    return;
  }
  const fd = new URLSearchParams();
  fd.append("rtsp_url", url);
  fd.append("file_name", name);
  fd.append("skip_frames", String(skip.value));
  fd.append("conf_val", String(conf.value));
  rtspSubmitting.value = true;
  try {
    const { data } = await http.post<{ task_id: string; file_name: string }>("/stream/live/start", fd, {
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
    });
    ElMessage({
      type: "success",
      message: `实时 RTSP 已启动：${data.file_name}。边录边分析，可在检索中查看并播放已录制片段。`,
      duration: 7000,
    });
  } catch (err: any) {
    ElMessage.error(err?.response?.data?.detail || "启动实时流失败");
  } finally {
    rtspSubmitting.value = false;
  }
}
</script>

<template>
  <div class="space-y-6">
    <div class="rounded-xl border border-slate-200/70 bg-white/85 p-4 shadow-sm backdrop-blur">
      <div class="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 class="text-xl font-semibold text-slate-800">视频导入及建模参数</h2>
          <p class="mt-1 text-sm text-slate-500">
            提交建模任务：本地上传、NVR 历史回放（ISAPI）、或 RTSP 实时拉流分析。
          </p>
        </div>
        <div class="grid grid-cols-3 gap-2">
          <div class="rounded-lg bg-slate-50 px-3 py-2 text-center shadow-sm">
            <div class="text-xs text-slate-500">抽帧步长</div>
            <div class="text-base font-semibold text-slate-700">{{ skip }}</div>
          </div>
          <div class="rounded-lg bg-sky-50 px-3 py-2 text-center shadow-sm">
            <div class="text-xs text-sky-600">检测灵敏度</div>
            <div class="text-base font-semibold text-sky-700">{{ conf.toFixed(2) }}</div>
          </div>
          <div class="rounded-lg bg-indigo-50 px-3 py-2 text-center shadow-sm">
            <div class="text-xs text-indigo-600">已选通道</div>
            <div class="text-base font-semibold text-indigo-700">{{ nvr.channels.length }}</div>
          </div>
        </div>
      </div>
    </div>

    <el-card shadow="never" class="rounded-xl border border-slate-200/70">
      <div class="mb-4 rounded-xl border border-slate-200 bg-slate-50/80 p-4">
        <div class="mb-3 text-sm font-medium text-slate-700">建模参数设置</div>
        <div class="grid grid-cols-1 gap-5 lg:grid-cols-2">
          <div class="rounded-lg border border-slate-200 bg-white p-3">
            <div class="mb-2 flex items-center justify-between">
              <div class="text-sm text-slate-600">抽帧步长</div>
              <span class="text-xs text-slate-400">范围 1 - 120</span>
            </div>
            <el-slider
              v-model="skip"
              :min="1"
              :max="120"
              :step="1"
              show-input
              :show-tooltip="false"
            />
          </div>
          <!--
          <div class="rounded-lg border border-slate-200 bg-white p-3">
            <div class="mb-2 flex items-center justify-between">
              <div class="text-sm text-slate-600">检测灵敏度</div>
              <span class="text-xs text-slate-400">范围 0.05 - 0.90</span>
            </div>
            <el-slider
              v-model="conf"
              :min="0.05"
              :max="0.9"
              :step="0.05"
              show-input
              :show-tooltip="false"
            />
          </div>
          -->
        </div>
      </div>

      <div class="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-4">
        <div class="text-sm text-slate-500">建模方式</div>
        <el-radio-group v-model="modelSubPage" size="large">
          <el-radio-button label="local" value="local">本地视频上传</el-radio-button>
          <el-radio-button label="isapi" value="isapi">NVR 历史回放</el-radio-button>
          <el-radio-button label="rtsp" value="rtsp">实时 RTSP</el-radio-button>
        </el-radio-group>
      </div>

      <div v-if="modelSubPage === 'local'" class="pt-5">
        <div class="mb-4 flex flex-wrap items-center justify-between gap-2">
          <div>
            <div class="text-base font-semibold text-slate-800">本地视频上传</div>
            <div class="mt-1 text-xs text-slate-500">上传本地视频后会立即创建建模任务，并进入视频源队列。切到其他页面不会中断上传。</div>
          </div>
          <el-tag type="info" effect="plain">本地视频上传</el-tag>
        </div>
        <div class="mb-4 rounded-lg border border-slate-200 bg-white p-3">
          <div class="mb-2 text-sm text-slate-600">视频拍摄开始时间（北京时间，可选）</div>
          <el-date-picker
            v-model="capturedAt"
            type="datetime"
            value-format="YYYY-MM-DDTHH:mm:ss"
            format="YYYY-MM-DD HH:mm:ss"
            placeholder="不填则报告只使用视频相对时间"
            clearable
            class="w-full"
          />
          <div class="mt-1 text-xs text-slate-400">
            请按实际拍摄开始时间填写。系统不会用文件创建时间或上传时间猜测。
          </div>
        </div>
        <el-upload
          class="model-upload"
          drag
          multiple
          :http-request="doLocalUpload"
          :show-file-list="false"
          accept=".mp4,.avi,.mov"
        >
          <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
          <div class="el-upload__text">拖拽到此处或 <em>点击上传</em></div>
          <template #tip>
            <div class="mt-2 text-xs text-slate-400">支持 mp4 / avi / mov，可一次选择多个文件，系统会按队列逐个上传。</div>
          </template>
        </el-upload>
        <div v-if="upload.uploading || upload.uploadPercent > 0" class="mt-4 rounded-lg border border-sky-100 bg-sky-50/70 p-3">
          <div class="mb-2 flex items-center justify-between gap-3 text-sm">
            <span class="truncate text-slate-600">
              {{ upload.uploading ? '正在上传' : '上传完成' }}：{{ upload.uploadFileName || '--' }}
              <span v-if="upload.queuedCount > 0" class="ml-2 text-slate-400">队列中 {{ upload.queuedCount }} 个</span>
            </span>
            <div v-if="upload.uploading" class="flex shrink-0 items-center gap-2">
              <el-button size="small" text type="danger" @click="cancelLocalUpload">取消当前</el-button>
              <el-button v-if="upload.queuedCount > 0" size="small" text type="danger" @click="cancelAllLocalUploads">取消全部</el-button>
            </div>
          </div>
          <el-progress :percentage="upload.uploadPercent" :status="upload.uploadPercent === 100 ? 'success' : undefined" />
        </div>
      </div>

      <div v-else-if="modelSubPage === 'isapi'" class="pt-5">
        <div class="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div>
            <div class="flex items-center gap-3 whitespace-nowrap">
              <span class="text-sm text-slate-500">监控厂商</span>
              <el-select v-model="selectedVendor" class="w-36" placeholder="请选择厂商">
                <el-option v-for="item in vendorOptions" :key="item.value" :label="item.label" :value="item.value" />
              </el-select>

            </div>
            <div class="mt-1 text-xs text-slate-500">支持自动识别通道，并按时间段拆分提交为多个任务。</div>
          </div>
          <el-button type="primary" :disabled="!canSubmitIsapi" @click="submitIsapi">提交建模</el-button>
        </div>

        <el-form label-width="110px" class="max-w-5xl">
          <el-row :gutter="16">
            <el-col :span="8">
              <el-form-item label="设备 IP">
                <el-input v-model="nvr.host" placeholder="192.168.1.3:8080" />
              </el-form-item>
            </el-col>
            <el-col :span="8">
              <el-form-item label="用户名">
                <el-input v-model="nvr.username" />
              </el-form-item>
            </el-col>
            <el-col :span="8">
              <el-form-item label="密码" required>
                <el-input v-model="nvr.password" type="password" show-password placeholder="必填" />
              </el-form-item>
            </el-col>
          </el-row>

          <el-row :gutter="16">
            <el-col :span="12">
              <el-form-item label="识别通道">
                <div class="flex w-full items-center gap-3">
                  <el-button :loading="probingChannels" @click="detectChannels">识别设备通道</el-button>
                  <span class="text-sm text-slate-500">填写账号密码后自动探测</span>
                </div>
              </el-form-item>
            </el-col>
            <el-col :span="12">
              <el-form-item label="通道多选" required>
                <el-select
                  v-model="nvr.channels"
                  multiple
                  placeholder="请选择一个或多个通道"
                  class="w-full"
                >
                  <el-option
                    v-for="item in channelOptions"
                    :key="item.id"
                    :label="`${item.name}（通道${item.id}）`"
                    :value="item.id"
                  />
                </el-select>
              </el-form-item>
            </el-col>
          </el-row>

          <el-row :gutter="16">
            <el-col :span="12">
              <el-form-item label="开始时间" required>
                <el-date-picker
                  v-model="nvr.startAt"
                  type="datetime"
                  value-format="YYYY-MM-DD HH:mm"
                  format="YYYY-MM-DD HH:mm"
                  class="w-full"
                />
              </el-form-item>
            </el-col>
            <el-col :span="12">
              <el-form-item label="结束时间" required>
                <el-date-picker
                  v-model="nvr.endAt"
                  type="datetime"
                  value-format="YYYY-MM-DD HH:mm"
                  format="YYYY-MM-DD HH:mm"
                  class="w-full"
                />
              </el-form-item>
            </el-col>
          </el-row>

          <el-form-item label="任务名称预览">
            <div class="w-full rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700">
              <div v-if="autoTaskNames.length" class="flex flex-wrap gap-2">
                <el-tag v-for="name in autoTaskNames" :key="name" type="info" effect="plain" class="max-w-full">
                  <span class="truncate">{{ name }}</span>
                </el-tag>
              </div>
              <div v-else class="text-slate-400">选择通道和时间后自动生成中文任务名</div>
            </div>
          </el-form-item>
        </el-form>
      </div>

      <div v-else class="pt-5">
        <div class="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div>
            <div class="text-base font-semibold text-slate-800">实时 RTSP</div>
            <div class="mt-1 text-xs text-slate-500">
              并行录制 H.264 到 video_archives；检索可边直播边看小图与播放已录片段（再次播放时长会更长）。
            </div>
          </div>
          <el-button
            type="primary"
            :loading="rtspSubmitting"
            :disabled="!canSubmitRtsp"
            @click="submitRtspLive"
          >
            开始实时分析
          </el-button>
        </div>
        <el-form label-width="110px" class="max-w-3xl">
          <el-form-item label="RTSP 地址" required>
            <el-input v-model="rtspLive.url" placeholder="rtsp://192.168.1.3:8554/live" />
          </el-form-item>
          <el-form-item label="任务名称" required>
            <el-input v-model="rtspLive.taskName" />
          </el-form-item>
        </el-form>
      </div>
    </el-card>

    <el-card shadow="never" class="rounded-xl border border-slate-200/70 bg-slate-50/70">
      <template #header>
        <span class="font-semibold text-slate-700">操作说明</span>
      </template>
      <div class="grid gap-2 text-sm text-slate-600 sm:grid-cols-3">
        <p>公共参数仅对当前建模方式提交生效。</p>
        <p>NVR 回放一次可选多个通道，会自动拆分为多个视频源任务。</p>
        <p>实时 RTSP 边录边分析；检索中可播放截至目前录制的 mp4。</p>
        <p>所有任务均可在「视频源列表」查看进度或入库数量。</p>
      </div>
    </el-card>
  </div>
</template>

<style scoped>
.model-upload :deep(.el-upload-dragger) {
  border-radius: 12px;
  border: 2px dashed #d9d9d9;
  background: #fafafa;
  padding: 26px 14px;
  transition: all 0.2s ease;
}

.model-upload :deep(.el-upload-dragger:hover) {
  border-color: #409eff;
  background: #ecf5ff;
}

.model-upload :deep(.el-upload-list__item) {
  border-radius: 8px;
}
</style>
