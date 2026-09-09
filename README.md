# 监控视频人物检索系统

面向历史录像检索与稽查辅助的全栈应用：将视频中的人物检测结果、外观特征和出现时间结构化，支持人物截图检索、文字描述检索及结果回放。

这是实习项目的源码整理版。保留前后端、异步分析、模型接入、数据库和检索主线；不包含业务录像、人物图片、真实配置或预训练权重。模型采用已有预训练模型，项目重点是应用集成与工程实现。

## 功能与边界

- 本地视频上传、后台分析和进度查询。
- YOLO 人物检测；OSNet 外观特征聚类；SigLIP 图文语义检索。
- MySQL 保存业务记录与向量；两个独立 FAISS 索引分别服务 OSNet 和 SigLIP。
- 人物出现记录、视频回放、区域停留、多目标共现、检索历史。
- 保留 NVR/ISAPI、RTSP 接入代码，但代理和中继服务不在本仓库中，需自行部署。
- 保留可选 Dify 报告功能；未提供已发布工作流或密钥，默认不配置。

目前应先复现“本地视频上传 → 分析 → 人物截图检索”。历史视频主流程采用直接检测与 OSNet 聚类，未运行 BoT-SORT；实时流有独立处理代码，不能假定与历史视频完全相同。

## 技术结构

```text
Vue 3 / TypeScript / Element Plus
                  ↓ HTTP
              FastAPI
          ↙                 ↘
Celery + Redis             检索服务
      ↓                       ↓
YOLO → crops → OSNet/SigLIP  查询向量 → FAISS → observation_id
      ↓                                          ↓
OSNet 聚类 → MySQL → FAISS                     MySQL → 展示结果
```

人物聚类使用 HNSW 寻找候选，再由匹配规则决定身份归属。普通图片检索使用 `IndexIDMap2(IndexFlatIP)`，按阈值过滤后查询业务信息，再按视频与时间整理展示结果。文字检索还包含关键词处理与重排，不应将所有检索分支描述为同一套简单 Top-K。

## 从哪里读代码

| 入口 | 作用 |
| --- | --- |
| `api_server.py` | `/analyze` 上传入队、`/search` 检索、状态及媒体接口 |
| `tasks.py` | Celery 任务、视频来源准备、状态更新、查询特征提取 |
| `tracking_v3/sidecar.py` | 组织分析、入库、跨视频关联及索引更新 |
| `tracking_experiments/detect_reid_video.py` | 抽帧、检测、特征提取和直接聚类 |
| `tracking_experiments/reid_match.py` | OSNet 身份匹配规则 |
| `tracking_v3/search_index.py` | FAISS 建立、增量更新、查询和重建 |
| `search_service.py` | 图片/文字检索与结果整理 |
| `db.py`、`migrations/` | 数据库连接及表结构 |
| `frontend/src/views/` | 上传、检索、回放和报告页面 |

`tracking_experiments/` 名称沿用早期实验阶段，但当前主流程仍引用其中的编码器、聚类与导入代码，不能整体删除。旧版 `track_video.py` 也包含生产流程复用的编码器。

## 安装与启动

详细步骤见 [运行说明](docs/SETUP.md)，模型文件要求见 [模型清单](docs/MODELS.md)。

需要 Python 3.10+、MySQL 8、Redis、FFmpeg/FFprobe、Node.js（满足 `frontend/package-lock.json` 中 Vite 的引擎要求），以及匹配的模型文件。NVIDIA GPU 建议用于实际视频处理；CUDA、PyTorch、ONNX Runtime、TensorRT 版本必须配套。

基础依赖与硬件依赖分开安装。`requirements/original-environment.txt` 是原环境快照，不代表在新机器上一条命令即可安装，尤其不应直接混装其中的 CUDA 12 PyTorch 与 CUDA 13 TensorRT 包。

配置并初始化后，在仓库根目录分别启动：

```bash
python -m uvicorn api_server:app --host 127.0.0.1 --port 8002
python -m celery -A tasks.app worker --loglevel=info --pool=threads --concurrency=1
```

前端：

```bash
cd frontend
npm ci
npm run dev
```

打开 `http://localhost:5174`。前端 `/api` 代理到本机 `8002` 端口。以上为本地开发启动方式；不是直接面向公网的部署配置。

## 验证与现有限制

源码整理检查可运行：

```bash
python scripts/check_repository.py
```

整理记录和验证范围见 [整理说明](docs/PREPARATION.md)。尚未在全新环境完成模型、数据库、NVR 和完整上传检索的端到端复现，不提供未经测量的准确率或性能提升数字。

已知边界包括同名视频替换行为、Celery 与数据库双状态同步、实时流与历史检索数据链路差异，以及对模型导出接口的依赖。这些保留为明确的后续改进项，不将源码整理描述为全面生产化改造。

公开前需确认实习代码的公开授权，并遵守依赖与预训练权重的许可证。本整理版未代替权利人选择开源许可证，也不包含可再分发的模型授权证明。
