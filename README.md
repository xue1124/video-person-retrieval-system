# 医保智能稽查分析系统

面向医保稽查场景的监控视频人物分析与检索系统。系统将视频中的人物、出现时间、所在区域和截图信息结构化，支持人物档案、图搜人、文搜人、停留记录与视频回放，帮助稽查人员快速定位目标人物及其活动轨迹。

该项目由本人在实习期间从 0 到 1 完成，主要负责需求落地、系统设计、前后端功能集成、视觉模型接入、数据库设计和部署验证。

## 项目功能

- 支持本地视频、NVR 历史回放和 RTSP 实时流接入。
- 使用 YOLO 检测视频中的人物并生成人物截图（crop）。
- 使用 OSNet 提取人物外观特征，完成人物 ID 聚类和主要的图搜图检索。
- 使用 SigLIP 实现文搜图和语义图搜图。
- 支持图搜人、人物档案、时间段整理、区域停留和视频回放。
- 支持双人同房间共现分析、检索日志和稽查分析报告。

## 系统流程

### 1. 视频分析与建模

```text
本地视频 / NVR 回放
  → FastAPI 创建任务
  → Redis + Celery 异步分析
  → 抽帧、YOLO 检测人物
  → OSNet 提取特征并聚类；SigLIP 提取图像特征
  → MySQL 保存结果，FAISS 建立索引
```

### 2. 人物检索

```text
图片 → OSNet 特征 → OSNet FAISS 索引
文字 / 图片 → SigLIP 特征 → SigLIP FAISS 索引

FAISS 返回 observation_id
  → MySQL 查询人物、时间和视频
  → 展示轨迹与视频回放
```

OSNet 是主要的图搜图方式；SigLIP 支持文搜图，也支持图搜图。两类索引都通过 `observation_id` 关联 MySQL。

## 技术栈

| 模块 | 技术 |
| --- | --- |
| 前端 | Vue 3、TypeScript、Element Plus、Axios、Vite |
| 后端 | Python、FastAPI、Uvicorn、SQLAlchemy |
| 异步任务 | Celery、Redis |
| 数据存储 | MySQL、FAISS |
| 视觉模型 | YOLO、OSNet、SigLIP |
| 视频处理 | OpenCV、FFmpeg/FFprobe |
| 模型部署 | PyTorch、ONNX Runtime、TensorRT、RKNN |

## 项目结构

```text
osnet-siglip/
├── api_server.py              # FastAPI应用入口和视频建模接口
├── tasks.py                   # Celery视频分析任务入口
├── run_api.py                 # 本地API启动入口
├── api/                       # 登录、检索、任务、档案和报告路由
├── services/
│   ├── analysis/              # 人物检测、特征提取、ID聚类和分析流程
│   ├── inference/             # YOLO、OSNet、SigLIP模型加载与推理
│   ├── search/                # 图像/文字检索与结果整理
│   ├── persistence/           # MySQL连接与数据清理
│   ├── tasks/                 # Celery配置、实时流和任务辅助逻辑
│   ├── media/                 # 视频回放、转码和时间处理
│   ├── integrations/          # NVR ISAPI与RTSP适配
│   ├── auth/                  # 登录和权限验证
│   └── reports/               # 分析报告服务
├── frontend/                  # Vue 3前端
├── sql/schema.sql             # 完整MySQL初始化结构
├── models/                    # 本地模型目录，权重默认不提交Git
├── deployment/                # ONNX、TensorRT和RKNN转换资料
├── tools/                     # 用户、数据库、模型和仓库检查工具
└── docs/                      # 运行、代码地图和部署说明
```

第一次阅读代码建议从 [代码地图](docs/CODE_MAP.md) 开始，再依次查看 `api_server.py`、`tasks.py` 和 `services/analysis/tracking/workflow.py`。

## 本地运行

运行环境需要 Python 3.10+、MySQL 8、Redis、FFmpeg/FFprobe 和 Node.js。实际视频分析建议使用 NVIDIA GPU。

1. 安装与配置步骤见 [本地运行说明](docs/SETUP.md)。
2. 模型文件与路径见 [模型清单](docs/MODELS.md)。
3. 模型格式转换见 [部署说明](docs/DEPLOYMENT.md)。

完成数据库和模型配置后，在项目根目录分别启动 API 与 Worker：

```bash
python run_api.py
python -m celery -A tasks.app worker --loglevel=info --pool=threads --concurrency=1
```

启动前端：

```bash
cd frontend
npm ci
npm run dev
```

浏览器访问 `http://localhost:5174`。

## 项目检查

```bash
python tools/check_repository.py
python -m unittest discover -s tests -v

cd frontend
npm run build
```

模型权重、监控视频、人物截图、向量索引和真实环境配置均由 Git 忽略。NVR 接入需要另外部署对应的 ISAPI 代理与 RTSP 中继服务。
