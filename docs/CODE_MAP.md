# 代码地图

先记住两个入口：`api_server.py` 接收前端请求，`tasks.py` 定义 Celery 视频任务。真正从 Redis 队列领取任务的是单独启动的 Celery Worker；它按照 `tasks.py` 中的定义执行服务。

## 一条视频如何完成建模

```text
前端上传视频
    ↓
api_server.py：保存文件、创建 pending 记录、发送 Celery 任务
    ↓
tasks.py：任务变为 processing，准备本地/ISAPI视频来源
    ↓
services/analysis/tracking/workflow.py：编排整条人物建模流程
    ↓
services/analysis/pipeline/detect_reid_video.py：YOLO检测、裁剪人物
    ↓
services/analysis/pipeline/reid_match.py：OSNet特征匹配和人物ID合并
    ↓
services/analysis/pipeline/persist_results.py：写入MySQL
    ↓
services/analysis/tracking/search_index.py：更新FAISS索引
```

## 一次检索如何返回结果

```text
前端提交图片或文字
    ↓
api/search.py /search
    ↓
services/search/service.py：提取OSNet或SigLIP查询向量
    ↓
services/analysis/tracking/search_index.py：FAISS Top-K和阈值过滤
    ↓
observation_id
    ↓
services/persistence/database.py：查询人物、视频、时间和crop路径
    ↓
按人物和时间段整理后返回前端
```

## 目录职责

| 目录 | 你需要记住的作用 |
| --- | --- |
| `api/` | 独立的FastAPI路由；主接口仍由根目录`api_server.py`统一注册 |
| `services/analysis/pipeline/` | 真正执行检测、特征提取、聚类、跨视频关联和停留计算 |
| `services/analysis/tracking/` | 把算法接进正式任务，管理模型、入库和FAISS更新 |
| `services/inference/` | 懒加载YOLO、OSNet和SigLIP，并提供统一的推理函数 |
| `services/search/` | 图搜图、文搜图以及结果聚合 |
| `services/persistence/` | MySQL连接、数据操作和删除清理 |
| `services/tasks/` | Celery应用配置、Worker公共逻辑、截图批处理和RTSP实时流 |
| `services/media/` | 视频回放、转码和时间处理 |
| `services/integrations/` | ISAPI代理和RTSP中继适配 |
| `services/reports/` | 分析报告和可选Dify调用 |
| `models/` | PT、ONNX、TensorRT、RKNN及tokenizer，本地存在但默认不提交Git |
| `deployment/` | 模型转换与Linux常驻服务资料 |
| `sql/schema.sql` | 一次性创建正式系统所需的完整数据库结构 |
| `tools/` | 建用户、检查源码和检查模型的独立工具 |

历史视频的正式检索数据在 `observations + observation_embeddings + FAISS`。RTSP 实时流目前仍把即时截图和向量写入 `gallery_meta`，主要用于实时任务记录和兼容展示，尚未并入同一套 FAISS 人物档案流程；面试时不要把两条链路说成完全一致。

## 面试优先读这七个文件

1. `api_server.py`：看应用组装、`/analyze` 和视频源接口。
2. `tasks.py`：看 `process_video_task` 如何调用建模流程。
3. `services/tasks/celery_app.py`：看 Worker 使用哪个 Redis broker/backend。
4. `services/analysis/tracking/workflow.py`：看整个视频分析如何串起来。
5. `services/analysis/pipeline/detect_reid_video.py`：看YOLO、crop和批量特征提取。
6. `services/analysis/pipeline/reid_match.py`：看相似度阈值和人物合并。
7. `services/analysis/tracking/search_index.py`：看FAISS索引如何存取`observation_id`。
8. `api/search.py` 与 `services/search/service.py`：看检索接口如何把查询向量变成前端结果。

部署转换、数据库维护和Dify细节可以后看。秋招面试更重要的是能顺着上面两条主流程定位问题，而不是背下每一行代码。
