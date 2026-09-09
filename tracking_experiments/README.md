# 检测、身份关联与历史实验代码

目录名称沿用开发阶段。当前生产流程仍复用这里的多个模块，不应整体作为废弃文件删除。

- `detect_reid_video.py`：当前无跟踪历史视频分析，包含检测、特征提取、HNSW 候选查询与身份聚类。
- `reid_match.py`：共享身份相似度判断。
- `track_video.py`：旧版 BoT-SORT 实验入口，同时定义当前主流程复用的 OSNet 编码器与取消异常。
- `import_to_v3.py`：将分析输出导入数据库。
- `match_cross_video.py`：跨视频人物关联。
- `generate_snapshots.py`、`compute_stays.py`、`annotate_rooms.py`：代表图、区域停留与区域标注工具。
- `render_global_video.py`：历史可视化辅助工具。

实验输出、测试视频、轨迹 JSON、向量 NPZ 均不随源码发布。没有保留可用于宣称准确率提升的定量评测报告。

如需复现实验，应使用自有或获得授权的素材，并为新旧方案保持一致的输入、参数和统计口径。旧实验入口直接运行时可能需要显式设置环境变量；正式启动配置见根目录运行说明。
