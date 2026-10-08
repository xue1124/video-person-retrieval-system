# 检测、身份关联与分析算法

这里仅保留当前正式建模流程。主线是 `detect_reid_video.py → reid_match.py → persist_results.py`。

- `detect_reid_video.py`：当前无跟踪历史视频分析，包含检测、特征提取、HNSW 候选查询与身份聚类。
- `reid_match.py`：共享身份相似度判断。
- `persist_results.py`：将分析输出写入数据库。
- `match_cross_video.py`：跨视频人物关联。
- `generate_snapshots.py`、`compute_stays.py`：生成人物代表图和区域停留记录。

OSNet 的 ONNX Runtime / TensorRT 编码器已经移到上一级 `osnet_encoder.py`，供正式分析流程复用。

实验输出、测试视频、轨迹 JSON、向量 NPZ 均不随源码发布。没有保留可用于宣称准确率提升的定量评测报告。

正式启动配置见根目录运行说明。评估模型效果时应使用自有或获得授权的素材，并固定输入、参数和统计口径。
