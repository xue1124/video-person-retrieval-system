# 模型文件约定

本仓库不包含权重，也不自动承诺任意同名模型可直接替换。需使用与预处理、输出维度、tokenizer 一致的导出模型；API 查询与 worker 建模必须采用相同版本。

| 文件/目录（默认位于仓库根目录） | 用途 |
| --- | --- |
| `yolo11m.pt` 或兼容的 `yolo11m.engine` | 历史视频人物检测 |
| `yolov10n.pt` 或 `yolov10n.engine` | `tasks.get_resources()` 中查询/实时相关资源加载依赖 |
| `osnet_ain_msmt17_dynamic.onnx` | OSNet 编码，代码期望行人 crop 输入，512 维特征 |
| `osnet_ain_msmt17_dynamic.onnx.data` | 如 ONNX 使用外置权重，必须一并提供且文件名匹配 |
| `siglip_vision.onnx` | 图片编码，当前预处理固定 256×256、RGB、[-1,1] |
| `siglip_text.onnx` 及其外置 `.data` | 文字编码，必须与图片编码器属于同一模型空间 |
| `siglip_v1/` | 与文字模型匹配的 tokenizer 配置及词表文件 |

当前 `tasks.get_resources()` 会联合加载多个模型，因此仅测试 OSNet 图片检索时也可能需要 SigLIP 和 YOLO 文件。它尚未拆成按查询类型独立懒加载。

OSNet 图片预处理以 `track_video.py` 和 `tasks.py` 中的实际实现为准。SigLIP 的导出输出名存在适配逻辑，不能仅按文件名判断模型兼容。替换编码器后需重算已有向量，不能混用旧索引。

## TensorRT / RKNN

TensorRT engine 与构建时硬件和软件环境有关，建议在目标环境重新构建和验证。基础配置优先使用 YOLO `.pt` 与 OSNet ONNX，未强制启用原生 TensorRT。

RKNN 转换辅助代码保留在 `rknn_export/`，它是部署实验资料，不表示现有 FastAPI 主流程已经支持直接加载任意 `.rknn` 文件。

原始权重来源、准确版本和许可证需要由项目作者进一步补全。在确认分发权限前，不将模型放入 Git 或 Release。本整理过程没有上传或下载权重。
