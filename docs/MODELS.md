# 模型文件约定

模型统一放在根目录 `models/`，具体清单见 [models/README.md](../models/README.md)。API查询与Celery建模必须使用同一版本的OSNet和SigLIP编码器，否则新查询向量与旧FAISS索引不在同一特征空间，需要重新建模或重建向量。

## 正式运行路径

| 路径 | 作用 |
| --- | --- |
| `models/pytorch/yolo11m.pt` | 历史视频人物检测的通用回退版本 |
| `models/pytorch/yolov10n.pt` | RTSP实时检测及兼容流程 |
| `models/onnx/osnet_ain_msmt17_dynamic.onnx` + `.data` | OSNet人物特征，输出512维向量 |
| `models/onnx/siglip_vision.onnx` | 图片语义特征 |
| `models/onnx/siglip_text.onnx` + `.data` | 文本语义特征 |
| `models/tokenizer/siglip_v1/` | 与SigLIP文本模型匹配的分词器 |

如果CUDA可用且存在兼容的 `models/tensorrt/yolo11m.engine`，历史分析会优先使用它；OSNet会从三个Engine版本中选择可用版本。CPU环境会回退到PT/ONNX。RTSP链路仍固定使用 `.pt`，用于规避线程中重复加载TensorRT引擎造成的不稳定。

## 部署展示版本

- `models/onnx/` 保存通用推理和转换输入。
- `models/tensorrt/` 保存NVIDIA GPU部署产物。
- `models/rknn/` 保存瑞芯微NPU部署产物。
- `deployment/rknn/convert_osnet.py` 保存OSNet转RKNN的实际辅助脚本。

Engine与GPU架构、CUDA和TensorRT版本有关，目标机器不一致时需要重新构建。RKNN文件也必须与目标芯片和Toolkit版本匹配。当前FastAPI主流程不会直接加载RKNN，它们属于边缘设备部署产物。

模型由 `services/inference/runtime.py` 按功能懒加载，因此只检查接口可不加载全部权重，但完整建模和检索仍需补齐对应模型。替换模型后应使用同一批样本核对尺寸、颜色通道、归一化、输出节点、NMS、特征维度及最终相似度。

本地检查：

```bash
python tools/check_models.py
python tools/check_models.py --all-formats
```

模型文件默认被Git忽略。公开分发前需要确认权重许可证；超过GitHub普通Git单文件限制的模型应使用Git LFS、Release、模型仓库或单独下载地址。
