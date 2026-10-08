# 模型部署路线

本项目保留四种模型格式，用来展示同一模型如何适配不同硬件：

```text
PyTorch .pt
     ↓ 导出
ONNX .onnx
     ├── NVIDIA GPU → TensorRT .engine
     └── 瑞芯微 NPU → RKNN .rknn
```

- `.pt`：训练框架权重，兼容性高，部署开销相对大。
- `.onnx`：通用计算图，作为跨框架推理和硬件转换的中间格式。
- `.engine`：TensorRT 针对特定 NVIDIA GPU、CUDA 和 TensorRT 环境优化后的产物。
- `.rknn`：通过 RKNN Toolkit 转换，供 RK3588 等瑞芯微 NPU 运行。

转换后需要使用同一批样本对比预处理、输出维度、检测框、余弦相似度和最终业务结果，并记录延迟、吞吐量、显存/内存占用。Engine 不应假定可以跨显卡或跨 TensorRT 版本直接复用。

现有 `rknn/convert_osnet.py` 覆盖 OSNet 动态 ONNX 转静态 ONNX、图优化、RKNN 构建和样本验证。YOLO 可通过 Ultralytics 导出 ONNX/Engine；其命令和版本需要在目标环境重新验证。现有 RKNN 文件是部署产物，不能据此声称所有模型的完整转换脚本都已经保留。

YOLO导出示例：

```bash
python deployment/export_yolo.py --format onnx --device cpu --dynamic
python deployment/export_yolo.py --format engine --device 0 --half
```

在NVIDIA部署环境启用历史分析Engine时，可设置：

```text
TRACKING_V3_YOLO_MODEL=models/tensorrt/yolo11m.engine
TRACKING_V3_OSNET_ENGINE=models/tensorrt/osnet_ain_msmt17_dyn_b1-8_fp16.engine
TRACKING_V3_OSNET_NATIVE_TRT=1
```

面试说明见 [部署准备](../docs/DEPLOYMENT.md)。
