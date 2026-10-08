# 模型目录

模型按部署格式分开保存。代码通过 `services/config.py` 统一取得路径；可用 `MODEL_ROOT` 或各模型环境变量覆盖。

```text
models/
├── pytorch/        # 原始 PyTorch/Ultralytics 权重
├── onnx/           # 跨平台中间格式，ONNX Runtime 可直接加载
├── tensorrt/       # NVIDIA GPU 的 TensorRT Engine
├── rknn/           # 瑞芯微 NPU 的 RKNN 产物
├── tokenizer/      # SigLIP 文本分词器
└── cache/          # TensorRT 运行缓存，不提交
```

## 本地保留的版本

| 格式 | 文件 | 用途 |
| --- | --- | --- |
| PT | `yolo11m.pt` | 历史视频人物检测主模型 |
| PT | `yolov10n.pt` | 实时流及兼容链路检测模型 |
| ONNX | `yolo11m.onnx`、`yolov10n.onnx` | 通用部署与转换来源 |
| ONNX | `osnet_ain_msmt17_dynamic.onnx` + `.data` | 动态批量 OSNet 推理 |
| ONNX | `osnet_ain_msmt17_static_1x3x256x128.onnx` | RKNN/静态部署转换来源 |
| ONNX | `siglip_vision.onnx`、`siglip_text.onnx` + `.data` | 图像与文字向量编码 |
| ONNX | `siglip_vision_static_1x3x256x256.onnx` | SigLIP 视觉静态部署来源 |
| Engine | `yolo11m.engine`、`yolov10n.engine` | NVIDIA TensorRT 检测 |
| Engine | 三个 `osnet_*.engine` | OSNet 动态 batch 1–8、1–16 和静态 batch 1 版本 |
| RKNN | `yolov10n.rknn` | 瑞芯微 NPU 目标检测 |
| RKNN | `osnet_ain_msmt17.rknn` | 瑞芯微 NPU 人物特征提取 |
| RKNN | `siglip_vision.rknn` | 瑞芯微 NPU 图像编码 |

`check*.onnx`、`*_tmp.onnx*`、`*.bak` 属于转换中间文件或失败备份，不进入最终目录。`siglip_v1/model.safetensors` 是完整 PyTorch 模型，本系统使用 ONNX 推理，文本侧只保留 tokenizer 文件。

## GitHub说明

这些模型本地合计约 2.5 GB，已被 `.gitignore` 排除。普通 GitHub 仓库不能直接提交超过 100 MiB 的文件。公开仓库保留模型清单和转换说明，模型需通过 Git LFS、Release、模型仓库或单独下载地址分发，并先确认权重许可证。

检查本机模型是否齐全：

```bash
python tools/check_models.py
python tools/check_models.py --all-formats
```
