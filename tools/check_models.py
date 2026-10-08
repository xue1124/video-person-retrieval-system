"""检查运行模型或完整部署展示模型是否齐全。"""
from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

RUNTIME_FILES = (
    "pytorch/yolo11m.pt",
    "pytorch/yolov10n.pt",
    "onnx/osnet_ain_msmt17_dynamic.onnx",
    "onnx/osnet_ain_msmt17_dynamic.onnx.data",
    "onnx/siglip_vision.onnx",
    "onnx/siglip_text.onnx",
    "onnx/siglip_text.onnx.data",
    "tokenizer/siglip_v1/tokenizer_config.json",
    "tokenizer/siglip_v1/tokenizer.json",
    "tokenizer/siglip_v1/spiece.model",
)

DEPLOYMENT_FILES = RUNTIME_FILES + (
    "onnx/yolo11m.onnx",
    "onnx/yolov10n.onnx",
    "onnx/osnet_ain_msmt17_static_1x3x256x128.onnx",
    "onnx/siglip_vision_static_1x3x256x256.onnx",
    "tensorrt/yolo11m.engine",
    "tensorrt/yolov10n.engine",
    "tensorrt/osnet_ain_msmt17_dyn_b1-8_fp16.engine",
    "tensorrt/osnet_ain_msmt17_dyn_b1-16_fp16.engine",
    "tensorrt/osnet_ain_msmt17_static_1x3x256x128_fp16.engine",
    "rknn/yolov10n.rknn",
    "rknn/osnet_ain_msmt17.rknn",
    "rknn/siglip_vision.rknn",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-formats", action="store_true", help="同时检查 Engine 和 RKNN 展示产物")
    args = parser.parse_args()
    expected = DEPLOYMENT_FILES if args.all_formats else RUNTIME_FILES
    missing = []
    total = 0
    for relative in expected:
        path = MODELS / relative
        if path.is_file():
            total += path.stat().st_size
            print(f"OK       {relative}")
        else:
            missing.append(relative)
            print(f"MISSING  {relative}")
    print(f"\nchecked={len(expected)} missing={len(missing)} size={total / 1024**3:.2f} GiB")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
