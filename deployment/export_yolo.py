"""使用 Ultralytics 将 YOLO PT 导出为 ONNX 或 TensorRT Engine。"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "pytorch" / "yolo11m.pt")
    parser.add_argument("--format", choices=("onnx", "engine"), required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="Engine通常使用GPU编号0；ONNX也可填cpu")
    parser.add_argument("--half", action="store_true", help="使用FP16导出，需目标格式和硬件支持")
    parser.add_argument("--dynamic", action="store_true", help="保留动态batch/输入能力")
    args = parser.parse_args()

    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    model = YOLO(str(args.weights))
    result = Path(
        model.export(
            format=args.format,
            imgsz=args.imgsz,
            device=args.device,
            half=args.half,
            dynamic=args.dynamic,
        )
    )
    destination = ROOT / "models" / ("onnx" if args.format == "onnx" else "tensorrt") / result.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if result.resolve() != destination.resolve():
        shutil.copy2(result, destination)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
