#!/usr/bin/env python3
"""
OSNet ONNX -> RKNN for RK3588 (SigLip_demo).

Matches services/inference/runtime.py preprocessing:
  BGR crop -> RGB 128x256 -> ImageNet normalize -> NCHW float32

Typical usage (from project root, in rknn_env):

  python deployment/rknn/convert_osnet.py \\
    --onnx models/onnx/osnet_ain_msmt17_static_1x3x256x128.onnx \\
    --rknn models/rknn/osnet_ain_msmt17.rknn \\
    --disable-rules convert_exnorm_to_exnorm_mul_add \\
    --rewrite-relu-to-clip \\
    --image video_crops/test3.mp4_30_0.jpg

Export static ONNX from dynamic first:

  python deployment/rknn/convert_osnet.py \\
    --from-dynamic \\
    --dynamic-onnx models/onnx/osnet_ain_msmt17_dynamic.onnx \\
    --onnx models/onnx/osnet_ain_msmt17_static_1x3x256x128.onnx \\
    --rknn models/rknn/osnet_ain_msmt17.rknn \\
    --disable-rules convert_exnorm_to_exnorm_mul_add \\
    --rewrite-relu-to-clip \\
    --image video_crops/test3.mp4_30_0.jpg
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

OSNET_INPUT_NAME = "input"
OSNET_STATIC_SHAPE = (1, 3, 256, 128)  # NCHW: H=256, W=128
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_osnet_bgr(bgr: np.ndarray) -> np.ndarray:
    """Same preprocessing as the runtime OSNet image encoder."""
    img = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (128, 256))
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)


def export_static_onnx(dynamic_onnx: Path, static_onnx: Path, simplify: bool) -> None:
    import onnx
    from onnxsim import simplify as onnx_simplify

    model = onnx.load(str(dynamic_onnx))
    model_simp, ok = onnx_simplify(
        model,
        overwrite_input_shapes={OSNET_INPUT_NAME: list(OSNET_STATIC_SHAPE)},
    )
    if not ok:
        print("WARN: onnxsim simplify returned ok=False, saving anyway")
    static_onnx.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model_simp, str(static_onnx))
    print(f"static ONNX saved: {static_onnx}")


def rewrite_relu_graph(onnx_path: Path, mode: str, clip_max: float, out_path: Path) -> Path:
    """Optional ONNX surgery before RKNN build."""
    if mode == "none":
        return onnx_path

    import onnx
    from onnx import helper

    model = onnx.load(str(onnx_path))
    new_nodes = []
    replaced = 0
    for node in model.graph.node:
        if node.op_type != "Relu":
            new_nodes.append(node)
            continue
        replaced += 1
        if mode == "clip":
            new_nodes.append(
                helper.make_node(
                    "Clip",
                    inputs=list(node.input),
                    outputs=list(node.output),
                    name=f"{node.name or 'relu'}_clip",
                    min=0.0,
                    max=float(clip_max),
                )
            )
        elif mode == "identity":
            new_nodes.append(
                helper.make_node(
                    "Identity",
                    inputs=list(node.input),
                    outputs=list(node.output),
                    name=f"{node.name or 'relu'}_id",
                )
            )
        else:
            new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    print(f"rewrite_relu={mode}, replaced {replaced} Relu nodes -> {out_path}")
    return out_path


def write_dataset_txt(image_path: Path, dataset_txt: Path) -> None:
    dataset_txt.write_text(str(image_path.resolve()) + "\n", encoding="utf-8")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def run_onnx(onnx_path: Path, inp: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return sess.run(None, {name: inp})[0]


def sim_check(
    ref_onnx: Path,
    rknn,
    image_path: Path,
    cosine_threshold: float,
) -> Tuple[float, int]:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    inp = preprocess_osnet_bgr(bgr)

    onnx_out = run_onnx(ref_onnx, inp).astype(np.float32)
    ret = rknn.init_runtime()
    if ret != 0:
        raise RuntimeError(f"rknn.init_runtime failed: {ret}")

    rknn_out = None
    for fmt in ("nchw", None):
        try:
            kw = {"data_format": fmt} if fmt else {}
            rknn_out = rknn.inference(inputs=[inp], **kw)[0]
            break
        except TypeError:
            rknn_out = rknn.inference(inputs=[inp])[0]
            break

    if rknn_out is None:
        raise RuntimeError("rknn inference returned no output")

    rknn_out = rknn_out.astype(np.float32)
    bad = int(np.isnan(rknn_out).sum() + np.isinf(rknn_out).sum())
    cos = cosine_similarity(onnx_out, rknn_out)
    print(
        f"sim_check: cosine={cos:.6f} bad={bad} "
        f"onnx[min,max]=({onnx_out.min():.4f},{onnx_out.max():.4f}) "
        f"rknn[min,max]=({np.nanmin(rknn_out):.4f},{np.nanmax(rknn_out):.4f})"
    )
    if bad > 0:
        raise RuntimeError(f"RKNN output has {bad} NaN/Inf values")
    if cos < cosine_threshold:
        raise RuntimeError(
            f"cosine {cos:.6f} < threshold {cosine_threshold}; conversion may be inaccurate"
        )
    return cos, bad


def build_rknn(
    onnx_path: Path,
    rknn_path: Path,
    *,
    target_platform: str,
    disable_rules: Sequence[str],
    optimization_level: int,
    quantize: bool,
    dataset_txt: Optional[Path],
) -> object:
    from rknn.api import RKNN

    rknn = RKNN(verbose=True)
    config_kw = dict(
        target_platform=target_platform,
        optimization_level=int(optimization_level),
    )
    if disable_rules:
        config_kw["disable_rules"] = list(disable_rules)
    print("rknn.config:", config_kw)
    rknn.config(**config_kw)

    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        raise RuntimeError(f"load_onnx failed: {ret}")

    build_kw = {}
    if quantize:
        if not dataset_txt or not dataset_txt.is_file():
            raise ValueError("quantize requires --dataset or --image")
        build_kw["do_quantization"] = True
        build_kw["dataset"] = str(dataset_txt)
    else:
        build_kw["do_quantization"] = False

    print("rknn.build:", build_kw)
    ret = rknn.build(**build_kw)
    if ret != 0:
        raise RuntimeError(f"build failed: {ret}")

    rknn_path.parent.mkdir(parents=True, exist_ok=True)
    ret = rknn.export_rknn(str(rknn_path))
    if ret != 0:
        raise RuntimeError(f"export_rknn failed: {ret}")
    print(f"RKNN exported: {rknn_path}")
    return rknn


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert OSNet ONNX to RKNN (RK3588)")
    p.add_argument("--onnx", type=Path, default=ROOT / "models" / "onnx" / "osnet_ain_msmt17_static_1x3x256x128.onnx")
    p.add_argument("--rknn", type=Path, default=ROOT / "models" / "rknn" / "osnet_ain_msmt17.rknn")
    p.add_argument("--from-dynamic", action="store_true", help="export static ONNX from dynamic first")
    p.add_argument(
        "--dynamic-onnx",
        type=Path,
        default=ROOT / "models" / "onnx" / "osnet_ain_msmt17_dynamic.onnx",
    )
    p.add_argument("--no-simplify", action="store_true", help="skip onnxsim when exporting static ONNX")
    p.add_argument(
        "--disable-rules",
        action="append",
        default=[],
        help="RKNN graph optimizer rules to disable (repeatable)",
    )
    p.add_argument(
        "--rewrite-relu-to-clip",
        action="store_true",
        help="replace ONNX Relu with Clip(0, clip-max) before build",
    )
    p.add_argument(
        "--rewrite-relu-with-identity",
        action="store_true",
        help="replace ONNX Relu with Identity before build",
    )
    p.add_argument("--clip-max", type=float, default=6.0)
    p.add_argument("--target-platform", default="rk3588")
    p.add_argument("--optimization-level", type=int, default=1, choices=[0, 1, 2, 3])
    p.add_argument("--quantize", action="store_true", help="INT8 quant (default: FP16/no quant)")
    p.add_argument("--dataset", type=Path, help="dataset.txt for quantization")
    p.add_argument("--image", type=Path, help="calibration / sim-check image")
    p.add_argument(
        "--ref-onnx",
        type=Path,
        help="ONNX for sim-check (default: --onnx before relu rewrite, or final onnx)",
    )
    p.add_argument("--skip-sim-check", action="store_true")
    p.add_argument("--cosine-threshold", type=float, default=0.95)
    p.add_argument("--workdir", type=Path, help="temp dir for rewritten onnx (default: system temp)")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    disable_rules: List[str] = list(args.disable_rules)
    if not disable_rules:
        disable_rules = ["convert_exnorm_to_exnorm_mul_add"]

    if args.rewrite_relu_to_clip and args.rewrite_relu_with_identity:
        print("ERROR: choose only one of --rewrite-relu-to-clip / --rewrite-relu-with-identity")
        return 2

    if args.from_dynamic:
        export_static_onnx(
            args.dynamic_onnx,
            args.onnx,
            simplify=not args.no_simplify,
        )

    if not args.onnx.is_file():
        print(f"ERROR: ONNX not found: {args.onnx}")
        return 1

    relu_mode = "none"
    if args.rewrite_relu_to_clip:
        relu_mode = "clip"
    elif args.rewrite_relu_with_identity:
        relu_mode = "identity"

    workdir_ctx = tempfile.TemporaryDirectory(prefix="osnet_rknn_")
    workdir = Path(args.workdir) if args.workdir else Path(workdir_ctx.name)
    workdir.mkdir(parents=True, exist_ok=True)

    build_onnx = args.onnx
    if relu_mode != "none":
        build_onnx = workdir / f"osnet_relu_{relu_mode}.onnx"
        rewrite_relu_graph(args.onnx, relu_mode, args.clip_max, build_onnx)

    dataset_txt = args.dataset
    if args.quantize and args.image and not dataset_txt:
        dataset_txt = workdir / "dataset.txt"
        write_dataset_txt(args.image, dataset_txt)

    rknn = None
    try:
        rknn = build_rknn(
            build_onnx,
            args.rknn,
            target_platform=args.target_platform,
            disable_rules=disable_rules,
            optimization_level=args.optimization_level,
            quantize=args.quantize,
            dataset_txt=dataset_txt,
        )

        if not args.skip_sim_check:
            ref = args.ref_onnx or args.onnx
            if not ref.is_file():
                print(f"WARN: ref-onnx missing ({ref}), skip sim-check")
            elif not args.image or not args.image.is_file():
                print("WARN: --image not set, skip sim-check")
            else:
                sim_check(ref, rknn, args.image, args.cosine_threshold)
    finally:
        if rknn is not None:
            rknn.release()
        if args.workdir is None:
            workdir_ctx.cleanup()

    print("OK:", args.rknn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
