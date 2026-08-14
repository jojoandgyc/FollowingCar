#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def _parse_triplet(raw: str, default):
    if not raw:
        return default
    vals = [float(x.strip()) for x in raw.split(",")]
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers")
    return vals


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert ONNX to RKNN with RKNN-Toolkit2.")
    parser.add_argument("onnx", help="Input ONNX model path.")
    parser.add_argument("output", help="Output .rknn path.")
    parser.add_argument("--target", default="rk3588")
    parser.add_argument("--dataset", default="", help="Dataset txt for INT8 quantization.")
    parser.add_argument("--dtype", choices=("fp", "i8", "u8"), default="fp")
    parser.add_argument("--mean-values", default="0,0,0")
    parser.add_argument("--std-values", default="255,255,255")
    parser.add_argument("--channel-mean-value", default="", help="Legacy RKNN config string if needed.")
    args = parser.parse_args()

    from rknn.api import RKNN

    onnx_path = Path(args.onnx)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rknn = RKNN(verbose=True)
    config_kwargs = {"target_platform": args.target}
    if args.channel_mean_value:
        config_kwargs["channel_mean_value"] = args.channel_mean_value
    else:
        config_kwargs["mean_values"] = [_parse_triplet(args.mean_values, [0.0, 0.0, 0.0])]
        config_kwargs["std_values"] = [_parse_triplet(args.std_values, [255.0, 255.0, 255.0])]

    print("Config:", config_kwargs)
    ret = rknn.config(**config_kwargs)
    if ret != 0:
        raise SystemExit(f"rknn.config failed: {ret}")

    print(f"Loading ONNX: {onnx_path}")
    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        raise SystemExit(f"rknn.load_onnx failed: {ret}")

    quant = args.dtype in {"i8", "u8"}
    if quant and not args.dataset:
        raise SystemExit("--dataset is required for INT8/UINT8 quantization")
    print(f"Building RKNN: quant={quant} dataset={args.dataset or '(none)'}")
    ret = rknn.build(do_quantization=quant, dataset=args.dataset or None)
    if ret != 0:
        raise SystemExit(f"rknn.build failed: {ret}")

    print(f"Exporting RKNN: {out_path}")
    ret = rknn.export_rknn(str(out_path))
    if ret != 0:
        raise SystemExit(f"rknn.export_rknn failed: {ret}")
    rknn.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
