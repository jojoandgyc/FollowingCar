#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from collections import OrderedDict
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a Torchreid OSNet checkpoint to fixed-shape ONNX.")
    parser.add_argument("--arch", default="osnet_x0_5", help="Torchreid OSNet arch, e.g. osnet_x0_5.")
    parser.add_argument("--weights", required=True, help="Input .pth checkpoint.")
    parser.add_argument("--osnet-source", required=True, help="Path to Torchreid torchreid/models/osnet.py.")
    parser.add_argument("--output", required=True, help="Output ONNX path.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=0, help="0 infers from classifier.bias in the checkpoint.")
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()

    import torch

    arch = _normalize_arch(args.arch)
    osnet_module = _load_module(Path(args.osnet_source))
    if not hasattr(osnet_module, arch):
        raise SystemExit(f"unsupported OSNet arch in {args.osnet_source}: {arch}")

    state_dict = _load_state_dict(torch, args.weights)
    num_classes = int(args.num_classes) if args.num_classes > 0 else _infer_num_classes(state_dict)
    model = getattr(osnet_module, arch)(num_classes=num_classes, pretrained=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"missing_keys={len(missing)} first={missing[:8]}")
    if unexpected:
        print(f"unexpected_keys={len(unexpected)} first={unexpected[:8]}")
    model.eval()

    dummy = torch.randn(int(args.batch), 3, int(args.height), int(args.width), dtype=torch.float32)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["embedding"],
        opset_version=int(args.opset),
        do_constant_folding=True,
    )
    print(
        "exported",
        f"arch={arch}",
        f"classes={num_classes}",
        f"input={[int(args.batch), 3, int(args.height), int(args.width)]}",
        f"output={out_path}",
    )
    return 0


def _normalize_arch(raw: str) -> str:
    arch = raw.strip().lower()
    if arch in {"x0_50", "x0_5", "osnet_x0_50"}:
        return "osnet_x0_5"
    if arch in {"x0_25", "osnet_x0_25"}:
        return "osnet_x0_25"
    if arch in {"x0_75", "osnet_x0_75"}:
        return "osnet_x0_75"
    if arch in {"x1_0", "osnet_x1_0"}:
        return "osnet_x1_0"
    return arch


def _load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("torchreid_osnet", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load OSNet source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_state_dict(torch: Any, path: str) -> OrderedDict:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    cleaned = OrderedDict()
    for key, value in state.items():
        key = str(key)
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    return cleaned


def _infer_num_classes(state_dict: OrderedDict) -> int:
    bias = state_dict.get("classifier.bias")
    if bias is None:
        weight = state_dict.get("classifier.weight")
        if weight is None:
            return 1000
        return int(weight.shape[0])
    return int(bias.shape[0])


if __name__ == "__main__":
    raise SystemExit(main())
