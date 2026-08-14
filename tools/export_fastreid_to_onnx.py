#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a FastReID checkpoint to fixed-shape ONNX.")
    parser.add_argument("--fastreid-root", required=True, help="Path to a FastReID source checkout.")
    parser.add_argument("--config-file", required=True, help="FastReID config path.")
    parser.add_argument("--weights", required=True, help="FastReID .pth checkpoint.")
    parser.add_argument("--output", required=True, help="Output ONNX path.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=0, help="0 infers from heads.classifier.weight.")
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()

    fastreid_root = Path(args.fastreid_root).resolve()
    sys.path.insert(0, str(fastreid_root))

    import torch
    from torch import nn

    from fastreid.config import get_cfg
    from fastreid.modeling.meta_arch import build_model
    from fastreid.utils.checkpoint import Checkpointer

    cfg = get_cfg()
    cfg.merge_from_file(str(Path(args.config_file)))
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.WEIGHTS = str(Path(args.weights))
    cfg.INPUT.SIZE_TEST = [int(args.height), int(args.width)]
    if cfg.MODEL.HEADS.POOL_LAYER == "fastavgpool":
        cfg.MODEL.HEADS.POOL_LAYER = "avgpool"
    num_classes = int(args.num_classes) if args.num_classes > 0 else _infer_num_classes(torch, args.weights)
    cfg.MODEL.HEADS.NUM_CLASSES = int(num_classes)
    cfg.freeze()

    model = build_model(cfg)
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()
    wrapper = _make_embedding_wrapper(model).eval()

    dummy = torch.randn(int(args.batch), 3, int(args.height), int(args.width), dtype=torch.float32)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        sample = wrapper(dummy)
    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["embedding"],
        opset_version=int(args.opset),
        do_constant_folding=True,
    )
    print(
        "exported",
        f"classes={num_classes}",
        f"input={[int(args.batch), 3, int(args.height), int(args.width)]}",
        f"embedding={list(sample.shape)}",
        f"output={out_path}",
    )
    return 0


def _make_embedding_wrapper(model: Any):
    from torch import nn

    class Wrapper(nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.backbone = inner.backbone
            self.heads = inner.heads
            self.register_buffer("pixel_mean", inner.pixel_mean.detach().clone())
            self.register_buffer("pixel_std", inner.pixel_std.detach().clone())

        def forward(self, images):
            images = (images - self.pixel_mean) / self.pixel_std
            return self.heads(self.backbone(images))

    return Wrapper(model)


def _infer_num_classes(torch: Any, weights: str) -> int:
    checkpoint = torch.load(weights, map_location="cpu")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    for key in ("heads.classifier.weight", "module.heads.classifier.weight"):
        value = state.get(key)
        if value is not None:
            return int(value.shape[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
