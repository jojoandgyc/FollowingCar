#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch static ONNX batch dimension.")
    parser.add_argument("input", help="Input ONNX path.")
    parser.add_argument("output", help="Output ONNX path.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--from-batch", type=int, default=0, help="Only replace this batch value; 0 replaces any positive dim.")
    args = parser.parse_args()

    import onnx

    model = onnx.load(args.input)
    changed = 0
    for value_info in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        dims = value_info.type.tensor_type.shape.dim
        if not dims:
            continue
        dim = dims[0]
        dim_value = int(dim.dim_value or 0)
        if dim_value <= 0:
            continue
        if args.from_batch and dim_value != int(args.from_batch):
            continue
        dim.dim_value = int(args.batch)
        changed += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, out_path)
    print(f"patched_batch={args.batch} changed_value_infos={changed} output={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
