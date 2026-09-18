#!/usr/bin/env python3
"""Explicit-manifest offline depth tracking. Stdout JSONL, never motor control.

See docs/depth_independent_tracking.md. Existing sparse diagnostic snapshots
cannot be represented as continuous frames or silently assigned zero ego-motion.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from car_control_modular.depth_track_shadow import CameraPose, DepthTrackShadow, Intrinsics


def replay(manifest, root):
    if manifest.get("schema_version") != 1:
        raise ValueError("schema_version=1 required")
    if manifest.get("depth_geometry") != "registered_rectified_orientation_verified":
        raise ValueError("Confirm registered depth grid, rectification and orientation first")
    tracker = DepthTrackShadow(Intrinsics(**manifest["intrinsics"]))
    for row in manifest["frames"]:
        # allow_pickle=False: external recordings never deserialize Python objects.
        with np.load(root / row["depth_file"], allow_pickle=False) as data:
            depth = data["depth_mm"]
        pose = CameraPose(**row["pose"])
        previous_timestamp = tracker.last_ts
        started = time.perf_counter()
        args = dict(timestamp=row["timestamp"], pose=pose, now=row["observed_at"])
        if "anchor" in row:
            anchor = row["anchor"]
            if row.get("searching", False) or row["active_uid"] != anchor["uid"]:
                tracker.reset()
                raise ValueError("visual seed cannot override search or a different active UID")
            result = tracker.seed(depth, **args, **anchor)
        else:
            result = tracker.update(depth, **args, active_uid=row["active_uid"],
                                    searching=row.get("searching", False))
        yield dict(asdict(result), frame_id=row["frame_id"],
                   processing_ms=(time.perf_counter()-started)*1000,
                   depth_age_ms=(row["observed_at"]-row["timestamp"])*1000,
                   visual_age_ms=(None if result.visual_timestamp is None else
                                  (row["observed_at"]-result.visual_timestamp)*1000),
                   physical_gap_ms=(None if previous_timestamp is None else
                                    (row["timestamp"]-previous_timestamp)*1000),
                   identity_update_allowed=False, motor_authority_created=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    counts = Counter()
    for result in replay(manifest, args.manifest.resolve().parent):
        counts[result["status"]] += 1
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    print(json.dumps(dict(mode="offline_shadow_only", statuses=counts)), file=sys.stderr)


if __name__ == "__main__":
    main()
