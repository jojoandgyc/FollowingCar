#!/usr/bin/env python3
"""Read-only SAME-IMAGE geometry regression, NOT a temporal/real-motion replay.

Each saved depth is seeded once then reused with synthetic 30Hz timestamps and
zero camera motion. This isolates ROI-induced false area/centroid changes; it
cannot measure real tracking frequency, identity accuracy or control safety.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from car_control_modular.depth_track_shadow import CameraPose, DepthTrackShadow, Intrinsics


def check(depth, metadata, hfov):
    h, w = depth.shape
    rw, rh = metadata['rgb_size']
    fx = w/(2*math.tan(math.radians(hfov)/2))
    tracker = DepthTrackShadow(Intrinsics(w,h,fx,fx,w/2,h/2))
    x1,y1,x2,y2 = metadata['anchor_bbox_rgb']
    bw,bh = x2-x1,y2-y1
    roi = ((x1+.32*bw)*w/rw,(y1+.22*bh)*h/rh,
           (x1+.68*bw)*w/rw,(y1+.68*bh)*h/rh)
    first = tracker.seed(depth,timestamp=10.,pose=CameraPose(10.,0,0,0),uid=1,
                         visual_timestamp=10.,torso_bbox=roi,
                         trusted_distance_m=metadata['baseline_distance_m'],
                         now=10.,identity_confirmed=True)
    records=[]
    for i in range(1,11):
        stamp=10+i/30
        start=time.perf_counter()
        result=tracker.update(depth,timestamp=stamp,pose=CameraPose(stamp,0,0,0),now=stamp,active_uid=1)
        records.append(dict(status=result.status,elapsed_ms=(time.perf_counter()-start)*1000,
                            displacement_from_seed_m=None if result.camera_xyz is None or first.camera_xyz is None
                                else float(np.linalg.norm(np.array(result.camera_xyz)-first.camera_xyz)),
                            diagnostics=result.diagnostics))
        if result.status!='tracked':break
    return dict(capture_frame_id=metadata['capture_frame_id'],seed_status=first.status,
                seed_diagnostics=first.diagnostics,updates=records,
                passed=first.status=='seeded' and len(records)==10
                    and all(r['status']=='tracked' and r['displacement_from_seed_m']<=.01 for r in records),
                motor_authority_created=False)


def report(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    results=[]
    for file in sorted(directory.glob('sample_*.npz')):
        with np.load(file,allow_pickle=False) as data:
            row=check(data['depth_mm'],json.loads(str(data['metadata'])),manifest['hfov_deg'])
        results.append(dict(row,file=file.name))
    return dict(mode='same_image_synthetic_time_zero_motion',not_a_temporal_replay=True,
                total=len(results),passed=sum(r['passed'] for r in results),results=results)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',help='depth_track_shadow snapshot directory')
    print(json.dumps(report(p.parse_args().directory),ensure_ascii=False,indent=2,allow_nan=False))
