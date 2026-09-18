#!/usr/bin/env python3
"""Read-only RPM/displacement audit. Never writes calibration or drives motors.

Independent measured travel must cover the EXACT requested CAP interval.
Depth closure is only a cross-check when the same target/surface is stationary.
"""
import argparse
import bisect
import csv
import json
import math
from pathlib import Path
import re


def integrate_revolutions(samples, start, end, max_gap=.15):
    """Trapezoidal integration of measured RPM, no extrapolation across gaps."""
    if not math.isfinite(start+end) or end <= start:
        raise ValueError('Invalid time interval')
    times=sorted(samples)
    if len(times)<2 or times[0]>start or times[-1]<end:
        raise ValueError('Encoder samples do not bracket requested interval')
    points=[start]+[t for t in times if start<t<end]+[end]
    def value(t):
        if t in samples:return samples[t]
        j=bisect.bisect_right(times,t)
        a,b=times[j-1],times[j]
        if b-a>max_gap+1e-9:raise ValueError('Encoder gap exceeds 150ms; no reliable integration')
        return samples[a]+(samples[b]-samples[a])*(t-a)/(b-a)
    total=0.
    for a,b in zip(points,points[1:]):
        va,vb=value(a),value(b)
        if b-a>max_gap+1e-9:raise ValueError('Encoder gap exceeds 150ms; no reliable integration')
        if not all(math.isfinite(v) and 0<=v<=200 for v in (va,vb)):
            raise ValueError('Invalid or reverse mean wheel feedback')
        total+=.5*(va+vb)*(b-a)/60.
    return total


def analyze(run, cap_start, cap_end, *, circumference=.60, measured_travel=None, fixed_target=False):
    if not math.isfinite(circumference) or circumference<=0:
        raise ValueError('Invalid wheel circumference')
    if measured_travel is not None and (not math.isfinite(measured_travel) or measured_travel<=0):
        raise ValueError('Independent travel must be positive and finite')
    run=Path(run)
    with (run/'camera_raw.frames.csv').open(newline='') as stream:
        rows={int(r['capture_frame_id']):r for r in csv.DictReader(stream)}
    try:
        start,end=[float(rows[c]['capture_monotonic_sec']) for c in (cap_start,cap_end)]
    except KeyError as exc:
        raise ValueError('Requested CAP not present') from exc
    samples={}; depths={}
    pattern=re.compile(r'feedback_forward_rpm=\(([-+\d.]+),\s*([-+\d.]+)\)')
    with (run/'request_0513_modular.log').open(encoding='utf-8',errors='replace') as stream:
        for line in stream:
            values=dict(re.findall(r'(\w+)=([^\s]+)',line))
            if 'visible_wheel_dispatch ' in line:
                match=pattern.search(line)
                try:
                    if match:
                        t=float(values['feedback_ts']);left,right=float(match[1]),float(match[2])
                        if math.isfinite(t):samples[t]=.5*(left+right) if min(left,right)>=0 else float('nan')
                except (KeyError,ValueError):pass
            if 'Astra depth timeline:' in line and values.get('temporal')=='new_sample':
                try:
                    t=float(values['sample_ts']);raw=float(values['raw'])
                    if math.isfinite(t) and math.isfinite(raw) and raw>0:
                        depths[t]=(raw,values.get('target'),values.get('selected_regions','unknown'))
                except (KeyError,ValueError):pass
    rev=integrate_revolutions(samples,start,end)
    result=dict(cap_start=cap_start,cap_end=cap_end,duration_sec=end-start,
                configured_circumference_m=circumference,encoder_revolutions=rev,
                encoder_travel_m=rev*circumference,independent_travel_m=measured_travel,
                independently_implied_circumference_m=measured_travel/rev if measured_travel and rev>0 else None,
                calibration_applied=False,depth_comparison=None)
    ds=sorted(t for t in depths if start<=t<=end)
    if fixed_target and len(ds)>=2:
        a,b=ds[0],ds[-1]
        ids={depths[t][1] for t in ds}
        if len(ids)==1 and None not in ids:
            try:
                travel=integrate_revolutions(samples,a,b)*circumference
                closure=depths[a][0]-depths[b][0]
                result['depth_comparison']=dict(duration_sec=b-a,raw_start_m=depths[a][0],
                    raw_end_m=depths[b][0],depth_closure_m=closure,encoder_travel_m=travel,
                    closure_to_encoder_ratio=closure/travel if travel>0 else None,
                    capture_endpoint_offsets_ms=[1000*(a-start),1000*(end-b)])
            except ValueError as exc:result['depth_comparison']=dict(unavailable=str(exc))
    result['limitations']=[
        'Assumes straight forward motion; wheel slip and RPM register scale remain unverified.',
        'Feedback is from existing dispatch logs; a value is not an independent ground-truth measurement.',
        'Depth closure requires the same fixed surface, aligned coordinates and reliable raw depth.',
        'Depth sub-interval is NOT necessarily the independent tape-measure interval.',
        'No automatic calibration; do not set circumference from depth ratio alone.',
    ]
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--cap-start',type=int,required=True)
    p.add_argument('--cap-end',type=int,required=True)
    p.add_argument('--circumference',type=float,default=.60)
    p.add_argument('--measured-travel-m',type=float)
    p.add_argument('--fixed-target',action='store_true')
    args=p.parse_args()
    try:
        result=analyze(args.run,args.cap_start,args.cap_end,circumference=args.circumference,
                       measured_travel=args.measured_travel_m,fixed_target=args.fixed_target)
    except (ValueError,OSError) as exc:p.error(str(exc))
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
