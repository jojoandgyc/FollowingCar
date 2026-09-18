#!/usr/bin/env python3
"""Read-only report of passive depth tracking. NOT proof of correct identity."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def report(directory):
    path = Path(directory)
    if not (path/"observations.jsonl").is_file():
        path = path/"depth_track_shadow"
    rows = [json.loads(line) for line in (path/"observations.jsonl").read_text().splitlines() if line.strip()]
    # Historical replays and repeated seeds do not inflate the online Hz.
    live = {(r['uid'], r['sample_timestamp']):r for r in rows
            if r.get('phase')=='update' and r.get('physical_sample_new')}
    successful = sorted((r for r in live.values() if r['status']=='tracked'), key=lambda r:r['sample_timestamp'])
    stamps = [r['sample_timestamp'] for r in successful]
    intervals = [r['tracked_interval_ms'] for r in successful if r.get('tracked_interval_ms') is not None]
    rejection_counts = Counter()
    for row in rows:
        audit = row.get('association_diagnostics') or {}
        if audit.get('scan_rejection'):
            rejection_counts[audit['scan_rejection']] += 1
        for candidate in audit.get('candidates', []):
            rejection_counts.update(candidate.get('reject_reasons', []))
    def distribution(values):
        return None if not values else dict(count=len(values),p50=float(np.percentile(values,50)),
                                            p95=float(np.percentile(values,95)),maximum=max(values))
    return dict(
        mode='shadow_only', geometry_verified=False, motor_authority_created=False,
        status_counts=dict(Counter(r['phase']+':'+r['status'] for r in rows)),
        candidate_rejection_counts=dict(rejection_counts),
        distinct_online_attempts=len(live), distinct_tracked_samples=len(successful),
        tracked_over_attempted_fraction=None if not live else len(successful)/len(live),
        # This duration spans available successful observations, not total
        # person-visible time. Inspect gaps/statuses, never call it camera FPS.
        tracked_hz_over_success_span=None if len(stamps)<2 or stamps[-1]==stamps[0]
            else (len(stamps)-1)/(stamps[-1]-stamps[0]),
        tracked_intervals_ms=distribution(intervals),
        tracked_gaps_over_180ms=sum(v>180 for v in intervals),
        potential_range_gap_fill_frames=sum(r.get('potential_range_gap_fill',False) for r in live.values()),
        tick_processing_ms=distribution([r['tick_processing_ms'] for r in rows if 'tick_processing_ms' in r]),
        live_depth_age_ms=distribution([r['sample_age_ms'] for r in live.values()]),
        live_visual_age_ms=distribution([r['visual_age_ms'] for r in live.values()]),
        caveats=['Potential range-gap fills are NOT granted motor authorizations.',
                 'Unique depth association is NOT proof of correct person; inspect snapshots/video.',
                 'Attempted frames exclude missing seed/pose and overload drops; ratio is conditional.',
                 'HFOV and encoder camera-at-axle geometry are uncalibrated; no verified world velocity.',
                 'tick_processing_ms excludes disk writes; worker summary records overload ticks.'],
    )


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',help='run directory or depth_track_shadow directory')
    args=parser.parse_args()
    print(json.dumps(report(args.directory),ensure_ascii=False,indent=2))
