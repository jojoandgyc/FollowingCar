#!/usr/bin/env python3
"""Offline, read-only raw-video LK trial, with recorded detector corrections.

No inference or camera is opened. Metrics describe image-flow availability and
agreement with saved YOLO boxes, not proof of target identity or control safety.
Recording gaps stay gaps; capture timestamps, not video FPS, drive expiry.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rk_vision.lk_shadow import LKShadowConfig, LKShadowSeed, LKShadowTracker


def percentiles(values):
    if not values:
        return dict(count=0, p50=None, p95=None, maximum=None)
    values = np.asarray(values, dtype=float)
    return dict(count=len(values), p50=float(np.percentile(values, 50)),
                p95=float(np.percentile(values, 95)), maximum=float(np.max(values)))


def load_recording(run_dir, first, last):
    run_dir = Path(run_dir).resolve()
    video = run_dir / 'camera_raw.avi'
    csv_path = run_dir / 'camera_raw.frames.csv'
    event_path = run_dir / 'reid_diagnostics/events.jsonl'
    for path in (video, csv_path, event_path):
        if not path.is_file():
            raise ValueError(f'required saved input is not a regular file: {path}')
    with csv_path.open(newline='') as stream:
        metadata = [r for r in csv.DictReader(stream)
                    if first <= int(r['capture_frame_id']) <= last]
    if not metadata:
        raise ValueError('no saved raw frames in requested capture interval')
    seen, previous_index, previous_ts, previous_cap = set(), -1, -float('inf'), -1
    for row in metadata:
        cap = int(row['capture_frame_id'])
        index = int(row['video_frame_index'])
        ts = float(row['capture_monotonic_sec'])
        if cap in seen or cap <= previous_cap or index <= previous_index or ts <= previous_ts:
            raise ValueError('recording metadata must be unique and ordered')
        seen.add(cap)
        previous_index, previous_ts, previous_cap = index, ts, cap
    events = {}
    with event_path.open() as stream:
        for line in stream:
            event = json.loads(line)
            cap = event.get('capture_frame_id', event.get('sample_metadata', {}).get('capture_frame_id'))
            if cap is not None and first <= cap <= last and event.get('detector_bbox'):
                events.setdefault(cap, []).append(event)
    return video, metadata, events


def intersection_over_union(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    area = np.maximum(0., np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    return float(area / max(1e-9, (a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod() - area))


def replay(run_dir, *, first=606, last=691, uid=1, correction_every=2,
           seed_scope='accepted', frame_scope='all-saved', config=None):
    if correction_every < 1:
        raise ValueError('correction_every must be positive')
    video, metadata, events = load_recording(run_dir, first, last)
    saved_capture_ids = {int(row['capture_frame_id']) for row in metadata}
    if frame_scope not in ('all-saved', 'diagnostic-captures'):
        raise ValueError('unsupported frame_scope')
    if frame_scope == 'diagnostic-captures':
        metadata = [row for row in metadata if int(row['capture_frame_id']) in events]
        if not metadata:
            raise ValueError('no saved images with identity diagnostics in requested interval')
    tracker = LKShadowTracker(config)
    rows, seed_ordinal, last_seed_box = [], 0, None
    # Passing only a validated regular-file path forbids camera device indices.
    reader = cv2.VideoCapture(str(video))
    if not reader.isOpened():
        raise ValueError(f'cannot decode saved video: {video}')
    reader.set(cv2.CAP_PROP_POS_FRAMES, int(metadata[0]['video_frame_index']))
    next_index = int(metadata[0]['video_frame_index'])
    start = time.perf_counter()
    try:
        for meta in metadata:
            cap, index = int(meta['capture_frame_id']), int(meta['video_frame_index'])
            ts = float(meta['capture_monotonic_sec'])
            decode_start = time.perf_counter()
            image = None
            while next_index <= index:
                ok, image = reader.read()
                if not ok:
                    raise ValueError(f'video ends before metadata frame {index}')
                next_index += 1
            decode_ms = (time.perf_counter() - decode_start) * 1000.
            candidates = events.get(cap, [])
            for event in candidates:
                event_ts = event.get('capture_timestamp')
                if event_ts is None:
                    event_ts = event.get('sample_metadata', {}).get('capture_timestamp')
                if event_ts is None or abs(float(event_ts) - ts) > 1e-6:
                    raise ValueError(f'CAP{cap}: detector box and raw image timestamps differ')
            seed_candidates = [e for e in candidates if seed_scope == 'all-detections' or e.get('uid') == uid]
            seed = None
            if len(seed_candidates) == 1:
                event = seed_candidates[0]
                event_ts = event.get('capture_timestamp')
                if event_ts is None:
                    event_ts = event['sample_metadata']['capture_timestamp']
                event_ts = float(event_ts)
                if seed_ordinal % correction_every == 0:
                    # CSV rounds to 6 decimals; use its paired event's exact clock.
                    ts = event_ts
                    seed = LKShadowSeed(int(event['uid']), int(event['raw_track_id']), cap, ts,
                                        tuple(event['detector_bbox']))
                    last_seed_box = seed.bbox
                seed_ordinal += 1
            cpu_start, thread_start = time.process_time_ns(), time.thread_time_ns()
            result = tracker.process(image, cap, ts, seed, frame_format='BGR')
            thread_ms = (time.thread_time_ns() - thread_start) / 1e6
            cpu_ms = (time.process_time_ns() - cpu_start) / 1e6
            row = dict(asdict(result), video_frame_index=index, cpu_ms=cpu_ms,
                       thread_cpu_ms=thread_ms, decode_wall_ms=decode_ms,
                       correction_applied=seed is not None, detector_comparison=None)
            if seed is None and result.status == 'tracked':
                same_raw = [e for e in candidates if e.get('raw_track_id') == result.seed_raw_track_id]
                if len(same_raw) == 1:
                    event = same_raw[0]
                    box = event['detector_bbox']
                    center = lambda b: (np.asarray(b[:2]) + np.asarray(b[2:])) / 2.
                    row['detector_comparison'] = dict(
                        logged_uid=event['uid'], logged_raw_track_id=event['raw_track_id'],
                        logged_bbox=box, iou=intersection_over_union(result.bbox, box),
                        center_error_px=float(np.linalg.norm(center(result.bbox) - center(box))),
                        held_yolo_center_error_px=float(np.linalg.norm(center(last_seed_box) - center(box))),
                        label='same_raw_track_logged_YOLO_agreement_not_ground_truth')
            rows.append(row)
    finally:
        reader.release()
    elapsed = time.perf_counter() - start
    statuses = Counter(row['status'] for row in rows)
    tracked = [row for row in rows if row['status'] == 'tracked']
    comparisons = [r['detector_comparison'] for r in rows if r['detector_comparison']]
    intervals = []
    for row in rows:
        if row['status'] == 'tracked':
            if intervals and intervals[-1]['seed_capture_id'] == row['seed_capture_id']:
                intervals[-1].update(last_capture_id=row['capture_id'],
                                     last_timestamp=row['capture_timestamp'])
                intervals[-1]['tracked_frames'] += 1
            else:
                intervals.append(dict(seed_capture_id=row['seed_capture_id'],
                    first_capture_id=row['capture_id'], last_capture_id=row['capture_id'],
                    first_timestamp=row['prev_capture_timestamp'],
                    last_timestamp=row['capture_timestamp'], tracked_frames=1))
    tracked_seconds = sum(r['capture_timestamp'] - r['prev_capture_timestamp'] for r in tracked)
    duration = rows[-1]['capture_timestamp'] - rows[0]['capture_timestamp']
    summary = dict(
        source='lk_shadow_prerecorded_trial', raw_video=str(video),
        first_capture=first, last_capture=last,
        scope='No NPU, no camera device, no hardware/control output, no target-truth labels.',
        timing_scope='Actual local CPU execution; excludes video decode; not live pipeline latency.',
        frame_scope=frame_scope,
        seed_scope=seed_scope, seed_uid=uid, correction_every_logged_observation=correction_every,
        config=asdict(tracker.config), opencv_threads=cv2.getNumThreads(),
        saved_frames=len(saved_capture_ids), processed_frames=len(rows),
        missing_capture_ids=sorted(set(range(first, last + 1)) - saved_capture_ids),
        skipped_by_frame_scope=sorted(saved_capture_ids - {r['capture_id'] for r in rows}),
        status_counts=dict(statuses), reason_counts=dict(Counter(r['reason'] for r in rows)),
        tracked_frame_coverage=len(tracked) / len(rows),
        seeded_or_tracked_frame_coverage=(len(tracked) + statuses['seeded']) / len(rows),
        tracked_interval_seconds=tracked_seconds, recorded_interval_seconds=duration,
        tracked_interval_time_coverage=tracked_seconds / duration if duration else 0.,
        successful_intervals=intervals,
        process_wall_ms=percentiles([r['wall_ms'] for r in rows]),
        process_cpu_ms=percentiles([r['cpu_ms'] for r in rows]),
        process_thread_cpu_ms=percentiles([r['thread_cpu_ms'] for r in rows]),
        tracked_wall_ms=percentiles([r['wall_ms'] for r in tracked]),
        decode_wall_ms=percentiles([r['decode_wall_ms'] for r in rows]),
        heldout_logged_detector_comparisons=len(comparisons),
        heldout_detector_iou=percentiles([r['iou'] for r in comparisons]),
        heldout_detector_center_error_px=percentiles([r['center_error_px'] for r in comparisons]),
        held_yolo_center_error_px=percentiles([r['held_yolo_center_error_px'] for r in comparisons]),
        replay_elapsed_seconds=elapsed,
        caveats=[
            'all-saved tests recording cadence, not the live processed_frames_only worker cadence.',
            'diagnostic-captures uses only frames with saved identity events; throttled diagnostics may omit processed/probe frames.',
            'Corrections are available at their saved capture image: this trial does not model YOLO return latency.',
            'Recorded raw-track/UID values are provenance, not new LK identity authorization.',
            'There is no ground-truth box on intervening captures; availability does not prove target correctness.',
            'Saved-video frames missing from the recording cannot be reconstructed.',
            'Timing uses this CPU and OpenCV configuration; an integrated live pipeline may contend for resources.',
        ])
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--first', type=int, default=606)
    parser.add_argument('--last', type=int, default=691)
    parser.add_argument('--uid', type=int, default=1)
    parser.add_argument('--correction-every', type=int, default=2)
    parser.add_argument('--seed-scope', choices=('accepted', 'all-detections'), default='accepted')
    parser.add_argument('--frame-scope', choices=('all-saved', 'diagnostic-captures'),
                        default='all-saved', help='Input cadence; neither mode models online scheduling.')
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--max-seed-age-sec', type=float, default=.75)
    parser.add_argument('--max-gap-sec', type=float, default=.25)
    parser.add_argument('--opencv-threads', type=int, default=1)
    parser.add_argument('--output', type=Path, help='Optional NEW report JSON path outside the source run.')
    args = parser.parse_args()
    if args.output:
        destination = args.output.resolve()
        if destination.is_relative_to(args.run_dir.resolve()) or destination.exists():
            parser.error('output must be a new file outside the source run directory')
    cv2.setNumThreads(max(1, args.opencv_threads))
    config = LKShadowConfig(width=args.width, max_seed_age_sec=args.max_seed_age_sec,
                            max_gap_sec=args.max_gap_sec)
    summary, rows = replay(args.run_dir, first=args.first, last=args.last, uid=args.uid,
                          correction_every=args.correction_every, seed_scope=args.seed_scope,
                          frame_scope=args.frame_scope, config=config)
    if args.output:
        with args.output.open('x') as stream:
            json.dump(dict(summary=summary, rows=rows), stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
