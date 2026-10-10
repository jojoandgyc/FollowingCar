#!/usr/bin/env python3
"""CAP606--691 score-driven identity-policy replay, without model/hardware I/O.

Saved detections, raw IDs, capture clocks, yaw, quality and competition are
replayed through IdentityBank.assign. Appearance vectors were not recorded:
synthetic vectors reproduce the logged full-gallery and torso minimum scores.
Their query-to-query similarity is not evidence of real visual continuity.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import logging
import math
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rk_vision.identity_bank import IdentityBank
from tools.replay_cap334_recovery import gallery_snapshot
from tools.replay_cap759_recovery import bank_config, metadata

REFERENCES = (21, 27)
WARMUP = 605
FIRST, LAST = 606, 691


class PreviousCandidatePolicy(IdentityBank):
    def _similar_follow_handoff_candidate(self, *args, **kwargs):
        return None


def recorded_full(event):
    assignment = event['assignment']
    choices = ((assignment.get('match_evidence') or {}).get('distance'),
               (event['sample_metadata'].get('identity_competition') or {}).get('distance'))
    for value in choices:
        if isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 2:
            return float(value)
    raise ValueError(f"CAP{event['capture_frame_id']}: no recorded full appearance score")


def vectors_for_distances(reference_distance, first, second):
    """Construct a synthetic query at specified distances from two references."""
    import numpy as np
    values = (reference_distance, first, second)
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 2 for v in values):
        raise ValueError('finite cosine distances are required')
    c = 1. - reference_distance
    s = math.sqrt(max(0., 1. - c*c))
    if s <= 1e-9:
        raise ValueError('distinct non-antipodal references are required')
    x = 1. - first
    y = ((1. - second)-c*x)/s
    residual = 1. - x*x-y*y
    if residual < -1e-6:
        raise ValueError('recorded scores are inconsistent with the two reference distances')
    return np.asarray([x, y, math.sqrt(max(0., residual))], dtype='float32')


def load_events(run_dir):
    path = Path(run_dir)/'reid_diagnostics/events.jsonl'
    all_events = [json.loads(line) for line in path.open()]
    events = {}
    anchor = None
    for event in all_events:
        cap = event['capture_frame_id']
        if cap in REFERENCES or WARMUP <= cap <= LAST:
            if cap in events:
                raise ValueError(f'ambiguous CAP{cap}; cannot select a target silently')
            events[cap] = event
        geometry = event['assignment'].get('reacquire_geometry') or {}
        reference = geometry.get('reference') or {}
        if reference.get('capture_frame_id') == 424:
            if anchor is not None and reference != anchor:
                raise ValueError('inconsistent recorded CAP424 protected geometry')
            anchor = copy.deepcopy(reference)
    if not set((*REFERENCES, WARMUP, FIRST, LAST)).issubset(events) or anchor is None:
        raise ValueError('missing approved templates, warmup, endpoints or protected CAP424 geometry')
    for cap in REFERENCES:
        if events[cap]['assignment'].get('bank_updated') is not True:
            raise ValueError(f'CAP{cap}: unapproved template cannot seed replay')
    if events[WARMUP]['assignment'].get('protected_search_anchor_cap') != 424:
        raise ValueError('CAP605 does not retain the expected protected CAP424 anchor')
    previous_stamp = -float('inf')
    for cap in sorted(c for c in events if c >= WARMUP):
        event = events[cap]
        if event['sample_metadata'].get('capture_frame_id') != cap or event['capture_timestamp'] <= previous_stamp:
            raise ValueError('unordered or mismatched capture provenance')
        previous_stamp = event['capture_timestamp']
        if event['raw_track_id'] not in (-1, 4, 5, 6):
            raise ValueError(f'CAP{cap}: unexpected raw-track lineage')
        recorded_full(event)
    return events, anchor


def seed_bank(config, events, anchor, bank_class=IdentityBank):
    import numpy as np
    bank = bank_class(config)
    initial, second = (events[cap] for cap in REFERENCES)
    full_separation = recorded_full(second)
    torso_separation = second['assignment']['partial_distance']
    base = np.asarray([1., 0., 0.], dtype='float32')
    bank._create_identity(base, initial['frame_index'], metadata(initial), base)
    entry = bank.identities[1]
    entry.add(vectors_for_distances(full_separation, full_separation, 0.),
              second['frame_index'], config.max_features, config.diversity_min_distance,
              config.diversity_replace_margin, metadata(second))
    entry.add_partial(vectors_for_distances(torso_separation, torso_separation, 0.),
                      second['frame_index'], config.partial_max_features,
                      config.diversity_min_distance, metadata(second))
    entry.last_strong_observation = copy.deepcopy(anchor)
    entry.last_seen_frame = anchor['frame_index']
    bank._reacquire_search_anchors[1] = copy.deepcopy(anchor)
    bank.track_to_uid[1] = 1
    bank._remember_track_seen(1, 1, anchor['frame_index'])
    # Existing time-based retirement is part of the checkpoint, not a write
    # caused by any candidate. Compare galleries after advancing to warmup.
    if entry.template_memory is not None:
        entry.template_memory.advance(metadata(events[WARMUP]))
    return bank, full_separation, torso_separation


def appearance(event, full_separation, torso_separation):
    full = recorded_full(event)
    per_reference = {}
    for row in (event['assignment'].get('match_evidence') or {}).get('nearest_samples', ()):
        cap = (row.get('metadata') or {}).get('capture_frame_id')
        if row.get('tier') == 'strong' and cap in REFERENCES:
            per_reference[cap] = row['distance']
    # With only the minimum logged (CAP675/677/679), both synthetic distances
    # are set to that minimum. No missing individual distance is claimed known.
    query = vectors_for_distances(full_separation,
        per_reference.get(21, full), per_reference.get(27, full))
    part = event['assignment'].get('partial_distance')
    torso = None if part is None else vectors_for_distances(torso_separation, part, part)
    return query, torso, sorted(per_reference)


def replay(config, events, anchor, *, previous=False):
    bank, full_separation, torso_separation = seed_bank(
        config, events, anchor, PreviousCandidatePolicy if previous else IdentityBank)
    before = gallery_snapshot(bank)
    rows = []
    for cap in sorted(c for c in events if c >= WARMUP):
        event = events[cap]
        m = metadata(event)
        full, torso, individual = appearance(event, full_separation, torso_separation)
        box = event['detector_bbox']
        raw = event['raw_track_id']
        # Preserve recorded search flags even if current policy now accepts.
        uid = bank.assign(track_id=raw, feature=full, partial_feature=torso,
            confidence=m['detector_confidence'], area=(box[2]-box[0])*(box[3]-box[1]),
            frame_index=event['frame_index'], candidate_count=m['candidate_count'],
            bbox_quality_ok=m['quality_bbox_ok'], bbox_quality_reason=m.get('bbox_quality_reason', ''),
            bbox_quality_tier=m['bbox_quality_tier'], sample_metadata=m,
            preferred_uid=1 if m.get('search_reacquire_context_active') else None,
            preferred_candidate_ok=bool(m.get('search_reacquire_context_active'))
                and m.get('search_direction_compatible') is not False)
        assignment = bank.last_assignments[raw]
        full_evaluated = (assignment.get('similar_follow') or {}).get('gallery_distance')
        if full_evaluated is not None and abs(full_evaluated-recorded_full(event)) > 1e-5:
            raise ValueError(f'CAP{cap}: synthetic full input no longer matches the recorded score')
        rows.append(dict(cap=cap, raw=raw, uid=uid, recorded_uid=event['uid'],
            reason=assignment['reason'], recorded_reason=event['assignment']['reason'],
            full_score=recorded_full(event), partial_score=event['assignment'].get('partial_distance'),
            individually_recorded_full_references=individual,
            quality_ok=m['quality_bbox_ok'], quality_reason=m.get('quality_bbox_reason'),
            similar_follow=assignment.get('similar_follow'),
            evaluated_full=full_evaluated,
            bank_updated=assignment.get('bank_updated'),
            learning_written_tiers=assignment.get('learning_written_tiers', [])))
    queried = [row for row in rows if row['cap'] >= FIRST]
    writes = [row['cap'] for row in rows if row['bank_updated'] or row['learning_written_tiers']]
    return dict(observation_count=len(queried), accepted_count=sum(row['uid'] == 1 for row in queried),
        accepted_caps=[row['cap'] for row in queried if row['uid'] == 1],
        reasons=dict(Counter(row['reason'] for row in queried)),
        per_raw={str(raw): dict(observed=sum(r['raw'] == raw for r in queried),
            accepted=sum(r['raw'] == raw and r['uid'] == 1 for r in queried))
            for raw in sorted({r['raw'] for r in queried})},
        gallery_write_caps=writes, gallery_before=before,
        gallery_after=gallery_snapshot(bank), rows=rows)


def report(run_dir, config_path):
    events, anchor = load_events(run_dir)
    config = bank_config(config_path)
    with patch('rk_vision.identity_bank.cropped_follow_continuous', return_value=False):
        baseline = replay(config, events, anchor, previous=True)
    current = replay(config, events, anchor)
    changes = [dict(cap=new['cap'], raw=new['raw'], previous_uid=old['uid'], current_uid=new['uid'])
               for old, new in zip(baseline['rows'], current['rows']) if old['uid'] != new['uid']]
    return dict(scope='recorded-score identity-policy replay; no saved appearance embeddings, '
                'no model execution, no raw-track association replay, no hardware/control replay',
        limitations=[
            'Only saved observations are replayed; missing detections and predictions are not reconstructed.',
            'Synthetic vectors reproduce gallery distances; query-to-query similarity is artificial.',
            'CAP675/677/679 have only minimum full scores and no saved torso scores.',
            'Raw IDs, search flags, boxes, yaw and timestamps stay exactly as logged.',
            'CAP21/27 seed approved templates; recorded CAP424 restores protected geometry only.',
            'Baseline disables only the new candidate raw handoff and edge crop exception; it is not historical binary execution.'],
        run_dir=str(Path(run_dir).resolve()),
        events_sha256=hashlib.sha256((Path(run_dir)/'reid_diagnostics/events.jsonl').read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        policy_sha256=hashlib.sha256((ROOT/'rk_vision/identity_bank.py').read_bytes()).hexdigest(),
        recorded_accepted_count=sum(e['uid'] == 1 for cap, e in events.items() if FIRST <= cap <= LAST),
        baseline_recorded_uid_agreement=sum(row['uid'] == row['recorded_uid']
            for row in baseline['rows'] if row['cap'] >= FIRST),
        missing_torso_caps=[cap for cap, e in events.items() if cap >= WARMUP and e['assignment'].get('partial_distance') is None],
        baseline=baseline, current=current, changed_decisions=changes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'car_control_modular/config/reid_runtime.ini')
    args = parser.parse_args(argv)
    previous_disable = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        result = report(args.run_dir, args.config)
    finally:
        logging.disable(previous_disable)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    main()
