#!/usr/bin/env python3
"""Offline CAP555 identity-policy comparison using saved RKNN crops only.

Reconstruct recorded approved learning through CAP321, resume the confirmed
CAP390 partial handoff, then compare the old early return with the fixed path.
Search state stays as logged: this is not detector, control, or motor replay.
"""
import argparse
import configparser
from dataclasses import fields
import inspect
import json
import logging
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.audit_reid_crops import car_processes, extract_saved, model_config
from rk_vision.identity_bank import IdentityBank, IdentityBankConfig, _geometry_observation


def old_policy():
    """Change only the added dispatch condition in an isolated subclass."""
    source = textwrap.dedent(inspect.getsource(IdentityBank._continue_observed_partial))
    condition = "if strong or pair.get('qualified'):"
    if source.count(condition) != 1:
        raise ValueError('CAP555 dispatch changed; review the baseline before replay')
    namespace = dict(IdentityBank._continue_observed_partial.__globals__)
    exec(source.replace(condition, "if pair.get('qualified'):"), namespace)
    return type('OldPartialExitBank', (IdentityBank,), {
        '_continue_observed_partial': namespace['_continue_observed_partial']})


def bank_config(path):
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(path)
    options = {}
    defaults = IdentityBankConfig()
    for field in fields(defaults):
        if not parser.has_option('identity_bank', field.name):
            continue
        value = getattr(defaults, field.name)
        getter = (parser.getboolean if isinstance(value, bool) else
                  parser.getint if isinstance(value, int) else parser.getfloat)
        options[field.name] = getter('identity_bank', field.name)
    return IdentityBankConfig(**options)


def replay(cls, config, by_cap, vectors, templates, queries):
    bank = cls(config)
    for cap in templates:
        event, value = by_cap[cap], vectors[cap]
        metadata, frame = dict(event['sample_metadata']), event['frame_index']
        if not bank.identities:
            bank._create_identity(value['fused'], frame, metadata, value['torso'])
        else:
            entry = bank.identities[1]
            entry.add(value['fused'], frame, config.max_features,
                      config.diversity_min_distance, config.diversity_replace_margin, metadata)
            if event['assignment'].get('partial_template_update_reason') == 'trusted_update':
                entry.add_partial(value['torso'], frame, config.partial_max_features,
                                  config.diversity_min_distance, metadata)

    event = by_cap[390]
    metadata = dict(event['sample_metadata'], frame_index=event['frame_index'])
    bank._bind_reacquired_identity(1, 2, event['frame_index'], metadata)
    previous = _geometry_observation(metadata, event['frame_index'])
    bank.identities[1].last_strong_observation = previous
    bank._remember_track_seen(2, 1, event['frame_index'])
    row = bank._candidate_observations.observe(1, 2, metadata, previous, True, lambda _: True)
    row.update(confirmed=dict(previous), late_confirmed_source='partial')
    evidence = event['assignment']['reacquire_recent_partial_evidence']
    bank._appearance_verified[1] = dict(
        metadata=metadata, comparable_caps=evidence['comparable_caps'],
        comparison_mode=evidence['comparison_mode'], pending_used=False,
        scale_started=event['capture_timestamp'])

    rows = []
    for cap in queries:
        event, value = by_cap[cap], vectors[cap]
        metadata, box = dict(event['sample_metadata']), event['detector_bbox']
        search = bool(metadata.get('search_reacquire_context_active'))
        uid = bank.assign(
            track_id=2, feature=value['fused'], partial_feature=value['torso'],
            confidence=metadata['detector_confidence'],
            area=(box[2]-box[0])*(box[3]-box[1]), frame_index=event['frame_index'],
            candidate_count=metadata['candidate_count'],
            bbox_quality_ok=metadata['quality_bbox_ok'],
            bbox_quality_tier=metadata['bbox_quality_tier'], sample_metadata=metadata,
            preferred_uid=1 if search else None,
            preferred_candidate_ok=search and metadata.get('search_direction_compatible') is not False)
        assignment = bank.last_assignments[2]
        rows.append(dict(
            cap=cap, uid=uid, reason=assignment['reason'],
            quarantine=assignment.get('template_quarantine_reason'),
            streak=assignment.get('template_quarantine_streak'),
            updated=assignment['bank_updated'],
            partial_update=assignment.get('partial_template_update_reason'),
            full=assignment.get('authorization_full_distance_floor'),
            partial=(assignment.get('reacquire_recent_partial_evidence') or {}).get('distance')))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'car_control_modular/config/reid_runtime.ini')
    parser.add_argument('--infer', action='store_true', help='explicit saved-crop NPU inference')
    parser.add_argument('--all-rows', action='store_true')
    args = parser.parse_args()
    if not args.infer:
        parser.error('--infer required; no camera, serial, or motor is used')
    if car_processes():
        parser.error('car runtime is active; refuse offline inference')
    events = [json.loads(line) for line in (args.run_dir/'reid_diagnostics/events.jsonl').open()]
    by_cap = {event['capture_frame_id']: event for event in events}
    if (len(by_cap) != len(events) or 390 not in by_cap
            or by_cap[390]['assignment']['reason'] != 'preferred_search_late_reacquire'
            or by_cap[390]['raw_track_id'] != 2):
        parser.error('expected unambiguous CAP390/track2 partial-handoff recording')
    templates = [e['capture_frame_id'] for e in events
                 if e['capture_frame_id'] <= 321 and e['assignment']['bank_updated']]
    queries = [e['capture_frame_id'] for e in events if e['capture_frame_id'] > 390]
    logging.disable(logging.CRITICAL)
    vectors, provenance = extract_saved(args.run_dir, templates+queries, model_config(args.config))
    config = bank_config(args.config)
    result = dict(scope='saved approved crops; CAP390 checkpoint; fixed logged search state',
                  model_config=str(args.config), inferred_crops=len(provenance), templates=templates)
    for name, cls in (('old', old_policy()), ('fixed', IdentityBank)):
        rows = replay(cls, config, by_cap, vectors, templates, queries)
        result[name] = dict(
            observed_after_555=sum(r['cap'] >= 555 for r in rows),
            uid_after_555=sum(r['cap'] >= 555 and r['uid'] == 1 for r in rows),
            baseline_differences=[r['cap'] for r in rows if
                (r['uid'], r['reason']) != (by_cap[r['cap']]['uid'], by_cap[r['cap']]['assignment']['reason'])]
                if name == 'old' else None,
            rows=rows if args.all_rows else [r for r in rows if r['cap'] in
                (415, 417, 419, 423, 516, 554, 566, 575, 606, 608)])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
