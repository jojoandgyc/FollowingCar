#!/usr/bin/env python3
"""Offline identity replay of CAP832--901 and the independent CAP881 entry.

Restore the recorded approved gallery and confirmed CAP830/CAP879 checkpoints.
No camera, detector, serial, control loop, or motor is imported or started.
Saved-crop RKNN inference is opt-in and refuses an active car entrypoint.
The continuous-visible scope changes search metadata only: it is not a claim
about the images or motion a differently controlled robot would have produced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.audit_reid_crops import car_processes, cosine, extract_saved, model_config
from tools.replay_cap225_pose_transition import model_fingerprint, validate_feature_report
from tools.replay_cap555_quarantine import bank_config
from rk_vision.identity_bank import IdentityBank, _geometry_observation

APPROVED_CAPS = (
    20, 25, 40, 57, 88, 142, 159, 176, 192, 211, 226, 244, 263, 297,
    346, 355, 369, 386, 399, 414, 434, 449, 463, 478, 493, 511, 525,
    539, 556, 571, 587, 603, 617, 633, 651, 666, 680, 696,
)
CHECKPOINTS = {830: dict(arm=814, source='partial', start=816),
               879: dict(arm=875, source='strong', start=878)}
END_CAP = 901


class DisabledVerifiedContinuationBank(IdentityBank):
    """Disable only the new proof, not a snapshot of the entire old program."""

    def _evaluate_verified_continuation(self, *args, **kwargs):
        return None


def metadata(event):
    return dict(event['sample_metadata'], frame_index=event['frame_index'])


def load_events(run_dir):
    with (run_dir / 'reid_diagnostics/events.jsonl').open() as stream:
        rows = [json.loads(line) for line in stream]
    by_cap = {row['capture_frame_id']: row for row in rows}
    if len(rows) != len(by_cap):
        raise ValueError('ambiguous CAP identifiers; expected one saved target crop per CAP')
    approved = tuple(sorted(row['capture_frame_id'] for row in rows
                            if row['capture_frame_id'] <= max(CHECKPOINTS)
                            and row['assignment'].get('bank_updated')))
    if approved != APPROVED_CAPS:
        raise ValueError(f'approved gallery differs from the CAP828 episode: {approved}')
    for cap, spec in CHECKPOINTS.items():
        row = by_cap.get(cap, {})
        assignment = row.get('assignment', {})
        partial = assignment.get('reacquire_recent_partial_evidence') or {}
        candidate = assignment.get('candidate_observation') or {}
        if (row.get('uid') != 1 or row.get('raw_track_id') != 3
                or assignment.get('reason') != 'mapped_late_continuation'
                or assignment.get('authorization_match_source') != spec['source']
                or assignment.get('bank_updated') is not False
                or assignment.get('reacquire_partial_state') != 'match'
                or assignment.get('protected_search_anchor_cap') != 713
                or assignment.get('template_quarantine_streak') != 0
                or partial.get('comparable_caps') != [414]
                or partial.get('comparison_mode') != 'exact_coverage'
                or candidate.get('start_cap') != spec['start']):
            raise ValueError(f'expected recorded confirmed CAP{cap}/track3 checkpoint')
        arm = by_cap.get(spec['arm'], {})
        if (arm.get('uid') != 1 or arm.get('raw_track_id') != 3
                or arm.get('assignment', {}).get('template_quarantine_reason') != 'armed'):
            raise ValueError(f'CAP{cap}: missing recorded quarantine arm')
    anchor = (by_cap.get(814, {}).get('assignment', {}).get('reacquire_geometry') or {}).get('reference')
    if (not anchor or anchor.get('capture_frame_id') != 713
            or anchor.get('track_id') != 1 or anchor.get('capture_timestamp', 0) <= 0):
        raise ValueError('missing the recorded CAP713 protected geometry; never synthesize it')
    if 832 not in by_cap or 881 not in by_cap or END_CAP not in by_cap:
        raise ValueError('incomplete CAP832--901 observation sequence')
    return by_cap


def required_caps(by_cap):
    return sorted(set(APPROVED_CAPS) | set(CHECKPOINTS) |
                  {cap for cap in by_cap if min(CHECKPOINTS) < cap <= END_CAP})


def checkpoint_bank(config, by_cap, vectors, checkpoint, bank_class=IdentityBank):
    """Only logged approved samples enter the gallery; checkpoints never do."""
    if checkpoint not in CHECKPOINTS:
        raise ValueError('unsupported recorded checkpoint')
    bank = bank_class(config)
    for cap in APPROVED_CAPS:
        event, value = by_cap[cap], vectors[cap]
        m, frame = metadata(event), event['frame_index']
        if event['assignment'].get('bank_updated') is not True or cap >= checkpoint:
            raise ValueError(f'CAP{cap}: refusing unapproved or future gallery input')
        if not bank.identities:
            bank._create_identity(value['fused'], frame, m, value['torso'])
        else:
            entry = bank.identities[1]
            entry.add(value['fused'], frame, config.max_features,
                      config.diversity_min_distance, config.diversity_replace_margin, m)
            if event['assignment'].get('partial_template_update_reason') == 'trusted_update':
                entry.add_partial(value['torso'], frame, config.partial_max_features,
                                  config.diversity_min_distance, m)
    event, spec = by_cap[checkpoint], CHECKPOINTS[checkpoint]
    assignment = event['assignment']
    m, frame = metadata(event), event['frame_index']
    geometry = _geometry_observation(m, frame)
    bank.identities[1].last_strong_observation = geometry
    bank.identities[1].last_seen_frame = frame
    bank._remember_track_seen(3, 1, frame)
    bank.track_to_uid[3] = 1
    arm = by_cap[spec['arm']]
    bank._quarantine_decisions[1] = bank._reacquire_quarantine.arm(
        1, 3, spec['arm'], arm['capture_timestamp'], arm['frame_index'])
    bank._reacquire_search_anchors[1] = dict(
        by_cap[814]['assignment']['reacquire_geometry']['reference'])
    evidence = assignment['reacquire_recent_partial_evidence']
    bank._appearance_verified[1] = dict(
        metadata=m, comparable_caps=list(evidence['comparable_caps']),
        comparison_mode=evidence['comparison_mode'], pending_used=False,
        scale_started=event['capture_timestamp'])
    observation = assignment['candidate_observation']
    start = by_cap[observation['start_cap']]
    bank._candidate_observations.rows[(1, 3)] = dict(
        start_ts=start['capture_timestamp'], start_cap=observation['start_cap'],
        direction=m.get('search_direction'), from_compatible=False,
        crossed=observation['crossed'], count=observation['count'],
        run_start_frame=start['frame_index'], last=dict(geometry),
        confirmed=dict(geometry), late_confirmed_source=spec['source'])
    return bank


def replay(config, by_cap, vectors, checkpoint, *, logged_search,
           bank_class=IdentityBank):
    bank = checkpoint_bank(config, by_cap, vectors, checkpoint, bank_class)
    rows = []
    for cap in sorted(c for c in by_cap if checkpoint < c <= END_CAP):
        event, value = by_cap[cap], vectors[cap]
        m, box = metadata(event), event['detector_bbox']
        if not logged_search:
            m.update(search_reacquire_context_active=False,
                     search_direction=None, search_direction_compatible=None)
        search = bool(m.get('search_reacquire_context_active'))
        uid = bank.assign(
            track_id=event['raw_track_id'], feature=value['fused'], partial_feature=value['torso'],
            confidence=m['detector_confidence'], area=(box[2]-box[0])*(box[3]-box[1]),
            frame_index=event['frame_index'], candidate_count=m['candidate_count'],
            bbox_quality_ok=m['quality_bbox_ok'], bbox_quality_reason=m.get('bbox_quality_reason', ''),
            bbox_quality_tier=m['bbox_quality_tier'], sample_metadata=m,
            preferred_uid=1 if search else None,
            preferred_candidate_ok=search and m.get('search_direction_compatible') is not False)
        assignment = bank.last_assignments[event['raw_track_id']]
        partial = assignment.get('reacquire_recent_partial_evidence') or {}
        rows.append(dict(
            cap=cap, uid=uid, reason=assignment['reason'], search=search,
            logged_uid=event['uid'], logged_reason=event['assignment']['reason'],
            full=(assignment.get('template_recent_evidence') or {}).get('distance'),
            partial=partial.get('distance'), comparable_caps=partial.get('comparable_caps'),
            comparison_mode=partial.get('comparison_mode'),
            partial_state=assignment.get('reacquire_partial_state'),
            verified_continuation=assignment.get('identity_continuation'),
            continuation_pair=assignment.get('identity_continuation_pair'),
            continuation_source=(bank._appearance_verified.get(1) or {}).get('continuation_source'),
            pending_deadline=(bank._appearance_verified.get(1) or {}).get('pending_continuation_deadline'),
            quarantine=assignment.get('template_quarantine_reason'),
            streak=assignment.get('template_quarantine_streak'),
            gallery_updated=assignment.get('bank_updated'),
            recent_updated=assignment.get('recent_bank_updated'),
            partial_update=assignment.get('partial_template_update_reason')))
    return dict(total_frames=len(rows), uid_frames=sum(r['uid'] == 1 for r in rows),
        accepted_caps=[r['cap'] for r in rows if r['uid'] == 1],
        gallery_write_caps=[r['cap'] for r in rows if r['gallery_updated'] or r['recent_updated']],
        differences_from_recording=[r['cap'] for r in rows if
            (r['uid'], r['reason']) != (r['logged_uid'], r['logged_reason'])], rows=rows)


def load_cached_vectors(archive_path, report_path, run_dir, caps, expected_model):
    import numpy as np
    provenance, report = validate_feature_report(
        report_path, archive_path, run_dir, caps, expected_model)
    digest = report.get('feature_archive_sha256')
    if digest and hashlib.sha256(archive_path.read_bytes()).hexdigest() != digest:
        raise ValueError('feature archive digest differs from audit')
    with np.load(archive_path, allow_pickle=False) as archive:
        vectors = {cap: {name: archive[f'cap_{cap}_{name}'].copy()
                        for name in ('fused', 'torso')} for cap in caps}
    for items in vectors.values():
        for vector in items.values():
            if vector.ndim != 1 or not np.isfinite(vector).all() or np.linalg.norm(vector) <= 1e-12:
                raise ValueError('invalid cached embedding')
    checked = set()
    for pair in report.get('pairs', []):
        query, template = pair['query'], pair['template']
        if query not in vectors or template not in vectors:
            continue
        for name in ('fused', 'torso'):
            if abs(cosine(vectors[query][name], vectors[template][name]) - pair[name]) > 1e-6:
                raise ValueError(f'CAP{query}/CAP{template}: cached {name} differs from audit')
        checked.update((query, template))
    if not digest and not set(caps).issubset(checked):
        raise ValueError('cache lacks a digest or pair-distance provenance for every crop')
    return vectors, provenance


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'car_control_modular/config/reid_runtime.ini')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--infer', action='store_true')
    source.add_argument('--features', '--feature-archive', dest='feature_archive', type=Path)
    parser.add_argument('--feature-report', type=Path)
    parser.add_argument('--feature-output', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--label', default='current_policy')
    args = parser.parse_args(argv)
    for path in (args.output, args.feature_output):
        if path is not None and path.exists():
            parser.error(f'refuse overwrite: {path}')
        if path is not None and args.run_dir.resolve() in path.resolve().parents:
            parser.error('refuse writing inside the historical run directory')
    if args.feature_output and not args.infer:
        parser.error('--feature-output requires --infer')
    if args.feature_archive and not args.feature_report:
        parser.error('--features requires --feature-report with model/crop/cache provenance')
    if args.feature_report and args.infer:
        parser.error('--feature-report is only used with --features')
    by_cap = load_events(args.run_dir)
    caps = required_caps(by_cap)
    logging.disable(logging.CRITICAL)
    encoder_config = model_config(args.config)
    model = model_fingerprint(encoder_config)
    if args.infer:
        if car_processes():
            parser.error('car runtime active; refuse saved-crop inference')
        vectors, provenance = extract_saved(args.run_dir, caps, encoder_config)
        if args.feature_output:
            import numpy as np
            args.feature_output.parent.mkdir(parents=True, exist_ok=True)
            with args.feature_output.open('xb') as stream:
                np.savez_compressed(stream, **{f'cap_{cap}_{name}': value
                    for cap, features in vectors.items() for name, value in features.items()
                    if value is not None})
    else:
        vectors, provenance = load_cached_vectors(
            args.feature_archive, args.feature_report, args.run_dir, caps, model)
    config = bank_config(args.config)
    archive = args.feature_archive or args.feature_output
    result = dict(label=args.label,
        scope='approved saved gallery; recorded confirmed CAP830/CAP879 checkpoints; not full-run replay',
        scope_notes=dict(
            logged_search='retain recorded search state, real capture times, yaw, competition, and boxes',
            continuous_visible_counterfactual='force only search metadata inactive; not a prediction of images/motion',
            disabled_verified_continuation='disable only _evaluate_verified_continuation in a subclass; not a historical code snapshot',
            checkpoint='no checkpoint or unconfirmed crop enters gallery; CAP713 protected geometry and original quarantine arm retained'),
        run_dir=str(args.run_dir.resolve()), config=str(args.config), templates=list(APPROVED_CAPS),
        inferred_crops=len(provenance) if args.infer else 0, model=model, crops=provenance,
        feature_archive=str(archive) if archive else None,
        feature_report=str(args.feature_report) if args.feature_report else None,
        feature_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest() if archive else None,
        policy_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
            'rk_vision/identity_bank.py', 'rk_vision/template_memory.py',
            'rk_vision/reacquire_quarantine.py', 'rk_vision/verified_continuation.py')},
        checkpoints={str(checkpoint): {label: {
            'logged_search': replay(config, by_cap, vectors, checkpoint, logged_search=True, bank_class=cls),
            'continuous_visible_counterfactual': replay(config, by_cap, vectors, checkpoint,
                                                      logged_search=False, bank_class=cls)}
            for label, cls in (('current_policy', IdentityBank),
                               ('disabled_verified_continuation', DisabledVerifiedContinuationBank))}
            for checkpoint in CHECKPOINTS})
    payload = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            stream.write(payload+'\n')
        print(f'report={args.output}')
    else:
        print(payload)
    return 0


def main(argv=None):
    """Keep diagnostic suppression local, including parser/error exits."""
    previous_disable = logging.root.manager.disable
    try:
        return _main(argv)
    finally:
        logging.disable(previous_disable)


if __name__ == '__main__':
    raise SystemExit(main())
