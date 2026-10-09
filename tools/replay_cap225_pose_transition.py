#!/usr/bin/env python3
"""Replay saved CAP225 pose/crop observations from the recorded CAP221 state.

No camera, detector, serial, controller, or motor is used. Inference is opt-in
and checks that the car is stopped. The approved gallery and the recorded
CAP221 identity/geometry/quarantine are reconstructed, not the entire runtime.
Two independent scopes retain logged search metadata or hypothesize that the
visible identity never entered search. The latter is not the observed run.
Both run against current policy and an isolated disabled-pose control. That
control disables only the new proof, and is not a complete historical version.
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

from tools.audit_reid_crops import car_processes, cosine, crop_path, extract_saved, model_config
from tools.replay_cap555_quarantine import bank_config
from rk_vision.identity_bank import IdentityBank, _geometry_observation

TEMPLATES = (18, 25, 47, 63, 79, 97, 113, 129)
CHECKPOINT = 221
APPEARANCE_CONTEXT = (208, 212, 215, 217, 221)


class DisabledPoseRetentionBank(IdentityBank):
    """Disable the new pose proof without modifying production globals."""

    def _appearance_continuity_reference(self, *args, **kwargs):
        reference = super()._appearance_continuity_reference(*args, **kwargs)
        if reference is None:
            return None
        return {key: value for key, value in reference.items()
                if not key.startswith('pose_')}


def model_fingerprint(config):
    cfg, weight = config
    return dict(path=cfg.model_path,
        sha256=hashlib.sha256(Path(cfg.model_path).read_bytes()).hexdigest(),
        width=cfg.input_width, height=cfg.input_height, format=cfg.input_format,
        dtype=cfg.input_dtype, layout=cfg.input_layout, normalize=cfg.normalize,
        color_weight=weight)


def validate_feature_report(report_path, archive_path, run_dir, caps, expected_model):
    """Require the audit provenance produced alongside this feature cache."""
    with report_path.open() as stream:
        report = json.load(stream)
    if Path(report.get('feature_archive', '')).resolve() != archive_path.resolve():
        raise ValueError('audit report does not identify this feature archive')
    recorded_model = report.get('model', {})
    for key, value in expected_model.items():
        if key != 'path' and recorded_model.get(key) != value:
            raise ValueError(f'cached model/preprocessing mismatch: {key}')
    recorded_crops = {row['cap']: row for row in report.get('crops', [])}
    provenance = []
    for cap in caps:
        path = crop_path(run_dir, cap)
        recorded = recorded_crops.get(cap)
        if not recorded or recorded.get('sha256') != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f'CAP{cap}: saved crop does not match feature provenance')
        provenance.append(dict(recorded, path=str(path.resolve())))
    return provenance, report


def metadata(event):
    return dict(event['sample_metadata'], frame_index=event['frame_index'])


def load_events(run_dir):
    with (run_dir / 'reid_diagnostics/events.jsonl').open() as stream:
        events = [json.loads(line) for line in stream]
    by_cap = {event['capture_frame_id']: event for event in events}
    if len(by_cap) != len(events):
        raise ValueError('ambiguous capture IDs; this replay expects one target per capture')
    checkpoint = by_cap[CHECKPOINT]
    if (checkpoint['uid'] != 1 or checkpoint['raw_track_id'] != 1
            or checkpoint['assignment']['reason'] != 'skip_update_reacquire_quarantine'
            or checkpoint['assignment'].get('reacquire_partial_state') != 'match'
            or checkpoint['assignment'].get('template_quarantine_streak') != 0):
        raise ValueError('expected the recorded, confirmed CAP221/UID1 checkpoint')
    approved = tuple(e['capture_frame_id'] for e in events
                     if e['capture_frame_id'] <= CHECKPOINT and e['assignment']['bank_updated'])
    if approved != TEMPLATES:
        raise ValueError(f'approved gallery differs from this episode: {approved}')
    queries = sorted(c for c in by_cap if CHECKPOINT < c <= 275)
    return by_cap, queries


def checkpoint_bank(config, by_cap, vectors, bank_class=IdentityBank):
    """Restore only recorded positive state; CAP221 is never a learned sample."""
    bank = bank_class(config)
    for cap in TEMPLATES:
        event, value = by_cap[cap], vectors[cap]
        m, frame = metadata(event), event['frame_index']
        if not bank.identities:
            bank._create_identity(value['fused'], frame, m, value['torso'])
        else:
            entry = bank.identities[1]
            entry.add(value['fused'], frame, config.max_features,
                      config.diversity_min_distance, config.diversity_replace_margin, m)
            if event['assignment'].get('partial_template_update_reason') == 'trusted_update':
                entry.add_partial(value['torso'], frame, config.partial_max_features,
                                  config.diversity_min_distance, m)
    event = by_cap[CHECKPOINT]
    m, frame = metadata(event), event['frame_index']
    a = event['assignment']
    geometry = _geometry_observation(m, frame)
    bank.identities[1].last_strong_observation = geometry
    bank.identities[1].last_seen_frame = frame
    bank._remember_track_seen(1, 1, frame)
    bank.track_to_uid[1] = 1
    # The last recorded quarantine arm was CAP189; elapsed time must not be
    # restarted at this checkpoint, nor may the frozen gallery be unfrozen.
    arm = by_cap[189]
    bank._quarantine_decisions[1] = bank._reacquire_quarantine.arm(
        1, 1, 189, arm['capture_timestamp'], arm['frame_index'])
    anchor_cap = a['protected_search_anchor_cap']
    anchor = by_cap[anchor_cap]
    bank._reacquire_search_anchors[1] = _geometry_observation(
        metadata(anchor), anchor['frame_index'])
    value = vectors[CHECKPOINT]
    bank._observe_template_quarantine(
        uid=1, track_id=1, feature=value['fused'],
        confidence=m['detector_confidence'],
        area=m['detector_area_ratio'] * m['image_width'] * m['image_height'],
        frame_index=frame, bbox_quality_ok=m['quality_bbox_ok'],
        bbox_quality_tier=m['bbox_quality_tier'], metadata=m)
    evidence = a['reacquire_recent_partial_evidence']
    bank._appearance_verified[1] = dict(
        metadata=m, comparable_caps=evidence['comparable_caps'],
        comparison_mode=evidence['comparison_mode'], pending_used=False,
        scale_started=event['capture_timestamp'])
    # Preserve the logged observation run and its last explicit partial
    # continuation. A normal strong observation is not a new partial handoff.
    observation = a.get('candidate_observation')
    if observation:
        start = by_cap[observation['start_cap']]
        prior = max((e for e in by_cap.values() if e['capture_frame_id'] <= CHECKPOINT
                     and e['assignment']['reason'] == 'mapped_late_continuation'),
                    key=lambda e: e['capture_frame_id'])
        bank._candidate_observations.rows[(1, 1)] = dict(
            start_ts=start['capture_timestamp'], start_cap=observation['start_cap'],
            direction=m.get('search_direction'), from_compatible=False,
            crossed=observation['crossed'], count=observation['count'],
            run_start_frame=start['frame_index'], last=dict(geometry),
            confirmed=_geometry_observation(metadata(prior), prior['frame_index']),
            late_confirmed_source=prior['assignment']['authorization_match_source'])
    return bank


def replay(config, by_cap, vectors, queries, *, logged_search, bank_class=IdentityBank):
    bank = checkpoint_bank(config, by_cap, vectors, bank_class)
    rows = []
    for cap in queries:
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
        a = bank.last_assignments[event['raw_track_id']]
        partial = a.get('reacquire_recent_partial_evidence') or {}
        rows.append(dict(
            cap=cap, search=search, uid=uid, reason=a['reason'],
            logged_uid=event['uid'], logged_reason=event['assignment']['reason'],
            full=(a.get('template_recent_evidence') or {}).get('distance'),
            partial=partial.get('distance'), comparable_caps=partial.get('comparable_caps'),
            comparison_mode=partial.get('comparison_mode'), partial_state=a.get('reacquire_partial_state'),
            pose_caps=partial.get('pose_bridge_caps'),
            pose_remaining_ms=partial.get('pose_retention_remaining_ms'),
            quarantine=a.get('template_quarantine_reason'), streak=a.get('template_quarantine_streak'),
            gallery_updated=a.get('bank_updated'), recent_updated=a.get('recent_bank_updated'),
            partial_update=a.get('partial_template_update_reason')))
    return dict(
        uid_frames=sum(r['uid'] == 1 for r in rows), total_frames=len(rows),
        accepted_caps=[r['cap'] for r in rows if r['uid'] == 1],
        gallery_write_caps=[r['cap'] for r in rows if r['gallery_updated'] or r['recent_updated']],
        differences_from_recording=[r['cap'] for r in rows if
            (r['uid'], r['reason']) != (r['logged_uid'], r['logged_reason'])], rows=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=ROOT/'car_control_modular/config/reid_runtime.ini')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--infer', action='store_true')
    source.add_argument('--features', '--feature-archive', dest='feature_archive', type=Path,
                        help='saved .npz vectors; requires --feature-report audit provenance')
    parser.add_argument('--feature-report', type=Path,
                        help='audit_reid_crops JSON produced with the supplied cache')
    parser.add_argument('--feature-output', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--label', default='current_policy')
    args = parser.parse_args(argv)
    for path in (args.output, args.feature_output):
        if path is not None and path.exists():
            parser.error(f'refuse overwrite: {path}')
    if args.feature_output and not args.infer:
        parser.error('--feature-output requires --infer')
    if args.feature_archive and not args.feature_report:
        parser.error('--features requires --feature-report for model/crop validation')
    if args.feature_report and args.infer:
        parser.error('--feature-report applies only to an existing --features cache')
    by_cap, queries = load_events(args.run_dir)
    caps = sorted(set(TEMPLATES + APPEARANCE_CONTEXT + tuple(queries)))
    logging.disable(logging.CRITICAL)
    import numpy as np
    encoder_config = model_config(args.config)
    model = model_fingerprint(encoder_config)
    if args.infer:
        if car_processes():
            parser.error('car runtime active; refuse saved-crop NPU inference')
        vectors, provenance = extract_saved(args.run_dir, caps, encoder_config)
        if args.feature_output:
            args.feature_output.parent.mkdir(parents=True, exist_ok=True)
            with args.feature_output.open('xb') as stream:
                np.savez_compressed(stream, **{f'cap_{cap}_{name}': value
                    for cap, features in vectors.items() for name, value in features.items()
                    if value is not None})
    else:
        provenance, feature_report = validate_feature_report(
            args.feature_report, args.feature_archive, args.run_dir, caps, model)
        with np.load(args.feature_archive, allow_pickle=False) as archive:
            vectors = {cap: {name: archive[f'cap_{cap}_{name}'].copy()
                            for name in ('fused', 'torso')} for cap in caps}
        # Audit archives predate an embedded archive digest. Check their known
        # pair distances as well as model and crop hashes before replaying.
        for pair in feature_report.get('pairs', []):
            query, template = pair['query'], pair['template']
            if query not in vectors or template not in vectors:
                continue
            for name in ('fused', 'torso'):
                if abs(cosine(vectors[query][name], vectors[template][name]) - pair[name]) > 1e-6:
                    raise ValueError(f'CAP{query}/CAP{template}: cached {name} disagrees with audit')
    config = bank_config(args.config)
    frozen = checkpoint_bank(config, by_cap, vectors).identities[1].template_memory
    distances = []
    for cap in APPEARANCE_CONTEXT + tuple(queries):
        m, value = metadata(by_cap[cap]), vectors[cap]
        distances.append(dict(cap=cap,
            full=frozen.evidence(value['fused'], m),
            partial_all=frozen.evidence(value['torso'], m, 'partial'),
            partial_comparable=frozen.evidence(value['torso'], m, 'partial',
                                              reliable_only=True, comparable_only=True),
            full_to_cap221=cosine(value['fused'], vectors[221]['fused']),
            torso_to_cap221=cosine(value['torso'], vectors[221]['torso'])))
    result = dict(label=args.label,
        scope='approved saved gallery; recorded CAP221 confirmed identity/geometry/quarantine checkpoint; not full-run reconstruction',
        scope_notes={
            'logged_search': 'retain recorded search metadata, captures, yaw, geometry and competition',
            'continuous_visible_counterfactual': 'force search inactive after CAP221; keep actual captured images/yaw/geometry; not a prediction of what the robot would have seen',
            'disabled_pose_retention': 'isolated subclass strips pose proof fields; all other current code remains active; not a full historical implementation',
            'checkpoint': 'CAP221 observation is trusted but not inserted into the template gallery; original CAP189 quarantine arm and CAP173 protected search anchor retained'},
        run_dir=str(args.run_dir.resolve()), config=str(args.config), templates=list(TEMPLATES),
        inferred_crops=len(provenance) if args.infer else 0,
        model=model, feature_archive=str(args.feature_archive or args.feature_output),
        feature_report=str(args.feature_report) if args.feature_report else None,
        cache_directory=str((args.feature_archive or args.feature_output).resolve().parent)
            if args.feature_archive or args.feature_output else None,
        feature_archive_sha256=hashlib.sha256((args.feature_archive or args.feature_output).read_bytes()).hexdigest()
            if args.feature_archive or args.feature_output else None,
        crops=provenance,
        policy_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
            'rk_vision/identity_bank.py', 'rk_vision/template_memory.py', 'rk_vision/reacquire_quarantine.py')},
        distances_against_frozen_checkpoint_gallery=distances,
        policies={label: {
            'logged_search': replay(config, by_cap, vectors, queries, logged_search=True, bank_class=bank_class),
            'continuous_visible_counterfactual': replay(config, by_cap, vectors, queries, logged_search=False, bank_class=bank_class)}
            for label, bank_class in (('current_policy', IdentityBank),
                                     ('disabled_pose_retention', DisabledPoseRetentionBank))})
    payload = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            stream.write(payload+'\n')
        print(f'report={args.output}')
    else:
        print(payload)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
