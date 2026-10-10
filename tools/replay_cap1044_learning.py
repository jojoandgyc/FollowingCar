#!/usr/bin/env python3
"""Saved-crop CAP1044 learning-opportunity audit, not a complete identity replay.

Only recorded approved pre-CAP941 templates seed the bank. Query crops never
seed it. At each logged write opportunity, restore the logged mapped UID and
pre-frame geometry: this isolates learning from unsaved intermediate frames.
Those restored observations are NOT new appearance evidence. The sparse audit
cannot prove how a consecutive-frame gate behaves on the omitted frames.

--infer explicitly uses the NPU on saved PNGs, refuses an active car runtime,
and never imports the camera, serial, motor, or car entrypoint. No overlap
metadata is invented for this historical recording.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.audit_reid_crops import car_processes, cosine, model_config
from tools.replay_cap225_pose_transition import model_fingerprint
from tools.replay_cap555_quarantine import bank_config
from rk_vision.identity_bank import IdentityBank

APPROVED_BEFORE_941 = (21, 27, 40, 108, 142, 171, 184, 212, 261, 286,
    299, 326, 368, 385, 430, 566, 592, 617, 628, 643, 914)
WRITE_OPPORTUNITIES = (941, 958, 1044, 1060, 1078, 1098, 1114, 1128,
    1176, 1191, 1206, 1217)
POLLUTED_CAPS = WRITE_OPPORTUNITIES[2:]


def metadata(event):
    return dict(event['sample_metadata'], frame_index=event['frame_index'])


def load_events(run_dir):
    events = [json.loads(line) for line in
        (run_dir/'reid_diagnostics/events.jsonl').open()]
    rows = {}
    for event in events:
        cap, track = event['capture_frame_id'], event['raw_track_id']
        key = (cap, track)
        if key in rows:
            raise ValueError(f'ambiguous CAP/track event: {key}')
        rows[key] = event
    approved = tuple(e['capture_frame_id'] for e in events
        if e['capture_frame_id'] < 941 and e['assignment'].get('bank_updated'))
    if approved != APPROVED_BEFORE_941:
        raise ValueError('recorded approved clean gallery differs from CAP1044 episode')
    result = {}
    for cap in APPROVED_BEFORE_941 + WRITE_OPPORTUNITIES:
        selected = [e for (c, _), e in rows.items() if c == cap
            and e['uid'] == 1 and e['assignment'].get('bank_updated') is True]
        if len(selected) != 1:
            raise ValueError(f'CAP{cap}: expected exactly one recorded approved crop')
        event = selected[0]
        if cap in WRITE_OPPORTUNITIES:
            reference = (event['assignment'].get('reacquire_geometry') or {}).get('reference')
            if (not reference or event['raw_track_id'] != 3
                    or event['sample_metadata'].get('search_reacquire_context_active')
                    or reference.get('capture_frame_id', cap) >= cap):
                raise ValueError(f'CAP{cap}: missing recorded mapped pre-frame geometry')
        result[cap] = event
    return result


def crop_path(run_dir, event):
    folder = (run_dir/'reid_diagnostics').resolve()
    path = (folder/event['sample_path']).resolve()
    if path.parent != folder or not path.is_file():
        raise ValueError('saved crop must be an existing file inside reid_diagnostics')
    return path


def infer(run_dir, events, config):
    import cv2
    from rk_vision.reid import OSNetRKNNExtractor, _color_signature, _fuse_appearance_features
    from rk_vision.yolo11 import Detection
    if car_processes():
        raise RuntimeError('car runtime active; refuse offline inference')
    encoder_config, weight = config
    encoder = OSNetRKNNExtractor(encoder_config)
    vectors, crops = {}, []
    try:
        for cap, event in sorted(events.items()):
            if car_processes():
                raise RuntimeError('car runtime started; abort offline inference')
            path = crop_path(run_dir, event)
            crop = cv2.imread(str(path))
            if crop is None:
                raise ValueError(f'unreadable CAP{cap}')
            h, w = crop.shape[:2]
            full = encoder.extract(crop, [Detection((0, 0, w, h), 1., 0)], 'BGR')[0]
            if full is None or encoder.last_partial_features[0] is None:
                raise ValueError(f'missing descriptor CAP{cap}')
            vectors[cap] = dict(fused=_fuse_appearance_features(full, _color_signature(crop), weight),
                               torso=encoder.last_partial_features[0])
            crops.append(dict(cap=cap, track=event['raw_track_id'], path=str(path),
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    finally:
        encoder.release()
    return vectors, crops


def load_vectors(archive, report_path, run_dir, events, expected_model):
    import numpy as np
    report = json.loads(report_path.read_text())
    if (Path(report.get('feature_archive', '')).resolve() != archive.resolve()
            or report.get('feature_archive_sha256') != hashlib.sha256(archive.read_bytes()).hexdigest()):
        raise ValueError('feature cache digest/path mismatch')
    for key, value in expected_model.items():
        if key != 'path' and report.get('model', {}).get(key) != value:
            raise ValueError(f'feature model/preprocessing mismatch: {key}')
    provenance = {row['cap']: row for row in report.get('crops', [])}
    for cap, event in events.items():
        path = crop_path(run_dir, event)
        info = provenance.get(cap, {})
        if info.get('track') != event['raw_track_id'] or info.get('sha256') != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f'CAP{cap}: feature crop provenance mismatch')
    with np.load(archive, allow_pickle=False) as loaded:
        vectors = {cap: {name: loaded[f'cap_{cap}_{name}'].copy()
                   for name in ('fused', 'torso')} for cap in events}
    for values in vectors.values():
        for value in values.values():
            if value.ndim != 1 or not np.isfinite(value).all() or np.linalg.norm(value) <= 1e-12:
                raise ValueError('invalid cached embedding')
    return vectors, list(provenance.values())


def seed_bank(config, events, vectors):
    bank = IdentityBank(config)
    for cap in APPROVED_BEFORE_941:
        event, value = events[cap], vectors[cap]
        if not event['assignment'].get('bank_updated') or cap >= min(WRITE_OPPORTUNITIES):
            raise ValueError('refuse unapproved or query gallery seed')
        m, frame = metadata(event), event['frame_index']
        if not bank.identities:
            bank._create_identity(value['fused'], frame, m, value['torso'])
        else:
            entry = bank.identities[1]
            entry.add(value['fused'], frame, config.max_features, config.diversity_min_distance,
                      config.diversity_replace_margin, m)
            if event['assignment'].get('partial_template_update_reason') == 'trusted_update':
                entry.add_partial(value['torso'], frame, config.partial_max_features,
                                  config.diversity_min_distance, m)
    return bank


def gallery_caps(bank):
    entry = bank.identities[1]
    result = dict(strong=[m.get('capture_frame_id') for m in entry.feature_metadata],
                  partial=[m.get('capture_frame_id') for m in entry.partial_feature_metadata],
                  weak=[m.get('capture_frame_id') for m in entry.weak_feature_metadata])
    if entry.template_memory:
        for name in ('recent', 'representatives'):
            for tier, rows in getattr(entry.template_memory, name).items():
                result[f'{name}_{tier}'] = [m.get('capture_frame_id') for _, m in rows]
    return result


def clean_parent_pair_audit(vectors):
    """Distances only; later approved positives here are never replay seeds."""
    parents = APPROVED_BEFORE_941 + (941, 958)
    rows = []
    for query in POLLUTED_CAPS:
        qualified = []
        for parent in parents:
            full = cosine(vectors[query]['fused'], vectors[parent]['fused'])
            partial = cosine(vectors[query]['torso'], vectors[parent]['torso'])
            if full <= .30 and partial <= .30:
                qualified.append(dict(cap=parent, full=full, partial=partial))
        rows.append(dict(cap=query, same_source_clean_pairs=qualified))
    return dict(scope='diagnostic distances to logged approved correct templates, not extra replay seeds',
                full_limit=.30, partial_limit=.30, rows=rows)


def replay(config, events, vectors):
    bank = seed_bank(config, events, vectors)
    rows = []
    for cap in WRITE_OPPORTUNITIES:
        event, values = events[cap], vectors[cap]
        m = metadata(event)
        reference = dict(event['assignment']['reacquire_geometry']['reference'])
        # Sparse diagnostic crops omit intermediate mapped frames. Restore the
        # recorded control-side mapping/geometry, never gallery appearance or
        # a learning guard's pending proof. This is an explicit audit premise.
        bank.identities[1].last_strong_observation = reference
        bank.identities[1].last_seen_frame = reference['frame_index']
        bank.track_to_uid[3] = 1
        bank._remember_track_seen(3, 1, reference['frame_index'])
        box = event['detector_bbox']
        uid = bank.assign(track_id=3, feature=values['fused'], partial_feature=values['torso'],
            confidence=m['detector_confidence'], area=(box[2]-box[0])*(box[3]-box[1]),
            frame_index=event['frame_index'], candidate_count=m['candidate_count'],
            bbox_quality_ok=m['quality_bbox_ok'], bbox_quality_tier=m['bbox_quality_tier'],
            bbox_quality_reason=m.get('bbox_quality_reason', ''), sample_metadata=m)
        a = bank.last_assignments[3]
        caps = gallery_caps(bank)
        rows.append(dict(cap=cap, uid=uid, logged_uid=event['uid'], reason=a['reason'],
            bank_updated=a.get('bank_updated'), recent_bank_updated=a.get('recent_bank_updated'),
            stored_current=any(cap in items for items in caps.values()),
            partial_update=a.get('partial_template_update_reason'),
            learning={k:v for k,v in a.items() if 'learning' in k}, gallery_caps=caps,
            geometry_reference_cap=reference['capture_frame_id']))
    return dict(rows=rows, learned_caps=[r['cap'] for r in rows if r['stored_current']],
                uid_caps=[r['cap'] for r in rows if r['uid'] == 1])


def same_frame_uid_probe(config, events, vectors):
    """Compare identical pre-frame galleries, not two diverging histories.

    The disabled baseline alone advances history between opportunities. Each
    enabled branch is discarded after one frame. It demonstrates whether the
    learning guard directly changes that frame's return UID; it cannot prove
    future identity results are unaffected by a less contaminated gallery.
    """
    baseline = seed_bank(replace(config, template_learning_guard_enable=False), events, vectors)
    rows = []
    for cap in WRITE_OPPORTUNITIES:
        event, value = events[cap], vectors[cap]
        m = metadata(event)
        reference = dict(event['assignment']['reacquire_geometry']['reference'])
        baseline.identities[1].last_strong_observation = reference
        baseline.identities[1].last_seen_frame = reference['frame_index']
        baseline.track_to_uid[3] = 1
        baseline._remember_track_seen(3, 1, reference['frame_index'])
        enabled = copy.deepcopy(baseline)
        enabled.config = replace(config, template_learning_guard_enable=True)
        box = event['detector_bbox']
        kwargs = dict(track_id=3, feature=value['fused'], partial_feature=value['torso'],
            confidence=m['detector_confidence'], area=(box[2]-box[0])*(box[3]-box[1]),
            frame_index=event['frame_index'], candidate_count=m['candidate_count'],
            bbox_quality_ok=m['quality_bbox_ok'], bbox_quality_tier=m['bbox_quality_tier'],
            bbox_quality_reason=m.get('bbox_quality_reason', ''))
        disabled_uid = baseline.assign(**kwargs, sample_metadata=dict(m))
        enabled_uid = enabled.assign(**kwargs, sample_metadata=dict(m))
        disabled_reason = baseline.last_assignments[3]['reason']
        enabled_reason = enabled.last_assignments[3]['reason']
        ordinary = {'mapped', 'updated_diverse', 'skip_update_redundant', 'skip_update_distance'}
        rows.append(dict(cap=cap, disabled_uid=disabled_uid, enabled_uid=enabled_uid,
                         disabled_reason=disabled_reason, enabled_reason=enabled_reason,
                         ordinary_reason_preserved=(disabled_reason not in ordinary or enabled_reason in ordinary),
                         unchanged=disabled_uid == enabled_uid))
    return dict(scope='one-frame cloned prestate; baseline-only history advancement',
                rows=rows, changed_uid_caps=[r['cap'] for r in rows if not r['unchanged']],
                lost_ordinary_reason_caps=[r['cap'] for r in rows if not r['ordinary_reason_preserved']])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'car_control_modular/config/reid_runtime.ini')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--infer', action='store_true')
    source.add_argument('--features', type=Path)
    parser.add_argument('--feature-report', type=Path)
    parser.add_argument('--feature-output', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--all-rows', action='store_true')
    args = parser.parse_args(argv)
    for path in (args.output, args.feature_output):
        if path and (path.exists() or args.run_dir.resolve() in path.resolve().parents):
            parser.error('refuse overwrite or write inside the historical run directory')
    if args.features and not args.feature_report:
        parser.error('--features requires --feature-report')
    if args.feature_output and not args.infer:
        parser.error('--feature-output requires --infer')
    events = load_events(args.run_dir)
    encoder = model_config(args.config)
    model = model_fingerprint(encoder)
    logging.disable(logging.CRITICAL)
    if args.infer:
        vectors, crops = infer(args.run_dir, events, encoder)
    else:
        vectors, crops = load_vectors(args.features, args.feature_report, args.run_dir, events, model)
    report = dict(scope='sparse recorded learning opportunities; mapped UID/geometry restored; not full identity/control replay',
        limitations=['unsaved intermediate frames are not inferred or used as pending learning proof',
                     'historical detection overlap metadata absent; no synthetic overlap supplied',
                     'zero new motor commands; changed galleries can affect later identity matches'],
        templates=list(APPROVED_BEFORE_941), opportunities=list(WRITE_OPPORTUNITIES),
        model=model, crops=crops, clean_parent_pair_audit=clean_parent_pair_audit(vectors))
    if args.feature_output:
        import numpy as np
        with args.feature_output.open('xb') as stream:
            np.savez_compressed(stream, **{f'cap_{cap}_{name}':v
                for cap, values in vectors.items() for name, v in values.items()})
        report.update(feature_archive=str(args.feature_output.resolve()),
            feature_archive_sha256=hashlib.sha256(args.feature_output.read_bytes()).hexdigest())
    config = bank_config(args.config)
    if not hasattr(config, 'template_learning_guard_enable'):
        report['replay_status'] = 'guard_interface_not_yet_available; descriptors cached only'
    else:
        report['disabled'] = replay(replace(config, template_learning_guard_enable=False), events, vectors)
        report['enabled'] = replay(replace(config, template_learning_guard_enable=True), events, vectors)
        report['same_frame_uid_probe'] = same_frame_uid_probe(config, events, vectors)
    if args.output:
        with args.output.open('x') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
    display = {key:value for key,value in report.items() if key != 'crops'}
    if not args.all_rows:
        for mode in ('disabled', 'enabled'):
            if mode in display:
                display[mode] = {k:v for k,v in display[mode].items() if k != 'rows'}
    print(json.dumps(display, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    main()
