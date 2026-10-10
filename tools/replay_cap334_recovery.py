#!/usr/bin/env python3
"""Replay CAP334--388 identity policy without inference, camera, or motor I/O.

Only CAP19 is seeded into the gallery. Synthetic unit vectors reproduce each
logged full/torso cosine distance to that source, not actual OSNet embeddings.
The last saved accepted CAP271 supplies the checkpoint geometry. This is not
a replay of unsaved frames, feature extraction, or the robot's changed motion.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rk_vision.identity_bank import IdentityBank, _geometry_observation
from tools.replay_cap759_recovery import bank_config, feature_at_distance, metadata

REFERENCE = 19
CHECKPOINT = 271
FIRST_QUERY = 334
LAST_QUERY = 388
REQUIRED_CAPS = (REFERENCE, CHECKPOINT, FIRST_QUERY, LAST_QUERY)


class DisabledSimilarFollowBank(IdentityBank):
    """The A/B baseline disables only the new similar-follow entry point."""

    def _evaluate_similar_follow(self, *args, **kwargs):
        return None


def load_events(run_dir):
    selected = {}
    with (Path(run_dir)/'reid_diagnostics/events.jsonl').open() as stream:
        for line in stream:
            event = json.loads(line)
            cap = event['capture_frame_id']
            if cap not in REQUIRED_CAPS and not FIRST_QUERY <= cap <= LAST_QUERY:
                continue
            if FIRST_QUERY <= cap <= LAST_QUERY and event['raw_track_id'] != 3:
                continue
            if cap in selected:
                raise ValueError(f'ambiguous CAP{cap} saved target')
            selected[cap] = event
    if not set(REQUIRED_CAPS).issubset(selected):
        raise ValueError('missing CAP19/271 checkpoints or CAP334/388 observations')
    reference, checkpoint = selected[REFERENCE], selected[CHECKPOINT]
    if (reference['uid'] != 1 or reference['assignment'].get('bank_updated') is not True
            or reference['assignment'].get('reason') != 'created_confirmed'
            or not {'recent_strong', 'recent_partial'}.issubset(
                reference['assignment'].get('learning_written_tiers', ()))):
        raise ValueError('CAP19 is not the approved paired initial identity')
    if checkpoint['uid'] != 1 or checkpoint['raw_track_id'] != 1:
        raise ValueError('CAP271 is not the saved accepted source track')
    first = selected[FIRST_QUERY]
    reference_age = first['assignment'].get('reacquire_reference_age_sec')
    if (reference_age is None or abs(first['capture_timestamp']-reference_age
            - checkpoint['capture_timestamp']) > 1e-6):
        raise ValueError('CAP334 does not reference the saved CAP271 timestamp')
    for cap, event in selected.items():
        if FIRST_QUERY <= cap <= LAST_QUERY:
            if event['sample_metadata'].get('capture_frame_id') != cap:
                raise ValueError(f'CAP{cap}: capture metadata mismatch')
            feature_at_distance(event['assignment']['template_recent_evidence']['distance'])
            feature_at_distance(event['assignment']['partial_distance'])
    return selected


def checkpoint_bank(config, events, bank_class=IdentityBank):
    reference, checkpoint = events[REFERENCE], events[CHECKPOINT]
    if reference['assignment'].get('bank_updated') is not True or checkpoint['uid'] != 1:
        raise ValueError('refuse an unapproved replay checkpoint')
    bank = bank_class(config)
    bank._create_identity(feature_at_distance(0.), reference['frame_index'],
                          metadata(reference), feature_at_distance(0.))
    entry = bank.identities[1]
    entry.last_strong_observation = _geometry_observation(
        metadata(checkpoint), checkpoint['frame_index'])
    entry.last_seen_frame = checkpoint['frame_index']
    bank.track_to_uid[1] = 1
    bank._remember_track_seen(1, 1, checkpoint['frame_index'])
    return bank


def gallery_snapshot(bank):
    """Count every retained archive/recent/representative tier, not crop files."""
    rows = {}
    for uid, entry in bank.identities.items():
        memory = entry.template_memory
        tiers = {
            'archive_strong': entry.feature_metadata,
            'archive_weak': entry.weak_feature_metadata,
            'archive_partial': entry.partial_feature_metadata,
        }
        if memory is not None:
            tiers.update({'recent_' + tier: [m for _, m in values]
                          for tier, values in memory.recent.items()})
            tiers.update({'representative_' + tier: [m for _, m in values]
                          for tier, values in memory.representatives.items()})
        rows[str(uid)] = {tier: dict(count=len(values),
            captures=[m.get('capture_frame_id') for m in values])
            for tier, values in tiers.items()}
    return rows


def replay(config, events, *, bank_class=IdentityBank, caps=None,
           logged_context=False, variant=None):
    """Use real assign calls; optional changes are labelled negative probes.

    Default: after acceptance, the next saved observation is evaluated outside
    search. Rejection re-enters search on the next sample. This simulates only
    the identity/search handshake, never altered wheel speed, yaw, or imagery.
    --logged-context leaves every recorded search flag unchanged for a second
    useful policy check even after the new code accepts the candidate.
    """
    bank = checkpoint_bank(config, events, bank_class)
    before = gallery_snapshot(bank)
    if variant == 'retained_contradiction':
        bank._mapped_geometry_conflicts[3] = dict(uid=1, search_contradiction=True,
            reference=copy.deepcopy(bank.identities[1].last_strong_observation),
            rejected_frame=events[FIRST_QUERY]['frame_index'])
    caps = sorted(c for c in events if FIRST_QUERY <= c <= LAST_QUERY) if caps is None else caps
    following = False
    rows = []
    for ordinal, cap in enumerate(caps):
        event = events[cap]
        m, box = metadata(event), event['detector_bbox']
        recorded_search = bool(m.get('search_reacquire_context_active'))
        if not logged_context:
            m['search_reacquire_context_active'] = not following
            if following:
                m.update(search_direction=None, search_direction_compatible=None)
        track_id = 3
        if variant == 'explicit_conflict':
            m.update(quality_bbox_ok=False, bbox_quality_tier='reject',
                quality_bbox_reason='identity_swap_competing_track',
                bbox_quality_reason='identity_swap_competing_track')
        elif variant == 'competition_failed':
            m['identity_competition'].update(passed=False, reason='ambiguous')
        elif variant == 'nonunique':
            m['candidate_count'] = 2
            m['identity_competition'].update(candidate_count=2, passed=False, reason='ambiguous')
        elif variant == 'new_track_each_frame':
            track_id = 30 + ordinal
        full = event['assignment']['template_recent_evidence']['distance']
        partial = event['assignment']['partial_distance']
        if variant == 'full_bad':
            full = .70
        elif variant == 'stale':
            m['is_fresh'] = False
        uid = bank.assign(track_id=track_id, feature=feature_at_distance(full),
            partial_feature=feature_at_distance(partial), confidence=m['detector_confidence'],
            area=(box[2]-box[0])*(box[3]-box[1]), frame_index=event['frame_index'],
            candidate_count=m['candidate_count'], bbox_quality_ok=m['quality_bbox_ok'],
            bbox_quality_reason=m.get('bbox_quality_reason', ''),
            bbox_quality_tier=m['bbox_quality_tier'], sample_metadata=m,
            preferred_uid=1 if m.get('search_reacquire_context_active') else None,
            preferred_candidate_ok=bool(m.get('search_reacquire_context_active'))
                and m.get('search_direction_compatible') is not False)
        assignment = bank.last_assignments[track_id]
        following = uid == 1
        rows.append(dict(cap=cap, uid=uid, reason=assignment['reason'],
            recorded_uid=event['uid'], recorded_reason=event['assignment']['reason'],
            recorded_full=event['assignment']['template_recent_evidence']['distance'],
            input_full=full, recorded_partial=partial,
            logged_search=recorded_search,
            evaluated_search=bool(m.get('search_reacquire_context_active')),
            evaluated_full=(assignment.get('template_recent_evidence') or {}).get('distance'),
            evaluated_partial=assignment.get('partial_distance'),
            similar_follow=assignment.get('similar_follow'),
            authorization_source=assignment.get('authorization_match_source'),
            bank_updated=assignment.get('bank_updated'),
            recent_updated=assignment.get('recent_bank_updated'),
            learning_written_tiers=assignment.get('learning_written_tiers', [])))
    return dict(accepted_caps=[r['cap'] for r in rows if r['uid'] == 1],
        gallery_write_caps=[r['cap'] for r in rows if r['bank_updated']
                            or r['recent_updated'] or r['learning_written_tiers']],
        gallery_before=before, gallery_after=gallery_snapshot(bank), rows=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path,
        default=ROOT/'car_control_modular/config/reid_runtime.ini')
    parser.add_argument('--logged-context', action='store_true')
    args = parser.parse_args(argv)
    events, config = load_events(args.run_dir), bank_config(args.config)
    previous_disable = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        result = dict(scope='identity-policy replay using synthetic distance-calibrated vectors; '
            'not crop/NPU inference, full execution history, or closed-loop motor replay',
            scope_notes=dict(
                gallery='all gallery tiers are counted; only the approved CAP19 full/torso pair is seeded; '
                        'CAP271 restores saved accepted geometry, not a new template',
                checkpoint='CAP271 timestamp matches the recorded CAP334 reference age; '
                           'unsaved identity states are not reconstructed',
                features='logged distances to CAP19 are exact; query-to-query vector similarity is synthetic '
                         'and cannot establish real visual continuity',
                control=('recorded search flags retained even after new acceptance' if args.logged_context else
                         'accepted UID exits search on the next frame; rejection re-enters it on the next frame')
                        + '; saved boxes, timestamps, yaw and crops never change with this simulated handshake'),
            reference_cap=REFERENCE, checkpoint_cap=CHECKPOINT,
            first_query=FIRST_QUERY, last_query=LAST_QUERY,
            run_dir=str(args.run_dir.resolve()),
            events_sha256=hashlib.sha256((args.run_dir/'reid_diagnostics/events.jsonl').read_bytes()).hexdigest(),
            config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
            policy_sha256=hashlib.sha256((ROOT/'rk_vision/identity_bank.py').read_bytes()).hexdigest(),
            baseline=replay(config, events, bank_class=DisabledSimilarFollowBank,
                            logged_context=args.logged_context),
            current=replay(config, events, logged_context=args.logged_context))
        negative_caps = sorted(c for c in events if FIRST_QUERY <= c <= LAST_QUERY)[:3]
        result['negative_variants'] = {name: replay(config, events, caps=negative_caps,
            logged_context=args.logged_context, variant=name) for name in
            ('explicit_conflict', 'retained_contradiction', 'competition_failed',
             'nonunique', 'new_track_each_frame', 'full_bad', 'stale')}
        result['negative_variants']['duplicate'] = replay(config, events,
            caps=[negative_caps[0], negative_caps[0]], logged_context=args.logged_context)
        result['negative_variants']['out_of_order'] = replay(config, events,
            caps=[negative_caps[1], negative_caps[0]], logged_context=args.logged_context)
        cross_caps = [c for c in sorted(events) if FIRST_QUERY <= c <= LAST_QUERY
            and events[c]['sample_metadata'].get('search_direction_compatible') is False][:3]
        if cross_caps:
            result['negative_variants']['new_opposite_candidate'] = replay(config, events,
                caps=cross_caps, logged_context=True)
    finally:
        logging.disable(previous_disable)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    main()
