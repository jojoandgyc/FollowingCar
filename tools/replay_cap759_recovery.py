#!/usr/bin/env python3
"""Bounded CAP759--827 identity-policy replay; no model or hardware access.

The saved log supplies capture times, boxes, yaw, competition, and recent
appearance distances. Synthetic unit vectors reproduce those distance inputs
against approved CAP258/CAP335 full and CAP258 torso references. They are
not extracted embeddings and do not reproduce the complete historical gallery.
The actual IdentityBank.assign path replays CAP779--826 from the recorded
CAP773 mapping and CAP769 quarantine, keeping the protected CAP613 anchor.
The optional --pose-checkpoint restores the confirmed partial continuation
lineage at CAP773 and starts at CAP777. Neither scope replays robot motion that
another identity policy would have produced.

The historical A/B isolates the CAP759 pose/recovery changes: similar-follow,
introduced later for CAP334, is disabled on both sides. Pass
--current-runtime-policy to additionally evaluate the unmodified runtime
configuration; that separate result is not an isolated historical A/B.
"""
from __future__ import annotations

import argparse
import configparser
import copy
from dataclasses import fields, replace
import hashlib
import json
import logging
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rk_vision.identity_bank import IdentityBank, IdentityBankConfig, _geometry_observation

FULL_REFERENCE = 335
PARTIAL_REFERENCE = 258
ARM_CAP = 769
CHECKPOINT = 773
FIRST_QUERY = 779
LAST_QUERY = 826  # No CAP827 saved crop exists in this recording.
REQUIRED_CAPS = (258, 335, 613, 769, 773, 777, 779, 792, 796, 798, 801, 803, 818, 822, 826)


class DisabledPartialConflictRecoveryBank(IdentityBank):
    """Change only the new recovery eligibility; retain all other policy."""

    def _can_recover_partial_conflict(self, *args, **kwargs):
        return False


class DisabledPoseAndRecoveryBank(DisabledPartialConflictRecoveryBank):
    """Baseline for the separate confirmed pose-lineage checkpoint."""

    def _evaluate_pose_follow(self, *args, **kwargs):
        return None


def metadata(event):
    return dict(copy.deepcopy(event['sample_metadata']), frame_index=event['frame_index'])


def load_events(run_dir):
    events = [json.loads(line) for line in
              (Path(run_dir)/'reid_diagnostics/events.jsonl').open()]
    selected = {}
    for event in events:
        cap = event['capture_frame_id']
        if cap not in REQUIRED_CAPS and not FIRST_QUERY <= cap <= LAST_QUERY:
            continue
        if cap >= ARM_CAP and event['raw_track_id'] != 5:
            continue
        if cap in selected:
            raise ValueError(f'ambiguous saved CAP{cap} target')
        selected[cap] = event
    if not set(REQUIRED_CAPS).issubset(selected):
        raise ValueError('missing CAP759 recovery checkpoints or observations')
    for cap in (FULL_REFERENCE, PARTIAL_REFERENCE):
        if selected[cap]['assignment'].get('bank_updated') is not True:
            raise ValueError(f'CAP{cap}: reference was not approved')
    if selected[PARTIAL_REFERENCE]['assignment'].get('partial_template_update_reason') != 'trusted_paired_update':
        raise ValueError('CAP258: missing approved torso source')
    if (selected[ARM_CAP]['uid'] != 1
            or selected[ARM_CAP]['assignment'].get('template_quarantine_reason') != 'armed'
            or selected[CHECKPOINT]['uid'] != 1
            or selected[CHECKPOINT]['assignment'].get('reason') != 'skip_update_reacquire_quarantine'
            or selected[FIRST_QUERY]['assignment'].get('reason') != 'recent_partial_conflict'):
        raise ValueError('recording does not contain the expected CAP769/773/779 checkpoints')
    reference = (selected[803]['assignment'].get('reacquire_geometry') or {}).get('reference')
    if (not reference or reference.get('capture_frame_id') != 613
            or reference.get('capture_timestamp') != selected[613]['capture_timestamp']):
        raise ValueError('missing original CAP613 protected geometry')
    return selected


def bank_config(path):
    parser = configparser.ConfigParser(interpolation=None)
    if not parser.read(path):
        raise ValueError(f'missing configuration: {path}')
    defaults = IdentityBankConfig()
    options = {}
    for field in fields(defaults):
        if parser.has_option('identity_bank', field.name):
            value = getattr(defaults, field.name)
            getter = (parser.getboolean if isinstance(value, bool) else
                      parser.getint if isinstance(value, int) else parser.getfloat)
            options[field.name] = getter('identity_bank', field.name)
    return IdentityBankConfig(**options)


def feature_at_distance(distance):
    """Cosine distance from [1, 0], not an inferred appearance feature."""
    import numpy as np
    if distance is None or not math.isfinite(distance) or not 0 <= distance <= 2:
        raise ValueError('replay requires a finite recorded cosine distance')
    cosine = 1. - float(distance)
    return np.asarray([cosine, math.sqrt(max(0., 1. - cosine*cosine))], dtype='float32')


def full_feature_at_distance(distance, paired_distance=None):
    """Synthetic query distances to two distinct approved full references.

    The approved CAP258 source retains both full and torso descriptors so the
    existing continuation pair check can run. A .03 difference between the
    references prevents the real memory from collapsing them as duplicates.
    """
    import numpy as np
    feature_at_distance(distance)  # Validate finite cosine-distance input.
    if paired_distance is None:
        paired_distance = distance
    feature_at_distance(paired_distance)
    x = 1. - float(distance)
    y = ((1. - float(paired_distance)) - .97*x) / math.sqrt(1. - .97**2)
    residual = 1. - x*x - y*y
    if residual < -1e-8:
        raise ValueError('recorded distance cannot fit the bounded reference geometry')
    return np.asarray([x, y, math.sqrt(max(0., residual))], dtype='float32')


def checkpoint_bank(config, events, bank_class=IdentityBank, *, pose_checkpoint=False):
    import numpy as np
    bank = bank_class(config)
    full, torso = events[FULL_REFERENCE], events[PARTIAL_REFERENCE]
    if (full['assignment'].get('bank_updated') is not True
            or torso['assignment'].get('bank_updated') is not True
            or torso['assignment'].get('partial_template_update_reason') != 'trusted_paired_update'):
        raise ValueError('refuse unapproved reference seed')
    # Preserve capture order: the memory watermark forbids inserting CAP258
    # after CAP335. Neither source is a query or an unapproved observation.
    seed = np.asarray([.97, math.sqrt(1. - .97**2), 0.], dtype='float32')
    bank._create_identity(seed, torso['frame_index'], metadata(torso), feature_at_distance(0.))
    bank.identities[1].add(np.asarray([1., 0., 0.], dtype='float32'), full['frame_index'],
        config.max_features, config.diversity_min_distance,
        config.diversity_replace_margin, metadata(full))
    checkpoint = events[CHECKPOINT]
    bank.identities[1].last_strong_observation = _geometry_observation(
        metadata(checkpoint), checkpoint['frame_index'])
    bank.identities[1].last_seen_frame = checkpoint['frame_index']
    bank.track_to_uid[5] = 1
    bank._remember_track_seen(5, 1, checkpoint['frame_index'])
    arm = events[ARM_CAP]
    bank._quarantine_decisions[1] = bank._reacquire_quarantine.arm(
        1, 5, ARM_CAP, arm['capture_timestamp'], arm['frame_index'])
    bank._reacquire_search_anchors[1] = copy.deepcopy(
        events[803]['assignment']['reacquire_geometry']['reference'])
    if pose_checkpoint:
        # Restored checkpoint, not freshly earned confirmation. CAP769's
        # partial handoff is the original lineage through accepted CAP773.
        if arm['assignment'].get('authorization_match_source') != 'partial':
            raise ValueError('CAP769: missing confirmed partial continuation lineage')
        evidence = checkpoint['assignment']['reacquire_recent_partial_evidence']
        bank._appearance_verified[1] = dict(metadata=metadata(checkpoint),
            comparable_caps=copy.deepcopy(evidence['comparable_caps']),
            comparison_mode=evidence['comparison_mode'], pending_used=False,
            scale_started=checkpoint['capture_timestamp'], continuation_source='partial',
            continuation_origin_cap=ARM_CAP)
    return bank


def replay(config, events, *, bank_class=IdentityBank, caps=None,
           variant=None, pose_checkpoint=False, current_runtime_policy=False):
    """Run assign; pin the historical A/B unless current policy is requested.

    Do not change bank_config: other tools use the current runtime settings.
    Copying here also leaves callers' configuration intact.
    """
    policy_config = config if current_runtime_policy else replace(config, similar_follow_enable=False)
    bank = checkpoint_bank(policy_config, events, bank_class, pose_checkpoint=pose_checkpoint)
    if variant == 'retained_contradiction':
        bank._mapped_geometry_conflicts[5] = dict(
            uid=1, search_contradiction=True,
            reference=copy.deepcopy(bank._reacquire_search_anchors[1]),
            rejected_frame=events[FIRST_QUERY]['frame_index'])
    if caps is None:
        first = 777 if pose_checkpoint else FIRST_QUERY
        caps = sorted(c for c in events if first <= c <= LAST_QUERY)
    rows = []
    for cap in caps:
        event = events[cap]
        m, box = metadata(event), event['detector_bbox']
        full = event['assignment']['template_recent_evidence']['distance']
        partial = event['assignment']['reacquire_recent_partial_evidence']['distance']
        recorded_pair = event['assignment'].get('identity_continuation_pair') or {}
        paired_full = (recorded_pair.get('full_distance')
                       if recorded_pair.get('winner_cap') == PARTIAL_REFERENCE else None)
        if variant == 'nonunique' and cap != FIRST_QUERY:
            m['candidate_count'] = 2
            m['identity_competition'].update(candidate_count=2, passed=False, reason='ambiguous')
        elif variant == 'full_bad' and cap != FIRST_QUERY:
            full = .31
            paired_full = None
        elif variant == 'full_box':
            m['partial_observation'] = False
        uid = bank.assign(track_id=5, feature=full_feature_at_distance(full, paired_full),
            partial_feature=feature_at_distance(partial), confidence=m['detector_confidence'],
            area=(box[2]-box[0])*(box[3]-box[1]), frame_index=event['frame_index'],
            candidate_count=m['candidate_count'], bbox_quality_ok=m['quality_bbox_ok'],
            bbox_quality_reason=m.get('bbox_quality_reason', ''),
            bbox_quality_tier=m['bbox_quality_tier'], sample_metadata=m,
            preferred_uid=1 if m.get('search_reacquire_context_active') else None,
            preferred_candidate_ok=bool(m.get('search_reacquire_context_active'))
                and m.get('search_direction_compatible') is not False)
        assignment = bank.last_assignments[5]
        rows.append(dict(cap=cap, uid=uid, reason=assignment['reason'],
            recorded_uid=event['uid'], recorded_reason=event['assignment']['reason'],
            recorded_full=event['assignment']['template_recent_evidence']['distance'],
            input_full=full, recorded_partial=partial,
            recorded_pair_full=paired_full,
            logged_search=bool(event['sample_metadata'].get('search_reacquire_context_active')),
            evaluated_full=(assignment.get('template_recent_evidence') or {}).get('distance'),
            evaluated_partial=(assignment.get('reacquire_recent_partial_evidence') or {}).get('distance'),
            continuation=assignment.get('identity_continuation'),
            pose_continuation=assignment.get('identity_pose_continuation'),
            similar_follow=assignment.get('similar_follow'),
            recovery_streak=assignment.get('reacquire_control_recovery_streak'),
            recovery_source=assignment.get('reacquire_control_recovery_source'),
            geometry_source=assignment.get('reacquire_control_geometry_source'),
            protected_cap=assignment.get('protected_search_anchor_cap'),
            quarantined=assignment.get('template_update_quarantined'),
            bank_updated=assignment.get('bank_updated'),
            recent_updated=assignment.get('recent_bank_updated'),
            learning_written_tiers=assignment.get('learning_written_tiers', [])))
    return dict(similar_follow_enabled=bool(policy_config.similar_follow_enable),
        accepted_caps=[r['cap'] for r in rows if r['uid'] == 1],
        gallery_write_caps=[r['cap'] for r in rows if r['bank_updated']
                            or r['recent_updated'] or r['learning_written_tiers']], rows=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path,
        default=ROOT/'car_control_modular/config/reid_runtime.ini')
    parser.add_argument('--pose-checkpoint', action='store_true',
        help='also restore the accepted CAP773 partial lineage and replay from CAP777')
    parser.add_argument('--current-runtime-policy', action='store_true',
        help='also report current runtime configuration, without historical A/B isolation')
    args = parser.parse_args(argv)
    events = load_events(args.run_dir)
    config = bank_config(args.config)
    log = args.run_dir/'reid_diagnostics/events.jsonl'
    previous_disable = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        result = dict(scope='bounded policy replay using synthetic distances and saved metadata; '
                           'no crop inference, complete-gallery reconstruction, control, or motor replay',
            scope_notes=dict(
                policy='historical baseline/current A/B disables similar_follow on both sides; '
                       'optional current_runtime_policy result keeps the input configuration',
                search='recorded search flags remain unchanged, including search after a newly accepted frame; '
                       'this does not simulate changed control decisions, yaw, or crops',
                features='recorded recent distance minima and available CAP258 paired full distances are reproduced; '
                         'missing pair distances use the recent minimum as a synthetic policy input, not measured evidence',
                checkpoint='CAP773 mapping and original CAP769 quarantine are restored; '
                           'optional pose mode explicitly restores the confirmed partial continuation lineage'),
            checkpoint_cap=CHECKPOINT, first_query=FIRST_QUERY, last_query=LAST_QUERY,
            run_dir=str(args.run_dir.resolve()),
            events_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
            config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
            policy_sha256=hashlib.sha256((ROOT/'rk_vision/identity_bank.py').read_bytes()).hexdigest(),
            baseline=replay(config, events, bank_class=DisabledPartialConflictRecoveryBank),
            current=replay(config, events))
        result['negative_variants'] = {name: replay(config, events, caps=caps, variant=variant)
            for name, caps, variant in (
                ('nonunique', [779, 801, 803], 'nonunique'),
                ('retained_contradiction', [779, 801, 803], 'retained_contradiction'),
                ('duplicate', [779, 801, 801], None),
                ('out_of_order', [779, 803, 801], None),
                ('full_bad', [779, 801, 803], 'full_bad'))}
        if args.pose_checkpoint:
            result['pose_checkpoint'] = dict(
                scope='restored CAP773 confirmed partial continuation source from CAP769; '
                      'synthetic recorded distances; not a full-run or motion replay',
                baseline=replay(config, events, bank_class=DisabledPoseAndRecoveryBank,
                                pose_checkpoint=True),
                current=replay(config, events, pose_checkpoint=True))
        if args.current_runtime_policy:
            result['current_runtime_policy'] = replay(config, events,
                pose_checkpoint=args.pose_checkpoint, current_runtime_policy=True)
    finally:
        logging.disable(previous_disable)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    main()
