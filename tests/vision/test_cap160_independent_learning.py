"""Saved RGB crops re-encoded offline; tests use their distance-preserving projection.

This starts from the six actually approved CAP24--118 templates. It does not
replay skipped detector-only frames or robot motion. Fixture provenance records
the model, crop hashes, extraction settings and differences from live input.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from rk_vision.identity_bank import IdentityBank, IdentityBankConfig, _geometry_observation
from rk_vision.template_learning import TemplateLearningGuard
from rk_vision.template_memory import TemplateMemory


FIXTURE = json.loads((Path(__file__).parent / 'fixtures/cap160_independent_learning.json').read_text())
EVENTS = {(row['cap'], row['raw']): row for row in FIXTURE['events']}
TEMPLATES = (24, 30, 45, 65, 82, 118)


class PreviousLearningBank(IdentityBank):
    def _observe_independent_follow_learning(self, *args, **kwargs):
        return False


def vector(row, tier):
    return np.asarray(row[tier], dtype='float32')


def metadata(row):
    return dict(deepcopy(row['metadata']), frame_index=row['frame'], track_id=row['raw'])


def checkpoint(bank_class=IdentityBank):
    bank = bank_class(IdentityBankConfig(**FIXTURE['identity_config']))
    for cap in TEMPLATES:
        row = EVENTS[cap, 1]
        assert row['label'] == 'STORED'
        if cap == TEMPLATES[0]:
            assert bank._create_identity(vector(row, 'full'), row['frame'], metadata(row),
                                         vector(row, 'partial')) == 1
        else:
            entry = bank.identities[1]
            entry.add(vector(row, 'full'), row['frame'], bank.config.max_features,
                      bank.config.diversity_min_distance,
                      bank.config.diversity_replace_margin, metadata(row))
            entry.add_partial(vector(row, 'partial'), row['frame'],
                              bank.config.partial_max_features,
                              bank.config.diversity_min_distance, metadata(row))
    row = EVENTS[118, 1]
    bank.track_to_uid[1] = 1
    bank._remember_track_seen(1, 1, row['frame'])
    bank.identities[1].last_strong_observation = _geometry_observation(metadata(row), row['frame'])
    return bank


def send_saved(bank, cap, raw=1, *, changes=None):
    row = EVENTS[cap, raw]
    info = metadata(row)
    info.update(changes or {})
    box = info['detector_bbox']
    uid = bank.assign(track_id=raw, feature=vector(row, 'full'),
        partial_feature=vector(row, 'partial'), confidence=info['detector_confidence'],
        area=(box[2]-box[0])*(box[3]-box[1]), frame_index=row['frame'],
        candidate_count=info['candidate_count'], bbox_quality_ok=info['quality_bbox_ok'],
        bbox_quality_tier=info['bbox_quality_tier'],
        bbox_quality_reason=info.get('bbox_quality_reason', ''), sample_metadata=info,
        preferred_uid=1 if info.get('search_reacquire_context_active') else None,
        preferred_candidate_ok=info.get('search_reacquire_context_active') is True)
    return uid, bank.last_assignments[raw]


def source_caps(bank):
    return {m['capture_frame_id'] for m in bank._learning_source_metadata(bank.identities[1])}


def after_recorded_cap143_bridge(bank_class=IdentityBank):
    bank = checkpoint(bank_class)
    for cap in (123, 127, 131, 139, 143):
        send_saved(bank, cap)
    # The actual CAP143 assignment had detector_position_bridge and
    # not_high_quality_strong/streak=0 (events.jsonl:12). Replaying only the
    # saved full crops omits CAP141's detector-only update, so restore that
    # recorded learning-only checkpoint explicitly. Its strong follow state
    # remains accepted, with no pending independent pair or gallery write.
    bank._reacquire_quarantine.reset(1)
    bank._template_learning = TemplateLearningGuard()
    assert source_caps(bank) == set(TEMPLATES)
    return bank


def test_projection_and_logged_identity_template_scores_agree():
    assert max(FIXTURE['provenance']['max_gram_error'].values()) < 1e-7
    bank = checkpoint()
    for cap in (148, 154, 160, 444, 449, 451):
        row = EVENTS[cap, 1 if cap < 200 else 4]
        assert bank.identities[1].distance(vector(row, 'full')) == pytest.approx(row['logged_full'], abs=2e-6)
        assert bank.identities[1].partial_distance(vector(row, 'partial')) == pytest.approx(
            row['logged_partial'], abs=2e-6)


def test_real_recovered_window_updates_both_tiers_without_a_new_follow_pause():
    current = after_recorded_cap143_bridge()
    previous = after_recorded_cap143_bridge(PreviousLearningBank)
    history = []
    for cap in (148, 154, 160):
        uid, assignment = send_saved(current, cap)
        old_uid, _ = send_saved(previous, cap)
        assert uid == old_uid
        history.append((cap, assignment.get('bank_updated'),
                        assignment.get('independent_learning_review'),
                        assignment.get('template_quarantine_reason')))
    assert source_caps(previous) == set(TEMPLATES)
    assert 160 in source_caps(current), history
    assert current.last_assignments[1]['learning_written_tiers'] == ['recent_strong', 'recent_partial']
    assert set(current.last_assignments[1]['template_learning']['parent_caps']).issubset(TEMPLATES)
    assert EVENTS[160, 1]['frame'] % current.config.update_interval != 0
    for tier in ('strong', 'partial'):
        assert current.identities[1].template_memory.last_learning[tier] == (
            EVENTS[160, 1]['metadata']['capture_timestamp'], 160)


@pytest.mark.parametrize('adapted', [False, True])
@pytest.mark.parametrize('cap,raw', [(422, 5), (425, 5), (529, 10), (562, 10)])
def test_real_rejected_people_cannot_enter_independent_learning(cap, raw, adapted):
    bank = after_recorded_cap143_bridge() if adapted else checkpoint()
    if adapted:
        for accepted in (148, 154, 160):
            send_saved(bank, accepted)
        assert 160 in source_caps(bank)
    before = source_caps(bank)
    uid, assignment = send_saved(bank, cap, raw)
    assert uid == 0
    assert not assignment['bank_updated']
    assert source_caps(bank) == before


@pytest.mark.parametrize('risk', [
    {'observed': True, 'risky': True, 'reason': 'overlapping_people'},
    {'observed': False, 'risky': False, 'reason': 'unavailable'},
])
def test_unverified_crop_risk_does_not_collect_learning_proof(risk):
    bank = checkpoint()
    for cap in (123, 127, 131, 139, 143, 148, 154, 160):
        send_saved(bank, cap, changes={'template_learning_risk': risk})
    assert source_caps(bank) == set(TEMPLATES)


def test_real_recovery_pair_transaction_rolls_back_failed_torso(monkeypatch):
    bank = after_recorded_cap143_bridge()
    original = TemplateMemory.remember
    def fail_partial(memory, feature, info, tier):
        if info['capture_frame_id'] == 160 and tier == 'partial':
            return False
        return original(memory, feature, info, tier)
    monkeypatch.setattr(TemplateMemory, 'remember', fail_partial)
    for cap in (148, 154, 160):
        send_saved(bank, cap)
    assert source_caps(bank) == set(TEMPLATES)
    for tier in ('strong', 'partial'):
        assert bank.identities[1].template_memory.last_learning[tier][1] == 118


def test_real_new_samples_remain_pending_until_independent_quarantine_finishes():
    bank = after_recorded_cap143_bridge()
    for cap in (148, 154):
        uid, assignment = send_saved(bank, cap)
        assert uid == 1
        assert bank._reacquire_quarantine.is_held(1)
        assert source_caps(bank) == set(TEMPLATES)
        assert assignment['independent_learning_review']['commit_allowed'] is False
        assert not {148, 154}.intersection(
            bank._template_learning.diagnostics(1)['frozen_parent_caps'])


def test_current_recovery_cannot_commit_using_only_the_follow_reference():
    bank = after_recorded_cap143_bridge()
    wrong = EVENTS[562, 10]
    bank._follow_references[1] = [dict(cap=150.,
        timestamp=EVENTS[154, 1]['metadata']['capture_timestamp'],
        feature=vector(wrong, 'full'), metadata=metadata(wrong))]
    before = source_caps(bank)
    for cap in (148, 154, 160):
        row = EVENTS[cap, 1]
        info = metadata(row)
        pair = bank.identities[1].template_memory.paired_recent_evidence(
            vector(wrong, 'full'), vector(wrong, 'partial'), info)
        assert not pair['qualified']
        bank._observe_independent_follow_learning(bank.identities[1],
            vector(wrong, 'full'), vector(wrong, 'partial'), row['frame'], info,
            {'similar_follow': {}}, independently_verified=pair['qualified'])
    assert source_caps(bank) == before


def test_strict_configured_full_update_limit_still_vetoes_real_recovery():
    bank = after_recorded_cap143_bridge()
    bank.config = replace(bank.config, update_threshold=.01)
    for cap in (148, 154, 160):
        uid, assignment = send_saved(bank, cap)
        assert uid == 1
        assert not assignment['bank_updated']
        assert assignment['independent_learning_review']['eligible'] is False
    assert source_caps(bank) == set(TEMPLATES)
