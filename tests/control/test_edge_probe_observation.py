"""CAP867 weak edge probe must not interrupt the frozen-direction search."""
from dataclasses import replace
from pathlib import Path
import configparser

import pytest

from car_control_modular.search_candidate_gate import (
    CandidateObservation, SearchCandidateGate, SearchCandidateGateConfig,
)
from test_search_observation_arbitration import owner, NOW


BOX = (0., 137.1864776611328, 66.5595474243164, 412.0030517578125)


def make_gate(**changes):
    return SearchCandidateGate(replace(SearchCandidateGateConfig(
        probe_edge_min_score=.20, probe_edge_center_ratio=.15,
        probe_confirm_frames=2, hold_frames=2,
    ), **changes))


def tick(gate, items, t=1., **kw):
    return gate.update(timestamp=t, search_active=True, width=640, height=480,
                       probe_candidates=tuple(items), **kw)


@pytest.mark.parametrize('box', [BOX, (573.44, 137., 640., 412.)])
@pytest.mark.parametrize('score', [.10, .116791822, .1999])
def test_repeated_weak_edge_probe_cannot_enter_or_accumulate_hold(box, score):
    gate = make_gate()
    item = CandidateObservation(box, score)
    for i in range(5):
        decision = tick(gate, [item], 1.+i*.05)
        assert decision.reason == 'edge_low_score_probe_ignored'
        assert not decision.pause_rotation and not decision.entered
        assert decision.bbox is None and not gate.hold_active
        assert gate._probe_streak == 0
        assert gate.last_ignored_edge_probes == (item,)
    assert gate.select_current_candidate(width=640, height=480,
        probe_candidates=(item,)).bbox is None


@pytest.mark.parametrize('score', [.20, .249])
def test_score_boundary_is_not_globally_rejected(score):
    gate = make_gate()
    item = CandidateObservation(BOX, score)
    first = tick(gate, [item])
    assert first.reason == 'probe_candidate_confirming' and not first.pause_rotation
    assert tick(gate, [item], 1.05).entered


@pytest.mark.parametrize('box', [(220., 80., 350., 440.), (200., 0., 440., 479.)])
def test_central_weak_partial_person_still_gets_two_frame_observation(box):
    gate = make_gate()
    item = CandidateObservation(box, .117)
    assert not tick(gate, [item]).pause_rotation
    assert tick(gate, [item], 1.05).pause_rotation


def test_ignored_edge_does_not_hide_another_eligible_candidate():
    gate = make_gate()
    edge = CandidateObservation(BOX, .19)
    middle = CandidateObservation((250., 80., 370., 400.), .12)
    assert tick(gate, [edge, middle]).bbox == middle.bbox
    second = tick(gate, [edge, middle], 1.05)
    assert second.entered and second.bbox == middle.bbox


def test_strong_identity_preferred_probe_keeps_fast_observation_not_identity_claim():
    gate = make_gate()
    item = CandidateObservation(BOX, .117)
    decision = tick(gate, [item], preferred_bbox=BOX)
    assert decision.entered and decision.pause_rotation
    assert not decision.completed  # Only entry to bounded observation, not UID confirmation.
    assert not gate.last_ignored_edge_probes


def test_unrelated_preferred_box_does_not_exempt_weak_edge():
    gate = make_gate()
    result = tick(gate, [CandidateObservation(BOX, .117)],
                  preferred_bbox=(400., 80., 600., 450.))
    assert not result.pause_rotation and result.bbox is None


def test_probe_degrading_at_edge_releases_only_its_own_hold(owner):
    gate = make_gate()
    owner._search_candidate_gate = gate
    item = CandidateObservation(BOX, .22)
    tick(gate, [item], NOW-.1)
    entered = tick(gate, [item], NOW-.05)
    owner._apply_search_candidate_gate_decision(entered, prepare_only=True)
    assert owner._search_evidence_observation_active
    released = tick(gate, [CandidateObservation(BOX, .117)], NOW)
    assert released.completed and not released.pause_rotation
    owner._apply_search_candidate_gate_decision(released, prepare_only=True)
    assert not owner._search_evidence_observation_active
    assert owner.search_direction == 'left' and owner._follow_controller.active_target_id == 7
    assert owner._events == []


def test_weak_edge_cannot_cancel_formal_candidate_hold():
    gate = make_gate()
    first = tick(gate, [], formal_candidates=(CandidateObservation(BOX, .7),))
    assert first.entered
    second = tick(gate, [CandidateObservation(BOX, .117)], 1.05)
    assert second.source == 'formal' and second.reason == 'formal_candidate_observe_hold'


@pytest.mark.parametrize('second', [(450., 80., 570., 400.), (220., 80., 420., 440.)])
def test_two_frames_require_position_and_size_continuity(second):
    gate = make_gate()
    tick(gate, [CandidateObservation((250., 100., 330., 300.), .22)])
    result = tick(gate, [CandidateObservation(second, .22)], 1.05)
    assert not result.pause_rotation and result.probe_streak == 1


def test_single_edge_probe_does_not_send_stop_or_change_direction(owner):
    gate = make_gate()
    owner._search_candidate_gate = gate
    decision = tick(gate, [CandidateObservation(BOX, .117)], NOW)
    owner._apply_search_candidate_gate_decision(decision)
    assert owner._events == []
    assert not owner._search_evidence_pause_current_frame
    assert owner.search_direction == 'left'
    assert owner._follow_controller.active_target_id == 7
    # The filter does not bypass the separate runtime hazard path.
    owner._handle_hazard_safety_state = lambda _: owner._events.append(('hazard',)) or True
    owner._consume_track_records([], 640, 480, 'test')
    assert owner._events == [('hazard',)]


@pytest.mark.parametrize('name', ['reid_runtime.ini', 'reid_runtime_rotation_only.ini'])
def test_shipped_profiles_enable_observation_only_thresholds(name):
    parser = configparser.ConfigParser()
    parser.read(Path(__file__).resolve().parents[2] / 'car_control_modular/config' / name)
    assert parser.getfloat('follow', 'search_evidence_probe_edge_min_score') == .20
    assert parser.getfloat('follow', 'search_evidence_probe_edge_center_ratio') == .15
    assert parser.getint('follow', 'search_evidence_probe_confirm_frames') == 2
