"""Sub-percent parked PI output can request release, never motor authority."""
from dataclasses import replace

import pytest

from test_cap825_park_resume_runtime import parked, release, authority, setup, owner
from test_depth_authority_250 import advance, decide_commit


def subquantum_preview(a, rpm=1.):
    current = a.frame(2.3, rpm=rpm, capture_timestamp=a.clock.now-.04,
                      capture_frame_id=1924)
    a.feedback = current.steering_feedback
    a.owner._last_command_capture_frame = current.capture_frame_id
    a.owner._last_command_capture_timestamp = current.capture_timestamp
    a.owner._active_capture_frame_id = current.capture_frame_id
    decision = a.controller.decide(current.capture_frame_id, current, longitudinal_only=False)
    assert a.controller.last_distance_pid_result.pi_status == 'parked_preview'
    return current, decision


@pytest.mark.parametrize('rpm', [1., 1.5])
def test_one_rpm_preview_releases_park_without_reusing_it_as_a_grant(parked, rpm):
    a = parked
    current, decision = subquantum_preview(a, rpm)
    result = a.controller.last_distance_pid_result
    assert result.output_rpm == 1
    assert all(action.speed_percent == 0 for action in decision.actions)
    assert a.controller.parked_forward_resume_demand_rpm(current, 1) > 2
    assert release(a, current, decision)
    assert a.owner._near_yaw_park_request is None
    assert not a.controller._distance_pid._distance_pi._normal_parking
    assert a.owner._depth30_linear_snapshot is None
    assert not a.backend.pairs

    # The exact parked sample stays at zero percent even after release.
    a.owner._refresh_visual_depth_linear_authority(
        current, current.persons[0], decision, is_fresh_depth=True,
        target_steerable=True, low_quality_visible=False)
    assert a.owner._depth30_linear_snapshot is None
    assert a.controller.last_distance_pid_result is result
    assert a.controller._distance_pid._distance_pi._execution_suspended

    # Fresh post-release samples still start the ordinary measured ramp.
    approved = []
    for n in range(1, 3):
        advance(a, current.distance_state.sample_timestamp+n*.05)
        _, actions, _ = decide_commit(a, a.frame(2.3, rpm=0.))
        approved.append(max([x.speed_percent*2 for x in actions if x.kind == 'forward'] or [0]))
    assert approved[0] == 0 < approved[1] <= 12
    assert not a.backend.pairs


@pytest.mark.parametrize('case', [
    'old_depth', 'wrong_uid', 'hazard', 'raw_near', 'current_not_released',
    'other_brake_owner', 'nan_output', 'negative_output',
])
def test_subquantum_preview_retains_release_and_provenance_gates(parked, case):
    a = parked
    current, decision = subquantum_preview(a)
    if case == 'old_depth':
        a.clock.now += .181
    elif case == 'wrong_uid':
        current = replace(current, persons=[replace(current.persons[0], track_id=2)])
    elif case == 'hazard':
        current = replace(current, hazard=replace(current.hazard, active=True))
    elif case == 'raw_near':
        current = replace(current, distance_state=replace(current.distance_state, raw_distance_m=1.3))
    elif case == 'current_not_released':
        a.evidence.current_released_at = None
    elif case == 'other_brake_owner':
        a.owner._brake_hold_label = 'danger'
    else:
        a.controller.last_distance_pid_result = replace(
            a.controller.last_distance_pid_result,
            output_rpm=float('nan') if case == 'nan_output' else -1)
    assert not release(a, current, decision)
    assert a.owner._near_yaw_park_request is not None
    assert a.owner._depth30_linear_snapshot is None
    assert not a.backend.pairs


def test_executable_preview_keeps_the_existing_translation_release_path(parked):
    a = parked
    current, decision = subquantum_preview(a, 2.)
    assert a.controller.last_distance_pid_result.output_rpm == 2
    assert any(action.speed_percent == 1 for action in decision.actions)
    assert a.controller.parked_forward_resume_demand_rpm(current, 1) == 0
    assert release(a, current, decision)
    assert a.owner._depth30_linear_snapshot is None
    assert not a.backend.pairs
