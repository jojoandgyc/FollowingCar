"""CAP110 full yaw must not inherit CAP105's older measurement-ROI clock.

Only pure planner objects run here. Scalars are from run_20261010_234438:
plan CAP105 (log 1130), CAP110 yaw (1140), expiry STOP (1146).
The physical range deadline never moves; forward permission is independent.
"""
from dataclasses import replace

import pytest

from car_control_modular.short_follow import (
    ShortFollowConfig, ShortFollowController, ShortFollowObservation,
    ShortFollowYawObservation,
)


CAPTURE105 = 47419.574270166
DEPTH105 = 47419.765931233
CAPTURE110 = 47419.843048024
OLD_VISUAL_END = 47420.074270166
DEPTH_END = 47420.115931233
STOP_TIME = 47420.079425729
FULL_YAW = ShortFollowYawObservation(1, 110, CAPTURE110, .7539457321166992,
                                     -9.275275362736451)


def setup(*, distance=1.4801183074265976, raw=1.4974249514563107,
          longitudinal_allowed=True):
    core = ShortFollowController(ShortFollowConfig(enabled=True,
        depth_ttl_sec=.35, yaw_response_exponent=.5))
    assert core.activate(1, 47418.)
    source = ShortFollowObservation(1, 105, CAPTURE105, DEPTH105,
        distance, .6807779788970947, raw)
    plan = core.update(source, 47419.94, longitudinal_allowed=longitudinal_allowed)
    assert plan is not None
    assert plan.expires_at == OLD_VISUAL_END
    return core, plan, source


def counters(core):
    return (core._source_floor, core._last_depth, core._last_capture,
            core._last_capture_id, core._integral_m_s, core._integral_stamp,
            core._unacknowledged_integral, core._parked)


def test_actual_cap110_full_yaw_keeps_pivot_for_remaining_physical_depth_only():
    core, old, _ = setup()
    old_counters = counters(core)
    new = core.update_lateral(FULL_YAW, 47420.0312, allow_yaw_renewal=True)
    assert new is not None and new.pivot and not new.forwarding
    assert new.left_rpm == 7 and new.right_rpm == -7 and new.base_rpm == 0
    assert new.capture_id == 105 and new.capture_timestamp == CAPTURE105
    assert new.depth_timestamp == DEPTH105 and new.depth_expires_at == pytest.approx(DEPTH_END)
    assert new.yaw_capture_id == 110 and new.yaw_capture_timestamp == CAPTURE110
    assert new.longitudinal_expires_at == OLD_VISUAL_END
    assert new.expires_at == pytest.approx(DEPTH_END)
    assert not old.valid(STOP_TIME) and new.valid(STOP_TIME)
    assert (new.expires_at-STOP_TIME)*1000 == pytest.approx(36.505504)
    assert new.valid(DEPTH_END-.000001) and not new.valid(DEPTH_END)
    assert new.epoch == old.epoch and new.sequence > old.sequence
    assert counters(core) == old_counters  # no PI/Depth/source-clock refresh


def test_position_only_default_does_not_extend_old_deadline():
    core, old, _ = setup()
    new = core.update_lateral(FULL_YAW, 47420.0312)
    assert new.pivot and new.yaw_capture_id == 110
    assert new.expires_at == old.expires_at == OLD_VISUAL_END
    assert not new.valid(STOP_TIME)


def test_trusted_yaw_can_arrive_just_after_old_visual_expiry_but_before_physical_expiry():
    core, old, _ = setup()
    assert not old.valid(STOP_TIME)
    new = core.update_lateral(FULL_YAW, STOP_TIME, allow_yaw_renewal=True)
    assert new is not None and new.pivot and new.valid(STOP_TIME)
    assert new.depth_expires_at == old.depth_expires_at
    assert new.longitudinal_expires_at == old.longitudinal_expires_at


def test_current_full_yaw_does_not_extend_a_forward_plan():
    core, old, _ = setup(distance=2.1, raw=2.1)
    assert old.forwarding
    new = core.update_lateral(FULL_YAW, 47420.0312, allow_yaw_renewal=True)
    assert new.forwarding and new.base_rpm == old.base_rpm
    assert new.expires_at == old.expires_at == OLD_VISUAL_END
    assert not new.valid(STOP_TIME)


@pytest.mark.parametrize('renew', [False, True])
def test_explicit_no_translation_downgrades_pair_without_new_depth_or_pi(renew):
    core, old, _ = setup(distance=2.1, raw=2.1)
    old_counters = counters(core)
    new = core.update_lateral(FULL_YAW, 47420.0312,
        allow_yaw_renewal=renew, longitudinal_allowed=False)
    assert new is not None and new.yaw_only and new.pivot and not new.forwarding
    assert new.base_rpm == new.base_request_rpm == 0
    assert new.depth_timestamp == old.depth_timestamp and new.distance_m == old.distance_m
    assert new.longitudinal_expires_at == OLD_VISUAL_END
    assert new.expires_at == pytest.approx(DEPTH_END if renew else OLD_VISUAL_END)
    assert counters(core) == old_counters
    following = core.update_lateral(replace(FULL_YAW, capture_id=111,
        capture_timestamp=47420.04), 47420.05, allow_yaw_renewal=True)
    assert following.yaw_only and following.base_rpm == 0 and not following.forwarding
    # Only a genuinely new, qualified physical range update can restore forward.
    fresh = core.update(ShortFollowObservation(1, 111, 47420.04, 47420.06,
        2.1, .75, 2.1), 47420.07)
    assert fresh.forwarding and not fresh.yaw_only
    assert fresh.depth_timestamp > old.depth_timestamp


def test_later_physical_depth_uses_already_current_full_yaw_without_old_roi_yaw_expiry():
    core = ShortFollowController(ShortFollowConfig(enabled=True, depth_ttl_sec=.35))
    core.activate(1, 47418.)
    assert core.update_lateral(FULL_YAW, 47420.0312, allow_yaw_renewal=True) is None
    source = ShortFollowObservation(1, 105, CAPTURE105, DEPTH105, 1.48, .68, 1.497)
    new = core.update(source, 47420.04)
    assert new.pivot and new.yaw_capture_id == 110
    assert new.expires_at == pytest.approx(DEPTH_END)
    assert new.longitudinal_expires_at == OLD_VISUAL_END
    assert new.depth_timestamp == DEPTH105


def test_cached_position_only_yaw_does_not_extend_a_later_depth_plan():
    core = ShortFollowController(ShortFollowConfig(enabled=True, depth_ttl_sec=.35))
    core.activate(1, 47418.)
    assert core.update_lateral(FULL_YAW, 47420.0312) is None
    new = core.update(ShortFollowObservation(1, 105, CAPTURE105, DEPTH105,
        1.48, .68, 1.497), 47420.04)
    assert new.pivot and new.yaw_capture_id == 110
    assert new.expires_at == OLD_VISUAL_END


def test_first_depth_entry_still_requires_its_original_roi_visual_deadline():
    core = ShortFollowController(ShortFollowConfig(enabled=True, depth_ttl_sec=.35))
    core.activate(1, 47418.)
    assert core.update_lateral(FULL_YAW, STOP_TIME, allow_yaw_renewal=True) is None
    assert core.update(ShortFollowObservation(1, 105, CAPTURE105, DEPTH105,
        1.48, .68), STOP_TIME) is None
    assert core.snapshot().plan is None


@pytest.mark.parametrize('distance,raw', [(1.1, None), (1.05, None), (2., 1.09)])
@pytest.mark.parametrize('longitudinal_allowed', [False, True])
def test_near_physical_boundary_cannot_turn_or_renew(distance, raw, longitudinal_allowed):
    core, old, _ = setup(distance=distance, raw=raw)
    new = core.update_lateral(FULL_YAW, 47420.0312,
        allow_yaw_renewal=True, longitudinal_allowed=longitudinal_allowed)
    assert new is not None and not new.moving
    assert new.distance_m <= 1.1
    assert new.expires_at == old.expires_at == OLD_VISUAL_END
    assert not new.valid(STOP_TIME)


def test_centered_new_yaw_requests_zero_not_an_extended_pivot():
    core, old, _ = setup()
    new = core.update_lateral(replace(FULL_YAW, center_x_ratio=.5),
        47420.0312, allow_yaw_renewal=True)
    assert not new.moving and new.expires_at == old.expires_at


def test_measured_yaw_taper_cannot_extend_deadline_at_executor_read_time():
    core, _, _ = setup()
    new = core.update_lateral(FULL_YAW, 47420.0312, allow_yaw_renewal=True)
    executed = core.execution_plan(new, STOP_TIME,
        current_yaw_deg=FULL_YAW.capture_yaw_deg+20., yaw_rate_deg_s=20.)
    assert not executed.moving
    assert executed.expires_at == new.expires_at
    assert executed.depth_expires_at == new.depth_expires_at
    assert executed.longitudinal_expires_at == new.longitudinal_expires_at
    assert core.snapshot().plan is new


@pytest.mark.parametrize('now', [DEPTH_END, DEPTH_END+.01])
def test_new_yaw_never_renews_expired_physical_depth(now):
    core, old, _ = setup()
    assert core.update_lateral(FULL_YAW, now, allow_yaw_renewal=True) is None
    assert core.snapshot().plan is old and not old.valid(now)
    assert old.depth_timestamp == DEPTH105


@pytest.mark.parametrize('change', [
    {'capture_id': 105}, {'capture_timestamp': CAPTURE105},
    {'capture_timestamp': 47420.2}, {'capture_id': True}, {'uid': 2},
    {'center_x_ratio': float('nan')},
])
def test_invalid_or_out_of_order_yaw_cannot_use_renewal_flag(change):
    core, old, _ = setup()
    assert core.update_lateral(replace(FULL_YAW, **change),
        47420.0312, allow_yaw_renewal=True) is None
    assert core.snapshot().plan is old


def test_replayed_yaw_cannot_renew_or_be_upgraded_from_position_only():
    core, _, _ = setup()
    position = core.update_lateral(FULL_YAW, 47420.0312)
    assert core.update_lateral(FULL_YAW, STOP_TIME, allow_yaw_renewal=True) is None
    assert core.snapshot().plan is position and position.expires_at == OLD_VISUAL_END


@pytest.mark.parametrize('event', ['revoke', 'deactivate', 'brake_wait', 'uid_change', 'expiry'])
def test_cleared_plan_is_never_resurrected_from_hidden_depth_cache(event):
    core, _, source = setup()
    if event == 'revoke': core.revoke('identity_rejected', 47420.02)
    elif event == 'deactivate': core.deactivate('hazard', 47420.02)
    elif event == 'brake_wait': core.wait_for_existing_brake(47420.02)
    elif event == 'uid_change': core.activate(2, 47420.02)
    else: core.expire_observation(STOP_TIME)
    old = core.snapshot()
    floor = core._source_floor
    assert old.plan is None
    assert core.update_lateral(FULL_YAW, STOP_TIME, allow_yaw_renewal=True) is None
    assert core.snapshot() is old and core._source_floor == floor
    # Even a fresh yaw captured after a hard revoke has no physical range
    # permission to restore; source floors still reject the old ROI on update.
    assert core.update_lateral(replace(FULL_YAW, capture_id=112,
        capture_timestamp=47420.08), 47420.09, allow_yaw_renewal=True) is None
    if event in {'revoke', 'deactivate', 'uid_change'}:
        assert core.update(replace(source, depth_timestamp=47420.08), 47420.09) is None
    assert core.snapshot().plan is None


def test_reentrant_publication_guard_cannot_publish_over_hard_revocation():
    core, _, _ = setup()
    def guard():
        core.revoke('obstacle', 47420.0312)
        return True
    assert core.update_lateral(FULL_YAW, 47420.0312,
        allow_yaw_renewal=True, publication_guard=guard) is None
    assert core.snapshot().plan is None and core.snapshot().reason == 'obstacle'
    assert core._source_floor == 47420.0312 and core._latest_yaw is None


@pytest.mark.parametrize('bad', [1, None, 'true'])
@pytest.mark.parametrize('field', ['allow_yaw_renewal', 'longitudinal_allowed'])
def test_permission_flags_require_explicit_booleans(field, bad):
    core, old, _ = setup()
    assert core.update_lateral(FULL_YAW, 47420.0312, **{field: bad}) is None
    assert core.snapshot().plan is old


def test_extended_yaw_deadline_cannot_be_relabelled_as_forward_or_unbounded_depth():
    core, _, _ = setup()
    new = core.update_lateral(FULL_YAW, 47420.0312, allow_yaw_renewal=True)
    assert not replace(new, left_rpm=20, right_rpm=10, base_rpm=20).valid(STOP_TIME)
    assert not replace(new, expires_at=DEPTH_END+10.).valid(DEPTH_END)


def test_limited_cap116_retirement_does_not_reject_already_captured_cap118():
    core, old, _ = setup()
    saved_watermarks = counters(core)[:4]
    retired = core.retire_for_lateral_handoff(47420.30)
    assert not retired.active and retired.plan is None and retired.reason == 'lateral_handoff'
    assert retired.epoch == old.epoch+1
    assert counters(core)[:4] == saved_watermarks
    assert core._latest_yaw is None and core._latest_yaw_renewal is False
    assert core._integral_m_s == 0 and core._integral_stamp is None
    assert core.retire_for_lateral_handoff(47420.40) is retired
    assert counters(core)[:4] == saved_watermarks
    assert core.activate(1, 47420.41)
    # This fresh full frame was captured BEFORE CAP116's handoff callback.
    cap118 = ShortFollowObservation(1, 118, 47420.266545950,
        47420.268606232, 1.4974249514563107, .9188101768493653, 1.4654)
    plan = core.update(cap118, 47420.41)
    assert plan is not None and plan.pivot and not plan.forwarding
    assert plan.capture_timestamp < 47420.30
    assert plan.depth_timestamp == cap118.depth_timestamp
    assert plan.expires_at == pytest.approx(cap118.depth_timestamp+.35)
    assert plan.epoch > old.epoch


@pytest.mark.parametrize('fault', ['duplicate_depth', 'old_capture_id', 'old_capture_timestamp'])
def test_soft_handoff_retains_sample_ordering_watermarks(fault):
    core = ShortFollowController(ShortFollowConfig(enabled=True))
    core.activate(1, 99.)
    old = core.update(ShortFollowObservation(1, 105, 100., 100., 1.56, .8), 100.01)
    saved = counters(core)[:4]
    core.retire_for_lateral_handoff(100.05)
    assert core.activate(1, 100.06)
    assert counters(core)[:4] == saved
    fields = dict(uid=1, capture_id=106, capture_timestamp=100.06,
                  depth_timestamp=100.07, distance_m=1.56, center_x_ratio=.8)
    if fault == 'duplicate_depth': fields['depth_timestamp'] = old.depth_timestamp
    elif fault == 'old_capture_id': fields['capture_id'] = 104
    else: fields['capture_timestamp'] = 99.99
    assert core.update(ShortFollowObservation(**fields), 100.08) is None
    assert core.snapshot().plan is None


@pytest.mark.parametrize('event_order', ['hard_then_lateral', 'lateral_then_hard'])
def test_lateral_retirement_cannot_lower_an_intervening_hard_source_floor(event_order):
    core, _, _ = setup()
    if event_order == 'hard_then_lateral':
        core.revoke('identity_rejected', 47420.30)
        core.retire_for_lateral_handoff(47420.31)
    else:
        core.retire_for_lateral_handoff(47420.29)
        core.deactivate('hazard', 47420.30)
    assert core._source_floor == 47420.30
    core.activate(1, 47420.41)
    assert core.update(ShortFollowObservation(1, 118, 47420.266545950,
        47420.35, 2., .8), 47420.41) is None
    assert core.snapshot().plan is None
    assert core._source_floor == 47420.30


def test_retired_plan_late_ack_cannot_change_reactivated_controller():
    core, old, _ = setup(distance=1.56, raw=1.56)
    core.retire_for_lateral_handoff(47420.01)
    assert not core.acknowledge_output(old, 0.)
    core.activate(1, 47420.02)
    fresh = core.update(ShortFollowObservation(1, 118, 47420.03,
        47420.04, 1.56, .8), 47420.05)
    assert fresh is not None
    before = counters(core)
    assert not core.acknowledge_output(old, 0.)
    assert counters(core) == before and core.snapshot().plan is fresh


@pytest.mark.parametrize('now', [None, True, 0., -1., float('nan'), float('inf')])
def test_lateral_handoff_requires_actual_finite_clock(now):
    core, old, _ = setup()
    with pytest.raises(ValueError):
        core.retire_for_lateral_handoff(now)
    assert core.snapshot().plan is old
