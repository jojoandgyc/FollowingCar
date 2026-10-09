"""Parking preview must not repeatedly restart zero-budget brake recovery."""
import pytest
from types import SimpleNamespace

from test_brake_evidence_transition import controller, update
from test_distance_tracking_response import setup
from test_distance_pi_controller import configured, step
from test_lateral_zero_runtime import owner, _intent, NOW
from car_control_modular.near_yaw_parking import ParkSettlingEvidence
import request_0513_modular as runtime


def test_park_jitter_and_repeated_no_authority_do_not_latch_zero_recovery():
    c = controller()
    update(c, 100.)
    update(c, 100.05, raw_closure_valid=False, allow_motion_memory=False)
    assert c._brake_recovery_pending
    c.set_normal_parking(True)
    integral = c.integral_m_s
    for n, rpm in enumerate((0., 15., -15., 0., 20., 0.)):
        ts = 100.10+n*.05
        c.suspend(ts, 'no_live_grant_before_pi', reset_execution=True)
        r = update(c, ts, distance=2.3, raw_distance_m=2.3, ego_forward_rpm=rpm)
        assert r.status == 'parked_preview'
        assert r.output_rpm > 0 and r.output_rpm <= r.cap_rpm
        assert not r.brake_recovery_limited
        assert c.integral_m_s == integral
        assert c._approved_rpm == 0 and c._last_execution_ts is None
    c.set_normal_parking(False)
    r = update(c, 100.45, distance=2.3, raw_distance_m=2.3, ego_forward_rpm=0.)
    assert 0 < r.output_rpm <= r.cap_rpm
    assert not r.brake_recovery_limited
    assert r.sample_dt_sec == 0  # no parked-time integration


@pytest.mark.parametrize('condition', ['duplicate', 'stale', 'near', 'missing_feedback'])
def test_release_does_not_make_bad_depth_a_new_grant(condition):
    c = controller()
    c.set_normal_parking(True)
    update(c, 100.)
    c.set_normal_parking(False)
    if condition == 'duplicate':
        r = update(c, 100.)
    elif condition == 'stale':
        r = update(c, 100.05, execution_now=100.31)
    elif condition == 'near':
        r = update(c, 100.05, distance=1.2, raw_distance_m=1.2)
    else:
        r = update(c, 100.05, ego_forward_rpm=None, range_rate_m_s=None, raw_closure_valid=False)
    assert r.output_rpm == 0


def test_parking_survives_pid_reset_without_changing_physical_sample_time():
    c = controller()
    c.set_normal_parking(True)
    c.reset()
    r = update(c, 100.)
    assert r.status == 'parked_preview'
    assert c._last_sample_ts == 100.


def test_real_controller_parking_binding_and_fresh_release(setup):
    clock, c, frame = configured(setup, distance_pi_launch_request_rpm=180.,
        distance_pi_kp_per_sec=3., distance_pi_motion_memory_sec=.35)
    c._live_longitudinal_authority_reader = lambda uid: None
    c.set_normal_parking(True, 1)
    for _ in range(5):
        step(c, frame(2.3, rpm=0.))
        assert c.last_distance_pid_result.pi_status == 'parked_preview'
        assert c.last_distance_pid_result.output_rpm > 0
        clock.now += .05
    c.set_normal_parking(False, 1)
    step(c, frame(2.3, rpm=0.))
    assert c.last_distance_pid_result.pi_status != 'parked_preview'
    assert c.last_distance_pid_result.output_rpm > 0
    assert not c.last_distance_pid_result.pi_brake_recovery_limited


def test_wrong_uid_cannot_set_parking_for_current_controller(setup):
    _, c, _ = configured(setup)
    c.set_normal_parking(True, 2)
    assert not c._distance_pid._distance_pi._normal_parking


def test_real_producer_calls_real_controller_on_park_and_release(owner, setup, monkeypatch):
    _, c, _ = configured(setup, distance_pi_launch_request_rpm=180.)
    owner._follow_controller = c
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: NOW)
    owner._request_near_yaw_park(_intent(owner), 'center_hold')
    assert c._distance_pid._distance_pi._normal_parking
    request = owner._near_yaw_park_request
    evidence = ParkSettlingEvidence(request, NOW+.01)
    fb = SimpleNamespace(timestamp=NOW+.45, trustworthy=True,
                         left_forward_rpm=-15., right_forward_rpm=15.)
    owner._action_runtime = SimpleNamespace(near_yaw_park_motion_ready=
        lambda req, stamp, now: req is request and evidence.motion_handoff_ready(stamp, fb, now))
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: NOW+.46)
    assert not owner._release_near_yaw_park(capture_id=577, capture_timestamp=NOW+.44,
        reason='qualified_motion', target_id=1, qualified=True, translation_requested=True)
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: NOW+.52)
    assert owner._release_near_yaw_park(capture_id=577, capture_timestamp=NOW+.44,
        reason='qualified_motion', target_id=1, qualified=True, translation_requested=True)
    assert not c._distance_pid._distance_pi._normal_parking
    assert c._distance_pid._distance_pi._execution_suspended
    assert owner._depth30_linear_snapshot is None


def test_preview_same_sample_after_release_does_not_integrate_or_refresh_clock(setup):
    clock, c, frame = configured(setup, distance_pi_launch_request_rpm=180.)
    c.set_normal_parking(True, 1)
    observation = frame(2.3, rpm=0.)
    step(c, observation)
    stamp = c._distance_pid_last_sample_timestamp
    output = c.last_distance_pid_result.output_rpm
    c.set_normal_parking(False, 1)
    step(c, observation)
    assert c.last_distance_pid_result.output_rpm == output
    assert c._distance_pid_last_sample_timestamp == stamp
    assert c._distance_pid._distance_pi.integral_m_s == 0
    assert c._distance_pid._distance_pi._execution_suspended
