"""CAP269 post-ranging cache refresh; real decisions, no hardware or threads."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import request_0513_modular as runtime
from car_control_modular.control_types import DistanceState, HazardState, SteeringFeedback
from car_control_modular.distance_pi import DistancePiConfig, DistancePiController
from car_control_modular.follow_distance_hold import FollowDistanceHold
from test_depth_priority_admission import visible, context_owner
from test_depth_target_snapshot import persons, NOW
from test_distance_tracking_response import setup
from test_distance_pi_controller import configured, step


def feedback(stamp, rpm=90., **kwargs):
    return SteeringFeedback(timestamp=stamp, left_forward_rpm=rpm,
                            right_forward_rpm=rpm, trustworthy=True, **kwargs)


def owner(reader, enabled=True):
    obj = object.__new__(runtime.PersonTracker)
    obj._follow_controller = SimpleNamespace(distance_pi_enabled=enabled)
    obj._action_runtime = SimpleNamespace(get_steering_feedback=reader)
    return obj


def test_resample_keeps_physical_timestamps_and_logs_skew(monkeypatch, caplog):
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: 100.)
    old, new = feedback(99.8444), feedback(99.96)
    state = DistanceState(sample_timestamp=99.89, raw_distance_m=2.105, used_distance_m=2.105)
    calls = []
    obj = owner(lambda: calls.append('cache') or new)
    with caplog.at_level('INFO'):
        assert obj._refresh_distance_control_feedback(old, state, 269) is new
    assert calls == ['cache']
    assert state.sample_timestamp == 99.89 and old.timestamp == 99.8444
    assert 'old_age_ms=155.6' in caplog.text and 'new_age_ms=40.0' in caplog.text
    assert 'serial_read=False deadline_renewed=False' in caplog.text


@pytest.mark.parametrize('case', ['missing', 'same', 'older'])
def test_no_new_sample_never_retimestamps_old_feedback(monkeypatch, case):
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: 100.)
    old = feedback(99.8444)
    new = {'missing': None, 'same': old, 'older': feedback(99.8)}[case]
    result = owner(lambda: new)._refresh_distance_control_feedback(old, DistanceState(), 269)
    assert result is old and 100.-result.timestamp > .15


@pytest.mark.parametrize('new', [feedback(100.01), feedback(float('nan')), object()])
def test_invalid_cache_does_not_authorize_old_healthy_feedback(monkeypatch, new):
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: 100.)
    assert owner(lambda: new)._refresh_distance_control_feedback(feedback(99.95), DistanceState(), 1) is None


def test_newer_fault_is_not_hidden_by_healthy_old_sample(monkeypatch):
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: 100.)
    bad = replace(feedback(99.99), trustworthy=False, left_error=1)
    assert owner(lambda: bad)._refresh_distance_control_feedback(feedback(99.95), DistanceState(), 1) is bad


def test_cache_reader_failure_fails_closed(monkeypatch):
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: 100.)
    def fail():
        raise RuntimeError('test cache unavailable')
    assert owner(fail)._refresh_distance_control_feedback(feedback(99.95), DistanceState(), 1) is None


def test_legacy_controller_does_not_read_new_cache():
    old = feedback(99.95)
    obj = owner(lambda: pytest.fail('legacy path must be unchanged'), enabled=False)
    assert obj._refresh_distance_control_feedback(old, DistanceState(), 1) is old


def test_search_controller_does_not_change_feedback_path():
    old = feedback(99.95)
    obj = owner(lambda: pytest.fail('search policy must be unchanged'))
    obj._follow_controller.search_state = 'searching'
    assert obj._refresh_distance_control_feedback(old, DistanceState(), 1) is old


@pytest.mark.parametrize('source', ['depth30', 'vision'])
def test_real_entrypoint_resamples_after_ranging_without_changing_depth(visible, source):
    obj = visible
    obj._follow_controller.distance_pi_enabled = True
    obj._follow_controller.set_last_dispatched = lambda _: None
    obj._follow_controller._is_fresh_depth_state = lambda _: True
    obj._brake_hold_active = True
    obj._brake_hold_label = 'follow_distance_hold'
    obj._brake_hold_stop_mode = None
    obj._follow_distance_hold = FollowDistanceHold(1, NOW-.1)
    obj._last_dispatched_action = runtime.ACTION_STOP
    obj._get_obstacle_status = lambda: {}
    obj._current_hazard_state_for_controller = lambda: HazardState()
    old, new = feedback(NOW-.1556), feedback(NOW-.04)
    events, seen = [], []
    def read():
        events.append('read')
        return old if len(events) == 1 else new
    state = DistanceState(source='vision_depth', sample_timestamp=NOW-.10,
                          raw_distance_m=2.105, used_distance_m=2.105)
    def measure(*args, **kwargs):
        events.append('measure')
        assert kwargs['steering_feedback'] is old
        return state
    obj._action_runtime = SimpleNamespace(get_steering_feedback=read)
    obj._distance_runtime = SimpleNamespace(select_target=lambda ts: ts[0], get_frame_distance_state=measure)
    class Observed(Exception):
        pass
    def observe(frame, *args, **kwargs):
        seen.append(frame)
        raise Observed  # stop the real path just before any decision or writes
    obj._observe_follow_distance_hold = observe
    with pytest.raises(Observed):
        obj._process_detections_modular(640, 480, persons(), control_source=source, depth_use_latest=True)
    assert events == ['read', 'measure', 'read']
    assert seen[0].steering_feedback is new and seen[0].distance_state is state


@pytest.mark.parametrize('case', ['fresh', 'stale', 'bad', 'skew'])
def test_cap269_controller_replay_keeps_guards(setup, case):
    clock, c, frame = configured(setup, distance_pi_launch_request_rpm=180.,
                                depth_longitudinal_sample_max_age_sec=.25)
    for _ in range(2):
        f = frame(2.105, rpm=89.5, stamp=clock.now-.1)
        step(c, replace(f, steering_feedback=feedback(clock.now-.05, 89.5)))
        clock.now += .05
    if case == 'skew':
        clock.now += .1  # keep the175ms-old sample newer than the last one
    original = frame(2.105, rpm=89.5, stamp=clock.now-.1)
    old = feedback(clock.now-.1556, 89.5)
    new = feedback(clock.now-.04, 89.5)
    if case == 'stale': new = old
    if case == 'bad': new = replace(new, trustworthy=False, left_error=1)
    if case == 'skew':
        original = replace(original, distance_state=replace(original.distance_state, sample_timestamp=clock.now-.175))
        new = feedback(clock.now, 89.5)
    fresh = owner(lambda: new)._refresh_distance_control_feedback(old, original.distance_state, 269)
    step(c, replace(original, steering_feedback=fresh))
    result = c.last_distance_pid_result
    if case == 'fresh':
        assert result.pi_brake_source == 'raw_relative_motion'
        assert result.output_rpm > 100  # previously about8RPM, not a new floor
        assert result.pi_launch_floor_rpm == 180.
    else:
        assert result.pi_brake_source == 'stationary_fallback'
        assert result.pi_launch_floor_rpm == 0.


@pytest.mark.parametrize('distance', [1.6, 1.8, 2.4])
def test_p_trial_increases_distance_component_by_fifty_percent(distance):
    results = []
    for kp in (2., 3.):
        c = DistancePiController(DistancePiConfig(kp_per_sec=kp, launch_request_rpm=180.))
        results.append(c.update(distance, 1.5, sample_timestamp=100., execution_now=100.,
            deadband_m=.03, max_output_rpm=200., ego_forward_rpm=20., range_rate_m_s=.2,
            raw_closure_valid=True))
    a, b = results
    assert b.p_m_s == pytest.approx(a.p_m_s*1.5)
    assert a.cap_rpm == b.cap_rpm and b.output_rpm <= b.cap_rpm
    # Show why the trial does NOT promise50% more motor speed under180floor.
    assert a.output_rpm == b.output_rpm


def test_new_cache_cannot_revive_expired_depth(setup):
    clock, c, frame = configured(setup, distance_pi_launch_request_rpm=180.,
                                depth_longitudinal_sample_max_age_sec=.25)
    old = frame(2.5, rpm=90., stamp=clock.now-.251)
    refreshed = owner(lambda: feedback(clock.now))._refresh_distance_control_feedback(
        feedback(clock.now-.3), old.distance_state, 269)
    decision = step(c, replace(old, steering_feedback=refreshed))
    assert not any(a.kind == 'forward' and a.speed_percent > 0 for a in decision.actions)
