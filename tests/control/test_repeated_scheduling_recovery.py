"""Repeated short gaps must not trap a following car at its launch cap.

Controller-only simulation; no motor, camera, serial or runtime threads.
"""
from dataclasses import replace

import pytest

from test_distance_tracking_response import setup, decide
from test_scheduling_gap_evidence import missing, recovery


def start_low(setup, *, scale=100):
    clock, c, frame, f = recovery(setup, physical_gap=.30, processing_gap=.32, rpm=12)
    c.cfg = replace(c.cfg, forward_max_rpm=scale)
    c._longitudinal_motion_uid = 1  # Already following this UID, not initial lock.
    c._depth_gap_resume_hint = (*c._depth_gap_resume_hint[:3], 24., None, False)
    request = int(60 * 100 / scale)
    assert c._limit_depth_quality_forward_percent(f, request, clock.now) * scale/100 == 24
    return clock, c, frame, request


@pytest.mark.parametrize('scale', [100, 200])
def test_repeated_120ms_gaps_finish_original_ramp_not_restart_at_24(setup, caplog, scale):
    clock, c, frame, request = start_low(setup, scale=scale)
    end = c._depth_schedule_recovery[5]
    outputs = [24.]
    for _ in range(4):
        clock.now += .04
        before = c._depth_schedule_recovery
        c._note_depth_quality_failure(missing(frame), clock.now)
        assert c._depth_schedule_recovery is before
        clock.now += .08
        output = c._limit_depth_quality_forward_percent(
            frame(2.02, rpm=outputs[-1]-4, stamp=clock.now-.02), request, clock.now)
        outputs.append(output*scale/100)
        if c._depth_schedule_recovery is not None:
            assert c._depth_schedule_recovery[5] == end
    assert outputs == [24, 48, 60, 60, 60]
    assert c._depth_schedule_recovery is None
    assert c._depth_recovery_started_at is None
    assert caplog.text.count('event=start ') == 1
    assert 'ramp_completed=True' in caplog.text


def test_gap_does_not_shift_sample_clock_or_completion_time(setup):
    clock, c, frame, request = start_low(setup)
    original = c._depth_schedule_recovery
    for age in (.10, .19, .35, .49):
        clock.now = original[1]+age
        c._note_depth_quality_failure(missing(frame), clock.now)
        assert c._depth_schedule_recovery is original
    clock.now = original[1]+.501
    c._note_depth_quality_failure(missing(frame), clock.now)
    assert c._depth_schedule_recovery is None
    assert c._depth_gap_resume_hint is None


def test_reference_can_resume_after_expiry_but_missing_depth_cannot_drive(setup):
    clock, c, frame, request = start_low(setup)
    original = c._depth_schedule_recovery
    clock.now = original[1]+.25
    result = decide(c, missing(frame))
    assert not any(a.kind in ('forward', 'steer_left', 'steer_right') and a.speed_percent > 0
                   for a in result.actions)
    assert c._depth_schedule_recovery is original
    clock.now += .02
    output = c._limit_depth_quality_forward_percent(frame(2.03, rpm=0), request, clock.now)
    assert 0 < output <= 36  # New depth + measured acceleration envelope, not old command.
    assert c._depth_schedule_recovery[5] == original[5]


@pytest.mark.parametrize('case', ['uid', 'person', 'search', 'hazard', 'obstacle',
    'brake', 'latched', 'safety', 'raw', 'pixels', 'jump', 'replay_jump', 'disabled'])
def test_non_scheduling_failure_discards_active_episode(setup, case):
    clock, c, frame, request = start_low(setup)
    clock.now += .04
    f = missing(frame)
    if case == 'uid': c.active_target_id = 2
    elif case == 'person': f = replace(f, persons=[])
    elif case == 'search': c.search_state = 'searching'
    elif case == 'hazard': f = replace(f, hazard=replace(f.hazard, active=True))
    elif case == 'obstacle': f = replace(f, obstacles=replace(f.obstacles, front=True))
    elif case == 'brake': f = replace(f, distance_state=replace(f.distance_state, brake_latched=True))
    elif case == 'latched': c._target_stop_latched = True
    elif case == 'safety': f = replace(f, distance_state=replace(f.distance_state, safety_distance_m=.4))
    elif case == 'raw': f = replace(f, distance_state=replace(f.distance_state, raw_distance_m=3.))
    elif case == 'pixels': f = replace(f, distance_state=replace(f.distance_state, source_detail='depth_invalid_pixels'))
    elif case in ('jump', 'replay_jump'):
        f = replace(f, distance_state=replace(f.distance_state, source_detail='distance_jump_pending',
            temporal_status='duplicate' if case == 'replay_jump' else '',
            observation_timestamp=c._depth_schedule_recovery[1]))
    elif case == 'disabled': c.cfg = replace(c.cfg, depth_measured_recovery_enable=False)
    c._note_depth_quality_failure(f, clock.now)
    assert c._depth_schedule_recovery is None
    if case != 'disabled':
        assert c._depth_quality_degraded


def test_repeated_sample_does_not_advance_ramp_even_after_nominal_end(setup):
    clock, c, frame, request = start_low(setup)
    clock.now += .35
    fresh = frame(2.02, rpm=0)
    c._limit_depth_quality_forward_percent(fresh, request, clock.now)
    original = c._depth_schedule_recovery
    clock.now += .10
    assert clock.now > original[5]
    c._limit_depth_quality_forward_percent(fresh, request, clock.now)
    assert c._depth_schedule_recovery == original


def test_time_alone_never_releases_stalled_wheels_to_full_pid(setup):
    clock, c, frame, request = start_low(setup)
    end = c._depth_schedule_recovery[5]
    for _ in range(12):
        clock.now += .05
        assert c._limit_depth_quality_forward_percent(frame(2.02, rpm=0), request, clock.now) <= 36
    assert clock.now > end
    assert c._depth_schedule_recovery[5] == end


def test_direct_continuity_cannot_skip_an_active_measured_ramp(setup):
    clock, c, frame, request = start_low(setup)
    clock.now += .03
    c._note_depth_quality_failure(missing(frame), clock.now)
    clock.now += .02
    # Wheels match old approval; the direct-continuity branch must not bypass
    # the active ramp and jump straight from 24 to requested 90 RPM.
    output = c._limit_depth_quality_forward_percent(frame(2.02, rpm=24, stamp=clock.now-.02), 90, clock.now)
    assert 35 <= output <= 36
    assert c._depth_schedule_recovery is not None


def test_lower_request_immediately_limits_fresh_recovery(setup):
    clock, c, frame, request = start_low(setup)
    clock.now += .05
    assert c._limit_depth_quality_forward_percent(frame(2.02, rpm=24), 10, clock.now) == 10
    assert c._depth_schedule_recovery[4] == 10


def test_active_recovery_is_cleared_by_target_reset(setup):
    clock, c, frame, request = start_low(setup)
    c.clear_active_target('test')
    assert c._depth_schedule_recovery is None
    assert c._depth_gap_resume_hint is None


def test_replayed_reduction_cannot_be_undone_by_another_replay(setup):
    clock, c, frame, request = start_low(setup)
    original = c._depth_schedule_recovery
    sample = frame(2.02, rpm=24, stamp=original[1])
    assert c._limit_depth_quality_forward_percent(sample, 10, clock.now) == 10
    clock.now += .01
    assert c._limit_depth_quality_forward_percent(sample, 90, clock.now) == 10
    assert c._depth_schedule_recovery[1] == original[1]
    assert c._depth_schedule_recovery[5] == original[5]


def test_decision_path_keeps_ramp_through_repeated_missing_measurements(setup, monkeypatch):
    clock, c, frame, request = start_low(setup)
    # Hold PID request constant to exercise the real decision/observation/recovery
    # chain; actuator authorization is independently tested in three-clock tests.
    monkeypatch.setattr(c, '_forward_percent_for_distance', lambda *args, **kw: 60)
    end = c._depth_schedule_recovery[5]
    outputs = [24]
    for _ in range(4):
        clock.now += .04
        decide(c, missing(frame))
        assert c._depth_schedule_recovery[5] == end
        clock.now += .08
        result = decide(c, frame(2.02, rpm=outputs[-1]-4, stamp=clock.now-.02))
        assert result.reason == 'longitudinal_distance_pid'
        outputs.append(result.current_forward_percent)
    assert outputs == [24, 48, 60, 60, 60]
    assert c._depth_schedule_recovery is None
