"""Real action executor, simulated encoder jitter, fake serial writes only."""
import pytest

from test_follow_wheel_periodic import setup_periodic
from test_near_yaw_park_execution import request_park
from test_visible_wheel_continuity import feedback


@pytest.mark.parametrize('left,right', [(0, 0), (15, 16), (-15, 16), (-23, -20)])
def test_ordinary_release_transfers_motion_check_to_wheel_guard(monkeypatch, left, right):
    rt, owner, driver, _, clock, state = setup_periodic(monkeypatch)
    request = request_park(owner, clock)
    rt._service_follow_wheels()
    clock[0] = 10.50
    rt._service_follow_wheels()
    clock[0] = 10.52
    rt.get_steering_feedback = lambda: feedback(clock[0], left, right)
    assert rt.near_yaw_park_motion_ready(request, clock[0]-.01, clock[0])
    assert not rt.near_yaw_park_release_ready(request, clock[0]-.01, clock[0])
    # Simulate the producer's already-qualified release; executor still owns
    # wheel reversal and current-axis authority, not the parking qualifier.
    owner._near_yaw_park_request = None
    owner._brake_hold_active = False
    owner._brake_hold_label = 'brake'
    state[:2] = [52., -10.]
    state[2:] = [clock[0]+.15, clock[0]+.15]  # NEW post-park visual/depth leases
    rt._service_follow_wheels()
    if left >= 0 and right >= 0:
        assert driver.pairs[-1] == (42, -62)
    elif left < 0 and right < 0:
        assert driver.pairs[-1] == (0, 0)  # true reverse is still guarded


@pytest.mark.parametrize('condition', ['stale', 'untrusted', 'pre_stop', 'future', 'missing', 'old_image'])
def test_motion_release_still_needs_fresh_feedback_and_post_stop_image(monkeypatch, condition):
    rt, owner, _, _, clock, _ = setup_periodic(monkeypatch)
    request = request_park(owner, clock)
    rt._service_follow_wheels()
    clock[0] = 10.50
    rt._service_follow_wheels()
    clock[0] = 10.53
    fb = feedback(10.52, 15, -15)
    stamp = 10.52
    if condition == 'stale': fb.timestamp = 10.01
    if condition == 'untrusted': fb.trustworthy = False
    if condition == 'pre_stop': fb.timestamp = 10.
    if condition == 'future': fb.timestamp = 10.60
    if condition == 'missing': fb = None
    if condition == 'old_image': stamp = 9.99
    rt.get_steering_feedback = lambda: fb
    assert not rt.near_yaw_park_motion_ready(request, stamp, clock[0])


def test_normal_refresh_exits_current_once_without_restarting_episode(monkeypatch):
    rt, owner, driver, _, clock, _ = setup_periodic(monkeypatch)
    request_park(owner, clock)
    rt._service_follow_wheels()
    before = list(driver.pairs)
    evidence = rt._near_yaw_park_settling
    for t in (11.2, 12.4):
        clock[0] = t
        assert rt.send_percent_brake('normal', 'near_yaw_park')
    assert driver.pairs == before  # current-only exit, no extra speed packet
    assert driver.stops == [1, 0, 2]
    assert rt.backend.parking_current_a == 0
    assert rt._near_yaw_park_settling is evidence and evidence.sent_at == 10.
    owner._near_yaw_park_request = None
    assert not rt.send_percent_brake('normal', 'near_yaw_park')
    assert driver.stops == [1, 0, 2]


def test_failed_initial_stop_cannot_release_via_motion_path(monkeypatch):
    rt, owner, _, _, clock, _ = setup_periodic(monkeypatch)
    request = request_park(owner, clock)
    def fail(*args, **kwargs):
        raise OSError('simulated write failure')
    rt.backend.send_stop = fail
    with pytest.raises(OSError):
        rt._service_follow_wheels()
    clock[0] += .05
    assert not rt.near_yaw_park_motion_ready(request, clock[0]-.01, clock[0])


def test_emergency_during_normal_parking_never_reenters_speed_mode(monkeypatch):
    rt, owner, driver, _, clock, _ = setup_periodic(monkeypatch)
    request_park(owner, clock)
    rt._service_follow_wheels()
    before = len(driver.pairs)
    assert rt.send_percent_brake('emergency', 'safety_hold_front_ir')
    assert driver.stops[-1] == 1
    assert len(driver.pairs) == before
    assert not rt.backend.motion_armed
