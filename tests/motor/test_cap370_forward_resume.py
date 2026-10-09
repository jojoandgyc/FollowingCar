"""Fake clock/serial only: ordinary dwell is interruptible by new forward evidence."""
import pytest

from car_control_modular.near_yaw_parking import ParkSettlingEvidence
from test_follow_wheel_periodic import setup_periodic
from test_near_yaw_park_execution import request_park
from test_visible_wheel_continuity import feedback


def test_fresh_forward_can_release_current_before_500ms_without_speed_write(monkeypatch):
    r, o, d, _, clock, state = setup_periodic(monkeypatch)
    req = request_park(o, clock)
    r._service_follow_wheels()
    before = list(d.pairs)
    clock[0] = 10.1
    assert r.request_near_yaw_forward_resume(req, 10.08, 10.09, clock[0])
    assert not r.near_yaw_park_motion_ready(req, 10.08, clock[0])
    r._service_follow_wheels()
    assert r._near_yaw_park_settling.current_released_at == 10.1
    assert d.pairs == before  # hint releases current, not wheel authority
    clock[0] = 10.12
    r.get_steering_feedback = lambda: feedback(10.11, 0, 0)
    assert r.near_yaw_park_motion_ready(req, 10.08, clock[0])
    assert not r.near_yaw_park_release_ready(req, 10.08, clock[0])
    o._near_yaw_park_request = None
    o._brake_hold_active = False
    o._brake_hold_label = 'brake'
    state[:] = [60., 0., 10.25, 10.25]
    r._service_follow_wheels()
    assert d.pairs[-1] == (60, -60)


@pytest.mark.parametrize('capture,sample,now', [
    (9.99, 10.09, 10.1), (10.08, 9.99, 10.1),
    (10.11, 10.09, 10.1), (10.08, 10.11, 10.1),
    (10.01, 10.29, 10.3), (10.29, 10.01, 10.3),
    (float('nan'), 10.09, 10.1)])
def test_hint_requires_fresh_post_stop_physical_evidence(capture, sample, now):
    e = ParkSettlingEvidence(object(), 10.)
    assert not e.request_forward_resume(capture, sample, now)
    assert not e.forward_resume_live(now)


@pytest.mark.parametrize('veto', ['expired', 'identity', 'search', 'explicit', 'hard_stop'])
def test_hint_cannot_release_protected_or_expired_hold(monkeypatch, veto):
    r, o, d, _, clock, _ = setup_periodic(monkeypatch)
    req = request_park(o, clock)
    r._service_follow_wheels()
    clock[0] = 10.1
    assert r.request_near_yaw_forward_resume(req, 10.08, 10.09, clock[0])
    if veto == 'expired': clock[0] = 10.3
    if veto == 'identity': o._follow_controller.active_target_id = 2
    if veto == 'search': o.search_state = 'searching'
    if veto == 'explicit': o._explicit_stop_requested = True
    if veto == 'hard_stop': r.hard_stop_check = lambda _: True
    r._service_ordinary_park_exit(r._near_yaw_park_settling, 'near_yaw_park')
    assert r._near_yaw_park_settling.current_released_at is None


def test_post_stop_image_need_not_postdate_free_stop(monkeypatch):
    r, o, _, _, clock, _ = setup_periodic(monkeypatch)
    req = request_park(o, clock)
    r._service_follow_wheels()
    clock[0] = 10.55
    r._service_follow_wheels()
    r.get_steering_feedback = lambda: feedback(10.56, 0, 0)
    assert r.near_yaw_park_motion_ready(req, 10.45, 10.57)
    assert not r.near_yaw_park_release_ready(req, 10.45, 10.57)


@pytest.mark.parametrize('change', ['lower_forward', 'revision', 'lost_depth', 'identity', 'hard_stop'])
def test_current_release_race_drops_packet_without_reparking_valid_forward(monkeypatch, change, caplog):
    r, o, d, _, clock, state = setup_periodic(monkeypatch)
    state[:] = [80., 0., 10.18, 10.18]
    r.backend.normal_zero_hold = True
    original = r.backend.prepare_speed_mode
    changed = []
    def prepare():
        original()
        if changed:
            return
        changed.append(True)
        if change == 'lower_forward': state[0] = 60.
        if change == 'revision': o._lateral_yaw_revision += 1
        if change == 'lost_depth': state[2] = 9.
        if change == 'identity': o._follow_controller.active_target_id = 2
        if change == 'hard_stop': r.hard_stop_check = lambda _: True
    r.backend.prepare_speed_mode = prepare
    # Fixture's snapshot must use percent, while periodic axes use RPM.
    o._fresh_depth_linear_snapshot = lambda uid, now=None: (
        ('forward', state[0]) if clock[0] < state[2] else None)
    with caplog.at_level('INFO'):
        r._service_follow_wheels()
    if change in ('lower_forward', 'revision'):
        assert 'parking_exit_new_forward_grant' in caplog.text
        assert not d.stops
        # Discard the obsolete packet and rebuild the current grant once in
        # the same execution tick; no extra zero or next-tick delay.
        assert d.pairs == [(state[0], -state[0])]
    else:
        assert not any(l or rr for l, rr in d.pairs)
        assert d.stops or d.pairs == [(0, 0)]
