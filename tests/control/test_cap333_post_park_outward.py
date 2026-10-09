"""CAP333-366: bounded post-park turn response needs post-quiet evidence."""
from dataclasses import replace

import pytest

import request_0513_modular as runtime
from car_control_modular.control_types import SteeringFeedback
from test_cap196_post_park_recenter import controller
from test_cap443_capture_braking import sample


def _observe(c, cap, stamp, x, *, age=.08, raw=6, uid=1, yaw=None,
             feedback_age=.02, trustworthy=True):
    target, frame = sample(cap, stamp, x, raw=raw, uid=uid)
    now = stamp + age
    if yaw is not None:
        frame = replace(frame, steering_feedback=SteeringFeedback(
            timestamp=now-feedback_age, trustworthy=trustworthy,
            integrated_yaw_right_deg=yaw, yaw_rate_right_dps=0))
    return c._capture_steering_observation(target, frame, now)


def _ready_pair(c, monkeypatch):
    clock = [10.]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock[0])
    c.set_normal_parking(True, 1)
    _observe(c, 340, 10.20, .34)  # before actual encoder quiet confirmation
    c.note_post_park_settled(1, 10.25, 0.)
    _observe(c, 342, 10.30, .28)
    clock[0] = 10.38
    c.set_normal_parking(False, 1)
    assert c.post_park_recenter_limit(1) == 4
    return clock


def test_cap344_two_post_quiet_captures_lift_only_to_six(monkeypatch):
    c = controller()
    _ready_pair(c, monkeypatch)
    observation = _observe(c, 344, 10.40, .23, age=.13, yaw=.2)
    assert observation.reason == "capture_rate_valid"
    assert observation.first_timestamp < c._post_park_recenter_settled_at
    assert c.post_park_recenter_limit(1) == 6
    turn = c.refresh_parked_lateral_pid(
        x_ratio=.23, base_rpm=0, feedback=None, now=10.53,
        visual_age_sec=.13, near_distance_mode=True, max_correction_rpm=7)
    assert 4 < abs(turn.correction_rpm) <= 6
    assert c._distance_pi_admitted_grant is None
    # A new park episode restores the original cap; the lift is not global.
    c.set_normal_parking(True, 1)
    assert c.post_park_recenter_limit(1) == 4


def test_logged_cap340_342_344_timestamps_take_bounded_path(monkeypatch):
    c = controller()
    clock = [28597.1]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock[0])
    c.set_normal_parking(True, 1)
    _observe(c, 340, 28597.765731, .3164)
    c.note_post_park_settled(1, 28597.822239625, 31.28)
    _observe(c, 342, 28597.864764, .2624, age=.1171)
    clock[0] = 28597.983
    c.set_normal_parking(False, 1)
    obs = _observe(c, 344, 28597.967256, .2180,
                   age=.2062, yaw=31.28, feedback_age=.012)
    assert obs.first_timestamp < c._post_park_recenter_settled_at
    assert c.post_park_recenter_limit(1) == 6


@pytest.mark.parametrize("case", [
    "missing_settled", "pre_quiet", "duplicate", "wrong_track", "wrong_uid",
    "stale_image", "missing_feedback", "stale_feedback", "untrusted_feedback",
    "body_rotation", "inward", "not_released",
])
def test_bounded_lift_rejects_unqualified_pair(monkeypatch, case):
    c = controller()
    if case == "not_released":
        monkeypatch.setattr(runtime.time, "monotonic", lambda: 10.)
        c.set_normal_parking(True, 1)
        _observe(c, 340, 10.20, .34)
        c.note_post_park_settled(1, 10.25, 0.)
        _observe(c, 342, 10.30, .28)
    else:
        _ready_pair(c, monkeypatch)
    if case == "missing_settled":
        c._post_park_recenter_settled_at = None
    if case == "pre_quiet":
        c._post_park_recenter_settled_at = 10.35
    cap = 342 if case == "duplicate" else 344
    stamp = 10.30 if case == "duplicate" else 10.40
    x = .33 if case == "inward" else .23
    _observe(c, cap, stamp, x, uid=2 if case == "wrong_uid" else 1,
             raw=9 if case == "wrong_track" else 6,
             age=.27 if case == "stale_image" else .13,
             yaw=(3.0 if case == "body_rotation" else None
                  if case == "missing_feedback" else .2),
             feedback_age=.16 if case == "stale_feedback" else .02,
             trustworthy=case != "untrusted_feedback")
    assert c.post_park_recenter_limit(1) == 4
    assert c._distance_pi_admitted_grant is None


def test_full_post_quiet_three_capture_window_can_clear_old_ceiling(monkeypatch):
    c = controller()
    monkeypatch.setattr(runtime.time, "monotonic", lambda: 10.)
    c.set_normal_parking(True, 1)
    c.note_post_park_settled(1, 10.25, 0.)
    _observe(c, 342, 10.30, .34)
    c.set_normal_parking(False, 1)
    _observe(c, 344, 10.40, .29)
    _observe(c, 347, 10.50, .24)
    assert c.post_park_recenter_limit(1) is None
