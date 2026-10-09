"""CAP347: the aligned outer wheel must not restart an all-wheel stop wait.

Fake encoder, clock and driver only. Pair values in assertions on the driver
use its raw right-wheel polarity; all other pairs are forward-normalized.
"""
import pytest

from car_control_modular.forward_loss_handoff import ForwardLossHandoff
from car_control_modular.wheel_zero_cross import WheelZeroCrossGuard
from test_cap676_turn_continuity import runtime
from test_visible_wheel_continuity import feedback


@pytest.mark.parametrize("mirror", [False, True])
def test_cap347_outer_wheel_keeps_moving_without_duplicate_wait(monkeypatch, mirror):
    r, o, d, _, clock = runtime(monkeypatch)
    def pair(values):
        return values[::-1] if mirror else values
    def send(values):
        left, right = pair(values)
        r._send_follow_wheel_targets(left, -right, "FOLLOW20")
    r.get_steering_feedback = lambda: feedback(clock[0] - .01, *pair((3, 6)))
    send((2, 18))
    o._fresh_depth_linear_snapshot = lambda uid, now=None: None
    # The current, trustworthy sample predates the handoff by 18ms. It may
    # start the wheel guard, but CANNOT count as post-zero confirmation.
    clock[0] = 10.05
    r.get_steering_feedback = lambda: feedback(10.032, *pair((3, 9)))
    send((-10, 10))
    assert d.pairs[-1] == (0, 0)
    assert r._visible_wheel_guard.pending_signs is not None
    assert not r._forward_loss_handoff.armed
    for stamp in (10.08, 10.12):
        clock[0] = stamp + .01
        r.get_steering_feedback = lambda: feedback(stamp, *pair((3, 11)))
        send((-10, 10))
        if stamp == 10.08:
            assert d.pairs[-1] == (0, 0)
            send((-10, 10))  # duplicated sample is not a second confirmation
            assert d.pairs[-1] == (0, 0)
    left, right = pair((-10, 10))
    assert d.pairs[-1] == (left, -right)
    # Once established, same-direction live yaw remains continuous. No
    # Depth grant, manufactured forward arc or new NORMAL brake is needed.
    for i in range(12):
        clock[0] += .05
        r.get_steering_feedback = lambda: feedback(clock[0] - .01, *pair((-3, 11)))
        send((-10, 10))
        assert d.pairs[-1] == (left, -right)
    assert not d.stops
    o._has_fresh_lateral_yaw = lambda uid: False
    send((-10, 10))
    assert d.pairs[-1] == (0, 0)


@pytest.mark.parametrize("measured", [(5, 11), (30, 40), (-20, -3), (3, 40)])
def test_strong_opposing_or_whole_car_reverse_is_not_low_residual(measured):
    h = ForwardLossHandoff()
    h.note_sent((2, 18))
    assert h.limit((-10, 10), feedback(10, *measured), 10,
                   residual_turn_max_rpm=4)[0] == (0, 0)


@pytest.mark.parametrize("case", ["stale", "future", "untrusted", "nan", "pre_zero",
                                   "duplicate", "opposing", "changed_direction", "opt_out"])
def test_directional_residual_guard_still_requires_safe_new_evidence(case):
    g = WheelZeroCrossGuard()
    g.note_sent((2, 18), 9.95)
    kwargs = dict(allow_aligned_turn=True, residual_turn_max_rpm=4)
    for now in (10., 10.05):
        out, _ = g.limit((-10, 10), feedback(now - .01, 3, 11), now, **kwargs)
        g.note_sent(out, now)
        assert out == (0, 0)
    f = feedback(10.09, 3, 11)
    req = (-10, 10)
    if case == "stale": f.timestamp = 9
    elif case == "future": f.timestamp = 11
    elif case == "untrusted": f.trustworthy = False
    elif case == "nan": f.left_forward_rpm = float("nan")
    elif case == "pre_zero": f.timestamp = 9.99
    elif case == "duplicate": f.timestamp = 10.04
    elif case == "opposing": f.left_forward_rpm = 5
    elif case == "changed_direction": req = (10, -10)
    elif case == "opt_out": kwargs["allow_aligned_turn"] = False
    assert g.limit(req, f, 10.1, **kwargs)[0] == (0, 0)


def test_explicit_zero_is_never_converted_to_yaw():
    h = ForwardLossHandoff()
    h.note_sent((2, 18))
    assert h.limit((0, 0), feedback(10, 3, 11), 10,
                   residual_turn_max_rpm=4)[0] == (0, 0)


def test_recorded_cap347_prefix_releases_at_second_post_zero_sample(monkeypatch):
    r, o, d, _, clock = runtime(monkeypatch)
    clock[0] = 30082.313356
    r.get_steering_feedback = lambda: feedback(30082.265488813, 3, 6)
    r._send_follow_wheel_targets(2, -18, "FOLLOW20")
    o._fresh_depth_linear_snapshot = lambda uid, now=None: None
    # Original run: all three commands were 0/0; first nonzero at .774956.
    # Only replay the unchanged prefix. After a different command is sent,
    # later real-world feedback is counterfactual, not a measured response.
    trace = [
        (30082.381076, 30082.363286500, (3, 9), 10),
        (30082.430741, 30082.414791172, (3, 11), 9),
        (30082.472489, 30082.464698719, (4, 11), 9),
    ]
    for now, stamp, measured, yaw in trace:
        clock[0] = now
        r.get_steering_feedback = lambda: feedback(stamp, *measured)
        r._send_follow_wheel_targets(-yaw, -yaw, "FOLLOW20")
    assert d.pairs[-3:] == [(0, 0), (0, 0), (-9, -9)]
    assert (trace[-1][0] - trace[0][0]) * 1000 == pytest.approx(91.413)
    assert not d.stops
    # A subsequent stronger opposing sample must still revoke the bounded
    # handoff, rather than blindly keeping the response window alive.
    clock[0] += .05
    r.get_steering_feedback = lambda: feedback(clock[0] - .01, 5, 9)
    r._send_follow_wheel_targets(-9, -9, "FOLLOW20")
    assert d.pairs[-1] == (0, 0)
