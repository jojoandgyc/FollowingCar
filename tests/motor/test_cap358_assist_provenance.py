"""CAP384 ordinary attenuation must not borrow the stronger assist gates.

Fake clock/encoder/serial only. Raw right wheel sign is inverted by hardware.
"""
import pytest

from test_cap837_turn_buildup import setup, image_intent
from test_visible_wheel_continuity import feedback


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("age", [.249, .251, .29, .32])
def test_legal_yaw_only_attenuation_not_rejected_as_boost(monkeypatch, caplog, sign, age):
    r, o, d, _, clock = setup(monkeypatch)
    o._fresh_depth_linear_snapshot = lambda uid, now=None: None
    intent = o._lateral_intent_store.publish(image_intent(
        clock[0], 384, sign, capture_timestamp=clock[0]-age, valid_until=10.35))
    o._has_fresh_lateral_yaw = lambda uid: uid == 1 and intent.valid(clock[0])
    r.get_steering_feedback = lambda: feedback(clock[0], sign*7, -sign*7)
    with caplog.at_level("INFO"):
        assert r._send_follow_wheel_targets(sign*10, sign*10, "CAP384")
    assert d.pairs == [(sign*7, sign*7)]
    assert "response_adjusted=False ordinary_mode_limited=True" in caplog.text
    assert "buildup_evidence_expired_before_write" not in caplog.text


@pytest.mark.parametrize("loss", ["yaw", "uid", "danger", "feedback"])
def test_ordinary_attenuation_keeps_final_safety_checks(monkeypatch, loss):
    r, o, d, _, clock = setup(monkeypatch)
    o._fresh_depth_linear_snapshot = lambda uid, now=None: None
    o._lateral_intent_store.publish(image_intent(10., 384, -1, capture_timestamp=9.71))
    r.get_steering_feedback = lambda: feedback(clock[0], -7, 7)
    def safety(_):
        if loss == "yaw": o._has_fresh_lateral_yaw = lambda uid: False
        elif loss == "uid": o._follow_controller.active_target_id = 2
        elif loss == "feedback": clock[0] += .16
        return loss == "danger"
    r.hard_stop_check = safety
    r._send_follow_wheel_targets(-10, -10, "CAP384")
    assert all(pair == (0, 0) for pair in d.pairs)
    if loss == "danger": assert d.stops


def arm_boost(r, o, clock):
    for t in [10., 10.11]:
        clock[0] = t
        o._lateral_intent_store.publish(image_intent(t, 370))
        r._send_follow_wheel_targets(88, -76, "ARM")
    clock[0] = 10.16
    o._lateral_intent_store.publish(image_intent(clock[0], 371))


def test_safety_zero_is_not_vetoed_by_expired_boost_proof(monkeypatch, caplog):
    r, o, d, _, clock = setup(monkeypatch)
    arm_boost(r, o, clock)
    def wait(*args, **kwargs):
        clock[0] += .12
        o._lateral_turn_response_policy = (o._lateral_intent_store.snapshot().sequence, False)
        return (0, 0), "cross_wait_zero"
    r._visible_wheel_guard.limit = wait
    with caplog.at_level("INFO"):
        assert r._send_follow_wheel_targets(88, -76, "WAIT")
    assert d.pairs[-1] == (0, 0)
    assert "response_adjusted=True" in caplog.text
    assert "turn_response_veto" not in caplog.text


def test_true_boost_expiry_checked_after_final_safety_callback(monkeypatch, caplog):
    r, o, d, _, clock = setup(monkeypatch)
    arm_boost(r, o, clock)
    def slow_check(_):
        clock[0] += .11  # valid generic 150ms feedback, invalid 100ms assist
        return False
    r.hard_stop_check = slow_check
    previous = list(d.pairs)
    with caplog.at_level("INFO"):
        assert r._send_follow_wheel_targets(88, -76, "EXPIRED_BOOST") is False
    assert d.pairs == previous
    assert "buildup_evidence_expired_before_write" in caplog.text
