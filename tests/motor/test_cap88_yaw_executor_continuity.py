"""Anti-overshoot through the real paired writer; only FakeDriver is used.

The cases mirror CAP88/92's late centering transition, not physical vehicle
motion. An encoder-only projection may remove yaw, never renew the depth plan
or insert a stop into lawful forward translation.
"""
from types import SimpleNamespace

import pytest

from car_control_modular.short_follow import ShortFollowObservation, ShortFollowYawObservation
from test_short_follow_executor import publish, short_runtime


def feedback(rt, clock, *, heading=0., rate=0., confirmed=True, age=0., wheels=(0., 0.)):
    sample = SimpleNamespace(timestamp=clock[0]-age, trustworthy=True,
        left_forward_rpm=wheels[0], right_forward_rpm=wheels[1],
        integrated_yaw_right_deg=heading, yaw_rate_right_dps=rate,
        yaw_rate_confirmed=confirmed)
    rt.get_steering_feedback = lambda: sample
    return sample


def yaw(rt, clock, cap, x, *, capture_yaw=0.):
    return rt.owner._short_follow.update_lateral(ShortFollowYawObservation(
        1, cap, clock[0], x, capture_yaw), clock[0])


def pair(driver):
    left, wire_right = driver.pairs[-1]
    return left, -wire_right


def start_arc(monkeypatch, *, distance=2., x=.2):
    rt, owner, driver, symbols, clock = short_runtime(monkeypatch)
    clock[0] += .01
    publish(rt, clock, 83, distance=distance, x=x)
    clock[0] += .001
    plan = yaw(rt, clock, 88, x)
    assert plan is not None
    feedback(rt, clock)
    rt._service_short_follow()
    assert not driver.stops
    assert pair(driver)[0] < pair(driver)[1]
    return rt, owner, driver, symbols, clock, plan


def test_encoder_centering_removes_yaw_without_zeroing_forward(monkeypatch, caplog):
    rt, owner, driver, _, clock, source = start_arc(monkeypatch)
    initial_pair = pair(driver)
    clock[0] += .06
    feedback(rt, clock, heading=-30.)
    with caplog.at_level("INFO"):
        rt._service_short_follow()
    assert pair(driver) == (max(initial_pair), max(initial_pair))
    assert not driver.stops
    assert owner._short_follow.snapshot().plan is source
    actual = owner._short_follow_last_applied_plan
    assert actual.expires_at == source.expires_at
    assert actual.depth_timestamp == source.depth_timestamp
    assert actual.capture_timestamp == source.capture_timestamp
    assert actual.epoch == source.epoch
    assert "yaw_cap=88" in caplog.text


def test_pivot_centering_has_no_new_settle_barrier(monkeypatch):
    rt, owner, driver, _, clock, source = start_arc(monkeypatch, distance=1.4)
    clock[0] += .06
    feedback(rt, clock, heading=-30.)
    rt._service_short_follow()
    assert driver.stops == [1]
    writer = rt._short_follow_executor
    assert writer._entry_stop_at is None
    assert not getattr(owner, "_brake_hold_active", False)
    assert owner._short_follow.snapshot().epoch == source.epoch

    # A genuinely NEW position needs yaw again. No depth refresh is needed,
    # but its original deadline remains unchanged. Do not demand two quiet
    # encoder frames after an ordinary near-distance centering STOP.
    clock[0] += .06
    resumed = yaw(rt, clock, 95, .2, capture_yaw=-30.)
    assert resumed is not None
    feedback(rt, clock, heading=-30.)
    rt._service_short_follow()
    assert pair(driver)[0] < 0 < pair(driver)[1]
    assert len(driver.pairs) == 2
    assert driver.stops == [1]
    assert resumed.expires_at == source.expires_at


@pytest.mark.parametrize("invalid", ["unconfirmed", "missing", "nan_heading", "nan_rate", "stale"])
def test_unusable_yaw_is_not_a_new_stop_condition(monkeypatch, invalid):
    rt, owner, driver, _, clock, _ = start_arc(monkeypatch)
    original = pair(driver)
    clock[0] += .06
    sample = feedback(rt, clock, heading=-30.,
        confirmed=invalid != "unconfirmed", age=.151 if invalid == "stale" else 0.)
    if invalid == "missing":
        del sample.integrated_yaw_right_deg
    elif invalid == "nan_heading":
        sample.integrated_yaw_right_deg = float("nan")
    elif invalid == "nan_rate":
        sample.yaw_rate_right_dps = float("nan")
    rt._service_short_follow()
    assert not driver.stops
    if invalid == "stale":
        # Existing encoder-age policy may cap speed to 40, but never zero.
        assert min(pair(driver)) > 0 and pair(driver)[0] < pair(driver)[1]
    else:
        assert pair(driver) == original


def test_yaw_projection_never_extends_motion_deadline(monkeypatch):
    rt, owner, driver, _, clock, source = start_arc(monkeypatch)
    clock[0] = source.expires_at + .001
    feedback(rt, clock, heading=-30.)
    assert yaw(rt, clock, 92, .5, capture_yaw=-30.) is None
    before = len(driver.pairs)
    rt._service_short_follow()
    assert len(driver.pairs) == before
    assert driver.stops == [1]


def test_lateral_update_during_prepare_is_sent_as_latest_complete_pair(monkeypatch):
    rt, owner, driver, _, clock, source = start_arc(monkeypatch)
    old_prepare = rt.backend.prepare_speed_mode
    published = []
    clock[0] += .06
    feedback(rt, clock)
    def prepare():
        old_prepare()
        if not published:
            clock[0] += .001
            assert yaw(rt, clock, 92, .5) is not None
            published.append(True)
    rt.backend.prepare_speed_mode = prepare
    rt._service_short_follow()
    assert pair(driver)[0] == pair(driver)[1] > 0
    assert not driver.stops
    latest = owner._short_follow.snapshot().plan
    assert latest.yaw_capture_id == 92
    assert latest.depth_timestamp == source.depth_timestamp
    assert latest.expires_at == source.expires_at


def test_same_visual_frame_cannot_reexpand_executed_turn(monkeypatch):
    rt, owner, driver, _, clock, source = start_arc(monkeypatch)
    clock[0] += .06
    feedback(rt, clock, heading=-30.)
    rt._service_short_follow()
    assert pair(driver)[0] == pair(driver)[1]
    clock[0] += .06
    feedback(rt, clock, heading=0.)
    # A fresh depth from the same detector frame may legitimately raise the
    # forward base. It does not undo the already executed yaw taper.
    updated = owner._short_follow.update(ShortFollowObservation(
        1, source.capture_id, source.capture_timestamp, clock[0], 2.3, .2), clock[0])
    assert updated is not None
    assert updated.yaw_capture_id == source.yaw_capture_id
    rt._service_short_follow()
    assert pair(driver)[0] == pair(driver)[1] == int(updated.base_rpm)
    assert not driver.stops
    # A genuinely new visual observation is allowed to ask for yaw again.
    clock[0] += .06
    assert yaw(rt, clock, 92, .2) is not None
    feedback(rt, clock)
    rt._service_short_follow()
    assert pair(driver)[0] < pair(driver)[1]
    assert not driver.stops


def test_same_pivot_frame_cannot_restart_after_center_stop(monkeypatch):
    rt, owner, driver, _, clock, _ = start_arc(monkeypatch, distance=1.4)
    clock[0] += .06
    feedback(rt, clock, heading=-30.)
    rt._service_short_follow()
    assert driver.stops == [1]
    clock[0] += .06
    feedback(rt, clock, heading=0.)
    rt._service_short_follow()
    assert len(driver.pairs) == 1
    assert driver.stops == [1]
    assert rt._short_follow_executor._entry_stop_at is None


@pytest.mark.parametrize("direction", [-1, 1])
def test_confirmed_rate_tapers_only_toward_target_without_stopping_forward(monkeypatch, direction):
    rt, owner, driver, _, clock = short_runtime(monkeypatch)
    clock[0] += .01
    x = .5 + direction * .11
    publish(rt, clock, 83, distance=2., x=x)
    clock[0] += .001
    assert yaw(rt, clock, 88, x) is not None
    feedback(rt, clock, rate=-direction*20.)
    rt._service_short_follow()
    original = pair(driver)
    assert direction*(original[0]-original[1]) > 0
    clock[0] += .06
    feedback(rt, clock, rate=direction*20.)
    rt._service_short_follow()
    assert pair(driver)[0] == pair(driver)[1] > 0
    clock[0] += .06
    feedback(rt, clock, rate=0.)
    rt._service_short_follow()
    assert pair(driver)[0] == pair(driver)[1] > 0
    assert not driver.stops


@pytest.mark.parametrize("adverse", ["hard_stop", "identity_rejected", "feedback_fault"])
def test_yaw_smoothing_does_not_override_real_stop(monkeypatch, adverse):
    rt, owner, driver, _, clock, _ = start_arc(monkeypatch)
    clock[0] += .06
    sample = feedback(rt, clock, heading=-30.)
    if adverse == "hard_stop":
        rt.hard_stop_check = lambda _action: True
    elif adverse == "identity_rejected":
        owner._validated_visual_observation = False
    else:
        sample.left_error = 1
    before = len(driver.pairs)
    rt._service_short_follow()
    assert len(driver.pairs) == before
    assert driver.stops == [1]
