"""Independent, advancing-clock acceptance for the normal-follow short path.

These tests never open a camera or serial port.  A periodic writer may reuse a
plan only until its original capture-based deadline; repeated test callbacks do
not make perception, feedback, or authority perpetually fresh.
"""
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from car_control_modular.short_follow import (
    ShortFollowConfig,
    ShortFollowController,
    ShortFollowObservation,
)
from car_control_modular.short_follow_adapter import ShortFollowAdapter
from car_control_modular.detector_identity_lease import validated_visual_observation

sys.path.append(str(Path(__file__).resolve().parents[1] / "motor"))
from test_depth_drive_rpm import make_runtime


def observation(now, *, uid=1, capture_id=1, distance=2.0, center=0.5,
                capture_age=0.04, depth_age=0.02, raw=None):
    return ShortFollowObservation(
        uid=uid,
        capture_id=capture_id,
        capture_timestamp=now-capture_age,
        depth_timestamp=now-depth_age,
        distance_m=distance,
        center_x_ratio=center,
        raw_distance_m=raw,
    )


@pytest.fixture
def short():
    controller = ShortFollowController(ShortFollowConfig(enabled=True))
    controller.activate(1, now=99.0)
    return SimpleNamespace(controller=controller, now=100.0)


def submit(short, **kwargs):
    return short.controller.update(observation(short.now, **kwargs), now=short.now)


def test_three_second_distance_turn_stream_has_no_artificial_stop(short):
    start = short.now
    initial_epoch = short.controller.snapshot().epoch
    previous_sequence = None
    plans = []
    tick_commands = []
    for tick in range(60):
        short.now = start + tick * 0.05
        if tick % 4 == 0:
            index = tick // 4
            plan = submit(
                short, capture_id=100+index,
                distance=(1.65, 1.85, 2.30, 1.70, 2.10)[index % 5],
                center=(0.25, 0.50, 0.75, 0.53, 0.47)[index % 5],
            )
            assert plan is not None and plan.moving
            assert previous_sequence is None or plan.sequence > previous_sequence
            previous_sequence = plan.sequence
            plans.append(plan)
        snapshot = short.controller.snapshot()
        assert snapshot.active and snapshot.uid == 1
        assert snapshot.epoch == initial_epoch
        assert snapshot.plan is plans[-1]
        assert snapshot.plan.expires_at > short.now
        assert snapshot.plan.expires_at == pytest.approx(
            min(snapshot.plan.depth_timestamp+0.30,
                snapshot.plan.capture_timestamp+0.50))
        tick_commands.append((snapshot.plan.left_rpm, snapshot.plan.right_rpm))

    assert len(plans) == 15
    assert len(tick_commands) == 60
    assert all(left > 0 and right > 0 for left, right in tick_commands)
    assert any(left < right for left, right in tick_commands)
    assert any(left > right for left, right in tick_commands)
    assert any(left == right for left, right in tick_commands)
    assert len({(left+right)/2 for left, right in tick_commands}) > 1
    assert short.now-start == pytest.approx(2.95)


@pytest.mark.parametrize("change", ["duplicate", "new_capture_same_depth", "out_of_order",
                                   "stale_depth", "stale_capture", "wrong_uid"])
def test_rejected_update_preserves_plan_without_refreshing_epoch_or_deadline(short, change):
    sample = observation(short.now, capture_id=10)
    plan = short.controller.update(sample, now=short.now)
    epoch = short.controller.snapshot().epoch
    short.now += 0.05
    candidate = {
        "duplicate": sample,
        "new_capture_same_depth": replace(sample, capture_id=11,
                                            capture_timestamp=short.now-0.01,
                                            center_x_ratio=0.8),
        "out_of_order": replace(sample, capture_id=9,
                                 depth_timestamp=sample.depth_timestamp-0.01,
                                 capture_timestamp=sample.capture_timestamp-0.01),
        "stale_depth": observation(short.now, capture_id=11, depth_age=0.301),
        "stale_capture": observation(short.now, capture_id=11, capture_age=0.501),
        "wrong_uid": observation(short.now, uid=2, capture_id=11),
    }[change]
    assert short.controller.update(candidate, now=short.now) is None
    snapshot = short.controller.snapshot()
    assert snapshot.plan is plan
    assert snapshot.epoch == epoch
    assert snapshot.plan.expires_at == plan.expires_at


def test_distance_stop_resume_hysteresis_is_not_a_fixed_stop_timer(short):
    for index, (distance, moving) in enumerate([
        (1.54, False), (1.56, True), (1.51, True),
        (1.49, False), (1.53, False), (1.56, True),
    ]):
        short.now += 0.05
        plan = submit(short, capture_id=10+index, distance=distance)
        assert plan is not None
        assert plan.moving is moving
        if moving:
            assert plan.left_rpm > 0 and plan.right_rpm > 0
        else:
            assert plan.left_rpm == plan.right_rpm == 0


def test_fresh_close_raw_measurement_cannot_be_masked_by_far_smoothed_distance(short):
    assert submit(short).moving
    short.now += 0.05
    plan = submit(short, capture_id=2, distance=2.2, raw=1.45)
    assert plan is not None and not plan.moving
    assert plan.left_rpm == plan.right_rpm == 0


def test_revoke_rejects_inflight_measurement_but_newly_captured_evidence_can_resume(short):
    before = submit(short)
    short.now += 0.10
    short.controller.revoke("explicit_stop", now=short.now)
    stopped = short.controller.snapshot()
    assert stopped.active and stopped.epoch > before.epoch
    assert stopped.plan is None or not stopped.plan.moving
    stale = observation(short.now, capture_id=2, capture_age=0.03, depth_age=0.01)
    short.now += 0.05
    assert short.controller.update(stale, now=short.now) is None
    assert short.controller.snapshot().plan is None or not short.controller.snapshot().plan.moving
    new_plan = submit(short, capture_id=3, capture_age=0.01, depth_age=0.01)
    assert new_plan is not None and new_plan.moving
    assert new_plan.epoch == stopped.epoch


def test_deactivate_needs_new_activation_and_does_not_accept_old_results(short):
    previous = submit(short)
    short.now += 0.10
    short.controller.deactivate("identity_conflict", now=short.now)
    snapshot = short.controller.snapshot()
    assert not snapshot.active
    assert snapshot.plan is None or not snapshot.plan.moving
    short.now += 0.05
    assert submit(short, capture_id=2, capture_age=0.01, depth_age=0.01) is None
    short.controller.activate(2, now=short.now)
    assert short.controller.snapshot().epoch > previous.epoch
    short.now += 0.05
    assert submit(short, uid=1, capture_id=3, capture_age=0.01, depth_age=0.01) is None
    plan = submit(short, uid=2, capture_id=4, capture_age=0.01, depth_age=0.01)
    assert plan is not None and plan.moving and plan.uid == 2


@pytest.mark.parametrize("field,value", [
    ("distance_m", float("nan")), ("distance_m", float("inf")),
    ("distance_m", -1.0), ("center_x_ratio", float("nan")),
    ("capture_timestamp", float("nan")), ("depth_timestamp", float("nan")),
])
def test_malformed_sample_is_not_a_new_motion_authority(short, field, value):
    plan = submit(short)
    short.now += 0.05
    sample = replace(observation(short.now, capture_id=2), **{field: value})
    assert short.controller.update(sample, now=short.now) is None
    assert short.controller.snapshot().plan is plan


@pytest.mark.parametrize("first_expiry", ["depth", "visual"])
def test_plan_deadline_follows_oldest_source_even_if_other_source_is_fresh(short, first_expiry):
    sample = observation(short.now, capture_age=0.45 if first_expiry == "visual" else 0.01,
                         depth_age=0.01)
    plan = short.controller.update(sample, now=short.now)
    assert plan is not None and plan.valid(short.now)
    expected = sample.capture_timestamp+0.50 if first_expiry == "visual" else sample.depth_timestamp+0.30
    assert plan.expires_at == pytest.approx(expected)
    assert plan.valid(expected-0.001)
    assert not plan.valid(expected)
    assert not plan.valid(expected+0.001)
    # Replaying the former valid observation after silence cannot resurrect it.
    assert short.controller.update(sample, now=expected+0.001) is None
    assert short.controller.snapshot().plan is plan
    assert not short.controller.snapshot().plan.valid(expected+0.001)


@pytest.mark.parametrize("field", ["capture_timestamp", "depth_timestamp"])
def test_future_dated_source_is_rejected_without_renewal(short, field):
    previous = submit(short)
    short.now += 0.05
    sample = replace(observation(short.now, capture_id=2), **{field: short.now+0.001})
    assert short.controller.update(sample, now=short.now) is None
    assert short.controller.snapshot().plan is previous


class _QuietLogger:
    def info(self, *args, **kwargs):
        pass


def _adapter_chain():
    controller = ShortFollowController(ShortFollowConfig(enabled=True))
    owner = SimpleNamespace(
        running=True,
        _follow_controller=SimpleNamespace(active_target_id=1, search_state="none"),
        _validated_visual_observation=None,
        _detector_identity_lease=None,
        _runtime_shutdown_requested=False,
        _brake_hold_active=False,
        _reacquire_depth_pending=False,
    )
    adapter = ShortFollowAdapter(owner, controller, _QuietLogger())
    return SimpleNamespace(controller=controller, owner=owner, adapter=adapter, now=100.0)


def _adapter_frame(chain, *, capture_id=1, distance=2.0, center=0.5,
                   new_identity=True, sample_stamp=None):
    capture = chain.now-0.04
    depth = chain.now-0.02 if sample_stamp is None else sample_stamp
    if new_identity:
        chain.owner._validated_visual_observation = validated_visual_observation(
            uid=1, track_id=1, capture=capture_id, timestamp=capture,
            now=chain.now, visibility_window=0.50, capture_max_age_sec=0.50,
        )
    target = SimpleNamespace(track_id=1, center=(center*640, 240))
    frame = SimpleNamespace(
        width=640, capture_frame_id=capture_id, capture_timestamp=capture,
        distance_m=distance,
        distance_state=SimpleNamespace(sample_timestamp=depth, raw_distance_m=distance,
                                       safety_distance_m=None),
        hazard=SimpleNamespace(active=False),
        obstacles=SimpleNamespace(front=False, left=False, right=False),
    )
    return frame, target


def _deliver(chain, frame, target, *, fresh=True):
    return chain.adapter.handle(frame, target, is_fresh_depth=fresh,
                                control_source="depth30", target_steerable=True,
                                low_quality_visible=False, now=chain.now)


def test_real_adapter_three_second_publication_stream_bypasses_legacy_actions(monkeypatch):
    chain = _adapter_chain()
    monkeypatch.setattr("car_control_modular.detector_identity_lease.time.monotonic", lambda: chain.now)
    start = chain.now
    original_epoch = None
    previous = None
    for tick in range(60):
        chain.now = start+tick*0.05
        if tick % 4 == 0:
            frame, target = _adapter_frame(
                chain, capture_id=100+tick//4,
                distance=(1.70, 2.20, 1.80)[tick//4 % 3],
                center=(0.25, 0.50, 0.75)[tick//4 % 3],
            )
            assert _deliver(chain, frame, target)
            previous = chain.controller.snapshot().plan
            assert previous is not None and previous.moving
        else:
            # A normal failed ranging attempt is consumed by the short path,
            # but neither the visual lease nor source timestamps get refreshed.
            assert _deliver(chain, frame, target, fresh=False)
            assert chain.controller.snapshot().plan is previous
        state = chain.controller.snapshot()
        original_epoch = state.epoch if original_epoch is None else original_epoch
        assert state.active and state.epoch == original_epoch
        assert state.plan.valid(chain.now)
        assert chain.owner._short_follow_handled_frame
        assert chain.owner._depth30_linear_snapshot is None


def test_adapter_identity_silence_cannot_renew_old_plan_using_new_depth_only(monkeypatch):
    chain = _adapter_chain()
    monkeypatch.setattr("car_control_modular.detector_identity_lease.time.monotonic", lambda: chain.now)
    frame, target = _adapter_frame(chain)
    assert _deliver(chain, frame, target)
    old = chain.controller.snapshot().plan
    assert old is not None
    chain.now += 0.60
    frame, target = _adapter_frame(chain, capture_id=2, new_identity=False)
    _deliver(chain, frame, target)
    latest = chain.controller.snapshot().plan
    assert latest is None or latest is old
    assert latest is None or not latest.valid(chain.now)


@pytest.mark.parametrize("danger", ["front", "left", "right", "hazard", "shutdown"])
def test_adapter_hard_safety_does_not_preserve_normal_follow_owner(danger, monkeypatch):
    chain = _adapter_chain()
    monkeypatch.setattr("car_control_modular.detector_identity_lease.time.monotonic", lambda: chain.now)
    frame, target = _adapter_frame(chain)
    assert _deliver(chain, frame, target)
    assert chain.controller.snapshot().plan.moving
    chain.now += 0.05
    frame, target = _adapter_frame(chain, capture_id=2)
    if danger == "hazard":
        frame.hazard.active = True
    elif danger == "shutdown":
        chain.owner._runtime_shutdown_requested = True
    else:
        setattr(frame.obstacles, danger, True)
    assert not _deliver(chain, frame, target)
    state = chain.controller.snapshot()
    assert not state.active
    assert state.plan is None


def _writer_chain(monkeypatch):
    chain = _adapter_chain()
    runtime, owner, driver, symbols = make_runtime(max_rpm=200)
    owner.__dict__.update(chain.owner.__dict__)
    chain.owner = owner
    owner._short_follow = chain.controller
    chain.adapter = ShortFollowAdapter(owner, chain.controller, runtime.logger)
    chain.runtime, chain.driver, chain.symbols = runtime, driver, symbols
    chain.feedback = SimpleNamespace(timestamp=chain.now, left_forward_rpm=0.0,
                                     right_forward_rpm=0.0, trustworthy=True)
    # Feedback is an actual cached sample. It only becomes new when the test
    # explicitly publishes another sample; calling the reader does not date it.
    runtime.get_steering_feedback = lambda: chain.feedback
    monkeypatch.setattr("car_control_modular.short_follow_executor.time.monotonic", lambda: chain.now)
    chain.writer = runtime._short_follow_executor_instance()
    return chain


def _advance_writer(chain, now, *, feedback=True):
    chain.now = now
    if feedback:
        left, raw_right = chain.driver.pairs[-1] if chain.driver.pairs else (0, 0)
        chain.feedback = SimpleNamespace(timestamp=now, left_forward_rpm=float(left),
                                         right_forward_rpm=float(-raw_right), trustworthy=True)


def test_real_adapter_controller_writer_three_second_stream_has_no_zero_packets(monkeypatch):
    chain = _writer_chain(monkeypatch)
    start = chain.now
    for tick in range(60):
        _advance_writer(chain, start+tick*0.05)
        if tick % 4 == 0:
            frame, target = _adapter_frame(
                chain, capture_id=100+tick//4,
                distance=(1.70, 2.20, 1.80)[tick//4 % 3],
                center=(0.25, 0.50, 0.75)[tick//4 % 3],
            )
            assert _deliver(chain, frame, target)
        else:
            assert _deliver(chain, frame, target, fresh=False)
        assert chain.runtime._service_short_follow()
        plan = chain.controller.snapshot().plan
        assert chain.driver.pairs[-1] == (plan.left_rpm, -plan.right_rpm)
        assert not chain.driver.stops
    assert len(chain.driver.pairs) == 60
    assert all(left > 0 and right < 0 for left, right in chain.driver.pairs)
    # The paired writer preserves continuity without silently retaining the
    # rejected miniverify 40 RPM ceiling after the PI request is restored.
    assert any(max(left, -right) > 40 for left, right in chain.driver.pairs)
    assert all(max(left, -right) <= chain.controller.config.max_rpm
               for left, right in chain.driver.pairs)

    # No callback produces fresh evidence after the stream. The executor alone
    # must observe the old deadline and stop despite a frozen vision thread.
    latest = chain.controller.snapshot().plan
    before = len(chain.driver.pairs)
    _advance_writer(chain, latest.expires_at+0.001)
    assert chain.runtime._service_short_follow()
    assert chain.driver.stops
    assert len(chain.driver.pairs) == before
    stop_count = len(chain.driver.stops)
    _advance_writer(chain, chain.now+0.05)
    assert chain.runtime._service_short_follow()
    assert len(chain.driver.stops) == stop_count  # No repeated 20 Hz STOP spam.


def test_real_writer_distance_stop_then_new_evidence_resumes_without_fixed_hold(monkeypatch):
    chain = _writer_chain(monkeypatch)
    for index, (distance, expected_moving) in enumerate([
        (2.0, True), (1.49, False), (1.53, False), (1.60, True),
    ]):
        _advance_writer(chain, 100.0+index*0.05)
        frame, target = _adapter_frame(chain, capture_id=index+1, distance=distance)
        assert _deliver(chain, frame, target)
        before = len(chain.driver.pairs)
        assert chain.runtime._service_short_follow()
        if expected_moving:
            assert len(chain.driver.pairs) == before+1
            assert chain.driver.pairs[-1][0] > 0 and chain.driver.pairs[-1][1] < 0
        else:
            assert len(chain.driver.pairs) == before
            assert chain.driver.stops
    assert chain.now == pytest.approx(100.15)


@pytest.mark.parametrize("trigger", ["explicit_stop", "shutdown", "hard_stop", "identity_change"])
def test_real_writer_hard_stop_wins_before_periodic_refresh(monkeypatch, trigger):
    chain = _writer_chain(monkeypatch)
    frame, target = _adapter_frame(chain)
    assert _deliver(chain, frame, target)
    assert chain.runtime._service_short_follow()
    writes = len(chain.driver.pairs)
    _advance_writer(chain, chain.now+0.001)
    if trigger == "explicit_stop":
        chain.owner._explicit_stop_requested = True
    elif trigger == "shutdown":
        chain.owner._runtime_shutdown_requested = True
    elif trigger == "hard_stop":
        chain.runtime.hard_stop_check = lambda *_: True
    else:
        chain.owner._follow_controller.active_target_id = 2
    assert chain.runtime._service_short_follow()
    assert chain.driver.stops
    assert len(chain.driver.pairs) == writes


def test_partial_wheel_write_failure_latches_fault_and_cannot_be_cleared_by_new_observation(monkeypatch):
    chain = _writer_chain(monkeypatch)
    frame, target = _adapter_frame(chain)
    assert _deliver(chain, frame, target)
    left_write = chain.driver.set_left_speed
    failed = []

    def fail_first_nonzero_left(value):
        if value and not failed:
            failed.append(value)
            raise OSError("injected fake serial left ACK failure")
        left_write(value)

    monkeypatch.setattr(chain.driver, "set_left_speed", fail_first_nonzero_left)
    with pytest.raises(OSError, match="fake serial"):
        chain.runtime._service_short_follow()
    assert failed
    assert chain.runtime.backend.motion_write_fault
    assert chain.driver.stops
    assert not any(left > 0 or right < 0 for left, right in chain.driver.pairs)

    _advance_writer(chain, chain.now+0.10)
    frame, target = _adapter_frame(chain, capture_id=2)
    assert _deliver(chain, frame, target)
    assert chain.runtime._service_short_follow()
    assert chain.runtime.backend.motion_write_fault
    assert not any(left > 0 or right < 0 for left, right in chain.driver.pairs)


def test_normal_adapter_delivery_cannot_clear_a_new_explicit_emergency_stop(monkeypatch):
    chain = _writer_chain(monkeypatch)
    frame, target = _adapter_frame(chain)
    assert _deliver(chain, frame, target)
    assert chain.runtime._service_short_follow()
    before = len(chain.driver.pairs)

    _advance_writer(chain, chain.now+0.05)
    frame, target = _adapter_frame(chain, capture_id=2)
    # The stop arrives while this already-captured normal depth result awaits
    # commit, before the periodic writer has consumed the emergency flag.
    chain.owner._explicit_stop_requested = True
    chain.owner._last_explicit_stop_reason = "manual_emergency"
    _deliver(chain, frame, target)
    assert chain.runtime._service_short_follow()
    assert chain.driver.stops
    assert len(chain.driver.pairs) == before


def test_new_observation_cannot_hide_a_completed_external_motor_stop(monkeypatch):
    chain = _writer_chain(monkeypatch)
    frame, target = _adapter_frame(chain)
    assert _deliver(chain, frame, target)
    assert chain.runtime._service_short_follow()
    writes = len(chain.driver.pairs)
    _advance_writer(chain, chain.now+0.05)
    chain.runtime.backend.send_stop("external-emergency", mode="emergency")
    frame, target = _adapter_frame(chain, capture_id=2)
    assert _deliver(chain, frame, target)
    assert chain.runtime._service_short_follow()
    assert chain.driver.stops
    assert len(chain.driver.pairs) == writes
    assert chain.controller.snapshot().plan is None


def test_revoke_while_preparing_serial_prevents_old_pair_from_being_sent(monkeypatch):
    chain = _writer_chain(monkeypatch)
    frame, target = _adapter_frame(chain)
    assert _deliver(chain, frame, target)
    old_epoch = chain.controller.snapshot().epoch
    prepare = chain.runtime.backend.prepare_speed_mode
    revoked = []

    def revoke_during_prepare():
        prepare()
        if not revoked:
            chain.controller.revoke("hard_event_during_prepare", now=chain.now)
            revoked.append(True)

    monkeypatch.setattr(chain.runtime.backend, "prepare_speed_mode", revoke_during_prepare)
    assert chain.runtime._service_short_follow()
    assert revoked and chain.controller.snapshot().epoch > old_epoch
    assert chain.driver.stops
    assert not chain.driver.pairs


def test_real_writer_hardware_ceiling_prevents_integral_windup_without_zero(monkeypatch):
    chain = _writer_chain(monkeypatch)
    chain.runtime.backend.config = replace(chain.runtime.backend.config, max_target=20)
    plans = []
    for index in range(8):
        _advance_writer(chain, 100.+index*.1)
        frame, target = _adapter_frame(chain, capture_id=index+1, distance=1.56)
        assert _deliver(chain, frame, target)
        plans.append(chain.controller.snapshot().plan)
        assert chain.runtime._service_short_follow()
        assert chain.driver.pairs[-1] == (20, -20)
        assert not chain.driver.stops
    assert all(plan.base_rpm > 20 for plan in plans)
    # Per-sample request diagnostics include that sample's tentative I. The
    # actual write feeds saturation back so it cannot accumulate each frame.
    assert plans[1].i_rpm > 0
    assert all(plan.i_rpm == pytest.approx(plans[1].i_rpm) for plan in plans[1:])

