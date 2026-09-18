"""Matching plus correction, wakeup scheduling, actuator limit feedback."""
from dataclasses import replace
from types import SimpleNamespace
import threading
import pytest
import request_0513_modular as runtime
from car_control_modular.control_types import ControlAction
from car_control_modular.steering_pid import LongitudinalDistancePid, DistancePidConfig
from test_distance_tracking_response import setup, decide
from test_depth_target_snapshot import owner as context_owner, persons, NOW


@pytest.fixture
def limits(monkeypatch):
    for name, value in dict(FORWARD_MAX_RPM=100,
                            ASTRA_DEPTH_LONGITUDINAL_MAX_FORWARD_PERCENT=20,
                            ASTRA_DEPTH_LONGITUDINAL_FAR_FORWARD_PERCENT=60,
                            ASTRA_DEPTH_NEAR_GUARD_DISTANCE_M=1.8,
                            ASTRA_DEPTH_LONGITUDINAL_FAR_DISTANCE_M=2.2,
                            TARGET_DISTANCE=1.5, DISTANCE_PID_DEADBAND_M=.03).items():
        monkeypatch.setattr(runtime, name, value)


def capped(distance, requested, base=None, kind="forward"):
    return runtime.PersonTracker._cap_depth_longitudinal_actions(
        [getattr(ControlAction, kind)(requested, "test")], distance, base,
    )[0].speed_percent


def test_log_1887m_can_close_gap_not_just_match_3063_rpm(limits):
    assert capped(1.887, 41, 30.63) == 41
    assert capped(1.858, 38, 28.46) == 38
    assert capped(1.887, 41) == 29


@pytest.mark.parametrize("distance", [1.49, 1.5, 1.529, 1.53])
def test_no_extra_catchup_in_target_band(limits, distance):
    assert capped(distance, 80, 30) == 30


def test_extra_catchup_total_caps_and_reverse_unchanged(limits):
    assert capped(1.9, 99, 30) == 50
    assert capped(1.9, 99, 99) == 60
    assert capped(3.5, 99, 40) == 60
    assert capped(1.9, 12, 30) == 12
    assert capped(1.9, 80, 30, "backward") == capped(1.9, 80, None, "backward")
    assert capped(1.9, 0, 30) == 0


@pytest.mark.parametrize("base", [None, float("nan"), float("inf"), -1, 0])
def test_invalid_or_absent_matching_evidence_cannot_add_budget(limits, base):
    assert capped(1.9, 80, base) == 30


def test_catchup_rpm_budget_converts_with_max_rpm(limits, monkeypatch):
    monkeypatch.setattr(runtime, "FORWARD_MAX_RPM", 200)
    assert capped(1.9, 80, 60) == 40  # matching 60RPM + correction 20RPM


def test_new_roi_wakes_waiter_without_renewing_capture_deadline(context_owner):
    context_owner._longitudinal_wake_event = threading.Event()
    context_owner._publish_longitudinal_context(640, 480, persons())
    assert context_owner._longitudinal_wake_event.is_set()
    assert context_owner._longitudinal_context["capture_timestamp"] == NOW-.1


def test_stop_wakes_depth_worker(context_owner):
    context_owner._longitudinal_stop_event = threading.Event()
    context_owner._longitudinal_wake_event = threading.Event()
    context_owner._longitudinal_thread = None
    context_owner._clear_longitudinal_context = lambda **kw: None
    context_owner._stop_longitudinal_thread()
    assert context_owner._longitudinal_stop_event.is_set()
    assert context_owner._longitudinal_wake_event.is_set()


def test_wakeup_loop_rechecks_latest_context_and_stop(context_owner):
    context_owner._publish_longitudinal_context(640, 480, persons())
    stop = threading.Event()
    waits, calls = [], []
    context_owner._longitudinal_stop_event = stop
    context_owner._longitudinal_wake_event = SimpleNamespace(
        clear=lambda: None, wait=lambda duration: (waits.append(duration), stop.set()),
    )
    context_owner._queue_actions_for_persons = lambda *a, **k: calls.append(k)
    context_owner._longitudinal_control_loop()
    assert len(calls) == len(waits) == 1
    assert calls[0]["evidence_capture_timestamp"] == NOW-.1


def test_cap_feedback_freezes_new_integral_and_uses_approved_slew_origin():
    pid = LongitudinalDistancePid(DistancePidConfig(
        kp_rpm_per_m=24, ki_rpm_per_m_s=.8, kd_rpm_s_per_m=0,
        output_rise_rpm_per_sec=10,
    ))
    pid.update(2, 1.5, now=100)
    before = pid._integral_m_s
    second = pid.update(2, 1.5, now=100.1)
    assert second.integral_m_s > before
    assert pid.accept_output_limit(100.1, 20)
    assert pid._integral_m_s == before
    assert pid._last_output_rpm == 20
    assert not pid.accept_output_limit(100.1, 0)
    assert pid.update(2, 1.5, now=100.2).output_rpm <= 21


@pytest.mark.parametrize("stamp,approved", [(99, 20), (100, -1), (100, float("nan")), (100, 99)])
def test_invalid_or_non_limiting_feedback_does_not_mutate_pid(stamp, approved):
    pid = LongitudinalDistancePid(DistancePidConfig())
    pid.update(2, 1.5, now=100)
    before = pid._integral_m_s, pid._last_output_rpm
    assert not pid.accept_output_limit(stamp, approved)
    assert (pid._integral_m_s, pid._last_output_rpm) == before


def test_feedback_only_updates_current_physical_pid_sample(setup):
    clock, controller, frame = setup
    decide(controller, frame(1.9))
    controller.accept_longitudinal_limit(clock.now-.1, 0)
    assert controller._distance_pid._last_output_rpm > 0
    controller.accept_longitudinal_limit(clock.now, 20)
    assert controller._distance_pid._last_output_rpm == 20


@pytest.mark.parametrize("case,reason", [("yaw", "yaw_limit"), ("encoder", "encoder_untrusted"),
                                       ("age", "depth_expired_or_future")])
def test_velocity_reset_reason_without_relaxing_safety(setup, caplog, case, reason):
    clock, controller, frame = setup
    decide(controller, frame(1.9, rpm=30))
    clock.now += .03
    current = frame(1.9, rpm=30, yaw=10 if case == "yaw" else 0,
                    stamp=clock.now-.2 if case == "age" else clock.now)
    if case == "encoder":
        current = replace(current, steering_feedback=replace(current.steering_feedback, trustworthy=False))
    controller._observe_longitudinal_motion(current, current.persons[0])
    assert controller._longitudinal_motion_evidence is None
    assert "reason=" + reason in caplog.text
