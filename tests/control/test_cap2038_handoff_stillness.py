"""CAP2031..2038 zero receipts + independent encoder reads, no hardware."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from car_control_modular.control_types import SteeringFeedback
from car_control_modular.mssd_motor import MotorSpeedReceipt
from car_control_modular.search_reacquire_braking import (
    MovingHandoffEvidence, handoff_retirement_reason, note_handoff_zero_write,
)


def feedback(stamp, ls, lf, rs, **changes):
    return replace(SteeringFeedback(timestamp=stamp, trustworthy=True,
        left_read_started=ls, left_read_finished=lf, right_read_started=rs,
        right_read_finished=stamp, raw_yaw_rate_right_dps=0.), **changes)


def fixture(monkeypatch):
    clock = [16404.427]
    monkeypatch.setattr("car_control_modular.search_reacquire_braking.time.monotonic", lambda: clock[0])
    owner = SimpleNamespace(search_state="none", _search_handoff_uid=1,
        _search_handoff_started_capture_ts=16403.700467995, _search_handoff_direction="right",
        _follow_controller=SimpleNamespace(active_target_id=1))
    backend = SimpleNamespace(last_speed_receipt=None)
    runtime = SimpleNamespace(owner=owner, backend=backend)
    owner._action_runtime = runtime
    # Log 37342 and 37356: distinct quiet reads 62.83 ms apart. The zero
    # episode began no later than the completed packet logged at 02.736.
    first = feedback(16404.313822557, 16404.297689714, 16404.306230957, 16404.306231540)
    second = feedback(16404.376655304, 16404.363132229, 16404.370779828, 16404.370780994)
    current = feedback(16404.415847315, 16404.403344756, 16404.409247357, 16404.409248232)
    runtime.get_steering_feedback = lambda: current
    owner._search_handoff_moving_evidence = MovingHandoffEvidence(
        1, 9, 2038, 16404.270035745, .9435871601, "right", .21)
    args = dict(eligible=True, uid=1, raw_track_id=9, cap=2038,
        stamp=16404.270035745, bbox=(567.79156494, 127.11323547, 640., 475.48474121),
        width=640, direction="right")
    return owner, runtime, clock, first, second, args


def write(runtime, completed, fb=None, pair=(0, 0), sequence=None):
    old = runtime.backend.last_speed_receipt
    sequence = sequence if sequence is not None else getattr(old, "sequence", 0)+1
    runtime.backend.last_speed_receipt = MotorSpeedReceipt(sequence, *pair, completed)
    note_handoff_zero_write(runtime, uid=1, previous_receipt=old,
                            packet_written=True, feedback=fb, now=completed+.0001)


def seed(runtime, first, second):
    write(runtime, 16404.242929221)
    write(runtime, 16404.359519154, first)
    write(runtime, 16404.401283016, second)


def test_real_cap2038_retires_only_search_constraint_after_completed_zero(monkeypatch):
    owner, runtime, _, first, second, args = fixture(monkeypatch)
    seed(runtime, first, second)
    moving = owner._search_handoff_moving_evidence
    receipt = runtime.backend.last_speed_receipt
    assert args["stamp"]-owner._search_handoff_started_capture_ts < .75
    assert handoff_retirement_reason(owner, **args) == "confirmed_post_zero_stillness"
    # The predicate never writes a packet, changes UID or issues any grant.
    assert runtime.backend.last_speed_receipt is receipt
    assert owner._search_handoff_uid == 1
    assert owner._search_handoff_moving_evidence is moving


@pytest.mark.parametrize("bad", ["one_sample", "duplicate", "overlap", "before_zero", "timestamp_only",
    "error", "untrusted", "raw_yaw", "filtered_yaw", "wheel_moving", "unconfirmed_yaw",
    "nan", "wide_gap", "too_short", "out_of_order"])
def test_quiet_proof_requires_two_unique_complete_post_zero_reads(monkeypatch, bad):
    owner, runtime, _, first, second, args = fixture(monkeypatch)
    if bad == "duplicate": second = first
    if bad == "overlap": second = replace(second, left_read_started=first.timestamp-.001)
    if bad == "before_zero": first = replace(first, left_read_started=16404.23)
    if bad == "timestamp_only": first = replace(first, left_read_started=None)
    if bad == "error": first = replace(first, right_error=1)
    if bad == "untrusted": first = replace(first, trustworthy=False)
    if bad == "raw_yaw": first = replace(first, raw_yaw_rate_right_dps=2.01)
    if bad == "filtered_yaw": first = replace(first, yaw_rate_right_dps=2.01)
    if bad == "wheel_moving": first = replace(first, right_forward_rpm=-1.01)
    if bad == "unconfirmed_yaw": first = replace(first, yaw_rate_confirmed=False)
    if bad == "nan": first = replace(first, left_forward_rpm=float("nan"))
    if bad == "wide_gap": first = replace(first, timestamp=16404.22)
    if bad == "too_short": first = replace(first, timestamp=second.timestamp-.03, right_read_finished=second.timestamp-.03)
    if bad == "out_of_order": second = replace(first, timestamp=first.timestamp-.001)
    write(runtime, 16404.242929221)
    write(runtime, 16404.359519154, first)
    if bad != "one_sample": write(runtime, 16404.401283016, second)
    assert handoff_retirement_reason(owner, **args) is None


@pytest.mark.parametrize("bad", ["nonzero", "stop", "external_zero", "sequence_gap", "new_episode", "new_uid",
    "search", "controller_search", "park", "fault", "parking_fault", "pending_brake", "explicit_stop", "shutdown",
    "brake_hold", "stop_execution", "person_detected"])
def test_interruption_invalidates_zero_episode(monkeypatch, bad):
    owner, runtime, _, first, second, args = fixture(monkeypatch)
    seed(runtime, first, second)
    if bad == "nonzero": write(runtime, 16404.405, pair=(8, -8))
    if bad == "stop": runtime.backend.last_speed_receipt = None
    if bad == "external_zero":
        runtime.backend.last_speed_receipt = MotorSpeedReceipt(7, 0, 0, 16404.405)
    if bad == "sequence_gap": write(runtime, 16404.405, second, sequence=20)
    if bad == "new_episode": owner._search_handoff_started_capture_ts = 16403.71
    if bad == "new_uid": owner._follow_controller.active_target_id = 2
    if bad == "search": owner.search_state = "searching"
    if bad == "controller_search": owner._follow_controller.search_state = "searching"
    if bad == "park": owner._near_yaw_park_request = object()
    if bad == "fault": runtime.backend.motion_write_fault = "partial_write"
    if bad == "parking_fault": runtime.backend.parking_release_fault = "current_uncertain"
    if bad == "pending_brake": runtime._search_reacquire_brake_request = object()
    if bad == "explicit_stop": owner._explicit_stop_requested = True
    if bad == "shutdown": owner._runtime_shutdown_requested = True
    if bad == "brake_hold": owner._brake_hold_active = True
    if bad == "stop_execution": owner.stop_action_execution = True
    if bad == "person_detected": owner.person_detected_flag = True
    assert handoff_retirement_reason(owner, **args) is None


@pytest.mark.parametrize("bad", ["expired_image", "replay", "before_zero_capture", "unconfirmed",
    "raw_id", "expired_feedback", "current_moving", "current_error", "current_read_before_zero", "cache_wait"])
def test_current_confirmed_capture_and_current_quiet_feedback_still_required(monkeypatch, bad):
    owner, runtime, clock, first, second, args = fixture(monkeypatch)
    seed(runtime, first, second)
    current = runtime.get_steering_feedback()
    if bad == "expired_image": clock[0] = args["stamp"]+.2101
    if bad == "replay": owner._search_handoff_last_capture = (args["cap"], args["stamp"])
    if bad == "before_zero_capture":
        args["stamp"] = 16404.24
        owner._search_handoff_moving_evidence = replace(owner._search_handoff_moving_evidence, stamp=args["stamp"])
    if bad == "unconfirmed": args["eligible"] = False
    if bad == "raw_id": args["raw_track_id"] = 10
    if bad == "expired_feedback": clock[0] = 16404.48
    if bad == "current_moving": current = replace(current, left_forward_rpm=2.)
    if bad == "current_error": current = replace(current, left_error=1)
    if bad == "current_read_before_zero": current = replace(current, left_read_started=16404.2)
    def get():
        if bad == "cache_wait": clock[0] += .1
        return current
    runtime.get_steering_feedback = get
    assert handoff_retirement_reason(owner, **args) is None


@pytest.mark.parametrize("receipt", [None, SimpleNamespace(), SimpleNamespace(left_rpm=0, right_rpm=0)])
def test_legacy_backend_missing_receipt_fields_safely_disables_branch(monkeypatch, receipt):
    owner, runtime, _, first, _, args = fixture(monkeypatch)
    runtime.backend.last_speed_receipt = receipt
    note_handoff_zero_write(runtime, uid=1, previous_receipt=None,
                            packet_written=True, feedback=first, now=16404.4)
    assert runtime._search_handoff_zero_evidence is None
    assert handoff_retirement_reason(owner, **args) is None


def test_intervening_nonzero_then_new_zero_cannot_reuse_old_quiet_samples(monkeypatch):
    owner, runtime, _, first, second, args = fixture(monkeypatch)
    seed(runtime, first, second)
    runtime.backend.last_speed_receipt = MotorSpeedReceipt(4, 8, -8, 16404.403)
    write(runtime, 16404.41, second)
    assert runtime._search_handoff_zero_evidence.quiet_samples == ()
    assert handoff_retirement_reason(owner, **args) is None


@pytest.mark.parametrize("missing", ["trustworthy", "yaw_rate_confirmed", "left_error", "right_error"])
def test_partial_legacy_feedback_safely_refuses_retirement(monkeypatch, missing):
    owner, runtime, _, first, second, args = fixture(monkeypatch)
    fields = dict(vars(first))
    fields.pop(missing)
    seed(runtime, SimpleNamespace(**fields), second)
    assert handoff_retirement_reason(owner, **args) is None


def test_real_producer_retires_cap2038_without_renewing_depth_or_issuing_stop(monkeypatch):
    import request_0513_modular as main
    from car_control_modular.controllers import FollowPolicyConfig

    owner, runtime, _, first, second, args = fixture(monkeypatch)
    seed(runtime, first, second)
    owner._active_capture_frame_id = args["cap"]
    owner._active_capture_timestamp = args["stamp"]
    owner._search_handoff_cap_rpm = 7.
    owner._search_handoff_moving_active = True
    owner._follow_controller.cfg = FollowPolicyConfig()
    runtime.search_reacquire_brake_pending = lambda: False
    owner._fresh_depth_linear_snapshot = lambda *a, **kw: ("forward", 27, 1, 16404.290405815)
    before = owner._fresh_depth_linear_snapshot()
    receipt = runtime.backend.last_speed_receipt
    assert not main.PersonTracker._hold_search_reacquire_brake(owner,
        bbox=args["bbox"], width=args["width"], eligible=True, confirmed=True, raw_track_id=9)
    assert owner._search_handoff_uid is None
    assert owner._fresh_depth_linear_snapshot() == before
    assert runtime.backend.last_speed_receipt is receipt
