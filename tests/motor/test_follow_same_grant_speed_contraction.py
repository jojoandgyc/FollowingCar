"""Same-depth-grant speed contractions at the periodic motor writer.

Fake clocks and fake serial targets only: none of these tests open a motor port.
"""

import pytest
from types import SimpleNamespace

from test_follow_wheel_periodic import setup_periodic
from test_visible_wheel_continuity import feedback
from car_control_modular.final_yaw_coalescing import contract_forward_base


def _live_depth_grant(monkeypatch):
    runtime, owner, driver, _, clock, _ = setup_periodic(monkeypatch)
    grant = {"stamp": 10.0, "initial_percent": 50., "alive": True,
             "first_cap": 60., "middle_cap": 59., "late_cap": 58.,
             "terminal_cap": 58., "yaw": 0.}
    owner._depth_linear_max_age_sec = lambda kind: .25
    owner._depth30_linear_snapshot = ("forward", 50., 1, grant["stamp"])

    def cap(now):
        if not grant["alive"] or not grant["stamp"] <= now < grant["stamp"] + .25:
            return None
        if grant["initial_percent"] == 50.:
            return 50.
        # A fixed sample can acquire a progressively lower braking envelope
        # while identity, sample timestamp, and yaw are all unchanged.
        if now < 10.065:
            return grant["first_cap"]
        if now < 10.085:
            return grant["middle_cap"]
        if now < 10.105:
            return grant["late_cap"]
        return grant["terminal_cap"]

    owner._fresh_depth_linear_snapshot = lambda uid, now=None: (
        ("forward", cap(clock[0] if now is None else now), uid, grant["stamp"])
        if uid == 1 and cap(clock[0] if now is None else now) is not None else None)
    owner._follow_wheel_axes = lambda now: (
        1, owner._lateral_yaw_revision, cap(now) or 0., grant["yaw"])
    runtime.get_steering_feedback = lambda: feedback(clock[0], 20, 20)
    runtime._service_follow_wheels()
    assert driver.pairs == [(50, -50)]

    clock[0] = 10.05
    grant.update(stamp=10.05, initial_percent=60.)
    owner._depth30_linear_snapshot = ("forward", 60., 1, grant["stamp"])
    driver.pairs.clear()
    return runtime, owner, driver, clock, grant


def test_same_grant_two_speed_cap_drops_do_not_insert_motor_zero(monkeypatch):
    runtime, owner, driver, clock, grant = _live_depth_grant(monkeypatch)
    reads = [0]

    def read_feedback():
        reads[0] += 1
        # First read invalidates the 60 RPM plan, so it is rebuilt once.
        # Second read contracts it again while the *same* Depth grant lives.
        if reads[0] <= 2:
            clock[0] += .02
        return feedback(clock[0], 20, 20)

    runtime.get_steering_feedback = read_feedback
    runtime._service_follow_wheels()

    assert reads[0] >= 2
    assert grant["stamp"] == owner._depth30_linear_snapshot[3]
    assert clock[0] < grant["stamp"] + .25
    assert owner._fresh_depth_linear_snapshot(1, now=clock[0])[1] == 58.
    # A natural reduction is not a revocation. The physical write must obey
    # the latest lower cap, without a 0 RPM packet inserted between grants.
    assert driver.pairs
    assert all(left > 0 and right < 0 for left, right in driver.pairs)
    assert 0 < driver.pairs[-1][0] <= 58
    assert -driver.pairs[-1][1] == driver.pairs[-1][0]


def test_base_contract_accepts_pair_already_clamped_below_planned_axes():
    assert contract_forward_base(
        (58, 58), (1, 1, 60., 0.), (1, 1, 56., 0.)
    ) == (56, 56)


def test_depth_cap_drops_before_send_and_again_after_feedback_without_zero(monkeypatch):
    runtime, owner, driver, clock, grant = _live_depth_grant(monkeypatch)
    grant["terminal_cap"] = 56.
    sends = [0]
    reads = [0]
    real_send = runtime._send_follow_wheel_targets

    def send_after_clock_progress(*args, **kwargs):
        sends[0] += 1
        if sends[0] == 2:
            # The second attempt was planned at 60 RPM; the fresh Depth
            # reader inside _send will already limit its guarded pair to 58.
            clock[0] = 10.09
        return real_send(*args, **kwargs)

    def read_feedback():
        reads[0] += 1
        if reads[0] == 1:
            # Consume the one full rebuild without changing the 60 RPM base.
            owner._lateral_yaw_revision += 1
        elif reads[0] == 2:
            # Same grant contracts again before the early write check.
            clock[0] = 10.11
        return feedback(clock[0], 20, 20)

    runtime._send_follow_wheel_targets = send_after_clock_progress
    runtime.get_steering_feedback = read_feedback
    runtime._service_follow_wheels()

    assert sends[0] >= 2 and reads[0] >= 2
    assert owner._depth30_linear_snapshot == ("forward", 60., 1, 10.05)
    assert owner._fresh_depth_linear_snapshot(1, now=clock[0])[1] == 56.
    assert driver.pairs
    assert all(0 < left <= 56 and right == -left for left, right in driver.pairs)


def test_same_grant_cap_drops_during_final_safety_check_do_not_insert_zero(monkeypatch):
    runtime, owner, driver, clock, grant = _live_depth_grant(monkeypatch)
    calls = [0]
    reads = [0]

    def read_feedback():
        reads[0] += 1
        if reads[0] == 1:
            clock[0] = 10.07  # The first drop requests the one full rebuild.
        return feedback(clock[0], 20, 20)

    def safety_check(_action):
        calls[0] += 1
        # The third call is the rebuilt packet's final hard-stop check, after
        # its wheel guard but before its final physical-grant/axes check.
        if calls[0] == 3:
            clock[0] = 10.09
        return False

    runtime.get_steering_feedback = read_feedback
    runtime.hard_stop_check = safety_check
    runtime._service_follow_wheels()

    assert calls[0] >= 3
    assert reads[0] >= 2
    assert grant["stamp"] == owner._depth30_linear_snapshot[3]
    assert clock[0] < grant["stamp"] + .25
    assert owner._fresh_depth_linear_snapshot(1, now=clock[0])[1] == 58.
    assert driver.pairs
    assert all(left > 0 and right < 0 for left, right in driver.pairs)
    assert 0 < driver.pairs[-1][0] <= 58


def test_same_grant_base_drop_survives_yaw_revision_with_unchanged_zero_yaw(monkeypatch):
    """CAP528: a new lateral revision can accompany a smaller forward cap."""
    runtime, owner, driver, clock, grant = _live_depth_grant(monkeypatch)
    source_grant = owner._depth30_linear_snapshot
    reads = [0]
    safety_calls = [0]

    def read_feedback():
        reads[0] += 1
        if reads[0] == 1:
            clock[0] = 10.07  # 60 -> 59 asks for the one complete rebuild.
        return feedback(clock[0], 20, 20)

    def safety_check(_action):
        safety_calls[0] += 1
        if safety_calls[0] == 3:
            # Same published Depth sample, same UID and yaw. The lateral
            # producer only advanced its bookkeeping revision while braking
            # reduced the authorized base a second time: 59 -> 58.
            clock[0] = 10.09
            owner._lateral_yaw_revision += 1
        return False

    runtime.get_steering_feedback = read_feedback
    runtime.hard_stop_check = safety_check
    runtime._service_follow_wheels()

    assert reads[0] >= 2 and safety_calls[0] >= 3
    assert owner._depth30_linear_snapshot is source_grant
    assert owner._fresh_depth_linear_snapshot(1, now=clock[0])[1] == 58.
    assert clock[0] < grant["stamp"] + .25
    assert driver.pairs == [(58, -58)]
    assert not driver.stops


def test_same_grant_base_drop_survives_fresh_unchanged_turn_revision(monkeypatch):
    """CAP671: a bookkeeping revision must not zero a still-valid 10 RPM turn."""
    runtime, owner, driver, clock, grant = _live_depth_grant(monkeypatch)
    grant["yaw"] = 10.
    owner._has_fresh_lateral_yaw = lambda uid: uid == 1
    reads = [0]
    safety_calls = [0]

    def read_feedback():
        reads[0] += 1
        if reads[0] == 1:
            clock[0] = 10.07  # 60 -> 59, consume the full rebuild.
        return feedback(clock[0], 20, 20)

    def safety_check(_action):
        safety_calls[0] += 1
        if safety_calls[0] == 3:
            clock[0] = 10.09  # 59 -> 58 on the same physical depth grant.
            owner._lateral_yaw_revision += 1
        return False

    runtime.get_steering_feedback = read_feedback
    runtime.hard_stop_check = safety_check
    runtime._service_follow_wheels()

    assert reads[0] >= 2 and safety_calls[0] >= 3
    assert owner._fresh_depth_linear_snapshot(1, now=clock[0])[1] == 58.
    assert driver.pairs == [(68, -48)]  # forward base 58, yaw +10
    assert not driver.stops


@pytest.mark.parametrize("change", ["yaw_expired", "yaw_changed", "depth_expired"])
def test_turning_base_drop_keeps_real_evidence_veto(monkeypatch, change):
    runtime, owner, driver, clock, grant = _live_depth_grant(monkeypatch)
    grant["yaw"] = 10.
    yaw_live = [True]
    owner._has_fresh_lateral_yaw = lambda uid: uid == 1 and yaw_live[0]
    reads = [0]
    safety_calls = [0]

    def read_feedback():
        reads[0] += 1
        if reads[0] == 1:
            clock[0] = 10.07
        return feedback(clock[0], 20, 20)

    def safety_check(_action):
        safety_calls[0] += 1
        if safety_calls[0] == 3:
            clock[0] = 10.09
            owner._lateral_yaw_revision += 1
            if change == "yaw_expired":
                yaw_live[0] = False
            elif change == "yaw_changed":
                grant["yaw"] = -10.
            else:
                clock[0] = grant["stamp"] + .251
        return False

    runtime.get_steering_feedback = read_feedback
    runtime.hard_stop_check = safety_check
    runtime._service_follow_wheels()

    assert safety_calls[0] >= 3
    assert driver.pairs == [(0, 0)]


def test_idle_predictive_yaw_intent_does_not_cancel_straight_base_drop(monkeypatch):
    """CAP693: a prior yaw brake cannot turn a straight base drop into STOP."""
    runtime, owner, driver, clock, grant = _live_depth_grant(monkeypatch)
    intent = SimpleNamespace(park_requested=False, forward_countersteer=True,
                             countersteer_rpm=6, mode="forward", x_ratio=.5)
    owner._lateral_intent_store = SimpleNamespace(snapshot=lambda: intent)
    reads = [0]
    safety_calls = [0]

    def read_feedback():
        reads[0] += 1
        if reads[0] == 1:
            clock[0] = 10.07  # First cap reduction consumes full rebuild.
        return feedback(clock[0], 20, 20)

    def safety_check(_action):
        safety_calls[0] += 1
        if safety_calls[0] == 3:
            clock[0] = 10.09  # Final same-grant cap falls to 58.
        return False

    runtime.get_steering_feedback = read_feedback
    runtime.hard_stop_check = safety_check
    runtime._service_follow_wheels()

    assert reads[0] >= 2 and safety_calls[0] >= 3
    assert driver.pairs == [(58, -58)]
    assert not driver.stops


def test_depth_missing_visual_frame_keeps_live_same_uid_depth_grant(monkeypatch):
    """A missing depth on this image does not revoke a still-live older sample."""
    runtime, owner, driver, clock, grant = _live_depth_grant(monkeypatch)
    source_grant = owner._depth30_linear_snapshot
    owner._vision_control_state = "target_visible_depth_missing"
    reads = [0]
    safety_calls = [0]

    def read_feedback():
        reads[0] += 1
        if reads[0] == 1:
            clock[0] = 10.07
        return feedback(clock[0], 20, 20)

    def safety_check(_action):
        safety_calls[0] += 1
        if safety_calls[0] == 3:
            clock[0] = 10.09  # A second same-grant braking reduction.
        return False

    runtime.get_steering_feedback = read_feedback
    runtime.hard_stop_check = safety_check
    runtime._service_follow_wheels()

    assert reads[0] >= 2 and safety_calls[0] >= 3
    assert owner._depth30_linear_snapshot is source_grant
    assert owner._fresh_depth_linear_snapshot(1, now=clock[0])[1] == 58.
    assert clock[0] < grant["stamp"] + .25
    assert driver.pairs == [(58, -58)]
    assert not driver.stops


@pytest.mark.parametrize("unsafe_state", ["lost_confirming", "target_visible_low_quality"])
def test_nontracking_visual_state_cannot_borrow_same_grant_contraction(monkeypatch, unsafe_state):
    runtime, owner, driver, clock, _ = _live_depth_grant(monkeypatch)
    if unsafe_state == "target_visible_low_quality":
        # Exercise the enabled low-quality wheel handoff too: it may retain
        # yaw, but cannot inherit this forward contraction.
        runtime.config.follow_forward_loss_handoff_enable = True
    reads = [0]
    safety_calls = [0]

    def read_feedback():
        reads[0] += 1
        if reads[0] == 1:
            clock[0] = 10.07
        return feedback(clock[0], 20, 20)

    def safety_check(_action):
        safety_calls[0] += 1
        if safety_calls[0] == 3:
            clock[0] = 10.09
            owner._vision_control_state = unsafe_state
        return False

    runtime.get_steering_feedback = read_feedback
    runtime.hard_stop_check = safety_check
    runtime._service_follow_wheels()

    assert reads[0] >= 2 and safety_calls[0] >= 3
    assert driver.pairs == [(0, 0)]
    assert not driver.stops


@pytest.mark.parametrize("late_loss", ["depth_expired", "uid_conflict", "person_danger"])
def test_final_base_contraction_still_zeroes_real_authority_loss(monkeypatch, late_loss):
    runtime, owner, driver, clock, grant = _live_depth_grant(monkeypatch)
    reads = [0]
    safety_calls = [0]

    def read_feedback():
        reads[0] += 1
        if reads[0] == 1:
            clock[0] = 10.07  # First reduction consumes the one rebuild.
        elif reads[0] == 2:
            clock[0] = 10.28  # Feedback stays fresh across the TTL boundary.
        return feedback(clock[0], 20, 20)

    def safety_check(_action):
        safety_calls[0] += 1
        if safety_calls[0] == 3:
            if late_loss == "depth_expired":
                clock[0] = grant["stamp"] + .251
            elif late_loss == "uid_conflict":
                owner._follow_controller.active_target_id = 2
            else:
                owner.person_detected_flag = True
        return False

    runtime.get_steering_feedback = read_feedback
    runtime.hard_stop_check = safety_check
    runtime._service_follow_wheels()

    assert reads[0] >= 2 and safety_calls[0] >= 3
    assert driver.pairs == [(0, 0)]
    assert not driver.stops


def test_final_contraction_does_not_overwrite_intervening_stop(monkeypatch):
    runtime, _, driver, clock, _ = _live_depth_grant(monkeypatch)
    reads = [0]
    safety_calls = [0]

    def read_feedback():
        reads[0] += 1
        if reads[0] == 1:
            clock[0] = 10.07
        return feedback(clock[0], 20, 20)

    def safety_check(_action):
        safety_calls[0] += 1
        if safety_calls[0] == 3:
            clock[0] = 10.09  # Final speed contraction after rebuild.
        elif safety_calls[0] == 4:
            runtime.backend.send_stop("intervening_safety_stop", mode="emergency",
                                      preserve_zero=True)
        return False

    runtime.get_steering_feedback = read_feedback
    runtime.hard_stop_check = safety_check
    runtime._service_follow_wheels()

    assert safety_calls[0] >= 4
    assert driver.stops == [1]
    assert driver.pairs == []  # Neither a smaller speed nor a speed-zero re-arms STOP.


def test_last_activity_check_cannot_hide_newer_same_grant_cap(monkeypatch):
    runtime, owner, driver, clock, grant = _live_depth_grant(monkeypatch)
    grant["terminal_cap"] = 57.
    reads = [0]
    safety_calls = [0]
    final_activity_advanced = [False]

    def read_feedback():
        reads[0] += 1
        if reads[0] == 1:
            clock[0] = 10.07  # 60 -> 59: use the one full rebuild.
        return feedback(clock[0], 20, 20)

    def safety_check(_action):
        safety_calls[0] += 1
        if safety_calls[0] == 3:
            clock[0] = 10.09  # 59 -> 58: enter final base contraction.
        return False

    ordinary_active = runtime._periodic_follow_active

    def activity_check():
        if safety_calls[0] >= 4 and not final_activity_advanced[0]:
            # A routine active-state reader takes enough time for the very
            # same sample's braking cap to contract again, 58 -> 57. Before
            # the final-order fix, this happened *after* checked_linear and
            # latest_axes, letting a now-too-high 58 RPM packet reach the port.
            final_activity_advanced[0] = True
            clock[0] = 10.11
        return ordinary_active()

    runtime.get_steering_feedback = read_feedback
    runtime.hard_stop_check = safety_check
    runtime._periodic_follow_active = activity_check
    runtime._service_follow_wheels()

    assert reads[0] >= 2 and safety_calls[0] >= 4
    assert final_activity_advanced[0]
    assert owner._depth30_linear_snapshot == ("forward", 60., 1, 10.05)
    assert owner._fresh_depth_linear_snapshot(1, now=clock[0])[1] == 57.
    assert driver.pairs == [(57, -57)]


@pytest.mark.parametrize("republication", ["same_values_new_object", "changed_percent"])
def test_same_sample_republication_is_not_age_only_contraction(monkeypatch, republication):
    runtime, owner, driver, clock, grant = _live_depth_grant(monkeypatch)
    calls = [0]
    original_grant = owner._depth30_linear_snapshot

    def read_feedback():
        calls[0] += 1
        if calls[0] <= 2:
            clock[0] += .02
        if calls[0] == 2:
            percent = 60. if republication == "same_values_new_object" else 61.
            owner._depth30_linear_snapshot = ("forward", percent, 1, grant["stamp"])
            assert owner._depth30_linear_snapshot is not original_grant
        return feedback(clock[0], 20, 20)

    runtime.get_steering_feedback = read_feedback
    runtime._service_follow_wheels()
    # Equal-looking tuples are not the same published authorization. A new
    # producer write must be processed on its own path, not inherit this
    # contraction's proof from the previous canonical grant.
    assert calls[0] >= 2
    assert driver.pairs == [(0, 0)]


def test_forward_base_contraction_cannot_reverse_inner_wheel(monkeypatch):
    runtime, owner, driver, clock, grant = _live_depth_grant(monkeypatch)
    grant.update(middle_cap=30., late_cap=10., yaw=20.)
    owner._has_fresh_lateral_yaw = lambda uid: True
    reads = [0]

    def read_feedback():
        reads[0] += 1
        if reads[0] <= 2:
            clock[0] += .02
        return feedback(clock[0], 20, 20)

    runtime.get_steering_feedback = read_feedback
    runtime._service_follow_wheels()
    assert reads[0] >= 2
    assert driver.pairs == [(0, 0)]


@pytest.mark.parametrize("bad_feedback", ["stale", "reverse"])
def test_forward_base_contraction_needs_current_nonreversing_feedback(monkeypatch, bad_feedback):
    runtime, _, driver, clock, _ = _live_depth_grant(monkeypatch)
    reads = [0]

    def read_feedback():
        reads[0] += 1
        if reads[0] <= 2:
            clock[0] += .02
        if reads[0] == 2:
            if bad_feedback == "stale":
                return feedback(clock[0] - .151, 20, 20)
            return feedback(clock[0], -2, 20)
        return feedback(clock[0], 20, 20)

    runtime.get_steering_feedback = read_feedback
    runtime._service_follow_wheels()
    assert reads[0] >= 2
    assert driver.pairs == [(0, 0)]


@pytest.mark.parametrize("loss", ["expired", "revoked"])
def test_same_writer_still_zeroes_true_depth_authority_loss(monkeypatch, loss):
    runtime, owner, driver, clock, grant = _live_depth_grant(monkeypatch)

    def read_feedback():
        if loss == "expired":
            clock[0] = grant["stamp"] + .251
        else:
            grant["alive"] = False
            owner._depth30_linear_snapshot = None
        return feedback(clock[0], 20, 20)

    runtime.get_steering_feedback = read_feedback
    runtime._service_follow_wheels()
    assert driver.pairs == [(0, 0)]
