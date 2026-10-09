"""A same-grant cap can fall after the final wheel pair is calculated."""

import request_0513_modular as app
from car_control_modular.control_types import DepthLinearTiming
from test_follow_same_grant_speed_contraction import _live_depth_grant
from test_visible_wheel_continuity import feedback


def test_180ms_continuation_veto_after_final_pair_never_writes_old_speed(monkeypatch):
    """The 250ms physical lease does not keep a pre-180ms speed cap alive."""
    runtime, owner, driver, clock, grant = _live_depth_grant(monkeypatch)
    stamp = grant["stamp"]
    raw = owner._depth30_linear_snapshot
    timing = DepthLinearTiming(
        snapshot=raw,
        accepted_depth_timestamp=stamp,
        depth_expires_at=stamp + .25,
        continuation_distance_m=1.9,
        continuation_speed_bound_m_s=.6,
    )
    owner._depth30_linear_timing = timing
    owner._depth30_prepared_timing = None
    owner._last_vision_control_ts = stamp
    owner._follow_controller.distance_pi_enabled = False
    owner._action_runtime = runtime
    owner._depth_forward_continuation_required = lambda linear, timing, now: (
        app.PersonTracker._depth_forward_continuation_required(owner, linear, timing, now))
    owner._depth_forward_continuation_limit = lambda linear, timing, now, **kwargs: (
        app.PersonTracker._depth_forward_continuation_limit(
            owner, linear, timing, now, **kwargs))
    monkeypatch.setattr(app, "ASTRA_DEPTH_LONGITUDINAL_SAMPLE_MAX_AGE_SEC", .25)
    monkeypatch.setattr(app, "ASTRA_DEPTH_CONTINUATION_SPEED_CAP_ENABLE", False)
    monkeypatch.setattr(app, "ASTRA_DEPTH_RELATIVE_CONTINUATION_ENABLE", False)
    monkeypatch.setattr(app, "TARGET_DISTANCE", 1.5)
    monkeypatch.setattr(app, "DISTANCE_PID_DEADBAND_M", .05)
    monkeypatch.setattr(app, "FOLLOW_BRAKE_DISTANCE_M", .5)
    monkeypatch.setattr(app, "DISTANCE_APPROACH_DECELERATION_M_S2", .4)
    monkeypatch.setattr(app, "DISTANCE_APPROACH_RESPONSE_DELAY_SEC", .2)
    monkeypatch.setattr(app, "VISION_MMWAVE_FUSION_ENCODER_WHEEL_CIRCUMFERENCE_M", .6)
    monkeypatch.setattr(app, "FORWARD_MAX_RPM", 100)
    terminal_cap, _ = owner._depth_forward_continuation_limit(
        raw, timing, stamp + .181,
        feedback=feedback(stamp + .181, 20, 20), quiet=True)
    assert terminal_cap == 0

    # The earlier two reductions exercise the one allowed full rebuild and
    # then the final same-grant pair contraction. At 180ms, the real braking
    # continuation model rejects this original grant (insufficient margin).
    def cap(now):
        age = now - stamp
        if not 0 <= age < .25:
            return 0.
        if age >= .18:
            return 0.
        if age >= .178:
            return 58.
        if age >= .17:
            return 59.
        return 60.

    owner._fresh_depth_linear_snapshot = lambda uid, now=None: (
        ("forward", cap(clock[0] if now is None else now), uid, stamp)
        if uid == 1 and cap(clock[0] if now is None else now) > 0 else None
    )
    owner._follow_wheel_axes = lambda now: (
        1, owner._lateral_yaw_revision, cap(now), 0.
    )
    reads = [0]
    safety_calls = [0]
    post_cap_activity_calls = [0]
    crossed_after_pair = [False]
    ordinary_active = runtime._visible_wheel_control_active

    def read_feedback():
        reads[0] += 1
        if reads[0] == 1:
            clock[0] = stamp + .174  # 60 -> 59: consume the full rebuild.
        return feedback(clock[0], 20, 20)

    def safety_check(_action):
        safety_calls[0] += 1
        if safety_calls[0] == 3:
            clock[0] = stamp + .179  # 59 -> 58: enter final contraction.
        return False

    def activity_check():
        if safety_calls[0] >= 4:
            post_cap_activity_calls[0] += 1
            if post_cap_activity_calls[0] == 2:
                # The second check follows checked_linear and latest_axes.
                # Their 58 RPM cap is now obsolete, though Depth TTL survives.
                clock[0] = stamp + .181
                crossed_after_pair[0] = True
        return ordinary_active()

    runtime.get_steering_feedback = read_feedback
    runtime.hard_stop_check = safety_check
    runtime._visible_wheel_control_active = activity_check
    runtime._service_follow_wheels()

    assert reads[0] >= 2 and safety_calls[0] >= 4
    assert crossed_after_pair[0]
    assert clock[0] < timing.depth_expires_at
    assert owner._fresh_depth_linear_snapshot(1, now=clock[0]) is None
    # A silent return also leaves the previous 50 RPM command active; a
    # physical zero (or an already checked lower pair) must be sent.
    assert driver.pairs and driver.pairs[-1] == (0, 0)
