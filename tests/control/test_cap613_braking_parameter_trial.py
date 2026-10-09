"""CAP613/616 same-input parameter trial; never a simulated vehicle trajectory.

This historical profile changes a braking MODEL, not physical deceleration.
Replay parameters stay pinned to the captured trial; current INI assertions
live in tests/config/test_distance_braking_trial.py instead.
The isolated PI result, envelope cap and final approved request are deliberately
asserted separately. No cameras, serial devices or motor threads are started.
"""
from dataclasses import dataclass, replace

import pytest

import request_0513_modular as runtime
from car_control_modular.depth_continuation import (
    ContinuationMotionEvidence, relative_continuation_speed_cap,
)
from car_control_modular.distance_pi import DistancePiConfig, DistancePiController
from car_control_modular.longitudinal_approach import RawDepthMotionEvidence
from test_distance_pi_runtime import pi_owner  # fixture dependency
from test_lateral_zero_runtime import NOW, owner  # fixture dependency
from test_relative_depth_continuation_runtime import relative, publish


CIRCUMFERENCE = .816814


@dataclass(frozen=True)
class Sample:
    cap: int
    distance: float
    raw: float
    target_speed: float
    range_rate: float
    span: float
    age: float
    pi_age: float
    feedback_skew: float
    left: float
    right: float
    previous_request: int
    old_pi_cap: float
    new_pi_cap: float
    new_pi_request: int


# run_20260930_201448_199287_c5f4a965: physical-window values, not video RPM.
# Ages printed in the log are rounded to 0.1ms; cap assertions allow that error.
SAMPLES = (
    Sample(613, 1.997, 1.8224, -.12546447929305474, -.6529901876263883,
           .27238135899824556, .0226, .0203, .003367493, 33., 29., 24,
           25.0375, 35.067, 31),
    Sample(616, 1.8224, 1.7606, -.25233617030628736, -.6539363869729551,
           .09627191800245782, .0299, .0276, .002049486, 28., 28., 8,
           9.8307, 18.990, 18),
)


@pytest.fixture
def board():
    # Reproduce CAP613/616 at their ORIGINAL parameters, including the prior
    # human-motion policy. Do not silently replay old logs under a new trial.
    return dict(a=.70, delay=.20, target=1.4, ttl=.25, overshoot=.20, kp=3.)


def pi_replay(sample, board, deceleration):
    controller = DistancePiController(DistancePiConfig(
        kp_per_sec=board["kp"], deceleration_m_s2=deceleration,
        response_delay_sec=board["delay"], physical_ttl_sec=board["ttl"],
        launch_request_rpm=180., launch_full_error_m=.5, motion_memory_sec=.3))
    # Both log samples entered recovery without a live prior motor grant.
    # A new controller reproduces this zero ramp-budget admission, not history.
    return controller.update(sample.distance, board["target"],
        sample_timestamp=NOW, execution_now=NOW+sample.pi_age,
        deadband_m=.03, max_output_rpm=200., rise_rpm_per_sec=240.,
        ego_forward_rpm=(sample.left+sample.right)/2.,
        range_rate_m_s=sample.range_rate, raw_closure_valid=True,
        raw_motion_evidence=RawDepthMotionEvidence(
            NOW, sample.range_rate, sample.target_speed, sample.span, 2),
        raw_distance_m=sample.raw)


def prepare(state, sample, board, monkeypatch, *, deceleration=None):
    monkeypatch.setattr(runtime, "DISTANCE_APPROACH_DECELERATION_M_S2",
                        board["a"] if deceleration is None else deceleration)
    monkeypatch.setattr(runtime, "DISTANCE_APPROACH_RESPONSE_DELAY_SEC", board["delay"])
    # Use the existing real-commit fixture to initialize ownership, then install
    # the captured grant to isolate its final reader from upstream PI changes.
    stamp, _ = publish(state, distance=5., percent=21,
                       stamp=NOW-sample.age, rpm=28.)
    linear = ("forward", sample.previous_request//2, 1, stamp)
    timing = replace(state.owner._depth30_linear_timing, snapshot=linear,
        continuation_distance_m=sample.raw,
        continuation_speed_bound_m_s=42*CIRCUMFERENCE/60.,
        continuation_motion=ContinuationMotionEvidence(
            1, stamp, sample.target_speed, sample.range_rate, sample.span, 2))
    state.owner._publish_depth_linear_pair(linear, timing)
    state.feedback = replace(state.feedback, timestamp=stamp+sample.feedback_skew,
                            left_forward_rpm=sample.left, right_forward_rpm=sample.right)
    state.owner._last_vision_control_ts = state.clock.now-.01
    return stamp, linear, timing


def envelope(state, stamp, timing, board, age):
    return relative_continuation_speed_cap(
        distance=timing.continuation_distance_m, stop_distance=1.2,
        original_speed_bound=timing.continuation_speed_bound_m_s,
        sample_age=age, feedback=replace(state.feedback, timestamp=stamp+age),
        now=stamp+age, circumference=CIRCUMFERENCE, max_rpm=200.,
        deceleration=board["a"], response_delay=board["delay"],
        motion=timing.continuation_motion, target_id=1, sample_timestamp=stamp)


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda sample: f"CAP{sample.cap}")
def test_fresh_pi_cap_and_recovery_request_are_not_the_same_number(sample, board):
    previous = pi_replay(sample, board, .40)
    trial = pi_replay(sample, board, board["a"])
    assert previous.cap_rpm == pytest.approx(sample.old_pi_cap, abs=.02)
    assert trial.cap_rpm == pytest.approx(sample.new_pi_cap, abs=.02)
    assert trial.output_rpm == sample.new_pi_request
    assert trial.output_rpm <= trial.cap_rpm
    assert trial.brake_source == "raw_relative_motion"
    assert trial.status == "recovering"
    assert trial.sample_dt_sec == 0.
    assert trial.integral_m_s == 0.
    if sample.cap == 613:
        assert trial.final_limit_reason == "execution_recovery"
        assert trial.output_rpm == (sample.left+sample.right)/2.
    else:
        assert trial.final_limit_reason == "braking_envelope"


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda sample: f"CAP{sample.cap}")
@pytest.mark.parametrize("deceleration", [.40, .70])
def test_real_reader_relaxes_only_captured_request_not_to_model_cap(
        relative, sample, board, monkeypatch, deceleration):
    stamp, linear, timing = prepare(relative, sample, board, monkeypatch,
                                    deceleration=deceleration)
    approvals = list(relative.owner.approvals)
    live = relative.owner._fresh_depth_linear_snapshot(1)
    if deceleration == .40:
        assert live is None
    else:
        assert live == linear
        assert live[1]*2 == sample.previous_request  # 24/8, NOT 42/29RPM.
        cap, reason = envelope(relative, stamp, timing, board, sample.age)
        assert reason == "same_grant_relative_braking_cap"
        assert cap > live[1]*2
        assert cap == pytest.approx(42. if sample.cap == 613 else 29.107, abs=.02)
    assert relative.owner._depth30_linear_timing is timing
    assert timing.depth_expires_at == pytest.approx(stamp+board["ttl"])
    assert relative.owner.approvals == approvals


def test_cap616_trial_still_vetoes_at_51ms_without_proven_deceleration(
        relative, board, monkeypatch):
    stamp, linear, timing = prepare(relative, SAMPLES[1], board, monkeypatch)
    for age in (.0299, .05):
        relative.clock.now = stamp+age
        relative.feedback = replace(relative.feedback, timestamp=relative.clock.now)
        assert relative.owner._fresh_depth_linear_snapshot(1) == linear
    relative.clock.now = stamp+.051
    relative.feedback = replace(relative.feedback, timestamp=relative.clock.now)
    assert relative.owner._fresh_depth_linear_snapshot(1) is None
    # A later slow encoder must not resurrect the same rejected grant.
    relative.feedback = replace(relative.feedback, left_forward_rpm=0., right_forward_rpm=0.)
    assert relative.owner._fresh_depth_linear_snapshot(1) is None
    assert relative.owner._depth30_linear_timing is timing


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda sample: f"CAP{sample.cap}")
def test_trial_envelope_is_nonincreasing_with_age(sample, relative, board, monkeypatch):
    stamp, _, timing = prepare(relative, sample, board, monkeypatch)
    values = [envelope(relative, stamp, timing, board, ms/1000.)[0]
              for ms in range(251)]
    assert all(0. <= value <= 42. for value in values)
    assert all(after <= before+1e-9 for before, after in zip(values, values[1:]))
    # The helper now supports up to300ms; this recorded profile's actual
    # lease still ends at250ms and is checked by the runtime test below.
    assert envelope(relative, stamp, timing, board, .300001) == (0., "invalid_braking_model")


def test_physical_depth_expiry_is_not_relaxed_even_with_excess_braking_margin(
        relative, board, monkeypatch):
    stamp, linear, timing = prepare(relative, SAMPLES[0], board, monkeypatch)
    timing = replace(timing, continuation_distance_m=5.)
    relative.owner._publish_depth_linear_pair(linear, timing)
    for age in (.249, .251):
        relative.clock.now = stamp+age
        relative.owner._last_vision_control_ts = relative.clock.now-.01
        relative.feedback = replace(relative.feedback, timestamp=relative.clock.now)
        live = relative.owner._fresh_depth_linear_snapshot(1)
        assert (live == linear) if age < .25 else (live is None)
    assert timing.depth_expires_at == pytest.approx(stamp+.25)


@pytest.mark.parametrize("fault", [
    "missing", "stale", "future", "untrusted", "nonfinite", "reverse",
    "body_overspeed", "outer_momentum", "approaching_person", "too_close",
    "explicit_stop", "visibility_expired",
])
def test_trial_does_not_relax_other_vetoes(relative, board, monkeypatch, fault):
    _, linear, timing = prepare(relative, SAMPLES[0], board, monkeypatch)
    assert relative.owner._fresh_depth_linear_snapshot(1) == linear
    if fault == "missing":
        relative.feedback = None
    elif fault == "approaching_person":
        timing = replace(timing, continuation_motion=replace(
            timing.continuation_motion, target_speed_bound_m_s=-2.))
        relative.owner._publish_depth_linear_pair(linear, timing)
    elif fault == "too_close":
        relative.owner._publish_depth_linear_pair(
            linear, replace(timing, continuation_distance_m=1.2))
    elif fault == "explicit_stop":
        relative.owner._explicit_stop_requested = True
    elif fault == "visibility_expired":
        relative.owner._last_vision_control_ts = relative.clock.now-.251
    else:
        changes = {
            "stale": dict(timestamp=relative.clock.now-.151),
            "future": dict(timestamp=relative.clock.now+.001),
            "untrusted": dict(trustworthy=False),
            "nonfinite": dict(left_forward_rpm=float("nan")),
            "reverse": dict(left_forward_rpm=-4.),
            "body_overspeed": dict(left_forward_rpm=43., right_forward_rpm=43.),
            # Mean42RPM fits the captured bound, outer84RPM does not fit braking.
            "outer_momentum": dict(left_forward_rpm=0., right_forward_rpm=84.),
        }[fault]
        relative.feedback = replace(relative.feedback, **changes)
    assert relative.owner._fresh_depth_linear_snapshot(1) is None
