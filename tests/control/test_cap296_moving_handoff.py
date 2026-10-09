"""CAP287..314: confirmed moving handoff constrains yaw, not a new park cycle."""
from dataclasses import replace
import configparser
from pathlib import Path

import pytest

from car_control_modular.control_types import SteeringFeedback
from car_control_modular.search_reacquire_braking import (
    observe_moving_handoff, moving_handoff_yaw,
)
from test_cap881_search_handoff import handoff


def setup(monkeypatch, x=.55):
    owner, clock = handoff(monkeypatch, elapsed=.2)
    owner._follow_controller.cfg = replace(owner._follow_controller.cfg,
        visible_steering_pid_predictive_countersteer_max_correction_rpm=6.,
        visible_steering_pid_predictive_countersteer_gain_rpm_per_dps=.25)
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: ("forward", 21.)
    owner._action_runtime.get_steering_feedback = lambda: SteeringFeedback(
        timestamp=clock[0]-.02, trustworthy=True, left_forward_rpm=42., right_forward_rpm=24.,
        yaw_rate_right_dps=26., raw_yaw_rate_right_dps=29.)
    args = dict(bbox=(640*x-40, 0, 640*x+40, 470), width=640,
                eligible=True, confirmed=True, raw_track_id=1)
    return owner, clock, args


def test_confirmed_forward_predictive_stop_uses_yaw_constraint_not_500ms_park(monkeypatch):
    owner, _, args = setup(monkeypatch)
    assert not owner._hold_search_reacquire_brake(**args)
    assert owner._search_handoff_moving_active
    assert not owner.requests and not owner.clears
    assert owner._search_handoff_uid == 1


@pytest.mark.parametrize("missing", ["depth", "feedback", "unconfirmed", "ineligible"])
def test_missing_proof_cannot_select_moving_alternative(monkeypatch, missing):
    owner, _, args = setup(monkeypatch, x=.5)
    if missing == "depth": owner._fresh_depth_linear_snapshot = lambda *a, **kw: None
    if missing == "feedback": owner._action_runtime.get_steering_feedback = lambda: None
    if missing == "unconfirmed": args["confirmed"] = False
    if missing == "ineligible": args["eligible"] = False
    owner._hold_search_reacquire_brake(**args)
    assert not getattr(owner, "_search_handoff_moving_active", False)
    if missing in ("depth", "feedback"):
        assert len(owner.requests) == 1


def test_capture_replay_cannot_refresh_evidence(monkeypatch):
    owner, clock, args = setup(monkeypatch)
    owner._hold_search_reacquire_brake(**args)
    old = owner._search_handoff_moving_evidence
    clock[0] += .1
    owner._hold_search_reacquire_brake(**args)
    assert owner._search_handoff_moving_evidence is old


def test_feedback_cache_wait_cannot_retire_handoff_with_expired_image(monkeypatch):
    owner, clock, args = setup(monkeypatch)
    owner._search_handoff_started_capture_ts = 9.05
    get = owner._action_runtime.get_steering_feedback
    def delayed():
        clock[0] += .12  # 100ms image age -> 220ms, beyond existing 210ms gate
        return get()
    owner._action_runtime.get_steering_feedback = delayed
    assert not owner._hold_search_reacquire_brake(**args)
    assert owner._search_handoff_uid == 1
    assert owner._search_handoff_moving_evidence is None
    assert not owner.requests


def test_takeover_timer_cannot_remove_active_residual_constraint(monkeypatch):
    owner, _, args = setup(monkeypatch)
    owner._search_handoff_started_capture_ts = 9.05
    assert not owner._hold_search_reacquire_brake(**args)
    assert owner._search_handoff_uid == 1 and not owner.requests


def test_long_unresolved_residual_uses_existing_stop_not_unlimited_moving_mode(monkeypatch):
    owner, _, args = setup(monkeypatch)
    owner._search_handoff_moving_active = True
    owner._search_handoff_started_capture_ts = 8.5
    assert owner._hold_search_reacquire_brake(**args)
    assert owner.requests


@pytest.mark.parametrize("x,raw", [(.749,19.5264), (.673,21.1536), (.531,32.544),
                                     (.447,32.544), (.387,26.0352), (.289,22.7808)])
def test_cap296_to308_numeric_sequence_stays_bounded_nonreversing(monkeypatch, x, raw):
    owner, clock, args = setup(monkeypatch, x=x)
    cfg = owner._follow_controller.cfg
    evidence = observe_moving_handoff(owner, eligible=True, uid=1, raw_track_id=1,
        cap=296, stamp=9.85, bbox=args["bbox"], width=640, direction="right", now=10., max_age=.21)
    fb = SteeringFeedback(timestamp=9.98, trustworthy=True, left_forward_rpm=42., right_forward_rpm=24.,
        yaw_rate_right_dps=raw, raw_yaw_rate_right_dps=raw)
    result = moving_handoff_yaw(owner, evidence=evidence, uid=1, base=42., yaw=7.,
        feedback=fb, now=clock[0], policy=cfg)
    assert result is not None
    if x <= .673:
        assert -6 <= result[0] <= 0
    assert 42.+result[0] >= 0 and 42.-result[0] >= 0


@pytest.mark.parametrize("base", [.5, 1., 2., 42.])
def test_counter_differential_never_increases_base_or_reverses_wheel(monkeypatch, base):
    owner, clock, args = setup(monkeypatch)
    owner._hold_search_reacquire_brake(**args)
    result = moving_handoff_yaw(owner, evidence=owner._search_handoff_moving_evidence,
        uid=1, base=base, yaw=7., feedback=owner._action_runtime.get_steering_feedback(),
        now=clock[0], policy=owner._follow_controller.cfg)
    assert result is not None and -min(6,base) <= result[0] <= 0


def test_quiet_raw_feedback_does_not_keep_countersteering_from_old_image(monkeypatch):
    owner, clock, args = setup(monkeypatch, x=.5)
    owner._hold_search_reacquire_brake(**args)
    fb = SteeringFeedback(timestamp=clock[0], trustworthy=True,
        left_forward_rpm=24., right_forward_rpm=24., yaw_rate_right_dps=1., raw_yaw_rate_right_dps=0.)
    result = moving_handoff_yaw(owner, evidence=owner._search_handoff_moving_evidence,
        uid=1, base=24., yaw=7., feedback=fb, now=clock[0], policy=owner._follow_controller.cfg)
    assert result == (0., "center_same_direction_removed")


def test_quiet_center_retires_after_takeover_without_late_parking(monkeypatch):
    owner, _, args = setup(monkeypatch, x=.5)
    owner._search_handoff_started_capture_ts = 9.05
    owner._search_handoff_moving_active = True
    owner._action_runtime.get_steering_feedback = lambda: SteeringFeedback(
        timestamp=9.98, trustworthy=True, left_forward_rpm=24., right_forward_rpm=24.,
        yaw_rate_right_dps=0., raw_yaw_rate_right_dps=0.)
    assert not owner._hold_search_reacquire_brake(**args)
    assert owner._search_handoff_uid is None
    assert not owner.requests and not owner.clears


def test_cap294_and296_real_clock_and_runtime_parameters(monkeypatch):
    owner, _, _ = setup(monkeypatch)
    ini = configparser.ConfigParser()
    ini.read(Path(__file__).resolve().parents[2]/"car_control_modular/config/reid_runtime.ini")
    names = ("camera_hfov_deg", "camera_latency_sec", "predictive_brake_decel_dps2",
             "predictive_brake_margin_deg", "predictive_brake_response_sec",
             "predictive_countersteer_max_correction_rpm", "predictive_countersteer_gain_rpm_per_dps")
    cfg = replace(owner._follow_controller.cfg,
        **{"visible_steering_pid_"+name: ini.getfloat("steering_pid", name) for name in names},
        center_left_ratio=ini.getfloat("follow", "center_left_ratio"),
        center_right_ratio=ini.getfloat("follow", "center_right_ratio"))
    owner._search_handoff_started_capture_ts = 21800.404350012
    owner._search_handoff_moving_evidence = None
    # Actual logged capture/read/control timestamps, not a simulated new path.
    samples = [
        (294, 21800.54070685, .7734380154288781, 21800.715309, 21800.67469279, 5., -7., 19.5264, 19.5264),
        (296, 21800.642941011, .7492576554194161, 21800.8372, 21800.831324, 16., 1., 24.408, 21.1536),
    ]
    results = []
    for cap, stamp, x, now, ft, left, right, raw, filtered in samples:
        evidence = observe_moving_handoff(owner, eligible=True, uid=1, raw_track_id=1,
            cap=cap, stamp=stamp, bbox=(640*x-40,0,640*x+40,470), width=640,
            direction="right", now=now, max_age=.21)
        fb = SteeringFeedback(timestamp=ft, trustworthy=True, left_forward_rpm=left,
            right_forward_rpm=right, raw_yaw_rate_right_dps=raw, yaw_rate_right_dps=filtered)
        results.append(moving_handoff_yaw(owner, evidence=evidence, uid=1, base=42., yaw=3.,
            feedback=fb, now=now, policy=cfg, execution_delay_sec=.05))
    assert results[0] == (3., "tracking")
    assert results[1][1] == "residual_counter_differential"
    assert -.5 < results[1][0] < 0
    assert (round(42+results[1][0]), round(42-results[1][0])) == (42,42)
