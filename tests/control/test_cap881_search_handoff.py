"""CAP867->887 brake latch / CAP900->901 weak probe, no hardware."""
from dataclasses import replace
from types import SimpleNamespace
import ast
from pathlib import Path

import pytest
import request_0513_modular as runtime
from car_control_modular.search_reacquire_braking import HandoffObservation
from car_control_modular.search_candidate_gate import (
    SearchCandidateGate, SearchCandidateGateConfig, CandidateObservation,
)
from car_control_modular.control_types import SteeringFeedback
from test_search_handoff_braking import tracker


def handoff(monkeypatch, elapsed=4.):
    t, clock = tracker(monkeypatch)
    t.search_state = t._follow_controller.search_state = "none"
    t.search_direction = None
    t._follow_controller.active_target_id = 1
    t._search_handoff_uid = 1
    t._search_handoff_direction = "right"
    t._search_handoff_started_capture_ts = 9.9-elapsed
    t._search_handoff_cap_rpm = 7.
    t._follow_controller.cfg = replace(t._follow_controller.cfg,
        visible_steering_pid_camera_hfov_deg=66.,
        visible_steering_pid_predictive_brake_decel_dps2=60.)
    t._action_runtime.get_steering_feedback = lambda: SteeringFeedback(
        timestamp=clock[0]-.02, trustworthy=True,
        yaw_rate_right_dps=43.9344, raw_yaw_rate_right_dps=43.9344)
    return t, clock


def tick(t, **kw):
    args = dict(bbox=(462.2447, 0., 639., 479.), width=640,
                eligible=True, confirmed=True, raw_track_id=4)
    args.update(kw)
    return t._hold_search_reacquire_brake(**args)


def test_cap791_old_handoff_cannot_brake_cap867(monkeypatch):
    t, _ = handoff(monkeypatch)
    assert not tick(t)
    assert t._search_handoff_uid is None
    assert not t.requests and not t.clears  # No zero or restored authority.


def test_residual_search_predictive_stop_is_preserved(monkeypatch):
    t, _ = handoff(monkeypatch, elapsed=.1)
    assert tick(t)
    assert t.requests[0][2] == "candidate_predictive_stop"


@pytest.mark.parametrize("bad", ["stale", "future", "unconfirmed", "ineligible", "wrong_uid", "invalid_box"])
def test_no_retirement_without_current_trusted_evidence(monkeypatch, bad):
    t, _ = handoff(monkeypatch)
    kw = {}
    if bad == "stale": t._active_capture_timestamp = 9.
    if bad == "future": t._active_capture_timestamp = 10.1
    if bad == "unconfirmed": kw["confirmed"] = False
    if bad == "ineligible": kw["eligible"] = False
    if bad == "wrong_uid": t._follow_controller.active_target_id = 2
    if bad == "invalid_box": kw["bbox"] = (20, 0, 10, 40)
    assert not tick(t, **kw)
    assert t._search_handoff_uid == 1
    assert not t.requests


def test_pending_brake_is_never_cancelled_by_takeover(monkeypatch):
    t, _ = handoff(monkeypatch)
    t._action_runtime.search_reacquire_brake_pending = lambda: True
    assert tick(t)
    assert t._search_handoff_uid == 1 and not t.requests


def test_replayed_capture_cannot_retire_handoff(monkeypatch):
    t, _ = handoff(monkeypatch)
    t._search_handoff_last_capture = (t._active_capture_frame_id, t._active_capture_timestamp)
    # Prevent a legitimate predictive stop from obscuring the retirement check.
    t._action_runtime.get_steering_feedback = lambda: None
    assert not tick(t)
    assert t._search_handoff_uid == 1


@pytest.mark.parametrize("sign", [-1, 1])
def test_outward_confirmed_chain_retires_before_deadline(monkeypatch, sign):
    t, clock = handoff(monkeypatch, elapsed=0)
    t._search_handoff_direction = "right" if sign == 1 else "left"
    # Seed two independent observations without an encoder braking request.
    t._action_runtime.get_steering_feedback = lambda: None
    for i, x in enumerate([.75, .78, .81]):
        clock[0] = 10. + i*.1
        t._active_capture_timestamp = clock[0]-.1
        t._active_capture_frame_id = 800+i
        x = .5+sign*(x-.5)
        assert not tick(t, bbox=(x*640-40, 0., x*640+40, 470.))
    assert t._search_handoff_uid is None
    assert not t.requests and not t.clears


@pytest.mark.parametrize("bad", ["inward", "jitter", "duplicate", "gap", "raw", "area", "center"])
def test_bad_outward_evidence_never_retires(bad):
    o = HandoffObservation()
    answers = []
    for i,x in enumerate([.75,.78,.81]):
        cap, stamp, raw, half = 10+i, 10+i*.1, 4, 40
        if bad == "inward": x = 1.5-x
        if bad == "jitter" and i == 2: x = .76
        if bad == "duplicate": cap,stamp = 10,10.
        if bad == "gap" and i == 2: stamp += .3
        if bad == "raw" and i == 2: raw = 5
        if bad == "area" and i == 2: half = 10
        if bad == "center": x -= .25
        answers.append(o.outward(uid=1, raw_track_id=raw, cap=cap, stamp=stamp,
            bbox=(x*640-half,0,x*640+half,470), width=640, direction="right"))
    assert not any(answers)


def gate():
    return SearchCandidateGate(SearchCandidateGateConfig(
        probe_observation_min_score=.15, probe_edge_min_score=.2,
        probe_confirm_frames=2, hold_frames=2))


def update(g, score, timestamp=10., preferred=False, formal=False):
    box = (427.3,178.5,488.8,273.4)
    args = dict(timestamp=timestamp, search_active=True, width=640,height=480,
        preferred_bbox=box if preferred else None)
    args["formal_candidates" if formal else "probe_candidates"] = (CandidateObservation(box,score),)
    return g.update(**args)


def test_cap900_901_background_never_stops_search():
    g = gate()
    for i,score in enumerate([.1402,.1107]*5):
        d = update(g,score,10.+i*.05)
        assert not d.pause_rotation and not g.hold_active
        assert d.reason == "weak_probe_observation_only"
        assert g.last_ignored_weak_probes


@pytest.mark.parametrize("score,preferred,formal", [(.18,False,False),(.111,True,False),(.3,False,True)])
def test_real_candidates_keep_bounded_observation(score,preferred,formal):
    g = gate()
    result = [update(g,score,10+i*.1,preferred,formal) for i in range(8)]
    assert any(r.entered for r in result)
    assert sum(r.pause_rotation for r in result) <= 2
    assert not result[-1].pause_rotation


def test_weakening_probe_releases_only_its_own_hold():
    g = gate()
    update(g,.18); assert update(g,.18,10.1).entered
    d = update(g,.11,10.2)
    assert d.completed and not d.pause_rotation
    g = gate()
    assert update(g,.3,formal=True).entered
    assert update(g,.11,10.1).pause_rotation  # Formal hold not erased.


def test_production_binding_and_handoff_arming():
    tree = ast.parse(Path(runtime.__file__).read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call)
             and isinstance(n.func,ast.Name) and n.func.id == "SearchCandidateGateConfig"]
    assert len(calls) == 1
    assert next(k.value.value for k in calls[0].keywords if k.arg == "probe_observation_min_score") == .15
    source = Path(runtime.__file__).read_text()
    assert "self._search_handoff_started_capture_ts = float(self._active_capture_timestamp)" in source
    assert 'raw_track_id=getattr(current["rec"], "track_id", None)' in source


def test_real_confirmed_target_entry_retires_old_handoff(monkeypatch):
    t, _ = handoff(monkeypatch)
    t.frame_index = 450
    candidate = dict(stable_id=1, bbox=(462.,0.,639.,479.),
        rec=SimpleNamespace(track_id=4,time_since_update=0))
    assert not t._hold_for_confirmed_search_reacquire([candidate],width=640)
    assert t._search_handoff_uid is None and not t.requests and not t.clears


@pytest.mark.parametrize("present", [False, True])
def test_search_diagnostics_report_normalized_feedback_without_io(present):
    from car_control_modular.search_diagnostics import (
        SearchDiagnosticsObserver, SearchDiagnosticsConfig, SearchDiagnosticSample,
        SearchControlObservation, RecognitionObservation, MotionObservation,
        TransportObservation,
    )
    messages = []
    logger = SimpleNamespace(info=lambda fmt,*args: messages.append(fmt % args))
    observer = SearchDiagnosticsObserver(SearchDiagnosticsConfig(snapshot_enabled=False),logger)
    motion = (MotionObservation(left_speed_rpm=7,right_speed_rpm=7,
        left_forward_rpm=7,right_forward_rpm=-7,feedback_timestamp=10.,
        left_position_deg=12,right_position_deg=13) if present else MotionObservation())
    observer.observe(SearchDiagnosticSample(frame_index=1,timestamp=10.1,width=640,height=480,
        recognition=RecognitionObservation(),
        control=SearchControlObservation(state_after="searching",direction_after="right"),
        motion=motion,transport=TransportObservation()))
    line = next(m for m in messages if m.startswith("search_motion_diag"))
    assert "speed_fields_above=raw_serial" in line
    if present:
        assert "feedback_forward_rpm=(7.000,-7.000)" in line
        assert "feedback_timestamp=10.0 position_deg=(12,13)" in line
    else:
        assert "feedback_forward_rpm=(none,none)" in line
