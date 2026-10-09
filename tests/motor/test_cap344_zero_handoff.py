"""Normal following must not enter position-lock parking during a handoff."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_visible_wheel_continuity import feedback, visible_runtime
from test_follow_wheel_periodic import setup_periodic


@pytest.mark.parametrize("mode", [None, "zero"])
@pytest.mark.parametrize("turn", [(8, -8), (-8, 8)])
def test_zero_handoff_never_parks_and_still_confirms_feedback(monkeypatch, mode, turn):
    r, owner, driver, _, clock = visible_runtime(monkeypatch)
    r.config.follow_cross_brake_enable = True
    if mode is not None:
        r.config.follow_cross_brake_mode = mode
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: None
    for t, wheels in [(10., (21, 20)), (10.3, (12, 12)), (10.35, (0, 0)),
                      (10.35, (0, 0)), (10.4, (0, 0))]:
        clock[0] = t
        r.get_steering_feedback = lambda: feedback(clock[0], *wheels)
        r._send_follow_wheel_targets(turn[0], -turn[1], "TEST")
    assert driver.pairs == [(0, 0)] * 4 + [(turn[0], -turn[1])]
    assert not driver.stops and not r.backend.normal_zero_hold


def test_periodic_zero_handoff_recovers_latest_forward_without_parking(monkeypatch):
    r, owner, driver, _, clock, state = setup_periodic(monkeypatch)
    r.config.follow_cross_brake_enable = True
    r.config.follow_cross_brake_mode = "zero"
    r.config.follow_forward_handoff_enable = True
    state[:] = [24., 8., 10.01, 11.]
    r._service_follow_wheels()
    for t in (10.011, 10.06, 10.11):
        clock[0] = t
        r._service_follow_wheels()
    assert driver.pairs == [(32, -16)] + [(0, 0)] * 3
    state[:] = [62., 2., 11., 11.]
    clock[0] = 10.16
    r.get_steering_feedback = lambda: feedback(clock[0], 12, 11)
    r._service_follow_wheels()
    assert driver.pairs[-1] == (64, -60)
    assert not driver.stops and not r.backend.normal_zero_hold


@pytest.mark.parametrize("stop_mode,code", [("normal", 0), ("emergency", 1)])
def test_zero_handoff_does_not_disable_other_explicit_stops(monkeypatch, stop_mode, code):
    r, owner, driver, _, clock = visible_runtime(monkeypatch)
    r.config.follow_cross_brake_enable = True
    r.config.follow_cross_brake_mode = "zero"
    r.get_steering_feedback = lambda: feedback(clock[0], 21, 20)
    r._send_follow_wheel_targets(8, 8, "TEST")
    assert driver.pairs[-1] == (0, 0) and not driver.stops
    r.backend.send_stop("explicit_near_or_safety", mode=stop_mode)
    assert driver.stops == ([1, 0] if code == 0 else [code])


def test_real_binding_defaults_to_zero_and_loads_explicit_mode():
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root / "request_0513_modular.py").read_text())
    expressions = [k.value for n in ast.walk(tree) if isinstance(n, ast.Call)
                   for k in n.keywords if k.arg == "follow_cross_brake_mode"]
    assert len(expressions) == 1
    for env, expected in [({}, "zero"), ({"FOLLOW_CROSS_BRAKE_MODE": "zero"}, "zero"),
                          ({"FOLLOW_CROSS_BRAKE_MODE": " NORMAL "}, "normal")]:
        value = eval(compile(ast.Expression(expressions[0]), "binding", "eval"),
                     {"os": SimpleNamespace(environ=env)})
        assert value == expected
