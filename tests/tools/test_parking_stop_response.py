"""Fake encoder, clock and serial only. Never opens a real motor connection."""
import json
from types import SimpleNamespace

import pytest

from tools import test_parking_stop_response as stop


class Clock:
    def __init__(self): self.t = 10.
    def __call__(self): return self.t
    def sleep(self, dt): self.t += dt


class FakeSession:
    def __init__(self, clock, braking=500, fail=None):
        self.clock, self.braking, self.fail = clock, braking, fail
        self.mode, self.amps = "speed", 0
        self.desired = dict(left=0, right=0)
        self.rpm = dict(left=0., right=0.)
        self.pos = dict(left=0., right=0.)
        self.last = dict(left=clock(), right=clock())
        self.calls = []
        self.forward_signs = dict(left=1, right=1)
        self.closed = self.opened = False
        self.controller_diagnostics = {}

    def open(self): self.opened = True
    def close(self, ensure_stop=True): self.closed = True
    def stop(self, *a): self.calls.append(("emergency",)); self.desired = dict(left=0, right=0)
    def current(self, amps, trial):
        self.calls.append(("current", amps))
        self.clock.sleep(.004)
        self.amps = amps
    def speed(self, desired, trial):
        self.calls.append(("speed", desired.copy()))
        self.desired, self.mode = desired.copy(), "speed"
        self.clock.sleep(.004)
    def normal(self, trial):
        self.calls.append(("normal", self.amps))
        if self.fail == "write": raise OSError("partial stop write")
        self.mode = "hold"
        self.clock.sleep(.004)
    def read_wheel(self, side):
        started = self.clock()
        self.clock.sleep(.002)
        if self.fail == "serial" and self.desired[side]: raise OSError("serial lost")
        dt = self.clock()-self.last[side]
        target = 0 if self.mode == "hold" else self.desired[side]
        accel = self.braking if self.mode == "hold" else 400
        if self.fail == "slow": accel = 1
        old = self.rpm[side]
        self.rpm[side] += max(-accel*dt, min(accel*dt, target-old))
        self.pos[side] += (old+self.rpm[side])/2*6*dt
        self.last[side] = self.clock()
        return dict(side=side, read_started=started, timestamp=self.clock(),
            raw_rpm=round(self.rpm[side]), forward_rpm=round(self.rpm[side]),
            position_deg=0 if self.fail == "frozen_encoder" else round(self.pos[side]), current_a=self.amps,
            temperature_c=30, error_code=1 if self.fail == "fault" else 0)


def args(*extra):
    return stop.parser().parse_args(["--motion", "forward", "--setup", "bench",
                                  "--rated-current-a", "12", *extra])


def trial(rpm=60, amps=5, hold=300):
    return dict(rpm=rpm, current_a=amps, hold_ms=hold, status="running", events=[], samples=[])


@pytest.mark.parametrize("rpm,amps,hold", [(r,a,h) for r in (60,100) for a in (5,10) for h in (300,500,700,900)])
@pytest.mark.parametrize("motion", ["forward", "left", "right"])
def test_matrix_qualifies_actual_rpm_and_measures_hold_then_zero(rpm, amps, hold, motion):
    c = Clock(); s = FakeSession(c); t = trial(rpm, amps, hold)
    a = args("--motion", motion)
    stop.run_trial(s, a, t, c, c.sleep)
    assert t["status"] == "complete"
    assert t["pre_stop_rpm"] == stop.targets(rpm, motion)
    assert t["release_started"]-t["normal_completed"] >= hold/1000-1e-8
    assert s.calls == [("current",0), ("speed",dict(left=0,right=0)),
        ("speed",stop.targets(rpm,motion)), ("current",amps), ("normal",amps),
        ("current",0), ("speed",dict(left=0,right=0))]
    assert stop.analyze(t)["verdict"] == "encoder_stable"


def test_slow_braking_is_not_falsely_reported_stopped():
    c = Clock(); s = FakeSession(c, braking=50); t = trial(100)
    stop.run_trial(s, args(), t, c, c.sleep)
    assert stop.analyze(t)["wheels"]["left"]["stable_by_hold_deadline"] is False
    assert stop.analyze(t)["verdict"] == "not_stable"


@pytest.mark.parametrize("fail", ["slow", "serial", "fault", "write", "frozen_encoder"])
def test_failure_never_reports_a_successful_test(fail):
    c = Clock(); s = FakeSession(c, fail=fail); t = trial()
    with pytest.raises((RuntimeError, OSError)):
        stop.run_trial(s, args(), t, c, c.sleep)
    assert stop.analyze(t)["verdict"] == "aborted"
    if fail == "slow": assert not any(x[0] == "normal" for x in s.calls)


def row(t, pos=0, rpm=0):
    return dict(trusted=True, timestamp=t, read_started=t-.001,
                position_deg=pos, forward_rpm=rpm)


@pytest.mark.parametrize("kind", ["one", "gap", "late", "untrusted", "read_overlap"])
def test_missing_or_late_evidence_is_unknown(kind):
    rows = [row(x) for x in (10.21,10.24,10.27,10.299)]
    if kind == "one": rows = rows[:1]
    if kind == "gap": rows = [row(x) for x in (10.201,10.28,10.299)]
    if kind == "late": rows = [row(10.301+i*.02) for i in range(4)]
    if kind == "untrusted":
        for r in rows: r["trusted"] = False
    if kind == "read_overlap":
        for r in rows: r["read_started"] = 10.19
    assert stop.window_stable(rows,10.2,10.3) is None


@pytest.mark.parametrize("positions,rpms", [([0,1,3,4],[0]*4), ([0,1,0,1],[0]*4), ([0]*4,[0,0,2,0])])
def test_zero_snapshot_or_position_oscillation_not_stability(positions,rpms):
    rows = [row(t,p,r) for t,p,r in zip((10.21,10.24,10.27,10.299),positions,rpms)]
    assert stop.window_stable(rows,10.2,10.3) is False


@pytest.mark.parametrize("argv", [["--rpms","101"], ["--currents","15"], ["--holds-ms","100"],
    ["--rpms","60,60"], ["--drive-timeout","nan"], ["--max-travel","inf"],
    ["--rated-current-a","5"], ["--rated-current-a","nan"], ["--execute"]])
def test_bad_or_unconfirmed_parameters_rejected(argv):
    with pytest.raises(ValueError): stop.validate(stop.parser().parse_args(argv))


def test_default_plan_never_opens_motor(monkeypatch):
    monkeypatch.setattr(stop, "ParkingSession", lambda: pytest.fail("opened motor"))
    assert stop.main([]) == 0
    assert len(stop.validate(stop.parser().parse_args([]))) == 8


def test_requested_60rpm_longer_hold_plan_is_exactly_four_trials():
    assert stop.validate(args("--rpms", "60", "--holds-ms", "700,900")) == [
        (60,5,700), (60,5,900), (60,10,700), (60,10,900)]


@pytest.mark.parametrize("fault", [None,"serial","write","cancel"])
def test_real_main_cleans_up_and_preserves_partial_results(monkeypatch,tmp_path,fault):
    c = Clock(); s = FakeSession(c,fail=fault)
    monkeypatch.setattr(stop, "ParkingSession", lambda:s)
    monkeypatch.setattr(stop, "acquire_test_lock", lambda p:SimpleNamespace(close=lambda:None))
    monkeypatch.setattr(stop, "_ensure_follow_runtime_stopped", lambda:None)
    monkeypatch.setattr("car_control_modular.config_loader.load_config_to_env", lambda p:None)
    answers = iter(["NO"] if fault == "cancel" else ["PARK","RUN"])
    monkeypatch.setattr("builtins.input", lambda _:next(answers))
    real = stop.run_trial
    monkeypatch.setattr(stop, "run_trial", lambda s,a,t:real(s,a,t,c,c.sleep))
    argv = ["--execute","--motion","forward","--setup","bench","--rated-current-a","12",
            "--rpms","60","--currents","5","--holds-ms","300","--output-dir",str(tmp_path)]
    status = stop.main(argv)
    assert s.closed
    assert s.opened is (fault != "cancel")
    assert (status == 0) is (fault is None)
    records = [json.loads(p.read_text()) for p in tmp_path.glob("*.json")]
    assert records
    if fault in ("serial","write"):
        assert any(r["status"] == "aborted" and r["trials"][0]["samples"] for r in records)


def test_save_includes_raw_csv_and_unknown_not_fake_pass(tmp_path):
    p = stop.save(tmp_path, dict(trials=[trial()], parameters={}))
    data = json.loads(p.read_text())
    assert data["trials"][0]["analysis"]["verdict"] == "aborted"
    assert p.with_suffix(".csv").exists()


@pytest.mark.parametrize("problem", ["heat", "current", "nan", "wrong_direction", "slow_read", "travel"])
def test_live_sample_safety_checks_abort(problem):
    c = Clock(); s = FakeSession(c); t = trial(); a = args()
    original = s.read_wheel
    def read(side):
        r = original(side)
        if problem == "heat": r["temperature_c"] = 60
        if problem == "current": r["current_a"] = 13
        if problem == "nan": r["forward_rpm"] = float("nan")
        if problem == "wrong_direction" and s.desired[side]: r["forward_rpm"] = -10
        if problem == "slow_read": c.sleep(.2)
        return r
    s.read_wheel = read
    if problem == "travel": a.max_travel = .01
    with pytest.raises(RuntimeError): stop.run_trial(s,a,t,c,c.sleep)


def test_missed_release_deadline_cannot_be_a_pass():
    c = Clock(); s = FakeSession(c); t = trial()
    stop.run_trial(s,args(),t,c,c.sleep)
    t["release_started"] = t["hold_deadline"]+.08
    assert stop.analyze(t)["verdict"] == "inconclusive"


def test_real_current_writes_are_volatile_verified_and_normal_has_no_zero_packet(monkeypatch):
    c = Clock(); events = []; registers = {}
    monkeypatch.setattr(stop.time,"monotonic",c)
    def write(name,value,persist):
        assert persist is False
        registers[name] = value
        events.append(("current",name,value))
        c.sleep(.002)
    driver = SimpleNamespace(write_register=write, read_register=lambda n:registers[n],
        stop=lambda side,mode:events.append(("stop",side,mode)))
    s = stop.ParkingSession(); s.backend = SimpleNamespace(driver=driver)
    t = trial()
    s.current(10,t)
    s.normal(t)
    assert events[-2:] == [("stop","right",0),("stop","left",0)]
    assert len(t["events"]) == 6
    driver.read_register = lambda _:5
    with pytest.raises(RuntimeError,match="回读不匹配"): s.current(10,t)
