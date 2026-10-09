"""No hardware: stop sequencing, read-only isolation and abort behavior."""
from types import SimpleNamespace

import pytest

from tools import test_stop_modes_diagnostic as mod


class Clock:
    def __init__(self): self.t = 10.
    def __call__(self): return self.t
    def sleep(self, seconds): self.t += seconds


class Driver:
    def __init__(self, clock):
        self.clock = clock
        self.calls = []
        self.amps = dict(left=0., right=0.)
        self.rpm = dict(left=0, right=0)
        self.pos = dict(left=0., right=0.)
        self.last = dict(left=clock(), right=clock())
        self.fail_stop = None
        self.current_readback_wrong = False
    def write_register(self, name, value, persist):
        assert not persist
        self.calls.append(("current", name, value))
        self.amps[name.split("_")[0]] = value
    def read_register(self, name):
        return 99 if self.current_readback_wrong else self.amps[name.split("_")[0]]
    def set_speed(self, side, value):
        self.calls.append(("speed", side, value))
        self.rpm[side] = value
    def stop(self, side, value):
        self.calls.append(("stop", side, value))
        if side == self.fail_stop: raise OSError("partial STOP")
        self.rpm[side] = 0
    def read_motor_status(self, side):
        self.clock.sleep(.001)
        self.pos[side] += (self.clock()-self.last[side])*6*self.rpm[side]
        self.last[side] = self.clock()
        return SimpleNamespace(speed_rpm=self.rpm[side], position_degree=round(self.pos[side]),
             phase_current_a=self.amps[side], pwm_percent=0., board_temperature_c=25, error_code=0)


def setup():
    clock = Clock()
    driver = Driver(clock)
    result = dict(events=[], samples=[], status="running")
    return mod.Diagnostic(driver, dict(left=1, right=1), result, clock, clock.sleep), driver, result


def test_read_only_and_close_cleanup_never_write():
    s, d, r = setup()
    s.observe("read_only", 1.)
    assert s.quiet("read_only")
    s.cleanup()
    assert not d.calls and not r["events"]


@pytest.mark.parametrize("case", ["normal-free", "free", "emergency"])
@pytest.mark.parametrize("rpm", [0, 8])
def test_sequences(case, rpm):
    s, d, r = setup()
    s.run(case, rpm, "forward")
    assert r["post_stop_quiet"]
    stops = [c for c in d.calls if c[0] == "stop"]
    assert stops == ([("stop", "right", 0), ("stop", "left", 0),
                      ("stop", "right", 2), ("stop", "left", 2)] if case == "normal-free" else
                     [("stop", "right", 2 if case == "free" else 1),
                      ("stop", "left", 2 if case == "free" else 1)])
    first_stop = next(i for i, c in enumerate(d.calls) if c[0] == "stop")
    assert not any(c[0] == "speed" for c in d.calls[first_stop:])
    if case == "normal-free":
        assert r["release_completed"] - r["normal_completed"] >= .5
        assert r["quiet_at_500ms"]


def test_unstable_baseline_blocks_all_writes():
    s, d, r = setup()
    d.rpm["left"] = 4
    with pytest.raises(RuntimeError, match="baseline not quiet"):
        s.run("normal-free", 8, "forward")
    s.cleanup()
    assert not d.calls


def test_partial_cleanup_failure_still_attempts_other_wheel_and_current():
    s, d, r = setup()
    s.modified = True
    d.fail_stop = "right"
    s.cleanup()
    assert d.calls[:2] == [("stop", "right", 1), ("stop", "left", 1)]
    assert [c[0] for c in d.calls] == ["stop", "stop", "current", "current"]
    assert r["status"] == "unsafe_or_unknown"


def test_readback_mismatch_prevents_drive():
    s, d, r = setup()
    d.current_readback_wrong = True
    with pytest.raises(RuntimeError, match="readback mismatch"):
        s.run("normal-free", 8, "forward")
    assert not any(c[0] == "speed" for c in d.calls)


def test_speed_limit_immediately_aborts():
    s, d, r = setup()
    d.rpm["right"] = 21
    with pytest.raises(RuntimeError, match="speed/travel limit"):
        s.observe("post_stop", 1.)


def test_repeated_opposite_motion_aborts():
    s, d, r = setup()
    s.expected["right"] = 1
    d.rpm["right"] = -4
    with pytest.raises(RuntimeError, match="unexpected reversal"):
        s.observe("post_stop", 1.)


def test_feedback_current_limit_aborts():
    s, d, r = setup()
    d.amps["left"] = 6.1
    with pytest.raises(RuntimeError, match="current limit"):
        s.observe("post_stop", 1.)


def test_encoder_motion_with_zero_speed_is_not_quiet():
    s, d, r = setup()
    s.observe("post_stop", .5)
    for i, row in enumerate(r["samples"]): row["position_deg"] = i * 2
    assert not s.quiet("post_stop")


def test_plan_does_not_open_hardware():
    assert mod.main([]) == 0


def test_execute_requires_independent_cutoff_confirmation():
    with pytest.raises(SystemExit, match="独立切断"):
        mod.main(["--execute"])


@pytest.mark.parametrize("case,mode", [("5a-free", 2), ("5a-emergency", 1)])
def test_5a_stop_holds_setting_then_clears_without_speed_write(case, mode):
    s, d, r = setup()
    s.run(case, 8, "forward")
    stop_index = next(i for i, c in enumerate(d.calls) if c[0] == "stop")
    assert d.calls[stop_index-2:stop_index] == [
        ("current", "right_parking_current", 5), ("current", "left_parking_current", 5)]
    assert d.calls[stop_index:stop_index+2] == [("stop", "right", mode), ("stop", "left", mode)]
    assert not any(c[0] == "speed" for c in d.calls[stop_index:])
    assert r["current_cleared"]-r["release_completed"] >= .5
    assert d.amps == {"left": 0, "right": 0}
    assert r["post_stop_quiet"]


def source_fixture(tmp_path, extra=""):
    (tmp_path / "camera_raw.frames.csv").write_text(
        "capture_frame_id,capture_unix_sec\n114,1790160875.689165\n")
    lines = ["2026-09-23 18:54:35,674 - LZ30EMA 停车命令: 模式=normal parking_current_a=5.0 pre_zero=True post_zero=False"]
    lines.append("2026-09-23 18:54:36,214 - LZ30EMA 驻车电流已确认: 右轮=0.0A 左轮=0.0A")
    for timestamp, value in [("36,224", 2), ("38,241", 1), ("39,300", 1),
                             ("40,503", 1), ("41,710", 1), ("41,957", 1)]:
        lines.append(f"2026-09-23 18:54:{timestamp} - LZ30EMA 停车命令: stop_value={value} pre_zero=False post_zero=False")
    lines.append("2026-09-23 18:54:41,993 - LZ30EMA 驻车电流已确认: 右轮=0.0A 左轮=0.0A")
    (tmp_path / "request_0513_modular.log").write_text("\n".join(lines) + "\n" + extra)
    return tmp_path


def test_replay_extract_and_execute_never_replays_feedback_or_bursts(tmp_path):
    plan = mod.extract_plan(source_fixture(tmp_path))
    assert len(plan["events"]) == 8
    assert plan["normal_to_cap_sec"] == pytest.approx(.015165, abs=1e-6)
    assert plan["events"][0]["offset"] == pytest.approx(.524835, abs=1e-6)
    s, d, r = setup()
    s.run("cap114-replay", 0, "forward", plan)
    assert not any(c[0] == "speed" and c[2] != 0 for c in d.calls)
    assert [c[2] for c in d.calls if c[0] == "stop"] == [0,0,2,2] + [1]*10
    assert len(r["replay_dispatches"]) == 8
    assert all(0 <= e["lateness_ms"] <= 10 for e in r["replay_dispatches"])
    assert r["post_stop_quiet"]


def test_replay_rejects_source_with_speed_write(tmp_path):
    with pytest.raises(ValueError, match="速度写入"):
        mod.extract_plan(source_fixture(tmp_path, "2026-09-23 18:54:39,000 - LZ30EMA 电机命令: 左轮=99转/分"))


def test_replay_rejects_unsupported_current(tmp_path):
    path = source_fixture(tmp_path)
    log = path / "request_0513_modular.log"
    log.write_text(log.read_text().replace("右轮=0.0A 左轮=0.0A", "右轮=10.0A 左轮=10.0A"))
    with pytest.raises(ValueError, match="清0A"):
        mod.extract_plan(path)


def test_replay_missing_plan_blocks_writes():
    s, d, r = setup()
    with pytest.raises(ValueError, match="validated"):
        s.run("cap114-replay", 8, "forward")
    assert not d.calls


@pytest.mark.parametrize("drive,observe", [(2,1), (.5,20), (float('nan'),1), (.5,float('inf'))])
def test_rejects_unbounded_durations_before_writes(drive, observe):
    s, d, r = setup()
    with pytest.raises(ValueError):
        s.run("5a-free", 8, "forward", drive_sec=drive, observe_sec=observe)
    assert not d.calls


def test_longer_observation_does_not_extend_5a_hold():
    s, d, r = setup()
    s.run("5a-emergency", 8, "forward", drive_sec=.8, observe_sec=5)
    assert .5 <= r["current_cleared"]-r["release_completed"] < .55
    rows = [row for row in r["samples"] if row['phase']=='post_stop']
    assert rows[-1]['timestamp']-rows[0]['timestamp'] >= 4.9


def test_60rpm_requires_exact_explicit_profile():
    assert mod.trial_limits("cap114-60", "cap114-replay", 60, "forward", 1.8) == dict(
        command_rpm=60, feedback_rpm=75, travel_deg=900, current_a=6)
    for profile, case, rpm, motion, seconds in (
        ("low-speed", "cap114-replay", 60, "forward", 1.8),
        ("cap114-60", "free", 60, "forward", 1.8),
        ("cap114-60", "cap114-replay", 60, "left", 1.8),
        ("cap114-60", "cap114-replay", 60, "forward", 2),
        ("cap114-60", "cap114-replay", 61, "forward", 1.8)):
        with pytest.raises(ValueError):
            mod.trial_limits(profile, case, rpm, motion, seconds)


def test_60rpm_forward_replay_retains_limits_and_no_extra_motion(tmp_path):
    plan = mod.extract_plan(source_fixture(tmp_path))
    s, d, r = setup()
    s.run("cap114-replay", 60, "forward", plan, 1.8, 1., "cap114-60")
    assert [c for c in d.calls if c[0] == "speed" and c[2] != 0] == [
        ("speed", "right", 60), ("speed", "left", 60)]
    assert r['post_stop_quiet']
    drive = [row for row in r['samples'] if row['phase']=='drive']
    assert drive[-1]['timestamp']-drive[0]['timestamp'] >= 1.77
    d.amps['left'] = 6.1
    with pytest.raises(RuntimeError, match="current limit"):
        s.sample("replay")


def test_60rpm_profile_retains_overspeed_and_total_travel_abort():
    s, d, r = setup()
    s.limits = mod.trial_limits("cap114-60", "cap114-replay", 60, "forward", 1.8)
    d.rpm['right'] = 76
    with pytest.raises(RuntimeError, match="speed/travel limit"):
        s.sample("drive")
    d.rpm['right'] = 0
    s.travel['right'] = 901
    with pytest.raises(RuntimeError, match="speed/travel limit"):
        s.sample("replay")


@pytest.mark.parametrize("limit", [None, 0, 4.9, 30.1, float('nan'), float('inf')])
def test_80rpm_requires_verified_bounded_current_limit(limit):
    with pytest.raises(ValueError, match="已核实"):
        mod.trial_limits("cap114-80", "cap114-replay", 80, "forward", 2., limit)


@pytest.mark.parametrize("case,rpm,motion,seconds", [
    ("free",80,"forward",2.), ("cap114-replay",81,"forward",2.),
    ("cap114-replay",80,"left",2.), ("cap114-replay",80,"forward",2.1)])
def test_80rpm_rejects_wrong_experiment(case,rpm,motion,seconds):
    with pytest.raises(ValueError, match="80RPM前进2秒"):
        mod.trial_limits("cap114-80",case,rpm,motion,seconds,10)


def test_80rpm_complete_fake_replay_keeps_parking_at_5a(tmp_path):
    plan = mod.extract_plan(source_fixture(tmp_path))
    s, d, r = setup()
    s.run("cap114-replay",80,"forward",plan,2.,1.,"cap114-80",10)
    assert r['safety_limits'] == dict(command_rpm=80, feedback_rpm=95, travel_deg=1350, current_a=10.)
    assert [c for c in d.calls if c[0]=='speed' and c[2]!=0] == [
        ('speed','right',80),('speed','left',80)]
    assert set(c[2] for c in d.calls if c[0]=='current') == {0,5}
    assert r['post_stop_quiet'] and len(r['replay_dispatches'])==8
    d.amps['left']=10.01
    with pytest.raises(RuntimeError,match="current limit"):
        s.sample('replay')


def test_80rpm_keeps_reversal_abort():
    s,d,r=setup()
    s.limits=mod.trial_limits('cap114-80','cap114-replay',80,'forward',2.,10)
    s.expected['right']=1
    d.rpm['right']=-4
    with pytest.raises(RuntimeError,match='unexpected reversal'):
        s.observe('replay',1.)


def test_80rpm_missing_current_stops_before_hardware(monkeypatch):
    monkeypatch.setattr(mod, 'acquire_test_lock', lambda port: pytest.fail('should not access hardware'))
    with pytest.raises(SystemExit,match='已核实'):
        mod.main(['--execute','--attended-power-cutoff','--profile','cap114-80',
                  '--case','cap114-replay','--rpm','80','--drive-sec','2'])


def zero_entry_fixture(tmp_path, bad=False):
    source_fixture(tmp_path)
    log = tmp_path/'request_0513_modular.log'
    entries=[]
    for ms in ('349','403','436','568'):
        rpm=1 if bad and ms=='436' else 0
        entries.append(f'2026-09-23 18:54:35,{ms} - LZ30EMA 电机命令: 左轮={rpm}转/分 右轮=0转/分')
    log.write_text('\n'.join(entries)+'\n'+log.read_text())
    plan=mod.extract_plan(tmp_path)
    plan['zero_entry']=mod.extract_zero_entry(plan)
    return plan


def test_repeated_zero_extract_and_order(tmp_path):
    plan=zero_entry_fixture(tmp_path)
    assert [e['offset'] for e in plan['zero_entry']['events']]==[0,.054,.087,.219]
    assert plan['zero_entry']['normal_offset']==.325
    s,d,r=setup()
    s.run('cap114-replay',80,'forward',plan,2.,1.,'cap114-80',10)
    assert len(r['zero_entry_dispatches'])==4
    assert [round(e['actual_offset'],3) for e in r['zero_entry_dispatches']]==[0,.054,.087,.219]
    # Fake writes are instantaneous: only the explicitly reserved 15ms is absent.
    assert r['first_zero_to_normal_ms']==pytest.approx(310)
    first_stop=next(i for i,c in enumerate(d.calls) if c[0]=='stop')
    drive=next(i for i,c in enumerate(d.calls) if c[0]=='speed' and c[2]==80)
    zeros=[c for c in d.calls[drive:first_stop] if c[0]=='speed' and c[2]==0]
    assert len(zeros)==10  # Four recorded groups plus NORMAL's own pre-zero.
    assert not any(c[0]=='speed' for c in d.calls[first_stop:])
    assert r['post_stop_quiet']


def test_repeated_zero_rejects_nonzero_in_entry(tmp_path):
    with pytest.raises(ValueError,match='zero-only'):
        zero_entry_fixture(tmp_path,bad=True)


def test_repeated_zero_rejects_changed_source(tmp_path):
    source_fixture(tmp_path)
    plan=mod.extract_plan(tmp_path)
    log=tmp_path/'request_0513_modular.log'
    log.write_text(log.read_text()+'changed')
    with pytest.raises(ValueError,match='source changed'):
        mod.extract_zero_entry(plan)


def test_replay_deadline_abort_does_not_burst_commands():
    s,d,r=setup()
    with pytest.raises(RuntimeError,match='deadline missed'):
        s.wait_replay_deadline(s.clock()-.2,'zero_entry')
    assert not d.calls


def test_repeated_zero_monitors_reversal_before_parking(tmp_path):
    plan=zero_entry_fixture(tmp_path)
    s,d,r=setup()
    s.expected['right']=1
    original=d.set_speed
    def stuck_reverse(side,value):
        original(side,value)
        if side=='right':d.rpm[side]=-4
    d.set_speed=stuck_reverse
    with pytest.raises(RuntimeError,match='unexpected reversal'):
        s.repeated_zero_entry(plan['zero_entry'])
    assert not any(c[0]=='stop' for c in d.calls)
