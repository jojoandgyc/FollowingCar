"""No ports, camera, RKNN or motors: fixed pulse/parking and visual angle checks."""
import json
import math
from types import SimpleNamespace

import pytest

from tools import test_low_speed_turn as turn
from test_parking_stop_response import Clock, FakeSession


class Vision:
    def __init__(self, clock, fail=False): self.clock,self.fail=clock,fail
    def sample(self, now):
        if self.fail and now > 10.6: raise RuntimeError("visual lost")
        return dict(timestamp=now,cap=int(round(now*1000)),x=.5,area=1000,bbox=[0,0,20,50],score=.9)


def args(*extra): return turn.parser().parse_args(["--rated-current-a","30",*extra])
def trial(rpm=4,direction="left"):
    return dict(rpm=rpm,direction=direction,status="running",events=[],samples=[],vision=[])


@pytest.mark.parametrize("rpm",[4,6])
@pytest.mark.parametrize("direction",["left","right"])
def test_one_second_target_then_three_second_5a_hold_then_checked_release(rpm,direction):
    c=Clock(); s=FakeSession(c); t=trial(rpm,direction)
    turn.run_trial(s,Vision(c),args(),t,c,c.sleep)
    assert t["status"] == "complete"
    assert t["brake_prepare_started"]-t["drive_completed"] == pytest.approx(1,abs=.025)
    assert t["release_started"]-t["normal_completed"] >= 3-1e-8
    assert t["finished"]-t["release_completed"] >= .35-1e-8
    assert s.calls == [("current",0),("speed",dict(left=0,right=0)),
        ("speed",turn.targets(rpm,direction)),("current",5),("normal",5),
        ("current",0),("speed",dict(left=0,right=0))]
    assert turn.analyze(t,66)["total_until_settled_deg"] == 0


def test_default_plan_does_not_open_hardware(monkeypatch):
    monkeypatch.setattr(turn,"ParkingSession",lambda:pytest.fail("hardware opened"))
    assert turn.main([]) == 0
    assert turn.plan(args()) == [(4,"left"),(4,"right"),(6,"left"),(6,"right")]


def test_async_vision_drain_keeps_fixed_window_and_does_not_restart_motor():
    c=Clock(); s=FakeSession(c); t=trial()
    class DelayedVision:
        def sample(self,now):
            # 15FPS captures delivered 200ms later, like the measured failure.
            stamp=math.floor((now-.2)*15)/15
            return dict(timestamp=stamp,cap=round(stamp*1000),x=.5,area=1000,
                        bbox=[0,0,20,50],score=.9)
    turn.run_trial(s,DelayedVision(),args(),t,c,c.sleep)
    assert t["finished"] >= t["visual_window_end"]+.2
    assert t["status"] == "complete"
    assert len([v for v in t["vision"] if t["visual_window_end"]-.3 <= v["timestamp"] <= t["visual_window_end"]]) >= 3
    assert s.calls[-2:] == [("current",0),("speed",dict(left=0,right=0))]
    assert turn.analyze(t,66)["visually_stable"]


def test_visual_drain_timeout_never_completes():
    c=Clock(); s=FakeSession(c); t=trial()
    class FrozenVision(Vision):
        def sample(self,now):
            r=super().sample(min(now,14.2))
            return r
    with pytest.raises(RuntimeError,match="视觉结果未收齐"):
        turn.run_trial(s,FrozenVision(c),args(),t,c,c.sleep)
    assert t["status"] != "complete"
    assert s.calls[-1] == ("speed",dict(left=0,right=0))


@pytest.mark.parametrize("extra",[["--execute"],["--rpms","8"],["--rpms","4,4"],
    ["--directions","forward"],["--repeats","4"],["--hfov","nan"],["--rated-current-a","nan"],
    ["--rated-current-a","4"]])
def test_parameter_rejection(extra):
    with pytest.raises(ValueError): turn.plan(turn.parser().parse_args(extra))


def item(stamp=10.,x=.5):
    return dict(capture_monotonic=stamp,frame_width=640,capture_index=1,
        detections=[dict(class_id=0,score=.9,bbox=[640*x-30,10,640*x+30,400])])


@pytest.mark.parametrize("bad",["stale","future","missing","two_people","edge","nan","low_score"])
def test_camera_guard_rejects_bad_evidence(bad):
    v=item()
    if bad == "stale": v["capture_monotonic"]=9.5
    if bad == "future": v["capture_monotonic"]=11
    if bad == "missing": v["detections"]=[]
    if bad == "two_people": v["detections"]*=2
    if bad == "edge": v=item(x=.06)
    if bad == "nan": v["detections"][0]["bbox"][0]=float("nan")
    if bad == "low_score": v["detections"][0]["score"]=.4
    with pytest.raises(RuntimeError): turn.person_sample(v,10.1)


def test_reader_preserves_partial_line_and_old_frame_does_not_renew(tmp_path):
    path=tmp_path/"vision.jsonl"
    path.write_text(json.dumps(item())+"\n",encoding="utf-8")
    r=turn.VisionReader(path)
    assert r.sample(10.1)["timestamp"] == 10.
    text=json.dumps(dict(item(10.2),capture_index=2))
    with path.open("a") as f: f.write(text[:20])
    assert r.sample(10.25)["cap"] == 1
    with path.open("a") as f: f.write(text[20:]+"\n")
    assert r.sample(10.3)["cap"] == 2
    with pytest.raises(RuntimeError): r.sample(11.)
    r.close()


@pytest.mark.parametrize("failure",["fault","serial","write","visual","moving_after_release"])
def test_failed_trial_never_reports_complete_or_continues(failure):
    c=Clock(); s=FakeSession(c,fail=failure); t=trial()
    if failure == "moving_after_release":
        original=s.speed
        def speed(desired,t):
            was_hold=s.mode == "hold"
            original(desired,t)
            if was_hold: s.desired=dict(left=3,right=3)
        s.speed=speed
    with pytest.raises((RuntimeError,OSError)):
        turn.run_trial(s,Vision(c,failure == "visual"),args(),t,c,c.sleep)
    assert turn.analyze(t,66)["verdict"] == "aborted"


def test_bearing_geometry_and_drive_vs_residual_are_separate():
    assert turn.bearing(0) == pytest.approx(-33)
    assert turn.bearing(.5) == 0
    assert turn.bearing(1) == pytest.approx(33)
    t=dict(status="complete",drive_started=10.,drive_completed=10.01,
        normal_completed=11.05,finished=14.4,vision=[])
    for i in range(92):
        stamp=9.9+i*.05
        angle=0 if stamp <= 10 else min(12,10*(stamp-10))
        x=(1-math.tan(math.radians(angle))/math.tan(math.radians(33)))/2
        t["vision"].append(dict(timestamp=stamp,x=x))
    r=turn.analyze(t,66)
    assert r["average_to_1s_deadline_dps"] == pytest.approx(10)
    assert r["total_until_settled_deg"] == pytest.approx(12)
    assert r["residual_after_normal_deg"] == pytest.approx(1.5)
    assert r["stop_overrun_ms"] == pytest.approx(40)
    assert r["observed_1deg_onset_ms"] == pytest.approx(100,abs=51)
    assert r["visually_stable"]
    t["vision"]=t["vision"][::8]
    assert turn.analyze(t,66)["verdict"] == "insufficient_visual_samples"


@pytest.mark.parametrize("error",[RuntimeError("serial fault"),KeyboardInterrupt()])
def test_main_aborts_remaining_trials_cleans_up_and_saves_partial_data(monkeypatch,tmp_path,error):
    from car_control_modular import config_loader
    c=Clock(); s=FakeSession(c)
    lock=SimpleNamespace(close=lambda:None)
    monkeypatch.setattr(turn,"ParkingSession",lambda:s)
    monkeypatch.setattr(turn,"acquire_test_lock",lambda p:lock)
    monkeypatch.setattr(turn,"_ensure_follow_runtime_stopped",lambda:None)
    monkeypatch.setattr(config_loader,"load_config_to_env",lambda p:None)
    monkeypatch.setattr("builtins.input",lambda prompt:"TURN")
    v=SimpleNamespace(sample=lambda now:dict(timestamp=now),close=lambda:None)
    monkeypatch.setattr(turn,"VisionReader",lambda p:v)
    called=[]
    def fail(session,vision,args,t):
        called.append(t)
        t["events"].append(dict(kind="partial_write"))
        raise error
    monkeypatch.setattr(turn,"run_trial",fail)
    assert turn.main(["--execute","--rated-current-a","30","--vision-jsonl","unused",
                      "--output-dir",str(tmp_path)]) == 1
    assert len(called) == 1 and s.closed
    result=json.loads(next(tmp_path.glob("*.json")).read_text())
    assert result["status"] == "aborted"
    assert result["trials"][0]["events"] == [dict(kind="partial_write")]
    assert result["trials"][0]["analysis"]["verdict"] == "aborted"
