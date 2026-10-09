"""Search-to-follow handoff limits, no hardware or identity relaxation."""
from types import SimpleNamespace

import pytest

from car_control_modular.search_reacquire_braking import limit_handoff_yaw
from test_search_handoff_braking import tracker, reason


@pytest.mark.parametrize("requested,expected", [(-15,-7), (15,7), (-3,-3), (0,0)])
def test_handoff_cannot_increase_search_yaw(requested, expected):
    owner = SimpleNamespace(_search_handoff_uid=1, _search_handoff_cap_rpm=7., search_state="none")
    assert limit_handoff_yaw(owner, 1, requested) == expected


@pytest.mark.parametrize("uid,state,obligation", [(2,"none",1), (1,"searching",1), (1,"none",None)])
def test_no_change_to_search_normal_follow_or_different_uid(uid,state,obligation):
    owner = SimpleNamespace(_search_handoff_uid=obligation, _search_handoff_cap_rpm=7., search_state=state)
    assert limit_handoff_yaw(owner, uid, -15) == -15


def test_execution_allowance_only_advances_stop_not_identity_or_stale_permission():
    # Without the future dispatch allowance .36 is still just outside braking.
    assert reason(.36) is None
    assert reason(.36, execution_delay_sec=.05) == "candidate_predictive_stop"
    assert reason(.36, execution_delay_sec=.05, eligible=False) is None
    assert reason(.36, execution_delay_sec=.05, capture_timestamp=9.7) is None


def test_request_write_veto_precedes_axis_revocation(monkeypatch):
    owner, _ = tracker(monkeypatch)
    events=[]
    owner._action_runtime.request_search_reacquire_brake=lambda *a: events.append("veto")
    owner._clear_lateral_intent=lambda *a: events.append("yaw_clear")
    owner._clear_longitudinal_context=lambda **a: events.append("depth_clear")
    assert owner._hold_search_reacquire_brake(bbox=(280,5,360,470),width=640,eligible=True)
    assert events == ["veto","yaw_clear","depth_clear"]


def test_failed_stop_request_does_not_remove_cap_or_clear_axes(monkeypatch):
    owner, _ = tracker(monkeypatch)
    owner._search_handoff_uid=1
    owner._action_runtime.request_search_reacquire_brake=lambda *a: False
    assert not owner._hold_search_reacquire_brake(bbox=(280,5,360,470),width=640,eligible=True)
    assert owner._search_handoff_uid == 1
    assert owner.clears == []
