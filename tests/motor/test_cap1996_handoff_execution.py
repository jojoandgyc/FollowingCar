"""Final wheel writer enforces cap, including producer/fast-loop bypass."""
import pytest

from test_follow_wheel_periodic import setup_periodic
from test_search_handoff_execution import arm


@pytest.mark.parametrize("base", [0.,28.])
@pytest.mark.parametrize("yaw", [-15.,15.])
def test_periodic_writer_keeps_base_but_limits_search_handoff_yaw(monkeypatch,base,yaw):
    rt,owner,driver,_,clock,state=setup_periodic(monkeypatch)
    owner._search_handoff_uid=1
    owner._search_handoff_cap_rpm=7.
    state[0],state[1]=base,yaw
    rt._service_follow_wheels()
    assert driver.pairs
    left,right=driver.pairs[-1]
    # Backend right motor has reversed mounting sign.
    assert abs((left+right)/2) <= 7
    assert (left-right)/2 == base


def test_search_stop_still_vetoes_capped_packet(monkeypatch):
    rt,owner,driver,_,clock,state=setup_periodic(monkeypatch)
    owner._search_handoff_uid=1
    owner._search_handoff_cap_rpm=7.
    state[0],state[1]=28.,-15.
    arm(rt,owner,clock)
    rt._service_follow_wheels()
    assert driver.stops == [1, 0]
    assert driver.pairs == [(0,0)]
