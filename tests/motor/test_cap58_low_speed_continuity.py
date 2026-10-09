"""No hardware: small pivots reach the normal periodic wheel writer intact."""
import pytest

from test_follow_wheel_periodic import setup_periodic
from test_visible_wheel_continuity import feedback


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("rpm", [2, 4])
def test_small_continuously_refreshed_pivot_is_not_floored_or_zeroed(monkeypatch, sign, rpm):
    r,o,d,s,clock,axes=setup_periodic(monkeypatch)
    axes[:2]=[0.,sign*rpm]
    r.get_steering_feedback=lambda:feedback(clock[0],0,0)
    for i in range(12):
        clock[0]=10.+i*.051
        axes[3]=clock[0]+.15  # simulate a fresh visual lease, not a minimum on-time
        o._lateral_yaw_revision+=1
        r._service_follow_wheels()
    assert len(d.pairs) == 12
    assert all(pair == (sign*rpm,sign*rpm) for pair in d.pairs)  # raw right-wheel sign
    assert not d.stops
    assert r.backend.parking_current_a == 0
    # No unconditional 0.5s continuation: loss of the visual lease stops yaw.
    clock[0]=axes[3]+.01
    r._service_follow_wheels()
    assert d.pairs[-1] == (0,0)
