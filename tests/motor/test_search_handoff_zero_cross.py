"""Search/opt-out keeps reversal protection; qualified forward may supersede it."""
import pytest
from car_control_modular.wheel_zero_cross import WheelZeroCrossGuard
from test_visible_wheel_continuity import feedback


@pytest.mark.parametrize("right", [True, False])
def test_without_forward_authority_base_restore_preserves_zero_cross_episode(right):
    guard = WheelZeroCrossGuard()
    turn = (8,-8) if right else (-8,8)
    forward = (43,33) if right else (33,43)
    assert guard.limit(turn, feedback(10.,24,24), 10.)[0] == (0,0)
    started = guard.started
    for t in (10.04,10.08):
        assert guard.limit(forward, feedback(t,24,24),t)[0] == (0,0)
        assert guard.started == started and guard.pending_signs == tuple(1 if v>0 else -1 for v in turn)
    assert guard.limit(turn, feedback(10.12,0,0),10.12)[0] == (0,0)
    assert guard.quiet_count == 1
    assert guard.limit(forward, feedback(10.12,0,0),10.13)[0] == (0,0)
    assert guard.limit(forward, feedback(10.16,0,0),10.16)[0] == forward
    assert guard.pending_signs is None


@pytest.mark.parametrize("requested", [(24,24),(33,43),(0,0)])
def test_straight_opposite_or_stop_does_not_replay_old_turn(requested):
    guard = WheelZeroCrossGuard()
    guard.limit((8,-8),feedback(10.,24,24),10.)
    result, _ = guard.limit(requested, feedback(10.05,24,24),10.05,allow_forward_handoff=True)
    assert result == requested
    assert guard.pending_signs is None


def test_normal_forward_without_pending_reversal_is_unchanged():
    guard = WheelZeroCrossGuard()
    assert guard.limit((43,33),feedback(10.,24,24),10.,allow_forward_handoff=True)[0] == (43,33)


def test_large_feedback_gap_cannot_supply_second_quiet_confirmation():
    guard = WheelZeroCrossGuard()
    guard.limit((8,-8),feedback(10.,24,24),10.)
    guard.limit((8,-8),feedback(10.05,0,0),10.05)
    assert guard.limit((43,33),feedback(10.3,0,0),10.3)[0] == (0,0)
    assert guard.quiet_count == 1
