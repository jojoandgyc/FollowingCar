from minimal_follow import LostPersonSearchConfig, LostPersonSearchPolicy, MinimalFollowCommand


def _policy(**overrides):
    return LostPersonSearchPolicy(LostPersonSearchConfig(**overrides))


def test_left_steering_loss_stops_once_then_searches_left():
    policy = _policy(turn_percent=9)
    policy.record_executed_follow_command(MinimalFollowCommand(6, 10, reason="steer_left"), 10.0)

    command, status = policy.target_missing(now=10.05, front_obstacle=False)
    assert command.reason == "search_transition_stop"
    assert status.state == "search_transition_stop"

    command, status = policy.target_missing(now=10.10, front_obstacle=False)
    assert command.reason == "search_rotate_left"
    assert (command.left_percent, command.right_percent) == (9, 9)
    assert (command.left_state, command.right_state) == (0x02, 0x01)
    assert status.direction == "left"


def test_right_search_is_bounded_and_front_ir_preempts():
    policy = _policy(timeout_sec=0.3)
    policy.record_executed_follow_command(MinimalFollowCommand(10, 6, reason="steer_right"), 20.0)
    policy.target_missing(now=20.01, front_obstacle=False)  # transition STOP
    command, _ = policy.target_missing(now=20.10, front_obstacle=False)
    assert command.reason == "search_rotate_right"
    assert (command.left_state, command.right_state) == (0x01, 0x02)

    command, status = policy.target_missing(now=20.31, front_obstacle=False)
    assert command.reason == "search_timeout"
    assert status.state == "search_timeout"

    command, status = policy.target_missing(now=20.32, front_obstacle=True)
    assert command.reason == "front_ir"
    assert status.state == "front_ir"


def test_reacquire_after_rotation_requires_one_stop_transition():
    policy = _policy()
    policy.record_executed_follow_command(MinimalFollowCommand(6, 10, reason="steer_left"), 25.0)
    policy.target_missing(now=25.01, front_obstacle=False)  # transition STOP
    policy.target_missing(now=25.06, front_obstacle=False)  # rotating left

    status = policy.visible(25.10)
    assert status.state == "search_reacquire_transition_stop"
    assert policy.visible(25.15).state == "tracking"


def test_stale_or_missing_direction_never_starts_rotation():
    policy = _policy(turn_memory_sec=0.2)
    policy.record_executed_follow_command(MinimalFollowCommand(6, 10, reason="steer_left"), 30.0)
    command, status = policy.target_missing(now=30.21, front_obstacle=False)
    assert command.reason == "person_missing_direction_unavailable"
    assert status.state == "direction_unavailable"
