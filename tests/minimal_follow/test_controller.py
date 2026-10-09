from minimal_follow import MinimalFollowConfig, MinimalFollowController


def _controller():
    return MinimalFollowController(MinimalFollowConfig())


def test_centered_far_target_moves_straight():
    command = _controller().step(
        frame_width=640, bbox=(280, 80, 360, 420), distance_m=2.0, front_obstacle=False
    )
    assert command.reason == "forward"
    assert command.left_percent == command.right_percent > 0


def test_left_target_uses_forward_differential_not_rotation():
    command = _controller().step(
        frame_width=640, bbox=(0, 80, 160, 420), distance_m=2.0, front_obstacle=False
    )
    assert command.reason == "steer_left"
    assert 0 < command.left_percent < command.right_percent


def test_close_missing_distance_or_front_ir_stops():
    controller = _controller()
    assert controller.step(frame_width=640, bbox=(280, 80, 360, 420), distance_m=1.5, front_obstacle=False).moving is False
    assert controller.step(frame_width=640, bbox=None, distance_m=2.0, front_obstacle=False).reason == "person_missing"
    assert controller.step(frame_width=640, bbox=(280, 80, 360, 420), distance_m=None, front_obstacle=False).reason == "distance_unavailable"
    assert controller.step(frame_width=640, bbox=(280, 80, 360, 420), distance_m=2.0, front_obstacle=True).reason == "front_ir"
