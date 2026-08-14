#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from car_control_modular.control_types import ControlAction, DistanceState, HazardState, ObstacleState, PersonTarget, SensorFrame
from car_control_modular.controllers import FollowPolicyConfig, FollowSafetyController


def main() -> int:
    cfg = FollowPolicyConfig(
        max_forward_percent=20,
        forward_speed_le_1_3_percent=12,
        forward_speed_le_1_7_percent=16,
        forward_speed_le_2_1_percent=20,
        forward_speed_le_2_6_percent=20,
        forward_speed_le_3_2_percent=20,
        forward_speed_le_3_8_percent=20,
        forward_speed_le_4_5_percent=20,
        forward_speed_far_percent=20,
    )
    controller = FollowSafetyController(cfg)
    person = PersonTarget((250, 120, 390, 430), track_id=1, confidence=0.9, area=43400)

    normal = SensorFrame(width=640, height=480, persons=[person], distance_m=2.0)
    d1 = controller.decide(1, normal)
    print("normal:", [(a.kind, a.speed_percent, a.reason) for a in d1.actions], d1.explicit_stop_requested, d1.reason)
    if not d1.actions:
        raise AssertionError("normal target should produce a control action")
    if max(a.speed_percent for a in d1.actions) > 20:
        raise AssertionError(f"speed cap exceeded: {d1.actions}")

    startup_cfg = FollowPolicyConfig(
        search_before_first_seen=False,
        initial_target_confirm_frames=2,
        max_forward_percent=20,
        forward_speed_far_percent=20,
    )
    startup_controller = FollowSafetyController(startup_cfg)
    d_start_empty = startup_controller.decide(1, SensorFrame(width=640, height=480, persons=[]))
    print("startup_wait:", d_start_empty.actions, d_start_empty.reason)
    if d_start_empty.actions or d_start_empty.explicit_stop_requested or d_start_empty.reason != "wait_first_person":
        raise AssertionError(f"startup with no person should idle in place, got {d_start_empty}")
    d_start_confirm_1 = startup_controller.decide(2, normal)
    print("startup_confirm_1:", d_start_confirm_1.actions, d_start_confirm_1.reason, startup_controller.active_target_id)
    if d_start_confirm_1.actions or startup_controller.active_target_id is not None:
        raise AssertionError(f"first visible uid should only enroll, got {d_start_confirm_1}")
    if d_start_confirm_1.reason != "initial_target_confirm_wait":
        raise AssertionError(f"first visible uid should explain enrollment wait, got {d_start_confirm_1.reason}")
    d_start_confirm_2 = startup_controller.decide(3, normal)
    print(
        "startup_confirm_2:",
        [(a.kind, a.speed_percent, a.reason) for a in d_start_confirm_2.actions],
        d_start_confirm_2.reason,
        startup_controller.active_target_id,
    )
    if startup_controller.active_target_id != 1 or not d_start_confirm_2.actions:
        raise AssertionError(f"stable uid should become active target, got {d_start_confirm_2}")
    startup_controller.clear_active_target("test_button")
    startup_other_person = PersonTarget((250, 120, 390, 430), track_id=2, confidence=0.9, area=43400)
    d_clear_confirm_1 = startup_controller.decide(
        4,
        SensorFrame(width=640, height=480, persons=[startup_other_person], distance_m=2.0),
    )
    if d_clear_confirm_1.actions or startup_controller.active_target_id is not None:
        raise AssertionError(f"manual clear should return to enrollment wait, got {d_clear_confirm_1}")
    d_clear_confirm_2 = startup_controller.decide(
        5,
        SensorFrame(width=640, height=480, persons=[startup_other_person], distance_m=2.0),
    )
    if startup_controller.active_target_id != 2 or not d_clear_confirm_2.actions:
        raise AssertionError(f"manual clear should allow a newly stable uid, got {d_clear_confirm_2}")

    edge_controller = FollowSafetyController(cfg)
    left_edge_person = PersonTarget((10, 120, 80, 430), track_id=1, confidence=0.9, area=24850)
    left_edge = SensorFrame(width=640, height=480, persons=[left_edge_person], distance_m=2.0)
    d_edge = edge_controller.decide(1, left_edge)
    print("left_edge:", [(a.kind, a.speed_percent, a.reason) for a in d_edge.actions], d_edge.reason)
    if not d_edge.actions or any(a.kind != "steer_left" for a in d_edge.actions):
        raise AssertionError(f"off-center visible target should use wheel-differential steer, got {d_edge.actions}")
    if d_edge.actions[0].steer_outer_ratio_percent <= d_edge.actions[0].steer_inner_ratio_percent:
        raise AssertionError(f"steer action should make the outer wheel faster, got {d_edge.actions[0]}")
    edge_controller.set_last_dispatched("rotate_left")
    d_edge_lost = edge_controller.decide(2, SensorFrame(width=640, height=480, persons=[], distance_m=2.0))
    print("edge_lost:", d_edge_lost.explicit_stop_requested, d_edge_lost.reason)
    if not d_edge_lost.waiting_lost_confirm:
        raise AssertionError(f"missing target after rotate should enter lost confirm, got {d_edge_lost.reason}")
    if d_edge_lost.explicit_stop_requested:
        raise AssertionError("missing target after a queued rotate should not cancel the rotate pulse")
    if d_edge_lost.reason != "lost_confirm_wait_keep_rotate":
        raise AssertionError(f"missing target after rotate should explain keep-rotate wait, got {d_edge_lost.reason}")

    hysteresis_cfg = FollowPolicyConfig(
        center_left_ratio=0.30,
        center_right_ratio=0.70,
        steer_enter_left_ratio=0.28,
        steer_enter_right_ratio=0.72,
        steer_release_left_ratio=0.35,
        steer_release_right_ratio=0.65,
        max_forward_percent=20,
        forward_speed_far_percent=20,
    )
    hysteresis_controller = FollowSafetyController(hysteresis_cfg)
    boundary_person = PersonTarget((680, 120, 740, 430), track_id=1, confidence=0.9, area=18600)
    boundary_frame = SensorFrame(width=1000, height=480, persons=[boundary_person], distance_m=2.0)
    d_enter = hysteresis_controller.decide(1, boundary_frame)
    print("steer_hysteresis_enter:", [(a.kind, a.reason) for a in d_enter.actions], d_enter.reason)
    if not d_enter.actions or any(a.kind != "forward" for a in d_enter.actions):
        raise AssertionError(f"target inside enter band should keep forward, got {d_enter.actions}")
    hysteresis_controller.set_last_dispatched("steer_right")
    d_release = hysteresis_controller.decide(2, boundary_frame)
    print("steer_hysteresis_release:", [(a.kind, a.reason) for a in d_release.actions], d_release.reason)
    if not d_release.actions or any(a.kind != "steer_right" for a in d_release.actions):
        raise AssertionError(f"active steer should continue until release band, got {d_release.actions}")

    visible_rotate_cfg = FollowPolicyConfig(
        center_left_ratio=0.30,
        center_right_ratio=0.70,
        steer_enter_left_ratio=0.28,
        steer_enter_right_ratio=0.72,
        visible_rotate_left_ratio=0.25,
        visible_rotate_right_ratio=0.75,
        max_forward_percent=20,
        forward_speed_far_percent=20,
    )
    visible_rotate_controller = FollowSafetyController(visible_rotate_cfg)
    steer_zone_person = PersonTarget((710, 120, 750, 430), track_id=1, confidence=0.9, area=12400)
    d_steer_zone = visible_rotate_controller.decide(
        1,
        SensorFrame(width=1000, height=480, persons=[steer_zone_person], distance_m=2.0),
    )
    print("visible_steer_zone:", [(a.kind, a.reason) for a in d_steer_zone.actions], d_steer_zone.reason)
    if not d_steer_zone.actions or d_steer_zone.actions[0].kind != "steer_right":
        raise AssertionError(f"target between steer and rotate boundary should steer, got {d_steer_zone.actions}")
    visible_rotate_controller.set_last_dispatched("forward")
    rotate_zone_person = PersonTarget((750, 120, 790, 430), track_id=1, confidence=0.9, area=12400)
    d_rotate_zone = visible_rotate_controller.decide(
        2,
        SensorFrame(width=1000, height=480, persons=[rotate_zone_person], distance_m=2.0),
    )
    print("visible_rotate_zone:", [(a.kind, a.reason) for a in d_rotate_zone.actions], d_rotate_zone.reason)
    if not d_rotate_zone.actions or d_rotate_zone.actions[0].kind != "rotate_right":
        raise AssertionError(f"target beyond visible rotate boundary should rotate, got {d_rotate_zone.actions}")
    if d_rotate_zone.reason != "person_right_rotate":
        raise AssertionError(f"visible rotate should have a distinct reason, got {d_rotate_zone.reason}")

    blocked_search_cfg = FollowPolicyConfig(
        brake_distance_m=0.8,
        search_before_first_seen=True,
        side_ir_blocks_rotation=False,
        search_rotate_front_block_enable=False,
        search_rotate_distance_block_enable=True,
    )
    blocked_search_controller = FollowSafetyController(blocked_search_cfg)
    near_empty = SensorFrame(
        width=640,
        height=480,
        persons=[],
        distance_m=0.72,
        distance_state=DistanceState(used_distance_m=0.72, brake_latched=True),
    )
    d_blocked = blocked_search_controller.decide(1, near_empty)
    print("search_rotate_blocked:", d_blocked.explicit_stop_requested, d_blocked.reason)
    if not d_blocked.explicit_stop_requested or d_blocked.actions:
        raise AssertionError(f"near obstacle should block free-search rotate, got {d_blocked}")
    if d_blocked.reason != "search_both_sides_blocked":
        raise AssertionError(f"blocked free-search should explain both sides blocked, got {d_blocked.reason}")

    front_empty = SensorFrame(width=640, height=480, persons=[], obstacles=ObstacleState(front=True))
    d_front_search = blocked_search_controller.decide(2, front_empty)
    print("front_ir_stops_search:", d_front_search.explicit_stop_requested, d_front_search.reason)
    if not d_front_search.explicit_stop_requested or d_front_search.actions or d_front_search.reason != "front_ir":
        raise AssertionError(f"front IR should stop free-search motion, got {d_front_search}")
    if not d_front_search.clear_action_queue or not d_front_search.stop_action_execution:
        raise AssertionError(f"IR stop should clear queued and current actions, got {d_front_search}")

    side_blocked_search_cfg = FollowPolicyConfig(
        search_before_first_seen=True,
        side_ir_blocks_rotation=True,
        search_rotate_front_block_enable=False,
        search_rotate_distance_block_enable=False,
    )
    right_blocked_search_controller = FollowSafetyController(side_blocked_search_cfg)
    d_right_blocked_search = right_blocked_search_controller.decide(
        1,
        SensorFrame(width=640, height=480, persons=[], obstacles=ObstacleState(right=True)),
    )
    print("right_ir_blocks_search_right:", d_right_blocked_search.explicit_stop_requested, d_right_blocked_search.reason)
    if (
        not d_right_blocked_search.explicit_stop_requested
        or d_right_blocked_search.actions
        or d_right_blocked_search.reason != "right_ir"
    ):
        raise AssertionError(f"right IR should unconditionally stop search motion, got {d_right_blocked_search}")

    left_blocked_search_controller = FollowSafetyController(side_blocked_search_cfg)
    left_blocked_search_controller.search_state = "searching"
    left_blocked_search_controller.search_direction = "left"
    d_left_blocked_search = left_blocked_search_controller.decide(
        1,
        SensorFrame(width=640, height=480, persons=[], obstacles=ObstacleState(left=True)),
    )
    print("left_ir_blocks_search_left:", d_left_blocked_search.explicit_stop_requested, d_left_blocked_search.reason)
    if (
        not d_left_blocked_search.explicit_stop_requested
        or d_left_blocked_search.actions
        or d_left_blocked_search.reason != "left_ir"
    ):
        raise AssertionError(f"left IR should unconditionally stop search motion, got {d_left_blocked_search}")

    center_person = PersonTarget((430, 120, 570, 430), track_id=1, confidence=0.9, area=43400)
    blocked_clear_cfg = FollowPolicyConfig(
        lost_confirm_frames=1,
        lost_confirm_sec=0.0,
        search_before_first_seen=False,
        side_ir_blocks_rotation=True,
        search_rotate_front_block_enable=False,
        search_rotate_distance_block_enable=False,
        blocked_turn_forward_enable=True,
        blocked_turn_forward_max_steps=2,
        distance_missing_forward_percent=35,
        min_forward_percent=10,
        max_forward_percent=60,
        forward_speed_far_percent=20,
    )
    blocked_clear_controller = FollowSafetyController(blocked_clear_cfg)
    blocked_clear_controller.decide(1, SensorFrame(width=1000, height=480, persons=[center_person], distance_m=2.0))
    d_clear_step_1 = blocked_clear_controller.decide(
        2,
        SensorFrame(width=1000, height=480, persons=[], obstacles=ObstacleState(right=True)),
    )
    print("right_ir_disables_clear_forward:", d_clear_step_1.explicit_stop_requested, d_clear_step_1.reason)
    if not d_clear_step_1.explicit_stop_requested or d_clear_step_1.actions or d_clear_step_1.reason != "right_ir":
        raise AssertionError(f"right IR must not generate a forward clear step, got {d_clear_step_1}")

    ir_policy_cfg = FollowPolicyConfig(
        center_left_ratio=0.30,
        center_right_ratio=0.70,
        visible_rotate_left_ratio=0.25,
        visible_rotate_right_ratio=0.75,
        distance_missing_forward_percent=35,
        min_forward_percent=10,
        max_forward_percent=60,
        forward_speed_le_2_1_percent=55,
        search_rotate_front_block_enable=False,
    )
    ir_center_controller = FollowSafetyController(ir_policy_cfg)
    d_front_center = ir_center_controller.decide(
        1,
        SensorFrame(width=1000, height=480, persons=[center_person], distance_m=2.0, obstacles=ObstacleState(front=True)),
    )
    print("front_ir_center:", d_front_center.explicit_stop_requested, d_front_center.reason)
    if not d_front_center.explicit_stop_requested or d_front_center.reason != "front_ir":
        raise AssertionError(f"front IR should block centered forward, got {d_front_center}")

    front_steer_person = PersonTarget((710, 120, 750, 430), track_id=1, confidence=0.9, area=12400)
    d_front_steer = ir_center_controller.decide(
        2,
        SensorFrame(width=1000, height=480, persons=[front_steer_person], distance_m=2.0, obstacles=ObstacleState(front=True)),
    )
    print("front_ir_steer_zone:", [(a.kind, a.reason) for a in d_front_steer.actions], d_front_steer.explicit_stop_requested, d_front_steer.reason)
    if not d_front_steer.explicit_stop_requested or d_front_steer.reason != "front_ir" or d_front_steer.actions:
        raise AssertionError(f"front IR should block steer without creating rotate, got {d_front_steer}")

    front_rotate_person = PersonTarget((910, 120, 960, 430), track_id=1, confidence=0.9, area=15500)
    d_front_rotate = ir_center_controller.decide(
        3,
        SensorFrame(width=1000, height=480, persons=[front_rotate_person], distance_m=2.0, obstacles=ObstacleState(front=True)),
    )
    print("front_ir_stops_visible_rotate:", d_front_rotate.explicit_stop_requested, d_front_rotate.reason)
    if not d_front_rotate.explicit_stop_requested or d_front_rotate.actions or d_front_rotate.reason != "front_ir":
        raise AssertionError(f"front IR should stop visible rotate_right, got {d_front_rotate}")

    right_steer_person = PersonTarget((710, 120, 750, 430), track_id=1, confidence=0.9, area=12400)
    ir_does_not_rewrite_controller = FollowSafetyController(ir_policy_cfg)
    d_side_ir_keeps_steer = ir_does_not_rewrite_controller.decide(
        1,
        SensorFrame(width=1000, height=480, persons=[right_steer_person], distance_m=2.0, obstacles=ObstacleState(left=True)),
    )
    print("left_ir_stops_opposite_steer:", d_side_ir_keeps_steer.explicit_stop_requested, d_side_ir_keeps_steer.reason)
    if not d_side_ir_keeps_steer.explicit_stop_requested or d_side_ir_keeps_steer.actions or d_side_ir_keeps_steer.reason != "left_ir":
        raise AssertionError(f"left IR should stop even an opposite-direction steer, got {d_side_ir_keeps_steer}")

    left_person = PersonTarget((40, 120, 90, 430), track_id=1, confidence=0.9, area=15500)
    right_person = PersonTarget((910, 120, 960, 430), track_id=1, confidence=0.9, area=15500)
    ir_left_block_controller = FollowSafetyController(ir_policy_cfg)
    d_left_block = ir_left_block_controller.decide(
        1,
        SensorFrame(width=1000, height=480, persons=[left_person], distance_m=2.0, obstacles=ObstacleState(left=True)),
    )
    print("left_ir_blocks_left_rotate:", d_left_block.explicit_stop_requested, d_left_block.reason)
    if not d_left_block.explicit_stop_requested or d_left_block.reason != "left_ir":
        raise AssertionError(f"left IR should block rotate_left toward the left side, got {d_left_block}")
    if not ir_left_block_controller._is_action_blocked(  # noqa: SLF001 - direct policy gate regression
        ControlAction.steer_left(35, 100, 130, "test_left_steer"),
        SensorFrame(width=1000, height=480, obstacles=ObstacleState(left=True)),
    ):
        raise AssertionError("left IR should block steer_left at the action gate")
    if not ir_left_block_controller._is_action_blocked(  # noqa: SLF001 - direct policy gate regression
        ControlAction.steer_right(35, 100, 130, "test_right_steer"),
        SensorFrame(width=1000, height=480, obstacles=ObstacleState(right=True)),
    ):
        raise AssertionError("right IR should block steer_right at the action gate")

    ir_opposite_controller = FollowSafetyController(ir_policy_cfg)
    d_left_allows_right = ir_opposite_controller.decide(
        1,
        SensorFrame(width=1000, height=480, persons=[right_person], distance_m=2.0, obstacles=ObstacleState(left=True)),
    )
    print("left_ir_stops_right_rotate:", d_left_allows_right.explicit_stop_requested, d_left_allows_right.reason)
    if not d_left_allows_right.explicit_stop_requested or d_left_allows_right.actions or d_left_allows_right.reason != "left_ir":
        raise AssertionError(f"left IR should stop even an opposite-direction rotate, got {d_left_allows_right}")

    ir_escape_controller = FollowSafetyController(ir_policy_cfg)
    d_side_escape = ir_escape_controller.decide(
        1,
        SensorFrame(
            width=1000,
            height=480,
            persons=[center_person],
            distance_m=2.0,
            obstacles=ObstacleState(left=True, right=True),
        ),
    )
    print("side_ir_center_stop:", d_side_escape.explicit_stop_requested, d_side_escape.reason)
    if not d_side_escape.explicit_stop_requested or d_side_escape.actions or d_side_escape.reason != "left_ir":
        raise AssertionError(f"centered target with side IR should stop, got {d_side_escape}")

    other_person = PersonTarget((250, 120, 390, 430), track_id=2, confidence=0.9, area=43400)
    switched = SensorFrame(width=640, height=480, persons=[other_person], distance_m=2.0)
    d_switch = controller.decide(2, switched)
    print(
        "switch_guard:",
        [(a.kind, a.speed_percent, a.reason) for a in d_switch.actions],
        d_switch.explicit_stop_requested,
        d_switch.reason,
    )
    if d_switch.actions:
        raise AssertionError("controller should not follow a different single track after locking a target")
    if not d_switch.waiting_lost_confirm:
        raise AssertionError(f"different single track should be treated as lost target, got {d_switch.reason}")
    if not d_switch.explicit_stop_requested:
        raise AssertionError("first lost-confirm frame should stop the current motion")
    if d_switch.reason != "lost_confirm_wait_stop":
        raise AssertionError(f"first lost-confirm frame should explain the stop, got {d_switch.reason}")

    predictive_cfg = FollowPolicyConfig(
        lost_confirm_sec=3.0,
        predictive_search_enabled=True,
        predictive_search_sec=1.0,
        max_forward_percent=20,
        forward_speed_far_percent=20,
    )
    predictive_controller = FollowSafetyController(predictive_cfg)
    predictive_controller.decide(1, normal)
    d_predict_left = predictive_controller.decide(
        2,
        SensorFrame(
            width=640,
            height=480,
            persons=[],
            distance_m=2.0,
            lost_intent="left_exit",
            lost_intent_age_sec=0.1,
        ),
    )
    print("lost_wait_predict:", [(a.kind, a.reason) for a in d_predict_left.actions], d_predict_left.reason)
    if not d_predict_left.waiting_lost_confirm:
        raise AssertionError(f"predictive lost wait should still be in lost-confirm window, got {d_predict_left.reason}")
    if d_predict_left.explicit_stop_requested:
        raise AssertionError("predictive lost wait should not stop when the predicted action is safe")
    if predictive_controller.active_target_id != 1:
        raise AssertionError(f"predictive lost wait should keep the locked target, got {predictive_controller.active_target_id}")
    if not d_predict_left.actions or d_predict_left.actions[0].kind != "rotate_left":
        raise AssertionError(f"left-exit predictive wait should rotate left, got {d_predict_left.actions}")
    if d_predict_left.reason != "lost_wait_predict_left_exit":
        raise AssertionError(f"predictive wait should explain the predicted action, got {d_predict_left.reason}")

    reacquire_cfg = FollowPolicyConfig(
        lost_confirm_frames=2,
        release_target_on_lost=True,
        max_forward_percent=20,
        forward_speed_far_percent=20,
    )
    reacquire_controller = FollowSafetyController(reacquire_cfg)
    reacquire_controller.decide(1, normal)
    d_wait = reacquire_controller.decide(2, switched)
    if not d_wait.waiting_lost_confirm:
        raise AssertionError(f"first missing locked target frame should wait, got {d_wait.reason}")
    if not d_wait.explicit_stop_requested:
        raise AssertionError("first missing locked target frame should explicitly stop")
    d_free = reacquire_controller.decide(3, switched)
    print("free_search:", [(a.kind, a.speed_percent, a.reason) for a in d_free.actions], d_free.reason)
    if reacquire_controller.active_target_id != 2:
        raise AssertionError(f"free search should lock the new visible target, got {reacquire_controller.active_target_id}")
    if not d_free.actions:
        raise AssertionError("free search should follow the new visible target immediately")

    locked_search_cfg = FollowPolicyConfig(
        lost_confirm_frames=2,
        release_target_on_lost=False,
        max_forward_percent=20,
        forward_speed_far_percent=20,
    )
    locked_search_controller = FollowSafetyController(locked_search_cfg)
    locked_search_controller.decide(1, normal)
    d_locked_wait = locked_search_controller.decide(2, switched)
    if not d_locked_wait.waiting_lost_confirm:
        raise AssertionError(f"locked search should wait before confirmed lost, got {d_locked_wait.reason}")
    d_locked_search = locked_search_controller.decide(3, switched)
    print(
        "locked_search_keeps_target:",
        [(a.kind, a.speed_percent, a.reason) for a in d_locked_search.actions],
        d_locked_search.reason,
        locked_search_controller.active_target_id,
    )
    if locked_search_controller.active_target_id != 1:
        raise AssertionError(
            f"locked search should keep looking for the original target, got {locked_search_controller.active_target_id}"
        )
    if d_locked_search.actions and any(a.kind == "forward" for a in d_locked_search.actions):
        raise AssertionError(f"locked search should not follow the wrong visible person, got {d_locked_search.actions}")
    locked_search_controller.clear_active_target("test_button")
    d_after_clear = locked_search_controller.decide(4, switched)
    if locked_search_controller.active_target_id != 2:
        raise AssertionError(f"manual clear should allow the visible person to become target, got {locked_search_controller.active_target_id}")
    if not d_after_clear.actions:
        raise AssertionError("manual clear should allow normal follow decisions again")

    timed_cfg = FollowPolicyConfig(
        lost_confirm_sec=3.0,
        max_forward_percent=20,
        forward_speed_far_percent=20,
    )
    timed_controller = FollowSafetyController(timed_cfg)
    timed_controller.decide(1, normal)
    d_timed_wait = timed_controller.decide(2, switched)
    if not d_timed_wait.waiting_lost_confirm:
        raise AssertionError(f"time-based lost-confirm should wait, got {d_timed_wait.reason}")
    if not d_timed_wait.explicit_stop_requested:
        raise AssertionError("time-based lost-confirm should stop immediately on the first lost frame")

    large_unassigned = PersonTarget((0, 0, 260, 430), track_id=-2, confidence=0.9, area=111800)
    mixed = SensorFrame(width=640, height=480, persons=[large_unassigned, person], distance_m=2.0)
    selected = controller.select_target_for_current_state(mixed.persons)
    if selected is None or selected.track_id != 1:
        raise AssertionError(f"locked controller should ignore larger fallback target, got {selected}")
    d_locked = controller.decide(3, mixed)
    print("locked_target:", [(a.kind, a.speed_percent, a.reason) for a in d_locked.actions], d_locked.reason)
    if controller.last_selected_target is None or controller.last_selected_target.track_id != 1:
        raise AssertionError(f"decision should keep the locked target, got {controller.last_selected_target}")
    if not d_locked.actions or any(a.kind != "forward" for a in d_locked.actions):
        raise AssertionError(f"locked target should drive from the matched person, got {d_locked.actions}")

    missing_distance_controller = FollowSafetyController(
        FollowPolicyConfig(distance_missing_forward_percent=35, initial_target_confirm_frames=1)
    )
    missing_distance = SensorFrame(width=640, height=480, persons=[person], distance_m=None)
    d_missing = missing_distance_controller.decide(1, missing_distance)
    if not d_missing.explicit_stop_requested or d_missing.actions or d_missing.reason != "distance_missing_stop":
        raise AssertionError(f"missing distance must stop even with a nonzero legacy fallback, got {d_missing}")

    timeout_controller = FollowSafetyController(
        FollowPolicyConfig(
            lost_confirm_frames=1,
            search_timeout_sec=4.0,
            release_target_on_lost=False,
        )
    )
    timeout_controller.decide(1, normal)
    timeout_controller._search_rotation_started_at = time.monotonic() - 4.1
    d_timeout = timeout_controller.decide(2, SensorFrame(width=640, height=480, persons=[]))
    if not d_timeout.explicit_stop_requested or d_timeout.reason != "search_timeout_stop":
        raise AssertionError(f"search timeout must stop and clear rotation, got {d_timeout}")
    if not d_timeout.clear_action_queue or not d_timeout.stop_action_execution:
        raise AssertionError(f"first search timeout must clear queued/current motion, got {d_timeout}")

    hazard = SensorFrame(
        width=640,
        height=480,
        persons=[person],
        distance_m=2.0,
        hazard=HazardState(active=True, reason="bunker"),
    )
    d2 = controller.decide(4, hazard)
    print("hazard:", [(a.kind, a.speed_percent, a.reason) for a in d2.actions], d2.explicit_stop_requested, d2.reason)
    if not d2.explicit_stop_requested:
        raise AssertionError("hazard should request an explicit stop")
    if any(a.kind != "stop" for a in d2.actions):
        raise AssertionError(f"hazard should only emit stop actions: {d2.actions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
