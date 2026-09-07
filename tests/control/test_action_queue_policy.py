import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from car_control_modular.action_queue_policy import (
    merge_pending_actions,
    should_drop_queued_action,
)


STOP = 3
ROTATE_LEFT = 1
ROTATE_RIGHT = 2
FORWARD = 0
ROTATE_ACTIONS = frozenset({ROTATE_LEFT, ROTATE_RIGHT})


def merge(pending, incoming, current=None):
    return merge_pending_actions(
        pending,
        incoming,
        stop_action=STOP,
        rotate_actions=ROTATE_ACTIONS,
        current_action=current,
    )


def test_same_direction_keeps_only_newest_pending_motion():
    result = merge([ROTATE_LEFT], [ROTATE_LEFT])
    assert result.actions == (ROTATE_LEFT,)
    assert result.replaced_count == 1
    assert not result.inserted_stop_barrier


def test_new_stop_supersedes_waiting_motion_and_is_never_dropped():
    result = merge([ROTATE_LEFT, FORWARD], [STOP])
    assert result.actions == (STOP,)
    assert result.replaced_count == 2


def test_explicit_stop_then_follow_up_preserves_follow_up_order():
    result = merge([ROTATE_RIGHT], [STOP, ROTATE_LEFT, STOP])
    assert result.actions == (STOP, ROTATE_LEFT)


def test_motion_after_queued_stop_stays_behind_stop_barrier():
    result = merge([STOP, ROTATE_RIGHT], [ROTATE_LEFT])
    assert result.actions == (STOP, ROTATE_LEFT)
    assert result.preserved_stop_barrier


def test_active_rotation_reversal_inserts_stop_before_new_direction():
    result = merge([], [ROTATE_LEFT], current=ROTATE_RIGHT)
    assert result.actions == (STOP, ROTATE_LEFT)
    assert result.inserted_stop_barrier


def test_repeated_reversal_does_not_accumulate_multiple_stops():
    first = merge([], [ROTATE_LEFT], current=ROTATE_RIGHT)
    second = merge(first.actions, [ROTATE_LEFT], current=ROTATE_RIGHT)
    assert second.actions == (STOP, ROTATE_LEFT)
    assert second.actions.count(STOP) == 1


def test_non_reversing_action_replaces_pending_motion():
    result = merge([ROTATE_RIGHT], [FORWARD], current=ROTATE_RIGHT)
    assert result.actions == (FORWARD,)
    assert not result.inserted_stop_barrier


def test_empty_update_does_not_destroy_pending_safety_action():
    result = merge([STOP, ROTATE_RIGHT], [])
    assert result.actions == (STOP, ROTATE_RIGHT)
    assert result.replaced_count == 0


def test_expired_motion_is_dropped_but_stop_is_never_expired():
    assert should_drop_queued_action(
        ROTATE_LEFT,
        0.46,
        ttl_sec=0.45,
        stop_action=STOP,
    )
    assert not should_drop_queued_action(
        STOP,
        10.0,
        ttl_sec=0.45,
        stop_action=STOP,
    )
    assert not should_drop_queued_action(
        ROTATE_LEFT,
        0.20,
        ttl_sec=0.45,
        stop_action=STOP,
    )
