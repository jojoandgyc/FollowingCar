"""Safety-aware merging for the motor action queue.

The motor executor owns the command currently in flight.  This module only
decides how a newly requested action should replace actions that are still
waiting in the queue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple


@dataclass(frozen=True)
class ActionQueueMerge:
    """Result of merging pending actions with a newer controller request."""

    actions: Tuple[int, ...]
    replaced_count: int
    inserted_stop_barrier: bool = False
    preserved_stop_barrier: bool = False


def should_drop_queued_action(
    action: int,
    age_sec: float,
    *,
    ttl_sec: float,
    stop_action: int,
) -> bool:
    """Return whether a waiting non-safety action has become obsolete."""

    if int(action) == int(stop_action) or float(age_sec) < 0.0:
        return False
    return float(age_sec) >= max(0.20, float(ttl_sec))


def merge_pending_actions(
    pending: Iterable[int],
    incoming: Sequence[int],
    *,
    stop_action: int,
    rotate_actions: frozenset[int],
    current_action: int | None = None,
) -> ActionQueueMerge:
    """Merge a fresh request without dropping a safety transition.

    Only actions which have not started sending are considered ``pending``.
    A queued STOP is a barrier: newer normal actions are placed after it.  A
    fresh request that reverses an active rotation gets a STOP barrier before
    the new direction.  For all other requests, pending actions are replaced
    by the newest requested state.
    """

    old = tuple(int(action) for action in pending)
    new = tuple(int(action) for action in incoming)
    if not new:
        return ActionQueueMerge(old, 0)

    # A safety STOP supersedes all waiting motion, but remains in the queue.
    if new[0] == stop_action:
        # Keep an explicitly requested follow-up action (for example
        # ``[STOP, rotate_left]``), while removing any duplicate STOPs.
        follow_up = tuple(action for action in new[1:] if action != stop_action)
        return ActionQueueMerge(
            (stop_action,) + follow_up,
            len(old),
            preserved_stop_barrier=stop_action in old,
        )

    # Keep an already queued STOP as a hard barrier.  Replace only the motion
    # suffix so repeated updates cannot accumulate stale rotations.
    if stop_action in old:
        merged = (stop_action,) + new
        return ActionQueueMerge(
            merged,
            len(old) - 1,
            preserved_stop_barrier=True,
        )

    # Direction reversal must stop before the opposite rotation.  The active
    # command is not removed here; the action thread will finish its current
    # write, then consume this barrier.
    if (
        current_action in rotate_actions
        and new[0] in rotate_actions
        and current_action != new[0]
    ):
        return ActionQueueMerge(
            (stop_action,) + new,
            len(old),
            inserted_stop_barrier=True,
        )

    return ActionQueueMerge(new, len(old))
