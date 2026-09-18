"""Evidence counter for ordinary distance parking, never a motor command."""
from dataclasses import dataclass
import math


def is_follow_distance_hold(owner):
    hold = getattr(owner, "_follow_distance_hold", None)
    controller = getattr(owner, "_follow_controller", None)
    return bool(
        isinstance(hold, FollowDistanceHold)
        and getattr(owner, "_brake_hold_active", False)
        and getattr(owner, "_brake_hold_label", None) == "follow_distance_hold"
        and not getattr(owner, "_brake_hold_stop_mode", None)
        and hold.uid == getattr(controller, "active_target_id", None)
        and getattr(owner, "search_state", None) == "none"
        and getattr(controller, "search_state", "none") == "none"
        and getattr(owner, "_vision_control_state", "") in {
            "target_visible", "target_visible_depth_valid", "target_visible_depth_missing"}
        and getattr(owner, "running", False)
        and not any(getattr(owner, key, False) for key in (
            "_explicit_stop_requested", "_runtime_shutdown_requested", "_reacquire_depth_pending"))
    )


@dataclass
class FollowDistanceHold:
    uid: int
    started: float
    last_stamp: float | None = None
    last_distance: float | None = None
    count: int = 0

    def reject(self):
        self.last_stamp = self.last_distance = None
        self.count = 0

    def observe(self, *, stamp, raw, used, now, target, brake, qualified):
        values = (stamp, raw, used, now, target, brake)
        if (not qualified or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                 or not math.isfinite(v) for v in values)
                or stamp <= self.started or not 0 <= now-stamp <= .18
                or min(raw, used) <= max(target+.03, brake+.1)
                or abs(raw-used) > .25):
            self.reject()
            return "rejected"
        if self.last_stamp is not None and stamp <= self.last_stamp:
            return "duplicate"
        continuous = (self.last_stamp is not None and stamp-self.last_stamp <= .18
                      and abs(raw-self.last_distance) <= .05+2.5*(stamp-self.last_stamp))
        self.count = self.count+1 if continuous else 1
        self.last_stamp, self.last_distance = stamp, raw
        return "ready" if self.count >= 2 else "wait"
