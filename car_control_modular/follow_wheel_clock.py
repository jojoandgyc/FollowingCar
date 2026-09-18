"""A non-blocking normal-follow clock. No stored motor actions or TTL renewal."""
from __future__ import annotations


class FollowWheelClock:
    def __init__(self, period_sec=0.05):
        self.period = max(0.05, float(period_sec))
        self.reset()

    def reset(self):
        self.next_due = None
        self.last_axes = None
        self.last_sent = None

    def reason(self, now, axes):
        # axes: UID, revision, signed forward base, signed right yaw.
        previous = self.last_axes
        if previous is not None:
            if axes[0] != previous[0]:
                return "uid_changed"
            if abs(axes[2]) < abs(previous[2]):
                return "base_reduced"
            if axes[3] == 0 and previous[3] != 0:
                return "yaw_revoked"
        if self.next_due is None or now + 1e-9 >= self.next_due:
            return "periodic"
        return None

    def sent(self, now, axes):
        if self.next_due is None:
            self.next_due = now + self.period
        elif now + 1e-9 >= self.next_due:
            # Skip missed periods; never replay a backlog after a stall.
            skipped = max(1, int((now - self.next_due + 1e-9) / self.period) + 1)
            self.next_due += skipped * self.period
        self.last_axes, self.last_sent = axes, now
