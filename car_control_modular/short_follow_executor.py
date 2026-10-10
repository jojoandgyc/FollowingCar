"""One periodic writer for paired normal following, including distance PI.

This is deliberately not another reader of the legacy forward/yaw grants.
The mailbox contains the complete immutable wheel pair, with acquisition-based
deadlines. The action worker services its watchdog even when perception stalls.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, replace

from .detector_identity_lease import read_visual_identity_evidence
from .wheel_zero_cross import wheel_feedback_valid
from .short_follow_yaw import ShortFollowYawResponse


@dataclass(frozen=True)
class _TurnResponse:
    """An executed turn's finite response interval, NEVER motion authority."""
    pair: tuple
    changed_at: float
    expires_at: float
    reverse_seen: bool = False


class ShortFollowExecutor:
    def __init__(self, runtime):
        self.runtime = runtime
        self.owner = runtime.owner
        self.backend = runtime.backend
        self._lock = threading.RLock()
        self._owned = False
        self._controller = None
        self._last_write_at = float("-inf")
        self._next_write_due = None
        self._last_stop_at = float("-inf")
        self._stop_key = None
        self._generation = self.backend.stop_write_generation
        self._entry_stop_at = None
        self._entry_sample = None
        self._entry_quiet_count = 0
        self._entry_stop_acknowledged = False
        self._entry_tail_receipt = None
        self._reverse_pending = None
        self._identity_check = None
        self._live_feedback_reason = None
        # Wheel-response evidence, never identity/observation authority. Only
        # completed small pivot ACKs can explain a negative inner-wheel tail.
        self._motion_receipt = None
        self._motion_pair = None
        self._motion_generation = self.backend.stop_write_generation
        self._turn_responses = ()
        self._park_tail_until = float("-inf")
        self._park_tail_reverse_seen = False
        self._park_tail_entry_stop_at = None
        # Same visual observation may taper an executed turn, but falling
        # rate feedback must not repeatedly restore that old turn request.
        # This is an output bound, not an observation/permission deadline.
        self._yaw_limit_key = None
        self._yaw_delta_limit = None
        self._yaw_response = ShortFollowYawResponse()

    def controller(self):
        controller = getattr(self.owner, "_short_follow", None)
        if controller is None or not controller.config.enabled:
            return None
        return controller

    def active(self):
        controller = self.controller()
        return controller is not None and controller.snapshot().active

    def blocks_legacy(self):
        # A deactivation must cross the physical STOP barrier before legacy
        # search/reverse can write. Merely clearing the mailbox is not enough.
        return self._owned or self.active()

    def _hard_reason(self):
        owner = self.owner
        if (getattr(owner, "_runtime_shutdown_requested", False)
                or not getattr(owner, "running", False)):
            return "shutdown"
        if (getattr(self.backend, "motion_write_fault", None)
                or getattr(self.backend, "parking_release_fault", None)):
            return "motor_fault"
        if self.explicit_stop_pending():
            return "explicit_stop"
        try:
            if self.runtime.hard_stop_check(self.runtime.symbols.forward):
                return "hard_stop"
        except Exception:
            self.runtime.logger.exception("short_follow safety check failed")
            return "hard_stop_check_failed"
        return None

    def explicit_stop_pending(self):
        reason = str(getattr(self.owner, "_last_explicit_stop_reason", "") or "explicit_stop")
        return bool(getattr(self.owner, "_explicit_stop_requested", False)
            and getattr(self.owner, "_explicit_stop_provenance", None) != ("controller_state", reason))

    def _consume_protected_stop(self, controller, now):
        command = self.runtime._take_short_follow_protected_stop()
        if command is None:
            return
        self.owner._explicit_stop_requested = True
        self.owner._last_explicit_stop_reason = command.reason or "protected_queue_stop"
        self.owner._explicit_stop_provenance = (getattr(command, "stop_origin", "unknown"), command.reason)
        controller.revoke(self.owner._last_explicit_stop_reason, now)

    def _checked_plan(self, controller):
        # The safety read may block and a visual publication may complete
        # meanwhile. Snapshot evidence FIRST, then time; never compare a new
        # proof's validated_at against time captured before the safety read.
        hard_reason = self._hard_reason()
        snapshot, now, reason = self._current_plan_check(controller)
        return snapshot, now, hard_reason or reason

    def _current_plan_check(self, controller):
        snapshot = controller.snapshot()
        evidence, now = read_visual_identity_evidence(self.owner)
        # Feedback settlement and new faults are observed even without a
        # plan. A later depth result must not erase an intervening reversal.
        entry_ready = self._entry_ready(now)
        self._live_feedback_reason = self._feedback_reason(now)
        reason = self._live_feedback_reason or self._plan_reason(snapshot, now, evidence)
        if reason is None and not entry_ready:
            reason = "entry_settling"
        if reason is None and not snapshot.plan.moving:
            reason = snapshot.plan.reason
        return snapshot, now, reason

    def _plan_reason(self, snapshot, now, evidence):
        if (getattr(self.runtime, "_search_reacquire_brake_request", None) is not None
                or (getattr(self.owner, "_brake_hold_active", False)
                    and getattr(self.owner, "_brake_hold_label", "") == "search_reacquire_brake")):
            return "existing_search_hold"
        if getattr(self.owner, "_brake_hold_active", False):
            return "external_brake_hold"
        plan = snapshot.plan
        if not snapshot.active or plan is None:
            return snapshot.reason or "waiting_observation"
        if plan.epoch != snapshot.epoch or plan.uid != snapshot.uid:
            return "epoch_changed"
        ctl = getattr(self.owner, "_follow_controller", None)
        if (getattr(ctl, "active_target_id", None) != plan.uid
                or getattr(ctl, "search_state", "none") != "none"
                or getattr(self.owner, "search_state", "none") != "none"):
            return "identity_or_search_changed"
        if not plan.valid(now):
            return "observation_expired"
        proof = evidence.observation
        self._identity_check = dict(
            checked_at=now, plan_cap=plan.capture_id, uid=plan.uid,
            depth_timestamp=plan.depth_timestamp, plan_expires_at=plan.expires_at,
            proof_cap=getattr(proof, "capture", None),
            proof_uid=getattr(proof, "uid", None),
            proof_validated_at=getattr(proof, "validated_at", None),
            proof_expires_at=getattr(proof, "expires_at", None),
            proof_kind=getattr(proof, "kind", None),
            proof_rejected=proof is False,
            lease_expires_at=getattr(evidence.lease, "expires_at", None),
            lease_rejected=evidence.lease is False,
            identity_live=evidence.live(plan.uid, now),
            sample_permitted=evidence.permits_depth(plan.uid, plan.depth_timestamp, now),
        )
        if not evidence.permits_depth(plan.uid, plan.depth_timestamp, now):
            return "identity_not_live"
        cfg = self.controller().config
        pair = plan.left_rpm, plan.right_rpm
        if any(type(v) is not int or abs(v) > cfg.max_rpm for v in pair):
            return "invalid_wheel_pair"
        if min(pair) < 0 and not (plan.pivot and plan.base_rpm == 0
                and max(map(abs, pair)) <= cfg.pivot_limit_rpm
                and plan.distance_m > cfg.braking_stop_distance_m):
            return "invalid_wheel_pair"
        return None

    def _feedback(self, now):
        sample = self.runtime.get_steering_feedback()
        return sample if wheel_feedback_valid(sample, max(now, time.monotonic())) else None

    def _receipt_pair(self, receipt):
        b = self.backend
        return (receipt.left_rpm * b.wheel_raw_state_to_target("left", 1, 0x01),
                receipt.right_rpm * b.wheel_raw_state_to_target("right", 1, 0x01))

    def _small_pivot(self, pair):
        # Legacy search uses up to 8 RPM per wheel too. Lower hardware/config
        # limits remain authoritative; an arbitrary reverse ACK is not a turn.
        controller = self.controller() or self._controller
        if controller is None:
            return False
        limit = min(8, controller.config.max_rpm, self.backend.config.max_target)
        return bool(pair and pair[0] == -pair[1] and 0 < abs(pair[0]) <= limit)

    def _note_motion_receipt(self, now):
        """Observe search ACKs even while paired control is inactive.

        A normal zero-speed pair may end a search pulse. Retain its preceding
        small pivot only for a fixed response window, not as a command lease.
        Unknown/external STOP generations break this lineage. Our own ordinary
        STOP retains only finite response diagnostics, never a reusable ACK.
        """
        generation = self.backend.stop_write_generation
        if generation != self._motion_generation:
            self._yaw_limit_key = self._yaw_delta_limit = None
            self._motion_receipt = self._motion_pair = None
            self._turn_responses = ()
            self._park_tail_until = float("-inf")
            self._park_tail_reverse_seen = False
            self._park_tail_entry_stop_at = None
            self._entry_tail_receipt = None
            self._motion_generation = generation
        receipt = self.backend.last_speed_receipt
        if receipt is None or receipt is self._motion_receipt:
            return
        pair = self._receipt_pair(receipt)
        previous = self._motion_receipt
        old_pair = self._motion_pair
        if (previous is not None and self._small_pivot(old_pair)
                and 0 <= now - previous.completed_at <= .5
                and not (self._small_pivot(pair) and pair[0] * old_pair[0] > 0
                         and abs(pair[0]) >= abs(old_pair[0]))):
            self._remember_turn_response(old_pair, receipt.completed_at, now)
            self.runtime.logger.info(
                "short_follow_turn_handoff previous_pair=%s new_pair=%s "
                "tail_deadline=%.9f stop_inserted=False", old_pair, pair, receipt.completed_at + .35)
        self._motion_receipt, self._motion_pair = receipt, pair

    def _remember_turn_response(self, pair, changed_at, now):
        # Keep distinct downward steps: 8 -> 5 -> 2 must not replace the
        # still-live 8 RPM response by only the latest 5 RPM response. Refreshes
        # at 2 RPM add nothing and cannot renew either historical deadline.
        feedback = self._feedback(now)
        reverse_seen = bool(feedback is not None
            and min(feedback.left_forward_rpm, feedback.right_forward_rpm) < -2)
        record = _TurnResponse(pair, changed_at, changed_at + .35, reverse_seen)
        self._turn_responses = (*self._turn_responses, record)[-16:]

    def _turn_response_overdue(self, now):
        return any(record.reverse_seen and now > record.expires_at for record in self._turn_responses)

    def _park_response_overdue(self, now):
        return self._park_tail_reverse_seen and now > self._park_tail_until

    def _expected_park_feedback(self, pair, now, sample_timestamp=None):
        # Fixed physical STOP-response history, independent of the plan TTL.
        # It can explain one small negative wheel while the other coasts;
        # it cannot produce identity, a distance observation or a wheel plan.
        if self._park_tail_entry_stop_at is not None:
            # A known motion -> zero -> acknowledged entry STOP can briefly
            # recoil on BOTH wheels. This is only diagnostic credit while the
            # original quiet-feedback barrier still owns a physically stopped
            # output; it never qualifies a moving pair or another STOP episode.
            return bool(now <= self._park_tail_until
                and self._entry_stop_acknowledged
                and self._entry_stop_at == self._park_tail_entry_stop_at
                and sample_timestamp is not None
                and sample_timestamp > self._park_tail_entry_stop_at
                and self.backend.last_speed_receipt is None
                and self.backend.last_speed_write is None
                and max(map(abs, pair)) <= 5)
        return (now <= self._park_tail_until and max(pair) >= -2 and min(pair) >= -5
                and max(pair) <= min(self.controller().config.max_rpm, self.backend.config.max_target))

    def _expected_turn_feedback(self, sample, now):
        measured = sample.left_forward_rpm, sample.right_forward_rpm
        limit = min(self.controller().config.max_rpm, self.backend.config.max_target)

        def compatible(command):
            # Only the wheel actually commanded backwards gets reverse credit.
            # Two backwards wheels or a growing/large reversal is not a pivot.
            return bool(max(measured) >= -2 and max(measured) <= limit
                and all(v >= -2 or (target < 0 and v >= target - 2)
                        for v, target in zip(measured, command)))

        receipt = self.backend.last_speed_receipt
        if receipt is not None and 0 <= now - receipt.completed_at <= .5:
            pair = self._receipt_pair(receipt)
            if self._small_pivot(pair) and compatible(pair):
                # A cached sample predating the new ACK cannot prove the new
                # target was reached. Retire larger history only when NEW
                # feedback actually catches up, not just within the +2 margin.
                if (sample.timestamp >= receipt.completed_at
                        and all(v >= target for v, target in zip(measured, pair) if target < 0)):
                    self._turn_responses = ()
                return True
        retained = []
        expected = False
        for record in self._turn_responses:
            if sample.timestamp > record.changed_at and min(measured) >= -2:
                continue
            if now <= record.expires_at and compatible(record.pair):
                expected = True
                record = replace(record, reverse_seen=record.reverse_seen or min(measured) < -2)
            elif now > record.expires_at and not record.reverse_seen:
                continue
            retained.append(record)
        self._turn_responses = tuple(retained)
        return expected

    def _entry_needed(self, now):
        self._note_motion_receipt(now)
        receipt = self.backend.last_speed_receipt
        known_forward = False
        known_turn_zero = False
        self._entry_tail_receipt = None
        limit = self.controller().config.max_rpm
        if receipt is not None:
            pair = self._receipt_pair(receipt)
            if ((min(pair) < 0 and not self._small_pivot(pair)) or max(pair) > limit):
                return True
            known_forward = min(pair) >= 0 and max(pair) > 0
            # Remember the actual ACK before positive feedback retires the
            # small-turn history below. A zero by itself proves no lineage.
            known_turn_zero = pair == (0, 0) and any(
                now <= record.expires_at for record in self._turn_responses)
        feedback = self._feedback(now)
        if feedback is None:
            return True
        pair = feedback.left_forward_rpm, feedback.right_forward_rpm
        if (receipt is not None and 0 <= now - receipt.completed_at <= .5
                and (known_forward or known_turn_zero) and min(pair) >= -2
                and max(pair) <= min(limit, self.backend.config.max_target)):
            self._entry_tail_receipt = receipt
        if (self._expected_turn_feedback(feedback, now)
                and max(map(abs, pair)) <= min(limit, 10)):
            return False
        if self._expected_park_feedback(pair, now, feedback.timestamp):
            return False
        if known_forward and min(pair) >= -2 and max(pair) <= limit:
            return False
        return max(map(abs, pair)) > 2

    def _feedback_reason(self, now):
        self._note_motion_receipt(now)
        sample = self.runtime.get_steering_feedback()
        now = max(now, time.monotonic())
        stamp = getattr(sample, "timestamp", None)
        if not isinstance(stamp, (int, float)) or not math.isfinite(stamp) or not 0 <= now-stamp <= .15:
            # Aging a suspect sample is not proof that the anomaly went away.
            if self._reverse_pending and now - self._reverse_pending[0] >= .15:
                return "feedback_reverse"
            if self._turn_response_overdue(now) or self._park_response_overdue(now):
                return "feedback_reverse"
            return None
        if getattr(sample, "left_error", 0) or getattr(sample, "right_error", 0):
            return "feedback_motor_error"
        if not wheel_feedback_valid(sample, now):
            return "feedback_invalid"
        if wheel_feedback_valid(sample, now):
            pair = sample.left_forward_rpm, sample.right_forward_rpm
            expected_turn = self._expected_turn_feedback(sample, now)
            if (stamp > self._park_tail_until - .35 and min(pair) >= -2
                    and (self._park_tail_entry_stop_at is None
                         or self._entry_sample is None or stamp >= self._entry_sample)):
                self._park_tail_until = float("-inf")
                self._park_tail_reverse_seen = False
                self._park_tail_entry_stop_at = None
            if min(pair) < -2:
                if not expected_turn and (self._turn_response_overdue(now) or self._park_response_overdue(now)):
                    return "feedback_reverse"
                receipt = self.backend.last_speed_receipt
                b = self.backend
                known_forward = bool(receipt is not None
                    and receipt.left_rpm * b.wheel_raw_state_to_target("left", 1, 0x01) >= 0
                    and receipt.right_rpm * b.wheel_raw_state_to_target("right", 1, 0x01) >= 0
                    and (receipt.left_rpm != 0 or receipt.right_rpm != 0))
                expected_park = self._expected_park_feedback(pair, now, stamp)
                if expected_park:
                    self._park_tail_reverse_seen = True
                if expected_turn or expected_park:
                    self._reverse_pending = None
                elif min(pair) < -5 or max(pair) < -2 or not known_forward:
                    return "feedback_reverse"
                # Experimental bounded tolerance for one wheel's -2..-5 RPM
                # transient during established forward motion. Keep a limited
                # pair, without accelerating; require NEW feedback to clear it.
                else:
                    pending = self._reverse_pending
                    if pending is None:
                        pending = (now, stamp, stamp, 1)
                        self.runtime.logger.info(
                            "short_follow_feedback_review left_rpm=%s right_rpm=%s "
                            "sample_ts=%.9f max_wait_ms=150 confirmation_ms=120", *pair, stamp)
                    elif stamp > pending[2]:
                        pending = (pending[0], pending[1], stamp, pending[3]+1)
                    self._reverse_pending = pending
                    if (now - pending[0] >= .15
                            or (pending[3] >= 2 and stamp - pending[1] >= .12)):
                        return "feedback_reverse"
            elif self._reverse_pending and stamp > self._reverse_pending[2]:
                self._reverse_pending = None
            # Compare against the absolute configured operating range, not
            # this observation's lower brake/request ceiling. A lawful old
            # high command is not a new overspeed fault when PI asks less.
            limit = min(self.controller().config.max_rpm, max(0, int(self.backend.config.max_target)))
            if max(pair) > limit + max(5., .10 * limit):
                return "feedback_overspeed"
        return None

    def _feedback_speed_cap(self, now):
        """Higher-speed requests need live own-wheel evidence, not person speed.

        Brief missing/stale feedback leaves a bounded low-speed command while
        the unchanged observation watchdog still runs. Fresh corrupt/reverse
        or faulted samples are rejected separately and never use this fallback.
        """
        limit = self.controller().config.max_rpm
        if self._reverse_pending:
            receipt = self.backend.last_speed_receipt
            prior = max(abs(receipt.left_rpm), abs(receipt.right_rpm)) if receipt else 0
            return min(limit, 40, prior)
        return min(limit, 40) if self._feedback(now) is None else limit

    @staticmethod
    def _yaw_feedback(sample):
        """Optional anti-overshoot evidence, never an extra motion gate.

        The caller supplies already qualified cached encoder feedback. Missing,
        stale or unconfirmed yaw leaves the existing lawful pair unchanged;
        it does not insert STOP or wait for a second feedback sample.
        """
        if sample is None or getattr(sample, "yaw_rate_confirmed", False) is not True:
            return None, None
        yaw = getattr(sample, "integrated_yaw_right_deg", None)
        rate = getattr(sample, "yaw_rate_right_dps", None)
        if any(not isinstance(value, (int, float)) or isinstance(value, bool)
               or not math.isfinite(value) for value in (yaw, rate)):
            return None, None
        return yaw, rate

    @staticmethod
    def _yaw_key(plan):
        if plan.yaw_capture_id <= 0 or plan.yaw_capture_timestamp <= 0:
            return None
        return plan.uid, plan.epoch, plan.yaw_capture_id, plan.yaw_capture_timestamp

    def _limit_retained_yaw(self, plan):
        key = self._yaw_key(plan)
        delta = plan.left_rpm-plan.right_rpm
        if (key is None or key != self._yaw_limit_key
                or self._yaw_delta_limit is None or abs(delta) <= self._yaw_delta_limit):
            return plan
        bound = max(0, int(self._yaw_delta_limit))
        if plan.base_rpm > 0:
            outer = max(plan.left_rpm, plan.right_rpm)
            left, right = ((outer, outer-bound) if delta > 0 else (outer-bound, outer))
            reason = plan.reason if bound else "forward"
        else:
            pivot = bound // 2
            left, right = ((pivot, -pivot) if delta > 0 else (-pivot, pivot))
            reason = plan.reason if pivot else plan.longitudinal_reason
        return replace(plan, left_rpm=left, right_rpm=right, reason=reason,
                       yaw_adjustment_reason="retained_yaw_taper")

    def _remember_executed_yaw(self, plan, left_rpm, right_rpm):
        # Only called after a completed pair/STOP. Failed or partially sent
        # output must never count as a physical centering intervention.
        key = self._yaw_key(plan)
        if key is None:
            return
        delta = abs(left_rpm-right_rpm)
        self._yaw_delta_limit = (min(self._yaw_delta_limit, delta)
            if key == self._yaw_limit_key and self._yaw_delta_limit is not None else delta)
        self._yaw_limit_key = key

    def _entry_ready(self, now):
        if self._entry_stop_at is None:
            return True
        if not self._entry_stop_acknowledged:
            return False
        sample = self._feedback(now)
        if sample is None or sample.timestamp <= self._entry_stop_at:
            return False
        if self._entry_sample is None or sample.timestamp > self._entry_sample:
            self._entry_sample = sample.timestamp
            quiet = max(abs(sample.left_forward_rpm), abs(sample.right_forward_rpm)) <= 2
            self._entry_quiet_count = self._entry_quiet_count + 1 if quiet else 0
        if self._entry_quiet_count < 2:
            return False
        self._entry_stop_at = None
        self._entry_stop_acknowledged = False
        return True

    @staticmethod
    def _acknowledge_stopped_output(controller):
        # A newer observation may arrive while a completed STOP is being
        # held/throttled. It does not get positive integral credit merely
        # because perception is still producing measurements during the wait.
        if controller is not None:
            current = controller.snapshot().plan
            if current is not None:
                controller.acknowledge_output(current, 0.)

    def _stop_locked(self, reason, epoch, now, *, force=False):
        controller = self.controller() or self._controller
        if controller is not None and reason in {
                "shutdown", "motor_fault", "explicit_stop", "hard_stop", "hard_stop_check_failed",
                "external_stop_generation", "stop_during_prepare", "identity_not_live",
                "identity_or_search_changed", "feedback_reverse", "feedback_overspeed",
                "feedback_motor_error", "feedback_invalid", "invalid_wheel_pair", "external_brake_hold"}:
            self._yaw_limit_key = self._yaw_delta_limit = None
            if controller.snapshot().plan is not None:
                controller.revoke(reason, now)
            epoch = controller.snapshot().epoch
        refresh = controller.config.stop_refresh_sec if controller is not None else 1.0
        key = (epoch, reason)
        feedback_stop = reason in {
            "feedback_reverse", "feedback_overspeed", "feedback_motor_error", "feedback_invalid"}
        new_feedback_stop = (feedback_stop and self._entry_stop_at is None
                             and self._live_feedback_reason == reason)
        if not force and not new_feedback_stop and self._stop_key == key and now - self._last_stop_at < refresh:
            self._acknowledge_stopped_output(controller)
            return
        # STOP is a mode command, not a zero-speed command which would release
        # the prior stop. No ordinary parking/500 ms holding lifecycle here.
        previous = self.backend.last_speed_receipt
        previous_pair = self._receipt_pair(previous) if previous is not None else None
        sample = self._feedback(now)
        entry_tail = bool(reason == "entry_settling"
            and self._entry_stop_at is not None and not self._entry_stop_acknowledged
            and previous is not None and previous is self._entry_tail_receipt
            and 0 <= now - previous.completed_at <= .5
            and not self._live_feedback_reason and not self._reverse_pending
            and not self._park_tail_reverse_seen and not self._turn_response_overdue(now)
            and sample is not None
            and min(sample.left_forward_rpm, sample.right_forward_rpm) >= -2
            and max(sample.left_forward_rpm, sample.right_forward_rpm)
                <= min(controller.config.max_rpm, self.backend.config.max_target))
        self.backend.send_stop("short_follow:" + reason, mode="emergency")
        self._yaw_response.reset()
        # The driver must first acknowledge both STOP writes. If it raises,
        # no fictitious zero output is reported to PI.
        self._acknowledge_stopped_output(controller)
        completed_at = time.monotonic()
        ordinary_stop = reason in {
            "awaiting_observation", "waiting_observation", "observation_expired",
            "target_distance_reached", "distance_restart_hysteresis",
            "identity_not_live", "identity_rejected", "identity_or_search_changed",
            "identity_or_search_handoff", "ownership_exit",
        }
        # A physical STOP cancels motion permission, not the physical fact
        # that we just commanded a turn. Keep only original finite diagnostic
        # intervals. A repeated STOP with no speed ACK cannot restart them.
        if ordinary_stop:
            if (previous is not None and self._small_pivot(previous_pair)
                    and 0 <= completed_at - previous.completed_at <= .5):
                self._remember_turn_response(previous_pair, completed_at, completed_at)
        else:
            self._turn_responses = ()
        self._motion_receipt = self._motion_pair = None
        self._motion_generation = self.backend.stop_write_generation
        retained_entry_tail = (reason == "entry_settling"
            and self._park_tail_entry_stop_at is not None
            and self._entry_stop_at == self._park_tail_entry_stop_at)
        if not ordinary_stop and not retained_entry_tail:
            self._park_tail_until = float("-inf")
            self._park_tail_reverse_seen = False
            self._park_tail_entry_stop_at = None
        elif previous_pair is not None:
            # An ordinary STOP does not turn small encoder braking tail
            # into an identity fault. Never grant this tolerance for a safety
            # STOP, unknown entry, or an already faulted drivetrain.
            if (min(previous_pair) >= 0 and max(previous_pair) > 0
                    and not self._park_tail_reverse_seen):
                self._park_tail_until = completed_at + .35
                sample = self._feedback(completed_at)
                self._park_tail_reverse_seen = bool(sample is not None
                    and min(sample.left_forward_rpm, sample.right_forward_rpm) < -2)
        if entry_tail:
            self._park_tail_until = completed_at + .35
            self._park_tail_entry_stop_at = completed_at
            self._park_tail_reverse_seen = False
            self.runtime.logger.info("short_follow_entry_stop_tail source_receipt=%s "
                "stop_ack=%.9f deadline=%.9f max_abs_rpm=5 motion_authorized=False",
                previous.sequence, completed_at, self._park_tail_until)
        self._entry_tail_receipt = None
        if feedback_stop and (new_feedback_stop or self._stop_key != key):
            self._entry_stop_at = completed_at
            self._entry_sample = None
            self._entry_quiet_count = 0
            self._entry_stop_acknowledged = True
            self._reverse_pending = None
        elif self._entry_stop_at is not None and not self._entry_stop_acknowledged:
            # Unknown takeover starts its clock at the first physical ACK.
            # This can be an awaiting-observation STOP too. Its real ACK and
            # subsequent quiet samples must not be discarded by a label change.
            self._entry_stop_at = completed_at
            self._entry_sample = None
            self._entry_quiet_count = 0
            self._entry_stop_acknowledged = True
        self._generation = self.backend.stop_write_generation
        self._stop_key = key
        self._last_stop_at = now
        self._next_write_due = None
        self.owner._short_follow_completed_stop_epoch = epoch
        self.owner._short_follow_completed_stop_at = completed_at
        self.owner.current_command = None
        self.owner.command_start_time = None
        self.owner._last_motor_dispatch_action = self.runtime.symbols.stop
        self.owner._last_dispatched_action = self.runtime.symbols.stop
        self.owner._last_motor_dispatch_source = "short_follow"
        self.owner._last_motor_dispatch_ts = completed_at
        self.owner._short_follow_last_applied_plan = None
        self.owner.is_forwarding = False
        self.runtime.logger.info("short_follow_stop reason=%s epoch=%s generation=%s",
                                 reason, epoch, self._generation)
        if reason == "identity_not_live":
            self.runtime.logger.info("short_follow_identity_check %s", self._identity_check)

    def _completed_stop_still_current(self):
        """Reuse our completed STOP barrier only if no I/O superseded it.

        A zero-speed ACK is NOT a STOP: it re-enters speed mode. A failed or
        partial write, external STOP, or fresh motor fault also forbids reuse.
        Must be checked under motor_io_lock, never used as motion authority.
        """
        return bool(self._stop_key is not None
            and self.backend.stop_write_generation == self._generation
            and self.backend.last_speed_receipt is None
            and self.backend.last_speed_write is None
            and not getattr(self.backend, "motion_write_fault", None)
            and not getattr(self.backend, "parking_release_fault", None))

    def external_stop(self, reason):
        """A genuine external stop revokes the mailbox before physical I/O."""
        controller = self.controller()
        if controller is None and not self._owned:
            return False
        now = time.monotonic()
        if controller is not None:
            controller.revoke(reason, now)
        with self._lock, self.owner.motor_io_lock:
            snapshot = controller.snapshot() if controller is not None else None
            self._stop_locked(reason, snapshot.epoch if snapshot else -1, now)
        return True

    def _retire_owner_locked(self, controller, latest, now, checked_reason=None):
        """One inactive handoff path, including deactivation during a tick.

        The caller holds motor I/O and the mailbox lock. A newly admitted
        successor goes through its canonical guard; neither the earlier active
        snapshot nor this helper grants it motion or changes its deadlines.
        """
        if controller is not None and latest.active:
            return True  # A newer activation superseded this pending exit.
        if controller is None and latest.active:
            self._controller.deactivate("ownership_exit", time.monotonic())
            latest = self._controller.snapshot()
        hard_reason = self._hard_reason()
        abnormal_reason = (checked_reason if checked_reason is not None
                           and checked_reason != latest.reason else None)
        transferred = False
        if (controller is not None and latest.reason in {
                "identity_or_search_handoff", "observation_expired", "lateral_handoff"}
                and hard_reason is None and abnormal_reason is None
                and self.backend.stop_write_generation == self._generation):
            self._owned = False
            before = self.backend.last_speed_receipt
            try:
                transferred = self.runtime._write_limited_yaw_successor()
                receipt = self.backend.last_speed_receipt
                if not transferred:
                    hard_reason = self._hard_reason()
                guarded_zero = bool(not transferred and receipt is not None and receipt is not before
                    and self._receipt_pair(receipt) == (0, 0)
                    and getattr(self.runtime, "_visible_wheel_waiting", False)
                    and self.backend.stop_write_generation == self._generation
                    and hard_reason is None)
                if guarded_zero:
                    # The successor already owns the required zero-crossing
                    # wait. Another STOP would discard its actual ACK/history;
                    # returning ownership grants no nonzero wheel permission.
                    transferred = True
                    self.runtime.logger.info("short_follow_ownership_exit epoch=%s "
                        "successor_guard_zero=True receipt=%s additional_stop=False",
                        latest.epoch, receipt.sequence)
            finally:
                self._owned = not transferred
        if not transferred:
            if hard_reason is None and abnormal_reason is None and self._completed_stop_still_current():
                self.runtime.logger.info(
                    "short_follow_ownership_exit epoch=%s reuse_completed_stop=True generation=%s",
                    latest.epoch, self._generation)
            else:
                self._stop_locked(hard_reason or abnormal_reason or "ownership_exit",
                                  latest.epoch, now, force=True)
        self._owned = False
        self._entry_stop_at = None
        self._entry_stop_acknowledged = False
        self._entry_tail_receipt = None
        self._last_write_at = float("-inf")
        self._next_write_due = None
        return True

    def service(self):
        """Return True while this writer owns motion (also while stopped).

        Called every action-loop iteration; only speed refreshes are 20 Hz.
        Deadline, hard-stop and ownership checks are never period-throttled.
        """
        with self._lock:
            controller = self.controller()
            snapshot = controller.snapshot() if controller is not None else None
            now = time.monotonic()
            if controller is not None:
                self._note_motion_receipt(now)
                self._consume_protected_stop(controller, now)
                snapshot = controller.snapshot()
            if snapshot is None or not snapshot.active:
                if controller is not None and self.explicit_stop_pending():
                    # A startup/search emergency still needs a real receipt;
                    # it must not wait for a UID to activate this mailbox.
                    with self.owner.motor_io_lock, controller.write_snapshot():
                        self._stop_locked("explicit_stop", controller.snapshot().epoch, now)
                    return True
                if not self._owned:
                    return False
                with self.owner.motor_io_lock:
                    # Reactivation while the old exit waited for I/O is a
                    # newer owner; an obsolete exit must not stop its pair.
                    mailbox = controller or self._controller
                    with mailbox.write_snapshot() as latest:
                        return self._retire_owner_locked(controller, latest, now)
            self._controller = controller
            snapshot = controller.snapshot()
            if not self._owned:
                self._owned = True
                self.runtime._retire_legacy_normal_executor()
                self._generation = self.backend.stop_write_generation
                self._last_write_at = float("-inf")
                self._next_write_due = None
                if self._entry_needed(now):
                    self._entry_stop_at = now
                    self._entry_sample = None
                    self._entry_quiet_count = 0
                    self._entry_stop_acknowledged = False
            reason = self._hard_reason()
            if reason is None and self.runtime._service_short_follow_entry_hold():
                # The sole exception is a REAL pre-takeover search STOP and
                # its encoder settlement, never an ordinary normal-follow hold.
                self._generation = self.backend.stop_write_generation
                return True
            if reason is None and self.backend.stop_write_generation != self._generation:
                reason = "external_stop_generation"
            if reason is not None:
                if snapshot.plan is not None:
                    controller.revoke(reason, now)
                with self.owner.motor_io_lock, controller.write_snapshot():
                    self._stop_locked(reason, controller.snapshot().epoch, now)
                return True
            snapshot, now, reason = self._current_plan_check(controller)
            if reason is not None:
                with self.owner.motor_io_lock, controller.write_snapshot():
                    # A ordinary stop plan may have been replaced while
                    # waiting for serial. Always re-read before writing it.
                    self._consume_protected_stop(controller, time.monotonic())
                    latest, _checked_at, latest_reason = self._checked_plan(controller)
                    if latest_reason == "existing_search_hold":
                        if latest.plan is not None:
                            controller.revoke(latest_reason, time.monotonic())
                        self._generation = self.backend.stop_write_generation
                        return True  # Next tick services the existing stop owner.
                    if not latest.active:
                        return self._retire_owner_locked(controller, latest, _checked_at, latest_reason)
                    if latest_reason is not None:
                        self._stop_locked(latest_reason, latest.epoch, time.monotonic())
                        return True
                # New normal evidence replaced the old stop. Fall through to
                # commit the current pair; never install the obsolete STOP.
            if self._next_write_due is not None and now + 1e-9 < self._next_write_due:
                return True
            with self.owner.motor_io_lock:
                generation = self.backend.stop_write_generation
                self.backend.prepare_speed_mode()
                # The mailbox's short commit guard orders revocation and the
                # actual dual-wheel write, not the expensive perception work.
                with controller.write_snapshot():
                    self._consume_protected_stop(controller, time.monotonic())
                    latest, now, reason = self._checked_plan(controller)
                    if self.backend.stop_write_generation != generation:
                        reason = "stop_during_prepare"
                    if reason == "existing_search_hold":
                        if latest.plan is not None:
                            controller.revoke(reason, now)
                        self._generation = self.backend.stop_write_generation
                        return True
                    if not latest.active:
                        return self._retire_owner_locked(controller, latest, now, reason)
                    if reason is not None:
                        self._stop_locked(reason, latest.epoch, now)
                        return True
                    source_plan = latest.plan
                    feedback_snapshot = self._feedback(now)  # Cached data; no serial read.
                    current_yaw, yaw_rate = self._yaw_feedback(feedback_snapshot)
                    plan = controller.execution_plan(source_plan, now,
                        current_yaw_deg=current_yaw, yaw_rate_deg_s=yaw_rate)
                    plan = self._limit_retained_yaw(plan)
                    if not plan.moving:
                        # Only a previously legal pivot can become all-zero
                        # here. A forward arc becomes straight, not stopped.
                        # Reuse the normal distance-hold stop lifecycle: no
                        # additional brake hold or quiet-feedback barrier.
                        self.runtime.logger.info(
                            "short_follow_yaw_center cap=%s yaw_cap=%s epoch=%s sequence=%s "
                            "source_pair=%s applied_pair=%s yaw_age_ms=%s "
                            "control_center=%s yaw_adjustment=%s",
                            plan.capture_id, getattr(plan, "yaw_capture_id", None),
                            plan.epoch, plan.sequence,
                            (source_plan.left_rpm, source_plan.right_rpm),
                            (plan.left_rpm, plan.right_rpm),
                            None if getattr(plan, "yaw_capture_timestamp", None) is None else
                                (now-plan.yaw_capture_timestamp)*1000.,
                            getattr(plan, "yaw_control_center_x_ratio", None),
                            getattr(plan, "yaw_adjustment_reason", "none"))
                        self._stop_locked(plan.reason, latest.epoch, now)
                        self._remember_executed_yaw(plan, 0, 0)
                        return True
                    feedback_cap = self._feedback_speed_cap(now)
                    limit = max(0, min(controller.config.max_rpm,
                                       int(self.backend.config.max_target), feedback_cap))
                    if limit == 0:
                        self._stop_locked("motor_limit_zero", latest.epoch, now)
                        return True
                    response_source_plan = plan
                    plan = self._yaw_response.adjust(plan, feedback_snapshot,
                        self.backend.last_speed_receipt, now, controller.config, wheel_limit=limit)
                    # Never let selecting this mode increase an existing
                    # hardware limit. Scale both wheels to preserve steering.
                    scale = min(1., limit / max(abs(plan.left_rpm), abs(plan.right_rpm)))
                    left_rpm, right_rpm = round(plan.left_rpm*scale), round(plan.right_rpm*scale)
                    left = self.backend.wheel_raw_state_to_target(
                        "left", abs(left_rpm), 0x02 if left_rpm < 0 else 0x01)
                    right = self.backend.wheel_raw_state_to_target(
                        "right", abs(right_rpm), 0x02 if right_rpm < 0 else 0x01)
                    send_started_at = time.monotonic()
                    self.backend.send_targets(left, right, "SHORT_FOLLOW", max_target_override=limit,
                                              history_uid=plan.uid)
                    self._yaw_response.acknowledge(response_source_plan,
                        self.backend.last_speed_receipt, left_rpm, right_rpm)
                    self._remember_executed_yaw(plan, left_rpm, right_rpm)
                    # Only a successfully completed wheel pair can feed back
                    # the output ceiling. Actual encoder lag is NOT windup.
                    applied_base = 0 if plan.pivot else max(left_rpm, right_rpm)
                    controller.acknowledge_output(plan, applied_base)
                    self._note_motion_receipt(time.monotonic())
                    self._generation = self.backend.stop_write_generation
                    self._last_write_at = time.monotonic()
                    period = controller.config.write_period_sec
                    self._next_write_due = send_started_at + period
                    if self._last_write_at + 1e-9 >= self._next_write_due:
                        skipped = max(1, int((self._last_write_at-self._next_write_due+1e-9)/period)+1)
                        self._next_write_due += skipped*period
                    self._stop_key = None
                    symbols = self.runtime.symbols
                    action = ((symbols.rotate_left if left_rpm < right_rpm else symbols.rotate_right)
                              if plan.pivot else
                              symbols.steer_left if left_rpm < right_rpm else
                              symbols.steer_right if left_rpm > right_rpm else symbols.forward)
                    self.owner.current_command = action
                    self.owner.command_start_time = time.time()
                    self.owner._last_motor_dispatch_action = action
                    self.owner._last_dispatched_action = action
                    self.owner._last_motor_dispatch_source = "short_follow"
                    self.owner._last_motor_dispatch_ts = self._last_write_at
                    self.owner._short_follow_last_applied_plan = replace(plan, left_rpm=left_rpm,
                        right_rpm=right_rpm, base_rpm=float(applied_base),
                        speed_cap_rpm=min(plan.speed_cap_rpm, float(limit)))
                    self.owner.is_forwarding = left_rpm + right_rpm > 0
                    self.runtime.logger.info("short_follow_write cap=%s uid=%s epoch=%s sequence=%s "
                        "left_rpm=%s right_rpm=%s pi_request_rpm=%.2f p_rpm=%.2f i_rpm=%.2f "
                        "speed_cap_rpm=%.2f feedback_speed_cap=%s depth_age_ms=%.1f expires_in_ms=%.1f "
                        "motion_kind=%s command_delta_rpm=%s feedback_delta_rpm=%s feedback_age_ms=%s "
                        "yaw_cap=%s yaw_age_ms=%s yaw_observed_center=%s yaw_control_center=%s "
                        "yaw_capture_deg=%s yaw_current_deg=%s yaw_rate_dps=%s yaw_adjustment=%s "
                        "yaw_common_reduction_rpm=%s yaw_response_reason=%s yaw_response_samples=%s",
                        plan.capture_id, plan.uid, plan.epoch, plan.sequence,
                        left_rpm, right_rpm, plan.base_request_rpm, plan.p_rpm, plan.i_rpm,
                        min(plan.speed_cap_rpm, float(limit)), feedback_cap,
                        (now-plan.depth_timestamp)*1000,
                        (plan.expires_at-now)*1000,
                        "pivot" if plan.pivot else "arc" if left_rpm != right_rpm else "straight",
                        left_rpm - right_rpm,
                        None if feedback_snapshot is None else feedback_snapshot.left_forward_rpm - feedback_snapshot.right_forward_rpm,
                        None if feedback_snapshot is None else (send_started_at-feedback_snapshot.timestamp)*1000.,
                        getattr(plan, "yaw_capture_id", None),
                        None if getattr(plan, "yaw_capture_timestamp", None) is None else
                            (now-plan.yaw_capture_timestamp)*1000.,
                        getattr(plan, "yaw_center_x_ratio", None),
                        getattr(plan, "yaw_control_center_x_ratio", None),
                        getattr(plan, "yaw_capture_yaw_deg", None), current_yaw, yaw_rate,
                        getattr(plan, "yaw_adjustment_reason", "none"),
                        plan.yaw_common_reduction_rpm, plan.yaw_response_reason,
                        plan.yaw_response_sample_count)
            return True
