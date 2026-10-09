"""Short physical speed history, never target identity or motion authority.

A new low request does not prove the wheels have finished responding to an
earlier high request.  Keep only verified, completed normal-follow writes for
650 ms (the observed 60 RPM step response reaches t90 around 600 ms). Records
remain for another 300 ms to prove past intervals without changing that MAX.
This is
an input to a conservative MAX with requested/measured speed, not a lower speed
limit, permission to accelerate, or a replacement for live encoder checks.
"""
from dataclasses import dataclass, replace
import math

from .mssd_motor import MotorSpeedReceipt, MotorSpeedWrite
from .depth_authority_timing import MAX_FORWARD_DEPTH_TTL_SEC


EXECUTED_SPEED_MEMORY_SEC = .65
EXECUTED_SPEED_HISTORY_SEC = EXECUTED_SPEED_MEMORY_SEC + MAX_FORWARD_DEPTH_TTL_SEC
MAX_EXECUTED_SPEED_RECORDS = 64


@dataclass(frozen=True)
class ExecutedSpeedRecord:
    uid: int
    outer_rpm: float
    receipt: MotorSpeedReceipt
    response_samples: tuple = ()
    response_anchor_valid: bool = True
    retired_at: float | None = None


def _finite(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _valid_uid(uid):
    return isinstance(uid, int) and not isinstance(uid, bool) and uid > 0


def record_completed_speed(history, *, uid, applied, signs, receipt,
                           previous_receipt, now, packet_written, feedback=None):
    """Return a copy-on-write tuple; rejected/planned writes leave it unchanged.

    The normal-follow executor calls this only after its dual-wheel write has
    returned.  Receipt object AND sequence must have advanced; a backend-held
    zero or skipped write cannot extend the history's physical timestamp.
    """
    if (not packet_written or not _valid_uid(uid) or not _finite(now)
            or not isinstance(receipt, MotorSpeedReceipt)
            or receipt is previous_receipt
            or not isinstance(receipt.sequence, int)
            or isinstance(receipt.sequence, bool) or receipt.sequence <= 0
            or not _finite(receipt.completed_at) or receipt.completed_at <= 0
            or not 0 <= now-receipt.completed_at <= EXECUTED_SPEED_MEMORY_SEC):
        return history
    if (not isinstance(applied, (tuple, list)) or len(applied) != 2
            or not isinstance(signs, (tuple, list)) or len(signs) != 2
            or not all(_finite(v) for v in (*applied, *signs,
                                           receipt.left_rpm, receipt.right_rpm))
            or any(s not in (-1, 1) for s in signs)
            or (receipt.left_rpm, receipt.right_rpm)
            != (applied[0]*signs[0], applied[1]*signs[1])):
        return history
    if (previous_receipt is not None
            and (not isinstance(previous_receipt, MotorSpeedReceipt)
                 or receipt.sequence <= previous_receipt.sequence)):
        return history
    if history:
        latest = history[-1].receipt
        if (receipt is latest or receipt.sequence <= latest.sequence
                or receipt.completed_at < latest.completed_at):
            return history
    # Physical momentum includes an outer wheel during a turn, not only the
    # average forward component.  A later zero retains (but never renews) the
    # timestamp of the earlier high write.
    entry = ExecutedSpeedRecord(uid, float(max(map(abs, applied))), receipt)
    kept = tuple(item for item in history
                 if item.uid == uid
                 and 0 <= now-item.receipt.completed_at <= EXECUTED_SPEED_HISTORY_SEC)
    if history and (previous_receipt is not history[-1].receipt
                    or receipt.sequence != history[-1].receipt.sequence + 1):
        # A non-follow write or mode/STOP transition broke this response's
        # provenance. Keep the old speed costs and their original timestamps,
        # but never collect more retirement samples against their old anchors.
        # Only this newly completed normal write can seed a fresh proof.
        kept = tuple(replace(item, response_samples=(), response_anchor_valid=False)
                     for item in kept)
    return _retire_responded_high_writes(
        (*kept, entry)[-MAX_EXECUTED_SPEED_RECORDS:], feedback, now)


def _retire_responded_high_writes(history, feedback, now):
    """Retire old high commands only after a verified lower response.

    Two distinct, fresh encoder samples must follow a completed lower write,
    remain below its outer-wheel target, and be nonincreasing.  Any later
    higher completed write prevents using that older low-write proof.  This
    updates only future braking-budget reads, never a frozen depth grant.
    """
    values = tuple(getattr(feedback, key, None) for key in (
        "timestamp", "left_forward_rpm", "right_forward_rpm"))
    if (not getattr(feedback, "trustworthy", False)
            or not all(_finite(v) for v in values)
            or not 0 <= now-values[0] <= .15
            or max(abs(v) for v in values[1:]) > 200):
        return tuple(replace(item, response_samples=()) if item.response_samples else item
                     for item in history)
    stamp, left, right = values
    outer = max(abs(left), abs(right))
    updated = list(history)
    later_max = 0.
    confirmed = []
    for index in range(len(history)-1, -1, -1):
        item = history[index]
        if item.retired_at is not None:
            # Retired records remain as historical interval costs, but must
            # not affect the original current-response retirement algorithm.
            continue
        samples = item.response_samples
        if (not item.response_anchor_valid or later_max > item.outer_rpm
                or stamp <= item.receipt.completed_at):
            samples = ()
        elif samples and stamp < samples[-1][0]:
            samples = ()  # An older observation cannot seed a new proof.
        elif samples and stamp == samples[-1][0]:
            # The same cache is not a second response; a conflicting payload
            # with the same timestamp invalidates the existing observation.
            if (left, right) != samples[-1][1:]:
                samples = ()
        elif outer > item.outer_rpm:
            samples = ()
        elif (not samples or stamp-samples[-1][0] > .15
              or outer > max(abs(v) for v in samples[-1][1:])):
            samples = ((stamp, left, right),)
        else:
            samples = (*samples[-1:], (stamp, left, right))
        updated[index] = replace(item, response_samples=samples)
        if len(samples) == 2:
            confirmed.append((index, item.outer_rpm, samples[-1][0]))
        later_max = max(later_max, item.outer_rpm)
    result = []
    for index, item in enumerate(updated):
        retirements = [stamp for low_index, low_outer, stamp in confirmed
                       if index < low_index and item.outer_rpm > low_outer]
        result.append(replace(item, retired_at=min(retirements))
                      if item.retired_at is None and retirements else item)
    return tuple(result)


def executed_speed_bound_rpm(history, *, uid, now):
    """Read immutable history without mutation, waiting, or motor operations."""
    if not _valid_uid(uid) or not _finite(now) or not history:
        return None
    # A different target must not borrow a prior target's source provenance.
    if history[-1].uid != uid:
        return None
    values = [item.outer_rpm for item in history
              if item.uid == uid
              and item.retired_at is None
              and 0 <= now-item.receipt.completed_at <= EXECUTED_SPEED_MEMORY_SEC]
    return max(values) if values else None


def submitted_execution_view(history, *, uid, now, receipt, submission, stop_generation):
    """Budget-only view of a normal follow transaction, without new authority.

    During I/O the previous ACK chain is used ONLY to cost past travel, with
    the submitted pair charged in full as a possible outer-wheel speed. After
    both ACKs, bridge the brief gap until runtime bookkeeping with the genuine
    receipt. Never substitute the previous receipt for final-write admission.
    The caller must recheck the immutable inputs after this pure calculation.
    """
    if (not isinstance(submission, MotorSpeedWrite) or submission.uid != uid
            or not _valid_uid(submission.uid)
            or not _valid_uid(uid) or not _finite(now)
            or type(stop_generation) is not int
            or type(submission.stop_generation) is not int
            or submission.stop_generation != stop_generation):
        return history, receipt, None
    if (not all(map(_finite, (submission.started_at, submission.left_rpm,
                             submission.right_rpm)))
            or not 0 < submission.started_at <= now):
        return history, receipt, None
    completed = submission.completed_receipt
    if completed is None:
        if (receipt is not None and receipt is not submission.previous_receipt
                or now-submission.started_at > MAX_FORWARD_DEPTH_TTL_SEC):
            return history, receipt, None
        return (history, submission.previous_receipt,
                float(max(abs(submission.left_rpm), abs(submission.right_rpm))))
    if (not isinstance(completed, MotorSpeedReceipt)
            or receipt is not None and receipt is not completed
            or (completed.left_rpm, completed.right_rpm)
                != (submission.left_rpm, submission.right_rpm)
            or not submission.started_at <= completed.completed_at <= now
            or now-completed.completed_at > EXECUTED_SPEED_MEMORY_SEC):
        return history, receipt, None
    if history and history[-1].receipt is completed:
        return history, completed, None
    # Genuine dual ACK, not a synthetic receipt for a planned/in-flight pair.
    view = record_completed_speed(history, uid=uid,
        applied=(completed.left_rpm, completed.right_rpm), signs=(1, 1),
        receipt=completed, previous_receipt=submission.previous_receipt,
        now=now, packet_written=True)
    return view, completed, None


def executed_interval_speed_bound_rpm(history, *, uid, sample_timestamp, now, receipt):
    """Prove a completed-command bound over a physical sample's past interval.

    This stricter, read-only query is not the instantaneous 650ms MAX reader.
    A pre-sample anchor, continuous completed receipts and the actual current
    backend receipt are required. STOP/untracked writes break the response
    chain even when the speed sequence is numerically consecutive. A command
    retired AFTER capture still contributed to this interval's upper bound.
    None means no proof, never a zero-speed observation or motion permission.
    """
    if (not _valid_uid(uid) or not all(map(_finite, (sample_timestamp, now)))
            or sample_timestamp <= 0
            or not 0 <= now-sample_timestamp <= MAX_FORWARD_DEPTH_TTL_SEC
            or not isinstance(history, tuple) or not history
            or not isinstance(history[-1], ExecutedSpeedRecord)
            or not isinstance(receipt, MotorSpeedReceipt)
            or receipt is not history[-1].receipt):
        return None
    anchor = None
    previous = None
    for index, item in enumerate(history):
        if (not isinstance(item, ExecutedSpeedRecord) or not _valid_uid(item.uid) or item.uid != uid
                or type(item.response_anchor_valid) is not bool
                or not isinstance(item.receipt, MotorSpeedReceipt)
                or type(item.receipt.sequence) is not int or item.receipt.sequence <= 0
                or not all(map(_finite, (item.outer_rpm, item.receipt.completed_at,
                                        item.receipt.left_rpm, item.receipt.right_rpm)))
                or item.outer_rpm < 0 or not 0 < item.receipt.completed_at <= now
                or item.outer_rpm != max(abs(item.receipt.left_rpm), abs(item.receipt.right_rpm))
                or (item.retired_at is not None and (
                    not _finite(item.retired_at)
                    or not item.receipt.completed_at < item.retired_at <= now))
                or (previous is not None and (
                    item.receipt.sequence <= previous.sequence
                    or item.receipt.completed_at < previous.completed_at))):
            return None
        if item.receipt.completed_at <= sample_timestamp:
            anchor = index
        previous = item.receipt
    if anchor is None:
        return None  # No observed command covers the start of this interval.
    for index in range(anchor, len(history)):
        item = history[index]
        if (not item.response_anchor_valid
                or (index > anchor and item.receipt.sequence != history[index-1].receipt.sequence+1)):
            return None
    # The instantaneous reader's 650ms expiry is not a measured response.
    # This stricter proof retains every available unretired high command;
    # only a verified retirement at/before capture can remove its past cost.
    values = [item.outer_rpm for item in history
              if item.retired_at is None or item.retired_at > sample_timestamp]
    return max(values) if values else None


def observe_completed_speed_response(history, *, feedback, now, receipt):
    """Apply fresh encoder response without requiring another motor write.

    A stopped/limited executor must not have to issue more speed packets just
    to learn that a previous high command has finished decelerating. Keep the
    same two-sample proof used by the writer, but let the feedback publisher
    advance it. No timestamp, receipt, motor lease or request is created here.
    A STOP/untracked write breaks the proof instead of making old history
    disappear. The caller serializes copy-on-write history publication and
    checks that the backend receipt did not change during this calculation.
    """
    if not history or not _finite(now):
        return history
    if (not isinstance(receipt, MotorSpeedReceipt)
            or receipt is not history[-1].receipt):
        return tuple(replace(item, response_samples=(), response_anchor_valid=False)
                     for item in history)
    return _retire_responded_high_writes(history, feedback, now)
