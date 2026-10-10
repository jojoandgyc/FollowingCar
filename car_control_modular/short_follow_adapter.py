"""Bridge verified perception into the single normal-follow wheel owner.

No hardware calls, ranging, PI, target-motion prediction, or recovery ramps.
The caller owns the perception/control mutex. The immutable controller snapshot
is the only object the periodic writer consumes.
"""
from __future__ import annotations

import math
import queue
import time

from .control_types import ControlDecision
from .action_command import ActionCommandSnapshot
from .detector_identity_lease import ValidatedVisualObservation, read_visual_identity_evidence
from .short_follow import ShortFollowObservation, ShortFollowYawObservation
from .wheel_zero_cross import wheel_feedback_valid
from .associated_position import AssociatedPosition


class ShortFollowAdapter:
    def __init__(self, owner, controller, logger):
        self.owner = owner
        self.controller = controller
        self.logger = logger

    @property
    def owned(self):
        return self.controller.snapshot().active

    def deactivate(self, reason, now):
        if self.owned:
            self.controller.deactivate(reason, now)

    @staticmethod
    def _detector_center(target, width, height, capture_id, capture_timestamp):
        """Only the identity-bound, current detector crop may bypass smoothing."""
        observation = getattr(target, "depth_observation", None)
        if (observation is None or observation.source != "yolo_detector"
                or observation.target_id != target.track_id
                or observation.capture_frame_id != capture_id
                or observation.capture_timestamp != capture_timestamp
                or width <= 0 or height <= 0):
            return None
        box = observation.bbox
        if (len(box) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                               or not math.isfinite(v) for v in box)):
            return None
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            return None
        return (x1 + x2) / (2. * width)

    @staticmethod
    def _yaw_feedback(feedback, now):
        if (not wheel_feedback_valid(feedback, now)
                or not getattr(feedback, "yaw_rate_confirmed", False)):
            return {}
        heading = getattr(feedback, "integrated_yaw_right_deg", None)
        rate = getattr(feedback, "yaw_rate_right_dps", None)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in (heading, rate)):
            return {}
        return dict(current_yaw_deg=heading, yaw_rate_deg_s=rate)

    def publish_visual_lateral(self, target, width, height, capture_id, capture_timestamp,
                               *, now, feedback=None):
        """Replace yaw in an existing pair, never range, PI or its deadline.

        This also runs at the early identity/ROI publication point, before the
        visual controller can wait for a Depth task. No motor or sensor I/O.
        """
        owner = self.owner
        evidence, checked_at = read_visual_identity_evidence(owner)
        now = max(now, checked_at)
        proof = evidence.observation
        if (not self.controller.config.enabled or target is None
                or not isinstance(proof, ValidatedVisualObservation)
                or proof.uid != target.track_id or proof.capture != capture_id
                or proof.timestamp != capture_timestamp
                or proof.continuation_sample_timestamp is not None):
            return None
        center = self._detector_center(target, width, height, capture_id, capture_timestamp)
        observation = getattr(target, "depth_observation", None)
        if center is None or observation is None or observation.raw_track_id != proof.track_id:
            return None

        def current():
            state = self.controller.snapshot()
            controller = owner._follow_controller
            return bool(state.active and state.uid == proof.uid
                and getattr(owner, "_validated_visual_observation", None) is proof
                and getattr(owner, "_detector_identity_lease", None) is evidence.lease
                and evidence.live(proof.uid, max(now, time.monotonic()))
                and getattr(controller, "active_target_id", None) == proof.uid
                and getattr(controller, "search_state", "none") == "none"
                and getattr(owner, "search_state", "none") == "none"
                and getattr(owner, "running", False)
                and (not getattr(owner, '_reacquire_depth_pending', False)
                     or self._pending_lateral_plan(state.plan))
                and not any(getattr(owner, key, False) for key in (
                    "_explicit_stop_requested", "_runtime_shutdown_requested",
                    "_brake_hold_active")))

        if not current():
            return None
        runtime = getattr(owner, "_action_runtime", None)
        heading_at = getattr(runtime, "get_steering_heading_at", None)
        # Only bracketed historical feedback is a capture heading. Never use
        # the processing-time heading as if it were sampled with the image.
        capture_heading = heading_at(capture_timestamp) if callable(heading_at) else None
        if feedback is None:
            reader = getattr(runtime, "get_steering_feedback", None)
            feedback = reader() if callable(reader) else None
        before = self.controller.snapshot().plan
        plan = self.controller.update_lateral(ShortFollowYawObservation(
            proof.uid, capture_id, capture_timestamp, center, capture_heading), now,
            publication_guard=current, **self._yaw_feedback(feedback, now))
        if plan is not None and plan is not before:
            self.logger.info("short_follow_yaw_publish capture_frame_id=%s uid=%s "
                "source=identity_bound_detector center=%.4f capture_yaw=%s "
                "left_rpm=%s right_rpm=%s depth_ts=%s expires_at=%s "
                "depth_renewed=False integral_updated=False parking_wait=False",
                capture_id, proof.uid, center, capture_heading,
                plan.left_rpm, plan.right_rpm, plan.depth_timestamp, plan.expires_at)
        return plan

    def publish_associated_lateral(self, position, width, *, identity_publication, now):
        """Use a qualified low-score position within the ORIGINAL live plan.

        No new full identity, range sample, forward lease, parking transition or
        PI update. The caller also supplies this observation to direction
        history separately, so losing full detection need not restore old yaw.
        """
        if not isinstance(position, AssociatedPosition) or width <= 0:
            return None
        owner = self.owner
        def current():
            checked = max(now, time.monotonic())
            observation = getattr(identity_publication, 'observation', None)
            follow = owner._follow_controller
            return bool(
                getattr(owner, '_visual_identity_evidence', None) is identity_publication
                and isinstance(observation, ValidatedVisualObservation)
                and observation.uid == position.uid and observation.track_id == position.track_id
                and position.reference_capture <= observation.capture < position.capture
                and position.reference_timestamp <= observation.timestamp < position.timestamp
                and identity_publication.live(position.uid, checked)
                and position.timestamp <= checked < position.expires_at
                and getattr(follow, 'active_target_id', None) == position.uid
                and getattr(follow, 'search_state', 'none') == 'none'
                and getattr(owner, 'search_state', 'none') == 'none'
                and getattr(owner, 'running', False)
                and (not getattr(owner, '_reacquire_depth_pending', False)
                     or self._pending_lateral_plan(self.controller.snapshot().plan))
                and not any(getattr(owner, key, False) for key in (
                    '_explicit_stop_requested', '_runtime_shutdown_requested',
                    '_brake_hold_active')))
        if not current():
            return None
        runtime = getattr(owner, '_action_runtime', None)
        heading_at = getattr(runtime, 'get_steering_heading_at', None)
        reader = getattr(runtime, 'get_steering_feedback', None)
        heading = heading_at(position.timestamp) if callable(heading_at) else None
        feedback = reader() if callable(reader) else None
        center = (position.bbox[0] + position.bbox[2]) / (2. * width)
        return self.controller.update_lateral(ShortFollowYawObservation(position.uid,
            position.capture, position.timestamp, center, heading), now,
            publication_guard=current, **self._yaw_feedback(feedback, now))

    @staticmethod
    def _pending_lateral_plan(plan):
        """Existing paired yaw only; never turn a forward grant into a bridge."""
        return bool(plan is not None and plan.base_rpm == 0
                    and plan.longitudinal_reason == 'reacquire_depth_pending'
                    and plan.left_rpm + plan.right_rpm == 0)

    def hold_explicit_stop(self, capture_timestamp, now):
        """Consume only a documented auto-clear stop AFTER physical STOP ack.

        Manual/unknown stops require their owner to explicitly clear its flag.
        An in-flight image predating the stop cannot authorize its own restart.
        """
        owner = self.owner
        if not getattr(owner, "_explicit_stop_requested", False):
            owner._short_follow_pending_explicit_stop = None
            return False
        reason = str(getattr(owner, "_last_explicit_stop_reason", "") or "explicit_stop")
        provenance = getattr(owner, "_explicit_stop_provenance", None)
        if provenance == ("controller_state", reason):
            # Startup/search use this old flag for ordinary per-frame holds.
            # Their own executor owns those transitions; they are not a
            # latched manual emergency. Match the reason as well so an
            # external request cannot accidentally inherit old provenance.
            owner._short_follow_pending_explicit_stop = None
            owner._explicit_stop_requested = False
            owner._last_explicit_stop_reason = ""
            owner._explicit_stop_provenance = None
            return False
        pending = getattr(owner, "_short_follow_pending_explicit_stop", None)
        if pending is None or pending[1] != reason:
            state = self.controller.revoke("explicit:" + reason, now)
            pending = (state.epoch, reason, now)
            owner._short_follow_pending_explicit_stop = pending
        auto_clear = provenance == ("sensor_safety", reason)
        if (auto_clear and getattr(owner, "_short_follow_completed_stop_epoch", -1) >= pending[0]
                and isinstance(capture_timestamp, (int, float))
                and math.isfinite(capture_timestamp)
                and max(pending[2], getattr(owner, "_short_follow_completed_stop_at", pending[2]))
                    < capture_timestamp <= now
                and not getattr(owner, "_runtime_shutdown_requested", False)):
            hazards = owner._current_hazard_state_for_controller()
            obstacles = owner._get_obstacle_status()
            if (not hazards.active and not any(obstacles.values())
                    and self._release_completed_sensor_hold(reason, pending[0], now)):
                owner._explicit_stop_requested = False
                owner._last_explicit_stop_reason = ""
                owner._explicit_stop_provenance = None
                owner._short_follow_pending_explicit_stop = None
                return False
        return True

    def _release_completed_sensor_hold(self, reason, epoch, now):
        """Release only an inherited, physically settled SENSOR stop.

        The caller already proved an acknowledged STOP and a newer clear
        image. Search/unknown/manual holds and backend faults remain with
        their own owner; matching a clear sensor is not a universal unlock.
        """
        owner = self.owner
        if not getattr(owner, "_brake_hold_active", False):
            return True
        if getattr(owner, "_brake_hold_label", "") not in (reason, "safety_hold_" + reason):
            return False
        runtime = getattr(owner, "_action_runtime", None)
        backend = getattr(runtime, "backend", None)
        if (backend is None or getattr(backend, "motion_write_fault", None)
                or getattr(backend, "parking_release_fault", None)
                or getattr(runtime, "_search_reacquire_brake_request", None) is not None
                or getattr(owner, "_near_yaw_park_request", None) is not None):
            return False
        feedback = runtime.get_steering_feedback()
        completed_at = getattr(owner, "_short_follow_completed_stop_at", now)
        if (not wheel_feedback_valid(feedback, now) or feedback.timestamp <= completed_at
                or max(abs(feedback.left_forward_rpm), abs(feedback.right_forward_rpm)) > 2):
            owner._short_follow_sensor_settle = None
            return False
        prior = getattr(owner, "_short_follow_sensor_settle", None)
        count = 1 if prior is None or prior[0] != epoch else prior[2] + (feedback.timestamp != prior[1])
        owner._short_follow_sensor_settle = (epoch, feedback.timestamp, count)
        if count < 2:
            return False
        owner._brake_hold_active = False
        owner._brake_hold_stop_mode = None
        owner._brake_hold_label = "brake"
        owner._last_brake_hold_send_ts = 0.
        owner._short_follow_sensor_settle = None
        self.logger.info("short_follow_sensor_hold_released reason=%s epoch=%s quiet_samples=%s",
                         reason, epoch, count)
        return True

    def _retire_legacy_normal(self):
        """Retire old plans, never send a transitional zero packet."""
        owner = self.owner
        pending = getattr(owner, "action_queue", None)
        pending_lock = getattr(owner, "action_queue_lock", None)
        if pending is not None and pending_lock is not None:
            with pending_lock:
                protected = []
                while True:
                    try:
                        command = pending.get_nowait()
                    except queue.Empty:
                        break
                    if isinstance(command, ActionCommandSnapshot) and command.protected_stop:
                        protected.append(command)
                for command in protected:
                    pending.put_nowait(command)
        # Do not call _clear_lateral_intent: its legacy implementation creates
        # another motor action. The paired writer replaces BOTH axes at once.
        store = getattr(owner, "_lateral_intent_store", None)
        if store is not None:
            store.clear()
        owner._action_command_revision = int(getattr(owner, "_action_command_revision", 0)) + 1
        owner._lateral_yaw_revision = int(getattr(owner, "_lateral_yaw_revision", 0)) + 1
        owner._depth30_linear_snapshot = None
        owner._depth30_linear_timing = None
        owner._last_depth30_translation_kind = None
        owner._last_depth30_translation_ts = 0.0
        owner._near_yaw_park_request = None
        owner._follow_distance_hold = None
        owner._lateral_intent_owned_frame = -1
        owner._lateral_turn_response_policy = None
        owner._forward_speed_latched_percent = None
        owner._search_handoff_uid = None
        owner._search_handoff_observation = None
        owner._search_handoff_moving_evidence = None
        owner._search_handoff_moving_active = False
        ctl = owner._follow_controller
        ctl._normal_parking_uid = None
        ctl._target_stop_latched = False
        ctl._reverse_active = False
        ctl.last_steering_pid_result = None
        reset_stale = getattr(ctl, "_reset_stale_direction_recovery", None)
        if callable(reset_stale):
            reset_stale("short_follow_verified_target")

    def _remember_visible(self, frame, target, control_source, now):
        owner = self.owner
        ctl = owner._follow_controller
        # A Depth task may finish with an older ROI than the latest visual
        # observation. Its complete wheel plan uses that ROI, but must not
        # roll back the direction history used by subsequent search.
        if control_source != "depth30":
            ctl.last_selected_target = target
            selection_capture = getattr(ctl, "_selection_capture", None)
            if callable(selection_capture):
                ctl._last_visual_selection_capture = selection_capture(frame)
            record = getattr(ctl, "_record_target_direction_evidence", None)
            if callable(record):
                record(frame, target, reliable=True)
            ctl.last_person_center_x = target.center[0]
            owner.last_person_center_x = target.center[0]
            ctl._lost_started_at = None
            ctl._lost_exit_direction = None
            ctl.lost_confirm_frames = 0
            ctl._has_seen_person = True
            owner.lost_confirm_frames = 0
            owner._waiting_lost_confirm = False
        owner.search_state = "none"
        owner.search_direction = None
        owner.person_detected_flag = False
        owner.stop_action_execution = False
        owner._use_soft_stop_next = False
        owner._current_rotate_raw_target = 0
        owner._current_rotate_raw_source = "short_follow"
        owner._current_rotate_pulse_enabled = False

    def handle(self, frame, target, *, is_fresh_depth, control_source,
               target_steerable, low_quality_visible, now):
        """True means the normal frame must not enter ANY legacy motor path.

        Startup/search identity selection remains with the original controller.
        Once it selects a verified UID, even the first normal frame is consumed
        here before its legacy PI/actions can publish a high-speed grant.
        """
        owner = self.owner
        ctl = owner._follow_controller
        uid = getattr(ctl, "active_target_id", None)
        if not self.controller.config.enabled:
            return False
        # ``now`` may have been taken before the latest off-lock identity
        # publication. Read its complete evidence first, then sample time;
        # retain a later caller clock but never judge a new proof using an
        # earlier control-entry time. All acquisition/deadline fields stay
        # unchanged, so this does not lengthen vision or Depth authorization.
        evidence, evidence_now = read_visual_identity_evidence(owner)
        now = max(now, evidence_now)
        proof = evidence.observation
        if self.hold_explicit_stop(frame.capture_timestamp, now):
            owner._short_follow_handled_frame = self.owned
            return self.owned
        hard = bool(frame.hazard.active or any((frame.obstacles.front,
                    frame.obstacles.left, frame.obstacles.right))
                    or getattr(owner, "_runtime_shutdown_requested", False)
                    or not getattr(owner, "running", False))
        if hard:
            self.deactivate("safety_or_shutdown", now)
            return False  # Existing explicit safety/STOP path still runs.
        normal = bool(type(uid) is int and uid > 0 and target is not None
                      and target.track_id == uid and target_steerable
                      and not low_quality_visible
                      and getattr(ctl, "search_state", "none") == "none")
        if not normal:
            prior = self.controller.snapshot()
            if (prior.active and prior.uid == uid and target is None
                    and getattr(ctl, "search_state", "none") == "none"
                    and not low_quality_visible and target_steerable
                    and prior.plan is not None and prior.plan.valid(now)
                    and evidence.live(uid, now)):
                # A single missing observation is not a negative identity
                # verdict. Keep ONLY the original plan until its fixed expiry.
                owner._short_follow_handled_frame = True
                return True
            self.deactivate("identity_or_search_handoff", now)
            return False
        if (proof is False or not evidence.motion_identity_live(uid, now)
                or (isinstance(proof, ValidatedVisualObservation) and proof.uid != uid)):
            self.deactivate("identity_rejected", now)
            return False
        self._remember_visible(frame, target, control_source, now)
        before = self.controller.snapshot().plan
        ds = frame.distance_state
        stamp = ds.sample_timestamp
        proof_ok = bool(isinstance(proof, ValidatedVisualObservation)
                        and proof.continuation_sample_timestamp is None
                        and evidence.permits_depth(uid, stamp, now))
        # The first enrolled frame may precede publication of its independent
        # identity lease. Own the zero/wait state; do not fall back to legacy PI.
        depth_pending = bool(getattr(owner, "_reacquire_depth_pending", False))
        lateral_confirmation = getattr(owner, "_current_search_reacquire_lateral_confirmation", None)
        confirmed_lateral = bool(depth_pending and callable(lateral_confirmation)
            and lateral_confirmation(uid, frame.capture_frame_id, frame.capture_timestamp, now))
        can_update = bool(is_fresh_depth and proof_ok and (not depth_pending or confirmed_lateral))
        awaiting_brake = False
        if getattr(owner, "_brake_hold_active", False):
            # Normal UID ownership cannot fall back into legacy PI just to
            # release a prior search stop (that could emit its first 180RPM
            # grant before this adapter runs again). Wait as paired/no-plan;
            # the existing physical settling contract alone may release it.
            observation = getattr(target, "depth_observation", None)
            distances = (frame.distance_m, ds.raw_distance_m)
            config = self.controller.config
            stop_at = config.target_distance_m + config.stop_margin_m
            ranges_valid = all(isinstance(d, (int, float)) and not isinstance(d, bool)
                               and math.isfinite(d) and d > 0 for d in distances)
            # A completed search STOP may hand off to a near-distance pivot,
            # not only to forward motion. This is solely release eligibility:
            # identity, post-STOP freshness and actual quiet confirmation still
            # go through the existing settlement contract below.
            resume_geometry = ranges_valid and (
                all(d > stop_at for d in distances)
                or config.pivot_allowed(min(distances),
                    target.center[0] / frame.width if frame.width > 0 else float("nan")))
            eligible = bool(can_update and getattr(owner, "_brake_hold_label", "") == "search_reacquire_brake"
                and observation is not None and observation.source == "yolo_detector"
                and observation.target_id == uid and observation.capture_frame_id == frame.capture_frame_id
                and observation.capture_timestamp == frame.capture_timestamp
                and sum(p.track_id == uid for p in frame.persons) == 1
                and ds.safety_distance_m is None
                and resume_geometry)
            release = getattr(getattr(owner, "_action_runtime", None),
                              "release_settled_search_brake_for_depth", None)
            pending = getattr(getattr(owner, "_action_runtime", None),
                              "search_reacquire_brake_pending", None)
            awaiting_stop = bool(eligible and callable(pending)
                                and pending(capture_timestamp=frame.capture_timestamp))
            released = bool(eligible and not awaiting_stop and callable(release) and release(
                uid=uid, capture_id=frame.capture_frame_id,
                capture_timestamp=frame.capture_timestamp, sample_timestamp=stamp,
                depth_max_age=self.controller.config.depth_ttl_sec,
                image_max_age=self.controller.config.visual_ttl_sec))
            if not released:
                awaiting_brake = True
                can_update = False
        safety_distance = ds.safety_distance_m
        # Share the writer's mailbox lock across activation and publication.
        # A valid search handoff must never expose an active/empty mailbox
        # whose awaiting_observation state could send an intervening STOP.
        # Brake settlement above may touch motor state and stays outside it.
        state = self.controller.snapshot()
        if not state.active or state.uid != uid:
            # Queue/store retirement may wait for another owner. Keep those
            # locks outside the short mailbox transaction; the executor must
            # remain able to check its watchdog and safety while we wait.
            self._retire_legacy_normal()
        entered = False
        with self.controller.write_snapshot() as state:
            if not state.active or state.uid != uid:
                self.controller.activate(uid, now)
                entered = True
            if awaiting_brake:
                if getattr(owner, "_brake_hold_label", "") == "search_reacquire_brake":
                    # The existing physical STOP/release contract already
                    # owns this wait. Do not turn each temporary range miss
                    # into a newer hard observation floor: a concurrently
                    # captured post-STOP image may precede this processing
                    # time and still legitimately supply the next fresh Depth.
                    self.controller.wait_for_existing_brake(now)
                else:
                    # Unrelated/unknown safety holds keep hard revocation.
                    self.controller.revoke("await_existing_brake_completion", now)
            if (isinstance(safety_distance, (int, float))
                    and not isinstance(safety_distance, bool)
                    and math.isfinite(safety_distance) and safety_distance > 0):
                self.controller.revoke("depth_safety_observation", now)
            elif not awaiting_brake:
                self.publish_visual_lateral(target, frame.width, getattr(frame, "height", 0),
                    frame.capture_frame_id, frame.capture_timestamp, now=now,
                    feedback=getattr(frame, "steering_feedback", None))
            if can_update and not (isinstance(safety_distance, (int, float))
                    and not isinstance(safety_distance, bool)
                    and math.isfinite(safety_distance) and safety_distance > 0):
                center = self._detector_center(target, frame.width, getattr(frame, "height", 0),
                    frame.capture_frame_id, frame.capture_timestamp)
                observation = ShortFollowObservation(
                    uid=uid, capture_id=int(frame.capture_frame_id),
                    capture_timestamp=float(frame.capture_timestamp),
                    depth_timestamp=stamp, distance_m=frame.distance_m,
                    center_x_ratio=(center if center is not None else
                        target.center[0] / frame.width if frame.width > 0 else float("nan")),
                    raw_distance_m=ds.raw_distance_m,
                )
                self.controller.update(observation, now,
                    longitudinal_allowed=not depth_pending,
                    **self._yaw_feedback(getattr(frame, "steering_feedback", None), now))
        if entered:
            self.logger.info("short_follow_owner uid=%s owner=paired_normal longitudinal=distance_pi "
                             "legacy_pi=False legacy_yaw=False", uid)
        plan = self.controller.snapshot().plan
        live = plan is not None and now < plan.expires_at
        reason = plan.reason if live else "short_follow_wait_measurement"
        owner._vision_control_state = "target_visible_depth_valid" if live else "target_visible_depth_missing"
        owner._last_control_decision_reason = reason
        # Diagnostics only: never read back as another longitudinal/yaw lease.
        owner._current_forward_percent = int(round((plan.left_rpm + plan.right_rpm) / 4.)) if live else 0
        owner._current_steer_base_percent = owner._current_forward_percent
        owner._current_steer_correction_rpm = int(round((plan.left_rpm - plan.right_rpm) / 2.)) if live else 0
        # A signed pivot is motion, but its net longitudinal request is zero.
        # Keep this diagnostic consistent with the wheel pair: classifying a
        # pivot as forward also mislabels recording/control handoff state.
        owner.is_forwarding = bool(live and plan.forwarding)
        decision = ControlDecision(reason=reason, is_forwarding=owner.is_forwarding,
                                   current_forward_percent=owner._current_forward_percent)
        record = getattr(owner, "_publish_follow_recording", None)
        if callable(record):
            record(frame, decision, control_source)
        if plan is not before:
            self.logger.info("short_follow_plan cap=%s uid=%s source=%s sequence=%s left_rpm=%s right_rpm=%s "
                             "reason=%s depth_ts=%s expires_at=%s pi_request_rpm=%s base_rpm=%s "
                             "speed_cap_rpm=%s p_rpm=%s i_rpm=%s integral_dt_sec=%s limit_reason=%s "
                             "identity_renewed=False",
                             frame.capture_frame_id, uid, control_source,
                             None if plan is None else plan.sequence,
                             None if plan is None else plan.left_rpm,
                             None if plan is None else plan.right_rpm,
                             reason, stamp, None if plan is None else plan.expires_at,
                             None if plan is None else plan.base_request_rpm,
                             None if plan is None else plan.base_rpm,
                             None if plan is None else plan.speed_cap_rpm,
                             None if plan is None else plan.p_rpm,
                             None if plan is None else plan.i_rpm,
                             None if plan is None else plan.integral_dt_sec,
                             None if plan is None else plan.limit_reason)
        owner._short_follow_handled_frame = True
        return True
