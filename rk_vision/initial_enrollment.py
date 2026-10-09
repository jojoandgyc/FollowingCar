"""Two-capture proof for the first eligible track, until explicit reset.

This is not a lost-target reacquisition path. A replacement raw track cannot
inherit the pending first identity merely because its appearance is similar.
An actual tracker retirement can open a short, independently checked rebind;
it never replaces the original person's anchors or restarts their lifetime.
An unconfirmed first pose is not a permanent appearance veto: a short,
source-bound colour/geometry bridge can connect it to two clear new crops.
The original anchors never roll, and continuity alone cannot repair a conflict.
"""
import math
import numpy as np

from .template_memory import TemplateMemory


class InitialEnrollment:
    POSE_WINDOW_SEC = 1.5
    TRACK_REBIND_WINDOW_SEC = 4.0

    def __init__(self):
        self.reset()

    def reset(self):
        self.candidate_track_id = None
        self.candidate_feature = None
        self.previous = None
        self.watermark = None
        self.seed = None
        self.candidate_color = None
        self.last_observation = None
        self.pose_continuous = False
        self.pose_clipped_side = None
        self.last_evidence = {}
        self._track_lifecycle = None
        self._lifecycle_watermark = None
        self._candidate_retired = None
        self._rebind_previous = None

    def observe_track_lifecycle(self, before, after, *, frame_index, metadata, observed_at):
        """Receive actual DeepSORT lifecycle events, not detector assertions.

        Only the wrapper owning the tracker calls this after its update. The
        frame binding prevents a stale deletion receipt from authorizing a
        later observation; reset() also discards all receipts.
        """
        try:
            cap, stamp = int(metadata['capture_frame_id']), float(metadata['capture_timestamp'])
            frame = int(frame_index)
            now = float(observed_at)
            if (cap <= 0 or not math.isfinite(stamp) or stamp <= 0
                    or not math.isfinite(now) or not 0 <= now - stamp <= .35):
                raise ValueError('invalid lifecycle capture')
            prior = self._lifecycle_watermark
            if prior and (frame <= prior[0] or cap <= prior[1] or stamp <= prior[2]):
                self._track_lifecycle = None
                self._rebind_previous = None
                return
            active = frozenset(int(track) for track in after)
            self._track_lifecycle = (frame, cap, stamp, active)
            self._lifecycle_watermark = (frame, cap, stamp)
            owner = self.candidate_track_id
            if (owner is not None and owner in before and owner not in active
                    and self.seed is not None and cap > self.seed[1]
                    and stamp > self.seed[2]):
                self._candidate_retired = (frame, cap, stamp)
                self._rebind_previous = None
            elif owner in active:
                # Raw IDs should not be reused, but an observed live owner
                # always revokes an earlier retirement claim.
                self._candidate_retired = None
                self._rebind_previous = None
        except (KeyError, TypeError, ValueError, OverflowError):
            self._track_lifecycle = None
            self._rebind_previous = None

    def _observe_retired_track_rebind(self, current, color, *, weak_observation):
        age = current[2] - self.seed[2]
        full_distance = self._distance(current[5], self.candidate_feature)
        color_distance = self._distance(color, self.candidate_color)
        lifecycle = self._track_lifecycle
        retirement = self._candidate_retired
        blocker = next((reason for failed, reason in (
            (retirement is None, 'original_track_not_retired'),
            (lifecycle is None or lifecycle[:3] != (current[3], current[1], current[2]),
             'track_lifecycle_not_current'),
            (lifecycle is not None and lifecycle[3] != frozenset((current[0],)),
             'other_live_tracks'),
            (retirement is not None and (current[1] <= retirement[1]
                                         or current[2] <= retirement[2]),
             'rebind_capture_not_after_retirement'),
            (not 0 < age <= self.TRACK_REBIND_WINDOW_SEC, 'startup_anchor_expired'),
            (weak_observation, 'rebind_crop_weak'),
            (full_distance is None or full_distance > .20, 'rebind_appearance_conflict'),
            (color_distance is None, 'rebind_color_unavailable'),
            (color_distance is not None and color_distance > .04, 'rebind_color_conflict'),
            (not self._geometry(current[4], self.seed[4]), 'rebind_geometry_conflict'),
        ) if failed), None)
        self.last_evidence.update(
            path='retired_track_rebind', seed_cap=self.seed[1], seed_age_sec=age,
            candidate_track_id=self.candidate_track_id, rebind_track_id=current[0],
            retired_capture=None if retirement is None else retirement[1],
            anchor_full_distance=full_distance, anchor_color_distance=color_distance,
            bridge_blocker=blocker)
        if blocker:
            return self._reject('initial_candidate_track_changed')
        old = self._rebind_previous
        pair_distance = None if old is None else self._distance(current[5], old[5])
        ready = bool(self._continuous(old, current) and pair_distance is not None
                     and pair_distance <= .20)
        self.previous = None
        self.pose_continuous = False
        self._rebind_previous = current
        reason = 'created_confirmed' if ready else 'pending_initial_track_rebind'
        self.last_evidence.update(reason=reason, confirmation_streak=2 if ready else 1,
                                  pair_full_distance=pair_distance)
        return ready, reason

    @staticmethod
    def _vector(feature):
        try:
            value = np.asarray(feature, dtype='float32').reshape(-1)
            norm = float(np.linalg.norm(value))
            if not np.isfinite(value).all() or not math.isfinite(norm) or norm <= 1e-12:
                return None
            return value / norm
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _distance(a, b):
        if a is None or b is None or a.shape != b.shape:
            return None
        return max(0., float(1. - a.dot(b)))

    @staticmethod
    def _geometry(a, b):
        area = (a[2]-a[0])*(a[3]-a[1])
        old_area = (b[2]-b[0])*(b[3]-b[1])
        intersect = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(
            0., min(a[3], b[3])-max(a[1], b[1]))
        return (area > 0 and old_area > 0
                and intersect/(area+old_area-intersect) >= .5
                and min(area, old_area)/max(area, old_area) >= .6)

    @classmethod
    def _continuous(cls, old, current):
        return bool(old and old[0] == current[0]
                    and current[1] > old[1] and current[3] == old[3]+1
                    and 0 < current[2]-old[2] <= .35
                    and cls._geometry(current[4], old[4]))

    @staticmethod
    def _weak_crop_observable(m):
        # A single clipped side during an uninterrupted startup pose change
        # may preserve observation continuity. It cannot seed or confirm UID.
        if (m.get('is_fresh') is not True or m.get('bbox_quality_tier') != 'weak'
                or m.get('quality_bbox_reason') != 'edge_touch>2'
                or m.get('initial_base_quality_ok', True) is not True):
            return False
        try:
            x1, y1, x2, y2 = map(float, m['detector_bbox'])
            w, h = float(m['image_width']), float(m['image_height'])
            return bool(all(math.isfinite(v) for v in (x1,y1,x2,y2,w,h))
                        and w > 0 and h > 0 and 0 <= x1 < x2 <= w
                        and 0 <= y1 < y2 <= h and x2-x1 >= 80 and y2-y1 >= 160
                        and (y2-y1)/h >= .85 and .25 <= (x2-x1)/(y2-y1) <= 1.2
                        and ((x1 <= .02*w) != (w-x2 <= .02*w)))
        except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
            return False

    def _reject(self, reason):
        self.previous = None
        self._rebind_previous = None
        self.pose_continuous = False
        self.last_evidence.update(reason=reason, confirmation_streak=0, pose_continuous=False)
        return False, reason

    def observe(self, track, feature, metadata):
        m = metadata or {}
        self.last_evidence = dict(seed_cap=None if self.seed is None else self.seed[1],
                                  candidate_track_id=self.candidate_track_id,
                                  path='original_appearance', confirmation_streak=0)
        rejection = None
        if m.get('candidate_count') != 1:
            rejection = 'initial_candidate_ambiguous'
        elif not TemplateMemory.initial_crop_usable(m, allow_near_vertical_crop=True):
            rejection = 'initial_crop_incomplete'
        try:
            cap, stamp = int(m['capture_frame_id']), float(m['capture_timestamp'])
            frame = int(m['frame_index'])
            if cap <= 0 or not math.isfinite(stamp) or stamp <= 0:
                raise ValueError('invalid capture')
        except (KeyError, TypeError, ValueError, OverflowError):
            return self._reject(rejection or 'initial_evidence_unavailable')
        if self.watermark and (cap <= self.watermark[0] or stamp <= self.watermark[1]):
            # A replay neither advances nor replaces the current proof. Keep
            # the crop/ambiguity diagnostic for other boxes in the same frame.
            self.last_evidence['reason'] = rejection or 'initial_capture_not_new'
            return False, self.last_evidence['reason']
        # Consume rejected physical captures too: changing their metadata on a
        # repeated call must not turn them into new enrollment evidence.
        self.watermark = (cap, stamp)
        weak_observation = (rejection == 'initial_crop_incomplete'
                            and m.get('candidate_count') == 1
                            and self._weak_crop_observable(m))
        if rejection is not None and not weak_observation:
            return self._reject(rejection)
        try:
            value = self._vector(feature)
            if value is None:
                raise ValueError('invalid feature')
            box = tuple(map(float, m['detector_bbox']))
        except (KeyError, TypeError, ValueError, OverflowError):
            return self._reject('initial_evidence_unavailable')
        current = (track, cap, stamp, frame, box, value.copy())
        color = (self._vector(m.get('initial_color_feature'))
                 if m.get('initial_color_source') == 'hsv_crop_v1_bgr' else None)
        if color is not None and color.shape != (16,):
            color = None
        if self.candidate_track_id is None:
            if weak_observation:
                return self._reject(rejection)
            self.candidate_track_id = track
            self.candidate_feature = value.copy()
            self.candidate_color = None if color is None else color.copy()
            self.seed = current
            self.last_observation = current
            self.pose_continuous = True
        elif track != self.candidate_track_id:
            return self._observe_retired_track_rebind(
                current, color, weak_observation=weak_observation)
        self._rebind_previous = None
        if weak_observation:
            side = 'left' if box[0] <= .02*float(m['image_width']) else 'right'
            if self.pose_clipped_side is not None and side != self.pose_clipped_side:
                self.last_evidence['bridge_blocker'] = 'clipped_side_changed'
                return self._reject('initial_candidate_appearance_mismatch')
            self.pose_clipped_side = side
        full_distance = self._distance(value, self.candidate_feature)
        color_distance = self._distance(color, self.candidate_color)
        age = stamp-self.seed[2]
        first = cap == self.seed[1]
        geometry_ok = self._geometry(box, self.seed[4])
        self.pose_continuous = bool(self.pose_continuous and (
            first or self._continuous(self.last_observation, current)))
        self.last_observation = current
        self.last_evidence.update(seed_cap=self.seed[1], candidate_track_id=self.candidate_track_id,
                                  seed_age_sec=age, anchor_full_distance=full_distance,
                                  anchor_color_distance=color_distance,
                                  anchor_geometry_ok=geometry_ok,
                                  pose_continuous=self.pose_continuous)
        original_match = full_distance is not None and full_distance <= .20
        # Neither colour nor two observations of the same candidate suffice
        # alone. Keep the original appearance ceiling and spatial anchor,
        # an unbroken unique-track corridor, and an absolute (not rolling) TTL.
        bridge_ok = bool(self.pose_continuous and 0 <= age <= self.POSE_WINDOW_SEC
                         and geometry_ok and full_distance is not None and full_distance <= .40
                         and color_distance is not None
                         and color_distance <= (.06 if weak_observation else .04))
        bridge_blocker = next((reason for failed, reason in (
            (full_distance is None or full_distance > .40, 'appearance_conflict'),
            (not geometry_ok, 'geometry_conflict'),
            (color_distance is None, 'color_unavailable'),
            (color_distance is not None and color_distance > (.06 if weak_observation else .04),
             'color_conflict'),
            (not self.pose_continuous, 'continuity_broken'),
            (age > self.POSE_WINDOW_SEC, 'pose_window_expired'),
        ) if failed), None)
        self.last_evidence['bridge_blocker'] = bridge_blocker
        if (weak_observation or not original_match) and not bridge_ok:
            return self._reject('initial_candidate_appearance_mismatch')
        if weak_observation:
            self.previous = None
            self.last_evidence.update(path='pose_observation_only', reason=rejection)
            return False, rejection
        path = 'original_appearance' if original_match else 'anchored_pose_pair'
        old = self.previous
        self.previous = current
        pair_distance = None if old is None else self._distance(value, old[5])
        continuous = bool(self._continuous(old, current)
                          and pair_distance is not None and pair_distance <= .20)
        # A local appearance conflict cannot be washed away by simply
        # repeating the new look, even when its clothing colour is similar.
        if (old is not None and old[1] != self.seed[1]
                and not original_match and not continuous):
            self.last_evidence['bridge_blocker'] = 'local_appearance_conflict'
            return self._reject('initial_candidate_appearance_mismatch')
        reason = 'created_confirmed' if continuous else 'pending_initial_identity'
        self.last_evidence.update(path=path, reason=reason, pair_full_distance=pair_distance,
                                  confirmation_streak=2 if continuous else 1)
        return continuous, reason
