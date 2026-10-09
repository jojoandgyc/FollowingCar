"""Bounded, capture-time based memory of *approved* template observations.

This module neither assigns identities nor refreshes motion authority. Queries
never renew a template. The identity bank alone decides whether learning is safe.
"""
from dataclasses import dataclass, field
import math
import numpy as np


def timestamp(metadata):
    try:
        value = float((metadata or {}).get("capture_timestamp"))
        return value if math.isfinite(value) and value >= 0 else None
    except (TypeError, ValueError):
        return None


@dataclass
class TemplateMemory:
    recent_sec: float = 30.0
    archive_sec: float = 120.0
    capacity: int = 8
    archive_capacity: int = 6
    recent: dict = field(default_factory=lambda: {"strong": [], "partial": []})
    # Approved regional representatives survive the recent window, but are
    # never merged into recent evidence or used to renew its age/permissions.
    representatives: dict = field(default_factory=lambda: {"strong": [], "partial": []})
    watermark: float = -1.0
    last_learning: dict = field(default_factory=dict)
    expired_count: int = 0

    def __post_init__(self):
        if not (math.isfinite(self.recent_sec) and math.isfinite(self.archive_sec)
                and 0 < self.recent_sec <= self.archive_sec):
            raise ValueError("template windows must be finite, positive and recent <= archive")

    def advance(self, metadata):
        stamp = timestamp(metadata)
        if stamp is None:
            return
        self.watermark = max(self.watermark, stamp)
        for tier, rows in self.recent.items():
            keep = [r for r in rows if 0 <= self.watermark - r[1]["capture_timestamp"] <= self.recent_sec]
            self.expired_count += len(rows) - len(keep)
            self.recent[tier] = keep
        for tier, rows in self.representatives.items():
            self.representatives[tier] = [r for r in rows
                if 0 <= self.watermark - r[1]["capture_timestamp"] <= self.archive_sec]

    def remember(self, feature, metadata, tier):
        stamp = timestamp(metadata)
        cap = (metadata or {}).get("capture_frame_id")
        # Legacy/missing timestamps cannot become apparently fresh evidence.
        if (stamp is None or cap is None or metadata.get("is_fresh") is not True
                or metadata.get("identity_control_rejected")
                or metadata.get("quality_bbox_ok") is False
                or metadata.get("bbox_quality_tier") == "reject"
                or (tier == "strong" and metadata.get("bbox_quality_tier") == "weak")
                or metadata.get("search_observation_only")
                or metadata.get("preferred_search_low_confidence")):
            return False
        previous = self.last_learning.get(tier)
        if stamp < self.watermark or (previous and (stamp <= previous[0] or cap <= previous[1])):
            return False
        self.advance(metadata)
        value = np.asarray(feature, dtype="float32").reshape(-1).copy()
        norm = float(np.linalg.norm(value))
        if not np.isfinite(value).all() or norm <= 1e-12:
            return False
        value /= norm
        rows = self.recent[tier]
        # Replace a duplicate with the ACTUAL new approved crop, not a renewed
        # timestamp on the old embedding. Preserve distinct recent poses.
        similar = [i for i, (v, _) in enumerate(rows)
                   if v.size == value.size and float(1 - v.dot(value)) < .02
                   and self.retention_key(rows[i][1]) == self.retention_key(metadata)]
        if similar:
            rows.pop(similar[0])
        rows.append((value, dict(metadata)))
        # Preserve a bounded representative of each observed crop coverage.
        # Only trusted remember() calls enter here; query hits never renew it.
        while len(rows) > max(1, self.capacity):
            keys = [self.retention_key(m) for _, m in rows]
            redundant = [i for i, key in enumerate(keys) if keys.count(key) > 1]
            rows.pop(redundant[0] if redundant else 0)
        self.recent[tier] = rows
        if self.partial_usable(metadata) and self.coverage_key(metadata) is not None:
            reps = self.representatives[tier]
            # One approved representative per coverage+side; no candidate hit
            # touches its timestamp, and no weak sample enters this archive.
            key = self.retention_key(metadata)
            reps[:] = [r for r in reps if self.retention_key(r[1]) != key]
            reps.append((value.copy(), dict(metadata)))
            del reps[:-max(1, self.archive_capacity)]
        self.last_learning[tier] = (stamp, cap)
        return True

    def evidence(self, feature, metadata, tier="strong", *, reliable_only=False,
                 comparable_only=False, shape_reference=None):
        stamp = timestamp(metadata)
        query = None if feature is None else np.asarray(feature, dtype="float32").reshape(-1)
        rows = []
        if stamp is not None and query is not None and np.isfinite(query).all():
            query = query / max(float(np.linalg.norm(query)), 1e-12)
            rows = [(float(1 - value.dot(query)), info) for value, info in self.recent[tier]
                    if value.size == query.size and 0 <= stamp - info["capture_timestamp"] <= self.recent_sec
                    and (not reliable_only or self.partial_usable(info))]
        rows.sort(key=lambda r: r[0])
        strict_caps = {r[1]["capture_frame_id"] for r in rows
                       if self.partial_shape_consistent(metadata, r[1]) is True} if tier == "partial" else set()
        retained_caps = set()
        if tier == "partial" and shape_reference:
            previous = shape_reference["metadata"]
            previous_ts = timestamp(previous)
            # Read-only hysteresis: only an independently verified, same-track
            # capture may retain an already admitted template. Never renew it.
            continuous = (stamp is not None and previous_ts is not None
                          and 0 < stamp - previous_ts <= .25
                          and metadata.get("is_fresh") is True
                          and metadata.get("track_id") == previous.get("track_id")
                          and self.coverage_key(metadata) == self.coverage_key(previous))
            if continuous:
                for _, info in rows:
                    ratio = self.partial_shape_ratio(metadata, info)
                    # Side-on narrowing may retain an already admitted torso
                    # reference only after independent partial reacquisition.
                    # This never admits a new template, old timestamp or crop
                    # from the opposite edge, and current descriptors are still
                    # checked against the confirmation threshold by the bank.
                    side_continuation = (shape_reference.get('partial_continuation') is True
                                         and self.side_shape_continuous(metadata, previous))
                    floor = .45 if side_continuation else .48
                    if (info["capture_frame_id"] in shape_reference.get("comparable_caps", ())
                            and ratio is not None and ratio >= floor):
                        retained_caps.add(info["capture_frame_id"])
        comparable = [r for r in rows if self.coverage_key(metadata) is not None
                      and self.coverage_key(metadata) == self.coverage_key(r[1])
                      and (tier != "partial" or self.crop_sides(metadata) == self.crop_sides(r[1]))
                      and (tier != "partial" or r[1]["capture_frame_id"] in strict_caps | retained_caps)]
        comparison_mode = "exact_coverage"
        scale_caps = []
        scale_remaining_ms = None
        pose_caps = []
        pose_remaining_ms = None
        if tier == "partial" and not comparable:
            comparable = [r for r in rows if self.vertical_border_comparable(metadata, r[1])]
            comparison_mode = "vertical_border_bridge" if comparable else "unavailable"
            if not comparable and shape_reference and shape_reference.get('scale_continuation'):
                previous = shape_reference['metadata']
                started = shape_reference.get('scale_started', timestamp(previous))
                if (stamp is not None and started is not None and 0 < stamp-started <= 2.
                        and 0 < stamp-timestamp(previous) <= .25
                        and metadata.get('capture_frame_id', 0) > previous.get('capture_frame_id', 0)
                        and metadata.get('track_id') == previous.get('track_id')
                        and self.scale_sequence_usable(metadata, previous)):
                    comparable = [r for r in rows
                        if r[1]['capture_frame_id'] in shape_reference.get('comparable_caps', ())
                        and self.scale_template_usable(metadata, r[1])]
                    scale_caps = [r[1]['capture_frame_id'] for r in comparable]
                    if comparable:
                        comparison_mode = 'verified_scale_continuation'
                        scale_remaining_ms = max(0., 2000.-(stamp-started)*1000.)
            if (shape_reference and shape_reference.get('pose_continuation') is True
                    and (not comparable or shape_reference.get('pose_scale_continuation') is True)):
                previous = shape_reference.get('metadata') or {}
                previous_ts = timestamp(previous)
                started = timestamp({'capture_timestamp': shape_reference.get('pose_started')})
                # This source-bound allowance is supplied only by IdentityBank
                # after its same-binding/geometry/competition checks. It is an
                # observation budget, not a template age or motor deadline.
                pose_gap = shape_reference.get('pose_sample_gap_sec', .25)
                if (isinstance(pose_gap, bool) or not isinstance(pose_gap, (int, float))
                        or not math.isfinite(pose_gap) or not 0 < pose_gap <= .35):
                    pose_gap = .25
                gap_ok = (stamp is not None and previous_ts is not None
                          and 0 < stamp-previous_ts
                          and (stamp < previous_ts+pose_gap if pose_gap != .25
                               else stamp-previous_ts <= .25))
                try:
                    capture = float(metadata.get('capture_frame_id'))
                    previous_capture = float(previous.get('capture_frame_id'))
                    new_capture = (math.isfinite(capture) and math.isfinite(previous_capture)
                                   and capture > previous_capture)
                except (TypeError, ValueError):
                    new_capture = False
                # The bank authorizes this narrowly scoped, fixed-duration
                # proof. Queries may retain only its original template set;
                # they cannot admit a new region or renew a template's age.
                if (stamp is not None and previous_ts is not None and started is not None
                        and started <= previous_ts and 0 < stamp-started <= 2.
                        and gap_ok and new_capture
                        and metadata.get('track_id') is not None
                        and metadata.get('track_id') == previous.get('track_id')
                        and self.pose_sequence_usable(metadata, previous)):
                    pose_rows = [r for r in rows
                        if r[1]['capture_frame_id'] in shape_reference.get('pose_caps', ())
                        and self.pose_template_usable(metadata, r[1],
                            allow_scale=shape_reference.get('pose_scale_continuation') is True)]
                    existing_caps = {r[1]['capture_frame_id'] for r in comparable}
                    retained = [r for r in pose_rows if r[1]['capture_frame_id'] not in existing_caps]
                    pose_caps = [r[1]['capture_frame_id'] for r in retained]
                    if retained:
                        # An incidental normal vertical-border match must not
                        # evict the fixed, independently admitted pose set.
                        # Exact-region evidence already won above this block;
                        # it cannot be hidden by this bounded retention.
                        comparable = sorted(comparable + retained, key=lambda r: r[0])
                        comparison_mode = 'verified_pose_continuation'
                        pose_remaining_ms = max(0., 2000.-(stamp-started)*1000.)
        if comparable_only:
            rows = comparable
        return {"count": len(rows), "distance": rows[0][0] if rows else None,
                "query_coverage": self.coverage_key(metadata),
                "winner_coverage": self.coverage_key(rows[0][1]) if rows else None,
                "comparable_count": len(comparable),
                "comparable_distance": comparable[0][0] if comparable else None,
                "comparable_caps": [r[1]["capture_frame_id"] for r in comparable],
                "comparison_mode": comparison_mode,
                "scale_bridge_caps": scale_caps,
                "scale_retention_remaining_ms": scale_remaining_ms,
                "pose_bridge_caps": pose_caps,
                "pose_retention_remaining_ms": pose_remaining_ms,
                "shape_hysteresis_caps": sorted(retained_caps - strict_caps),
                "winner_cap": rows[0][1]["capture_frame_id"] if rows else None,
                "winner_age_sec": stamp - rows[0][1]["capture_timestamp"] if rows else None,
                "winner_crop_shape_consistent": (
                    self.partial_shape_consistent(metadata, rows[0][1])
                    if tier == "partial" and rows else None)}

    @staticmethod
    def crop_sides(metadata):
        try:
            x1, _, x2, _ = metadata['detector_bbox']
            w = float(metadata['image_width'])
            if not all(math.isfinite(float(v)) for v in (x1, x2, w)) or w <= 0:
                return None
            return (x1 <= .02*w, w-x2 <= .02*w)
        except (KeyError, TypeError, ValueError):
            return None

    @classmethod
    def representative_key(cls, metadata):
        return cls.coverage_key(metadata), cls.crop_sides(metadata)

    @classmethod
    def retention_key(cls, metadata):
        """Reserve bounded slots for independently approved viewing scales."""
        try:
            b = metadata['detector_bbox']
            height = (b[3]-b[1])/float(metadata['image_height'])
            scale = 'near' if height >= .70 else 'mid' if height >= .40 else 'far'
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            scale = 'unknown'
        return cls.representative_key(metadata), scale

    @classmethod
    def scale_crop_usable(cls, m):
        if not cls.partial_usable(m) or cls.crop_sides(m) != (False, False):
            return False
        try:
            x1,y1,x2,y2 = m['detector_bbox']
            return (0 <= x1 < x2 <= m['image_width'] and 0 <= y1 < y2 <= m['image_height']
                    and x2-x1 >= 64 and y2-y1 >= 160
                    and .30 <= (x2-x1)/(y2-y1) <= .85
                    and float(m.get('detector_confidence', 0)) >= .80)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return False

    @classmethod
    def pose_crop_usable(cls, metadata):
        """Narrow side-view crop usable only by a separately verified proof."""
        if (not cls.partial_usable(metadata) or metadata.get('identity_control_rejected')
                or cls.crop_sides(metadata) != (False, False)):
            return False
        try:
            x1, y1, x2, y2 = map(float, metadata['detector_bbox'])
            w, h = float(metadata['image_width']), float(metadata['image_height'])
            confidence = float(metadata.get('detector_confidence', 0))
            return (all(math.isfinite(v) for v in (w, h, confidence))
                    and 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
                    and x2-x1 >= 64 and y2-y1 >= 160
                    and .25 <= (x2-x1)/(y2-y1) <= .85 and confidence >= .80)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return False

    @classmethod
    def pose_sequence_usable(cls, query, previous, *, entering=False):
        """Smooth same-view sequence across at most one nearby vertical edge.

        This is a crop-continuity proxy, not pose recognition or UID proof.
        Entry requires a vertical label change; continuation may keep the
        same labels. Scale-only and global border admission remain stricter.
        """
        if not cls.pose_crop_usable(query) or not cls.pose_crop_usable(previous):
            return False
        if (query['image_width'] != previous['image_width']
                or query['image_height'] != previous['image_height']):
            return False
        q, p = query['detector_bbox'], previous['detector_bbox']
        qh, ph = q[3]-q[1], p[3]-p[1]
        qa, pa = (q[2]-q[0])/qh, (p[2]-p[0])/ph
        if min(qh, ph)/max(qh, ph) < .85 or min(qa, pa)/max(qa, pa) < .80:
            return False
        qkey, pkey = cls.coverage_key(query), cls.coverage_key(previous)
        changed = [i for i, (a, b) in enumerate(zip(qkey.split('_')[:2], pkey.split('_')[:2]))
                   if a != b]
        if len(changed) > 1 or (entering and len(changed) != 1):
            return False
        # A large border jump is not merely quantization around the 2% tag.
        return not changed or abs(q[1 if changed[0] == 0 else 3]
                                  - p[1 if changed[0] == 0 else 3]) <= .05*query['image_height']

    @classmethod
    def pose_template_usable(cls, query, template, *, allow_scale=False):
        """Retain an admitted recent template; never bootstrap one here."""
        if (not cls.pose_crop_usable(query) or not cls.pose_crop_usable(template)
                or cls.partial_shape_ratio(query, template) < .50):
            return False
        qkey, tkey = cls.coverage_key(query), cls.coverage_key(template)
        if sum(a != b for a, b in zip(qkey.split('_')[:2], tkey.split('_')[:2])) > 1:
            return False
        q, t = query['detector_bbox'], template['detector_bbox']
        qh = (q[3]-q[1])/query['image_height']
        th = (t[3]-t[1])/template['image_height']
        # Entry remains .70. Only an already active, source-bound episode can
        # use the existing scale-retention floor after local/cumulative checks.
        return min(qh, th)/max(qh, th) >= (.45 if allow_scale else .70)

    @classmethod
    def pose_scale_sequence_usable(cls, query, previous, origin):
        """Scale during one verified pose epoch; no rolling origin/permission."""
        if not cls.scale_sequence_usable(query, previous):
            return False
        try:
            if (query['image_width'] != origin['image_width']
                    or query['image_height'] != origin['image_height']):
                return False
            q, o = query['detector_bbox'], origin['detector_bbox']
            qh, oh = float(q[3]-q[1]), float(o[3]-o[1])
            qw, ow = float(q[2]-q[0]), float(o[2]-o[0])
            if not all(math.isfinite(v) and v > 0 for v in (qw, ow, qh, oh)):
                return False
            # The immutable origin stores geometry only, not a mutable copy
            # of quality/identity metadata. Its provenance belongs to the bank.
            qa, oa = qw/qh, ow/oh
            return (min(qh, oh) > 0
                    and min(qh, oh)/max(qh, oh) >= .65
                    and min(qa, oa)/max(qa, oa) >= .75)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return False

    @classmethod
    def scale_sequence_usable(cls, query, previous):
        if (not cls.scale_crop_usable(query) or not cls.scale_crop_usable(previous)
                or cls.coverage_key(query) != cls.coverage_key(previous)):
            return False
        q,p = query['detector_bbox'],previous['detector_bbox']
        qh,ph = q[3]-q[1],p[3]-p[1]
        return (min(qh,ph)/max(qh,ph) >= .85
                and cls.partial_shape_ratio(query, previous) >= .75)

    @classmethod
    def scale_template_usable(cls, query, template):
        if (not cls.scale_crop_usable(query) or not cls.scale_crop_usable(template)
                or cls.partial_shape_ratio(query,template) < .70):
            return False
        q,t = cls.coverage_key(query),cls.coverage_key(template)
        # Only the single vertical-boundary difference previously admitted by
        # the normal bridge may be retained, never side fragments/new regions.
        if sum(a != b for a,b in zip(q.split('_')[:2],t.split('_')[:2])) != 1:
            return False
        qb,tb=query['detector_bbox'],template['detector_bbox']
        qh=(qb[3]-qb[1])/query['image_height'];th=(tb[3]-tb[1])/template['image_height']
        return min(qh,th)/max(qh,th) >= .45

    @classmethod
    def vertical_border_comparable(cls, query, template):
        """Conservative crop proxy, NOT anatomical/pose correspondence.

        Only one vertical boundary may differ; no side-cropped fragment or
        substantial scale/shape change can use this secondary selection path.
        Exact comparable evidence always takes precedence, even if conflicting.
        """
        qkey, tkey = cls.coverage_key(query), cls.coverage_key(template)
        if qkey is None or tkey is None or qkey == tkey:
            return False
        ratio = cls.partial_shape_ratio(query, template)
        if (ratio is None or ratio < .75
                or cls.crop_sides(query) != (False, False)
                or cls.crop_sides(template) != (False, False)):
            return False
        if sum(a != b for a, b in zip(qkey.split('_')[:2], tkey.split('_')[:2])) != 1:
            return False
        try:
            sizes = []
            for m in (query, template):
                x1,y1,x2,y2 = map(float, m['detector_bbox'])
                w,h = float(m['image_width']), float(m['image_height'])
                if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
                        and (x2-x1)/w >= .15 and (y2-y1)/h >= .55
                        and .30 <= (x2-x1)/(y2-y1) <= .85):
                    return False
                sizes.append(((x2-x1)/w, (y2-y1)/h))
            # Height checks coverage/scale; aspect already checks narrowing.
            # A separate width-ratio veto double-penalizes normal side views.
            return min(s[1] for s in sizes)/max(s[1] for s in sizes) >= .70
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return False

    def representative_evidence(self, feature, metadata, tier="partial"):
        """Expired-recent representatives are diagnostic support, never recent."""
        stamp = timestamp(metadata)
        out = {"count": 0, "distance": None, "winner_cap": None, "winner_age_sec": None,
               "permission": "observation_only"}
        if stamp is None or feature is None:
            return out
        query = np.asarray(feature, dtype="float32").reshape(-1)
        if not np.isfinite(query).all() or np.linalg.norm(query) <= 1e-12:
            return out
        query = query/max(float(np.linalg.norm(query)), 1e-12)
        rows = [(float(1-v.dot(query)), m) for v,m in self.representatives[tier]
                if v.size == query.size and self.partial_usable(metadata) and self.partial_usable(m)
                and self.recent_sec < stamp-m['capture_timestamp'] <= self.archive_sec
                and ((self.representative_key(metadata) == self.representative_key(m)
                      and self.partial_shape_consistent(metadata,m) is True)
                     or self.vertical_border_comparable(metadata,m))]
        rows.sort(key=lambda r:r[0])
        if rows:
            out.update(count=len(rows), distance=rows[0][0], winner_cap=rows[0][1]['capture_frame_id'],
                       winner_age_sec=stamp-rows[0][1]['capture_timestamp'])
        return out

    def paired_recent_evidence(self, full_feature, partial_feature, metadata, *, shape_reference=None):
        """Same approved capture must support both descriptors; no min mixing."""
        out = {"count": 0, "qualified": False, "winner_cap": None,
               "full_distance": None, "partial_distance": None}
        if full_feature is None or partial_feature is None or not self.partial_usable(metadata):
            return out
        full = np.asarray(full_feature,dtype='float32').reshape(-1).copy()
        part = np.asarray(partial_feature,dtype='float32').reshape(-1).copy()
        if any(not np.isfinite(v).all() or np.linalg.norm(v)<=1e-12 for v in (full,part)):
            return out
        full /= np.linalg.norm(full); part /= np.linalg.norm(part)
        eligible = self.evidence(part,metadata,'partial',reliable_only=True,comparable_only=True,
                                 shape_reference=shape_reference)
        stamps = {(m['capture_frame_id'],m['capture_timestamp']):(v,m)
                  for v,m in self.recent['strong']}
        pairs = []
        for v,m in self.recent['partial']:
            f = stamps.get((m['capture_frame_id'],m['capture_timestamp']))
            if (m['capture_frame_id'] not in eligible['comparable_caps'] or f is None
                    or v.size != part.size or f[0].size != full.size
                    or f[1].get('track_id') != m.get('track_id')
                    or f[1].get('detector_bbox') != m.get('detector_bbox')):
                continue
            fd,pd = float(1-f[0].dot(full)),float(1-v.dot(part))
            pairs.append((fd,pd,m['capture_frame_id']))
        qualified = [p for p in pairs if p[0] <= .30 and p[1] <= .26]
        out['count'] = len(pairs)
        if qualified:
            fd,pd,cap = min(qualified,key=lambda p:p[0]+p[1])
            out.update(qualified=True,winner_cap=cap,full_distance=fd,partial_distance=pd)
        return out

    def continuation_pair_evidence(self, full_feature, partial_feature, metadata,
                                   reliable, *, full_limit):
        """Current evidence from ONE approved recent view, not gallery learning.

        A confirmed source may use this for local continuity. It cannot mix
        an old full-body minimum with a different torso sample, or convert a
        bounded pose/scale retention into an unbounded verification proof.
        Quarantine learning retains its separate, stricter paired thresholds.
        """
        out = dict(qualified_comparison=False, winner_cap=None,
                   full_distance=None, partial_distance=None, count=0)
        if (reliable.get('comparison_mode') not in ('exact_coverage', 'vertical_border_bridge')
                or full_feature is None or partial_feature is None
                or not self.partial_usable(metadata)):
            return out
        full = np.asarray(full_feature, dtype='float32').reshape(-1).copy()
        part = np.asarray(partial_feature, dtype='float32').reshape(-1).copy()
        if any(not np.isfinite(v).all() or np.linalg.norm(v) <= 1e-12 for v in (full, part)):
            return out
        full /= np.linalg.norm(full)
        part /= np.linalg.norm(part)
        stamp = timestamp(metadata)
        retained = (set(reliable.get('shape_hysteresis_caps', ()))
                    | set(reliable.get('pose_bridge_caps', ()))
                    | set(reliable.get('scale_bridge_caps', ())))
        strong = {(m['capture_frame_id'], timestamp(m)): (v, m)
                  for v, m in self.recent['strong']}
        pairs = []
        for v, m in self.recent['partial']:
            ts = timestamp(m)
            f = strong.get((m['capture_frame_id'], ts))
            if (stamp is None or ts is None or not 0 <= stamp-ts <= self.recent_sec
                    or m['capture_frame_id'] not in reliable.get('comparable_caps', ())
                    or m['capture_frame_id'] in retained
                    or not self.partial_usable(m) or f is None
                    or f[1].get('track_id') != m.get('track_id')
                    or f[1].get('detector_bbox') != m.get('detector_bbox')
                    or f[0].size != full.size or v.size != part.size):
                continue
            fd, pd = float(1-f[0].dot(full)), float(1-v.dot(part))
            if fd <= full_limit:
                pairs.append((fd, pd, m['capture_frame_id']))
        out['count'] = len(pairs)
        if pairs:
            fd, pd, cap = min(pairs, key=lambda p: p[1])
            out.update(qualified_comparison=True, winner_cap=cap,
                       full_distance=max(0., fd), partial_distance=max(0., pd))
        return out

    @staticmethod
    def initial_crop_evidence(metadata, *, allow_near_vertical_crop=False):
        """Bounded crop proxy, not anatomical or identity proof.

        Only empty-gallery enrollment opts into the near/vertical crop path.
        Preserve the stricter shape rule for adding other identities.
        """
        m = metadata or {}
        result = dict(usable=False, reason='geometry_unavailable', path=None)
        try:
            x1, y1, x2, y2 = map(float, m['detector_bbox'])
            w, h = float(m['image_width']), float(m['image_height'])
            if (not all(math.isfinite(v) for v in (x1,y1,x2,y2,w,h))
                    or w <= 0 or h <= 0 or not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h)):
                return result
            bw, bh = x2-x1, y2-y1
            aspect = bw / bh
            vertical_cut = y1 <= .02*h or h-y2 <= .02*h
            side_cut = x1 <= .02*w or w-x2 <= .02*w
            result.update(aspect_ratio=aspect, height_ratio=bh/h,
                          area_ratio=bw*bh/(w*h), vertical_cut=vertical_cut,
                          side_cut=side_cut)
            # Report lateral clipping even when it also caused weak quality.
            if side_cut:
                result['reason'] = 'lateral_crop'
            elif m.get('is_fresh') is not True:
                result['reason'] = 'stale_observation'
            elif m.get('quality_bbox_ok') is not True or m.get('bbox_quality_tier') != 'strong':
                result['reason'] = 'bbox_quality'
            elif bw < 80 or bh < 160:
                result['reason'] = 'body_too_small'
            elif .25 <= aspect <= .85:
                result.update(usable=True, reason='eligible', path='standard')
            elif (allow_near_vertical_crop and .85 < aspect <= 1.20
                  and vertical_cut and bh/h >= .85 and bw*bh/(w*h) >= .30):
                result.update(usable=True, reason='eligible', path='near_vertical_crop')
            else:
                result['reason'] = 'aspect_ratio_out_of_range'
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            pass
        return result

    @staticmethod
    def initial_crop_usable(metadata, *, allow_near_vertical_crop=False):
        return TemplateMemory.initial_crop_evidence(
            metadata, allow_near_vertical_crop=allow_near_vertical_crop)['usable']

    @staticmethod
    def coverage_key(metadata):
        """Crop coverage, NOT pose/anatomical identity. Unknown stays unknown.

        Used only to select comparable evidence. A coverage change never
        excuses an already established identity contradiction.
        """
        try:
            m = metadata or {}
            w, h = float(m['image_width']), float(m['image_height'])
            x1, y1, x2, y2 = map(float, m['detector_bbox'])
            if not all(math.isfinite(v) for v in (w,h,x1,y1,x2,y2)) or min(w,h,x2-x1,y2-y1) <= 0:
                return None
            return 'top%d_bottom%d_side%d' % (
                y1 <= .02*h, h-y2 <= .02*h, x1 <= .02*w or w-x2 <= .02*w)
        except (KeyError, TypeError, ValueError):
            return None

    @classmethod
    def side_shape_continuous(cls, query, previous):
        """Bounded crop proxy for an already confirmed track, not pose proof."""
        if (not cls.partial_usable(query) or not cls.partial_usable(previous)
                or cls.coverage_key(query) != cls.coverage_key(previous)
                or cls.crop_sides(query) != (False, False)
                or cls.crop_sides(previous) != (False, False)):
            return False
        q, p = query['detector_bbox'], previous['detector_bbox']
        qh, ph = q[3]-q[1], p[3]-p[1]
        qa, pa = (q[2]-q[0])/qh, (p[2]-p[0])/ph
        return (.25 <= qa <= .85 and .25 <= pa <= .85
                and min(qh, ph)/max(qh, ph) >= .85
                and min(qa, pa)/max(qa, pa) >= .80)

    @classmethod
    def partial_shape_consistent(cls, query, template):
        """Strict admission for comparable fixed-ROI evidence, not identity."""
        ratio = cls.partial_shape_ratio(query, template)
        return None if ratio is None else ratio >= .5

    @classmethod
    def partial_shape_ratio(cls, query, template):
        if not cls.partial_usable(query) or not cls.partial_usable(template):
            return None
        def aspect(m):
            x1, y1, x2, y2 = m["detector_bbox"]
            return (x2 - x1) / (y2 - y1)
        a, b = aspect(query), aspect(template)
        return min(a, b) / max(a, b)

    @staticmethod
    def partial_usable(metadata):
        """Conservative crop usability, not a claim of anatomical alignment."""
        m = metadata or {}
        if (m.get("partial_feature_source") != "osnet_torso"
                or m.get("quality_bbox_ok") is not True
                or m.get("bbox_quality_tier") != "strong"
                or m.get("is_fresh") is not True):
            return False
        try:
            x1,y1,x2,y2 = m["detector_bbox"]
            edges = float(m.get("detector_edge_touch_count", 0))
            return (all(math.isfinite(float(v)) for v in (x1,y1,x2,y2,edges))
                    and x2-x1 >= 40 and y2-y1 >= 80 and edges <= 2)
        except (KeyError, TypeError, ValueError):
            return False

    def prune_archive(self, features, metadata, *, keep_anchor):
        if self.watermark < 0:
            return 0
        keep = [i for i in range(len(features)) if (keep_anchor and i == 0)
                or (i < len(metadata) and timestamp(metadata[i]) is not None
                    and self.watermark - timestamp(metadata[i]) <= self.archive_sec)]
        removed = len(features) - len(keep)
        features[:] = [features[i] for i in keep]
        metadata[:] = [metadata[i] if i < len(metadata) else {} for i in keep]
        return removed
