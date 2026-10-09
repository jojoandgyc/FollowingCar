"""Short, untrusted search tracklets. Never assign UID, learn, or command motors."""
import math


class CandidateObservationMemory:
    def __init__(self, max_gap=.35, max_span=3., capacity=16):
        self.max_gap, self.max_span, self.capacity = max_gap, max_span, capacity
        self.rows = {}

    def clear(self):
        self.rows.clear()

    def discard(self, uid, track):
        self.rows.pop((uid, track), None)

    def observe(self, uid, track, metadata, geometry, qualified, continuity,
                *, preserve_search_entry=False):
        key = (uid, track)
        cap, ts = metadata.get('capture_frame_id'), metadata.get('capture_timestamp')
        if not qualified or metadata.get('is_fresh') is not True or geometry is None:
            self.discard(uid, track)
            return None
        try:
            if cap is None or not math.isfinite(ts) or not math.isfinite(geometry['integrated_yaw_deg']):
                raise ValueError('missing capture/yaw')
        except (TypeError, ValueError, KeyError):
            self.discard(uid, track)
            return None
        for k, row in list(self.rows.items()):
            if ts - row['last']['capture_timestamp'] > self.max_gap:
                self.rows.pop(k)
        previous = self.rows.get(key)
        if previous and (cap <= previous['last']['capture_frame_id']
                         or ts <= previous['last']['capture_timestamp']):
            # Do not manufacture a crossing by reprocessing a capture.
            return None
        side = metadata.get('search_direction_compatible')
        direction = metadata.get('search_direction')
        entering_search = bool(previous and preserve_search_entry
                               and previous['direction'] is None
                               and direction in ('left', 'right'))
        retained_confirmation = None
        continuous = bool(previous and continuity(previous['last']))
        if (previous and continuous and ts - previous['start_ts'] <= self.max_span
                and direction != previous['direction'] and not entering_search
                and previous.get('confirmed')
                and previous.get('late_confirmed_source') in ('strong', 'partial')
                and 0 < ts - previous['confirmed']['capture_timestamp'] <= self.max_gap):
            # Search direction is not identity. Carry only the already verified
            # source/geometry, never a pending count or cross-side permission.
            retained_confirmation = {k: previous[k] for k in ('confirmed', 'late_confirmed_source')}
        if previous and ((direction != previous['direction'] and not entering_search)
                         or ts - previous['start_ts'] > self.max_span or not continuous):
            previous = None
        if previous is None:
            row = dict(start_ts=ts, start_cap=cap, direction=direction,
                       from_compatible=side is True, crossed=False, count=1,
                       run_start_frame=geometry.get('frame_index'))
            if retained_confirmation:
                row.update(retained_confirmation)
        else:
            row = dict(previous)
            row['count'] += 1
            if geometry.get('frame_index', -1) != previous['last'].get('frame_index', -3) + 1:
                row['run_start_frame'] = geometry.get('frame_index')
            if entering_search:
                # Carry observation geometry, never invent a crossing that
                # happened before this search direction was selected.
                row.update(direction=direction, from_compatible=side is True, crossed=False)
            row['crossed'] = bool(row['from_compatible'] and side is False)
        row['last'] = dict(geometry)
        self.rows[key] = row
        while len(self.rows) > self.capacity:
            self.rows.pop(min(self.rows, key=lambda k: self.rows[k]['last']['capture_timestamp']))
        return row

    def crossing(self, uid, track, metadata):
        row = self.rows.get((uid, track))
        return bool(row and row['crossed'] and row['count'] >= 2
                    and row['last']['capture_frame_id'] == metadata.get('capture_frame_id')
                    and row['last']['capture_timestamp'] == metadata.get('capture_timestamp')
                    and metadata.get('is_fresh') is True)
