"""Continuous measured-depth -> real PI -> grant -> fake wheel-stream checks.

Unlike motor-only fixtures, admission and speed here come from the production
controller and PersonTracker commit function.  No serial devices are opened.
"""
from dataclasses import replace

import pytest

import request_0513_modular as runtime

from test_authority_commit_integration import chain
from test_depth_authority_250 import authority, advance, decide_commit
from test_distance_tracking_response import setup
from test_lateral_zero_runtime import owner


@pytest.mark.parametrize("intermediate", ["no_sample", "duplicate", "none"])
def test_three_second_real_pi_stream_survives_ordinary_measurement_gaps(chain, intermediate):
    a = chain
    latest_stamp = a.owner._depth30_linear_timing.accepted_depth_timestamp
    distinct_samples = set()
    start = a.clock.now
    for tick in range(60):
        advance(a, start + tick*.05)
        if tick % 4 == 0:
            # Actual independently captured depth every 200 ms, with smaller
            # producer updates between samples.  Old evidence is never dated
            # again merely because the periodic writer needs another tick.
            current = a.frame(3.5 + .01*((tick//4) % 3), rpm=20.)
            _, actions, accepted = decide_commit(a, current)
            assert accepted and any(x.kind == "forward" and x.speed_percent > 0 for x in actions)
            latest_stamp = current.distance_state.sample_timestamp
            distinct_samples.add(latest_stamp)
        elif intermediate != "none":
            timing = a.owner._depth30_linear_timing
            integrator_stamp = a.controller._distance_pid._distance_pi._last_sample_ts
            if intermediate == "no_sample":
                missing = a.frame(None, rpm=20.)
                missing = replace(missing, distance_state=replace(
                    missing.distance_state, sample_timestamp=None,
                    source_detail="depth_detector_bbox_stale"))
                decide_commit(a, missing, fresh=False)
            else:
                decide_commit(a, a.frame(3.5, rpm=20., stamp=latest_stamp), fresh=False)
            assert a.owner._depth30_linear_timing.depth_expires_at == timing.depth_expires_at
            assert a.controller._distance_pid._distance_pi._last_sample_ts == integrator_stamp
        a.action._service_follow_wheels()
        assert a.backend.pairs[-1][0] > 0 and a.backend.pairs[-1][1] < 0
        assert not a.backend.stops
        live = a.owner._fresh_depth_linear_snapshot(1, quiet=True)
        assert live is not None and live[3] == latest_stamp
        assert a.owner._depth30_linear_timing.depth_expires_at == pytest.approx(latest_stamp+.25)
        assert not a.owner.motor_io_lock.locked()

    assert len(distinct_samples) == 15
    assert len(a.backend.pairs) >= 30
    assert not any(left == right == 0 for left, right, _ in a.backend.pairs)
    assert a.clock.now-start == pytest.approx(2.95)

    # Sensor silence after the successful stream still has a finite deadline.
    expires = a.owner._depth30_linear_timing.depth_expires_at
    advance(a, expires+.001)
    a.action._service_follow_wheels()
    assert a.backend.pairs[-1][:2] == (0, 0)


def test_real_pi_stream_cannot_treat_fresh_close_depth_as_an_ordinary_gap(chain):
    a = chain
    for _ in range(8):
        advance(a, a.clock.now+.05)
        _, _, accepted = decide_commit(a, a.frame(3.5, rpm=20.))
        assert accepted
        a.action._service_follow_wheels()
    assert all(left > 0 and right < 0 for left, right, _ in a.backend.pairs)
    before = len(a.backend.pairs)
    advance(a, a.clock.now+.05)
    decide_commit(a, a.frame(1., rpm=20.))
    a.action._service_follow_wheels()
    assert not any(left > 0 and right < 0 for left, right, _ in a.backend.pairs[before:])
    assert a.backend.stops or a.backend.pairs[-1][:2] == (0, 0)


def test_completed_stop_between_preview_and_publication_cannot_restore_old_grant(chain, monkeypatch):
    a = chain
    previous = a.owner._depth30_linear_snapshot
    reader = runtime.PersonTracker._fresh_depth_linear_snapshot
    stopped = []

    def stop_after_preview(self, *args, **kwargs):
        result = reader(self, *args, **kwargs)
        if kwargs.get("_candidate") is not None and not stopped:
            # Hardware STOP wins before its producer has published owner flags.
            # The candidate sample was already captured before that STOP.
            assert self._depth30_linear_snapshot is previous
            a.backend.send_stop("concurrent-stop-after-preview", mode="emergency")
            stopped.append(a.backend.stop_write_generation)
        return result

    monkeypatch.setattr(runtime.PersonTracker, "_fresh_depth_linear_snapshot", stop_after_preview)
    advance(a, a.clock.now+.05)
    _, actions, _ = decide_commit(a, a.frame(3.5, rpm=20.))
    assert stopped == [1]
    # The commit flag also covers a committed zero/revocation.  The actual
    # action and canonical grant, not that flag, define motion permission.
    assert not any(action.kind == "forward" and action.speed_percent > 0 for action in actions)
    assert a.owner._depth30_linear_snapshot is None
