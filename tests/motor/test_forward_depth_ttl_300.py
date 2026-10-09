"""300ms support must reach execution evidence and queue coalescing.

Fake serial driver only. The original per-grant deadline still wins when a
profile deliberately keeps the old 180/250ms deadline.
"""
import pytest

from test_forward_execution_anchor import execution_runtime, write_forward
from test_follow_queue_coalescing import following


@pytest.mark.parametrize("age", [.251, .275, .299])
def test_completed_packet_in_extended_band_remains_valid_evidence(monkeypatch, age):
    stamp = 10.-age
    rt, owner, driver, _, clock, _ = execution_runtime(monkeypatch, sample_timestamp=stamp)
    owner._depth30_linear_timing.depth_expires_at = stamp+.30
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: (
        owner._depth30_linear_snapshot if uid == 1 and
        0 <= (clock[0] if now is None else now)-stamp <= .30 else None)
    write_forward(rt, owner)
    assert driver.pairs == [(50, -30)]
    anchor = rt.forward_execution_anchor(1, stamp, clock[0])
    assert anchor is rt._forward_execution_anchor and anchor is not None
    proof = rt.recovery_forward_execution_anchor(1, clock[0])
    assert proof is not None and rt.forward_recovery_anchor_valid(1, proof, clock[0])
    # A recovery proof is not a lease renewal and never creates a write.
    assert owner._depth30_linear_timing.depth_expires_at == stamp+.30
    assert driver.pairs == [(50, -30)]


@pytest.mark.parametrize("ttl", [.18, .25, .30])
def test_execution_evidence_obeys_original_profile_deadline(monkeypatch, ttl):
    rt, owner, driver, _, clock, _ = execution_runtime(monkeypatch, sample_timestamp=9.84)
    owner._depth30_linear_timing.depth_expires_at = 9.84+ttl
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: (
        owner._depth30_linear_snapshot if uid == 1 and
        0 <= (clock[0] if now is None else now)-9.84 <= ttl else None)
    write_forward(rt, owner)
    clock[0] = 9.84+ttl-.01
    write_forward(rt, owner)
    anchor = rt._forward_execution_anchor
    # A recent receipt alone cannot make a profile's expired source live.
    clock[0] = 9.84+ttl+.001
    assert 0 < clock[0]-anchor.sent_at < .1
    assert rt.forward_execution_anchor(1, 9.84, clock[0]) is None
    assert driver.pairs == [(50, -30), (50, -30)]


@pytest.mark.parametrize("age,allowed", [(.251, True), (.275, True), (.299, True), (.301, False)])
def test_queue_refresh_uses_extended_supported_band_but_never_renews_it(monkeypatch, age, allowed):
    rt, owner, driver, symbols, clock, state = following(monkeypatch)
    state[2] = 10.30
    owner._fresh_depth_linear_snapshot = lambda uid, now=None: (
        ("forward", 24, uid, 10.) if 0 <= clock[0]-10. <= .30 else None)
    clock[0] = 10.+age
    assert rt.can_coalesce_follow_queue_refresh([symbols.forward]) is allowed
    assert driver.pairs == []
    assert owner._action_command_revision == 7
