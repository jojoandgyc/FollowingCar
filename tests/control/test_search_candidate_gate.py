#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from car_control_modular.search_candidate_gate import (
    CandidateObservation,
    SearchCandidateGate,
    SearchCandidateGateConfig,
)


def candidate(x1: float, score: float = 0.30) -> CandidateObservation:
    return CandidateObservation((x1, 80.0, x1 + 120.0, 400.0), score)


def main() -> int:
    source = (ROOT / "car_control_modular/search_candidate_gate.py").read_text(
        encoding="utf-8"
    )
    forbidden = ("action_runtime", "mssd_motor", "steering_pid", "rk_vision")
    present = [name for name in forbidden if name in source]
    if present:
        raise AssertionError(f"candidate gate crossed subsystem boundary: {present}")

    config = SearchCandidateGateConfig(hold_frames=3, probe_confirm_frames=2)
    gate = SearchCandidateGate(config)
    current = gate.select_current_candidate(
        width=640,
        height=480,
        probe_candidates=(candidate(20.0, 0.12), candidate(400.0, 0.20)),
    )
    if current.bbox != candidate(400.0, 0.20).bbox or current.score != 0.20:
        raise AssertionError(f"current-frame selection did not choose highest score: {current}")
    active_target = candidate(20.0, 0.31)
    wrong_high_score = candidate(400.0, 0.95)
    identity_preferred = gate.select_current_candidate(
        width=640,
        height=480,
        formal_candidates=(wrong_high_score, active_target),
        preferred_bbox=(25.0, 82.0, 145.0, 402.0),
    )
    if (
        identity_preferred.bbox != active_target.bbox
        or not identity_preferred.preferred_target_match
        or "active_target" not in identity_preferred.reason
    ):
        raise AssertionError(
            f"active identity did not outrank unrelated YOLO confidence: {identity_preferred}"
        )
    formal = candidate(20.0, 0.31)
    first = gate.update(
        timestamp=1.0,
        search_active=True,
        width=640,
        height=480,
        formal_candidates=(formal,),
    )
    second_bbox = candidate(50.0, 0.35)
    third_bbox = candidate(90.0, 0.40)
    second = gate.update(
        timestamp=1.1,
        search_active=True,
        width=640,
        height=480,
        formal_candidates=(second_bbox,),
    )
    third = gate.update(
        timestamp=1.2,
        search_active=True,
        width=640,
        height=480,
        formal_candidates=(third_bbox,),
    )
    if not (
        first.pause_rotation
        and first.entered
        and first.hold_frame == 1
        and second.pause_rotation
        and second.hold_frame == 2
        and third.pause_rotation
        and third.completed
        and third.hold_frame == 3
        and third.bbox == third_bbox.bbox
    ):
        raise AssertionError(f"formal bounded hold failed: {first}, {second}, {third}")
    try:
        first.pause_rotation = False
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("candidate decision must be immutable")

    blocked = gate.update(
        timestamp=1.3,
        search_active=True,
        width=640,
        height=480,
        formal_candidates=(candidate(250.0, 0.31),),
    )
    if blocked.pause_rotation or blocked.reason != "candidate_already_observed":
        raise AssertionError(f"persistent candidate retriggered indefinitely: {blocked}")
    moved_again = gate.update(
        timestamp=1.4,
        search_active=True,
        width=640,
        height=480,
        formal_candidates=(candidate(400.0, 0.31),),
    )
    if (
        moved_again.pause_rotation
        or moved_again.reason != "candidate_already_observed"
        or moved_again.bbox != candidate(400.0, 0.31).bbox
    ):
        raise AssertionError(f"moving candidate retriggered evidence hold: {moved_again}")
    for index in range(7):
        waiting = gate.update(
            timestamp=1.5 + index * 0.1,
            search_active=True,
            width=640,
            height=480,
        )
        if waiting.pause_rotation:
            raise AssertionError(f"missing-gap rearm must not stop rotation: {waiting}")
    still_blocked = gate.update(
        timestamp=2.2,
        search_active=True,
        width=640,
        height=480,
        formal_candidates=(formal,),
    )
    if still_blocked.pause_rotation or still_blocked.reason != "candidate_already_observed":
        raise AssertionError(f"short detector dropout rearmed candidate: {still_blocked}")
    for index in range(8):
        gate.update(
            timestamp=2.3 + index * 0.1,
            search_active=True,
            width=640,
            height=480,
        )
    retriggered = gate.update(
        timestamp=3.1,
        search_active=True,
        width=640,
        height=480,
        formal_candidates=(formal,),
    )
    if not retriggered.pause_rotation or not retriggered.entered:
        raise AssertionError(f"candidate did not retrigger after a real gap: {retriggered}")

    # A strong identity-prioritized bbox must supersede a stale blocked
    # detector candidate so the correct person can start a fresh hold.
    priority_gate = SearchCandidateGate(SearchCandidateGateConfig(hold_frames=2))
    wrong = candidate(420.0, 0.95)
    priority_gate.update(
        timestamp=5.0,
        search_active=True,
        width=640,
        height=480,
        formal_candidates=(wrong,),
    )
    priority_gate.update(
        timestamp=5.1,
        search_active=True,
        width=640,
        height=480,
        formal_candidates=(wrong,),
    )
    preferred = candidate(20.0, 0.35)
    preferred_result = priority_gate.update(
        timestamp=5.2,
        search_active=True,
        width=640,
        height=480,
        formal_candidates=(wrong, preferred),
        preferred_bbox=preferred.bbox,
    )
    if (
        not preferred_result.pause_rotation
        or not preferred_result.entered
        or preferred_result.bbox != preferred.bbox
    ):
        raise AssertionError(
            f"preferred identity bbox did not supersede blocked candidate: {preferred_result}"
        )

    # A tiny detector fragment must not consume the only observation window
    # when the next frame contains the same person's valid near-camera box.
    scale_gate = SearchCandidateGate(
        SearchCandidateGateConfig(hold_frames=2, blocked_rearm_area_ratio=3.0)
    )
    fragment = CandidateObservation((40.0, 180.0, 160.0, 500.0), 0.40)
    scale_gate.update(
        timestamp=5.0,
        search_active=True,
        width=640,
        height=480,
        formal_candidates=(fragment,),
    )
    scale_gate.update(
        timestamp=5.1,
        search_active=True,
        width=640,
        height=480,
        formal_candidates=(fragment,),
    )
    near_person = CandidateObservation((20.0, 20.0, 500.0, 470.0), 0.28)
    rearmed = scale_gate.update(
        timestamp=5.2,
        search_active=True,
        width=640,
        height=480,
        formal_candidates=(near_person,),
    )
    if not rearmed.pause_rotation or not rearmed.entered or rearmed.bbox != near_person.bbox:
        raise AssertionError(
            f"large valid candidate did not re-arm observation window: {rearmed}"
        )

    probe_gate = SearchCandidateGate(config)
    probe = candidate(250.0, 0.16)
    confirming = probe_gate.update(
        timestamp=4.0,
        search_active=True,
        width=640,
        height=480,
        probe_candidates=(probe,),
    )
    confirmed = probe_gate.update(
        timestamp=4.1,
        search_active=True,
        width=640,
        height=480,
        probe_candidates=(candidate(255.0, 0.17),),
    )
    if confirming.pause_rotation or confirming.probe_streak != 1:
        raise AssertionError(f"one weak frame must not stop search: {confirming}")
    if not confirmed.pause_rotation or not confirmed.entered or confirmed.source != "probe":
        raise AssertionError(f"consistent weak evidence did not start hold: {confirmed}")

    slow_gate = SearchCandidateGate(
        SearchCandidateGateConfig(
            hold_frames=10,
            max_hold_sec=0.30,
            probe_confirm_frames=2,
        )
    )
    slow_gate.update(
        timestamp=10.0,
        search_active=True,
        width=640,
        height=480,
        formal_candidates=(formal,),
    )
    timed_out = slow_gate.update(
        timestamp=10.31,
        search_active=True,
        width=640,
        height=480,
    )
    if timed_out.pause_rotation or not timed_out.completed or not timed_out.reason.endswith("_timeout"):
        raise AssertionError(f"time bound did not end a slow-frame observation: {timed_out}")

    reset = probe_gate.update(timestamp=4.2, search_active=False, width=640, height=480)
    if reset.pause_rotation or probe_gate.hold_active:
        raise AssertionError(f"inactive search did not clear evidence state: {reset}")

    scale_gate = SearchCandidateGate(config)
    far_edge_person = CandidateObservation((0.0, 155.0, 38.0, 298.0), 0.339)
    far_edge = scale_gate.update(
        timestamp=3.0,
        search_active=True,
        width=640,
        height=480,
        formal_candidates=(far_edge_person,),
    )
    if not far_edge.pause_rotation:
        raise AssertionError(f"1.8% far edge person was filtered out: {far_edge}")
    tiny_gate = SearchCandidateGate(config)
    tiny_fragment = CandidateObservation((0.0, 100.0, 20.0, 120.0), 0.90)
    tiny = tiny_gate.update(
        timestamp=3.0,
        search_active=True,
        width=640,
        height=480,
        formal_candidates=(tiny_fragment,),
    )
    if tiny.pause_rotation:
        raise AssertionError(f"20x20 fragment stopped search: {tiny}")

    print("search_candidate_gate_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
